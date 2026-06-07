import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.tracker.kalman_filter import KalmanFilter


def main():
    kf = KalmanFilter()
    device = kf.device

    # 초기 상태
    init_measurement = torch.tensor([100.0, 100.0, 40.0, 30.0], device=device)
    mean, covariance = kf.initiate(init_measurement)

    mean = mean.unsqueeze(0)
    covariance = covariance.unsqueeze(0)

    cases = [
        # name, measurement, score, area
        (
            "small residual / high score",
            torch.tensor([[103.0, 102.0, 40.0, 30.0]], device=device),
            0.90,
            40.0 * 30.0,
        ),
        (
            "large residual / high score",
            torch.tensor([[140.0, 130.0, 40.0, 30.0]], device=device),
            0.90,
            40.0 * 30.0,
        ),
        (
            "large residual / low score",
            torch.tensor([[140.0, 130.0, 40.0, 30.0]], device=device),
            0.20,
            40.0 * 30.0,
        ),
        (
            "very large residual / high score",
            torch.tensor([[200.0, 180.0, 40.0, 30.0]], device=device),
            0.90,
            40.0 * 30.0,
        ),
    ]

    print("Adaptive Q Test")
    print("=" * 70)

    for name, measurement, score, area in cases:
        test_mean = mean.clone()
        test_cov = covariance.clone()

        score_t = torch.tensor([score], device=device)
        area_t = torch.tensor([area], device=device)

        projected_mean, _ = kf.project(
            test_mean,
            test_cov,
            score=score_t,
            area=area_t
        )

        innovation = measurement - projected_mean

        q_scale = kf.compute_q_scale(
            innovation,
            score=score_t
        )

        updated_mean, updated_cov = kf.update(
            test_mean,
            test_cov,
            measurement,
            score=score_t,
            area=area_t
        )

        print(f"\nCase: {name}")
        print(f"score={score:.2f}, area={area:.1f}")
        print(
            "Innovation xy = "
            f"({innovation[0, 0].item():.2f}, {innovation[0, 1].item():.2f})"
        )
        print(f"Q scale={q_scale.item():.3f}")
        print(
            "Before mean xy = "
            f"({test_mean[0, 0].item():.2f}, {test_mean[0, 1].item():.2f})"
        )
        print(
            "Measurement xy = "
            f"({measurement[0, 0].item():.2f}, {measurement[0, 1].item():.2f})"
        )
        print(
            "After mean xy  = "
            f"({updated_mean[0, 0].item():.2f}, {updated_mean[0, 1].item():.2f})"
        )


def make_det(x, y, w=30.0, h=30.0, score=0.9, device='cuda'):
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    return torch.tensor([[x1, y1, x2, y2, score]], device=device)


def test_scenario1():
    """시나리오 1 - 정상 추적: 트랙 3개, 가림 없음, w_motion 높아야 함"""
    from src.tracker.tracker import Tracker
    from src.tracker.track import Track
    Track.reset_id()

    tracker = Tracker(high_thresh=0.5, max_lost=5)
    device = tracker.kf.device

    positions = [(100, 100), (200, 200), (300, 300)]

    for frame in range(5):
        dets = torch.cat([
            make_det(x + frame * 2, y + frame * 2, device=device)
            for x, y in positions
        ])
        active = tracker.update(dets)

    assert len(active) == 3, f"시나리오1: active tracks={len(active)}, expected 3"

    for track in active:
        w_motion, w_neighbor = tracker.gating.get_weights(track, active)
        assert w_motion > 0.7, f"시나리오1: w_motion={w_motion:.3f} 낮음 (가림 없음)"

    print("시나리오 1 통과: 정상 추적, w_motion 높음")


def test_scenario2():
    """시나리오 2 - 새-새 가림: 1개 가림 10프레임, 재등장 후 ID 유지"""
    from src.tracker.tracker import Tracker
    from src.tracker.track import Track
    Track.reset_id()

    tracker = Tracker(high_thresh=0.5, max_lost=30)
    device = tracker.kf.device

    base_positions = [(100 + i * 60, 100) for i in range(5)]

    # 5프레임 정상 추적
    for frame in range(5):
        dets = torch.cat([
            make_det(x + frame * 2, y, device=device)
            for x, y in base_positions
        ])
        tracker.update(dets)

    # 트랙 2 (index=1) ID 기록
    tracked_ids = {t.track_id for t in tracker.tracked_tracks}
    assert len(tracked_ids) == 5

    # 트랙 2 제외하고 10프레임 진행
    occluded_idx = 1
    for frame in range(5, 15):
        dets = torch.cat([
            make_det(base_positions[i][0] + frame * 2, base_positions[i][1], device=device)
            for i in range(5) if i != occluded_idx
        ])
        tracker.update(dets)

    # 가림 트랙에 imputed_positions가 쌓였는지 확인
    lost = tracker.lost_tracks
    assert len(lost) > 0, "시나리오2: lost_tracks가 비어 있음"
    imputed_track = lost[0]
    assert len(imputed_track.imputed_positions) > 0, "시나리오2: imputed_positions 없음"

    # 가림 중 w_neighbor 확인
    w_motion, w_neighbor = tracker.gating.get_weights(imputed_track, tracker.tracked_tracks)
    if len(tracker.tracked_tracks) > 0:
        assert w_neighbor > 0.0, f"시나리오2: w_neighbor={w_neighbor:.3f}"

    print(f"시나리오 2 통과: 가림 중 imputed_positions={len(imputed_track.imputed_positions)}, w_neighbor={w_neighbor:.3f}")


def test_scenario3():
    """시나리오 3 - 배경 가림(이웃 없음): 트랙 1개, 가림 10프레임, 칼만 폴백"""
    from src.tracker.tracker import Tracker
    from src.tracker.track import Track
    Track.reset_id()

    tracker = Tracker(high_thresh=0.5, max_lost=30)
    device = tracker.kf.device

    # 5프레임 정상 추적
    for frame in range(5):
        dets = make_det(100 + frame * 2, 100, device=device)
        tracker.update(dets)

    assert len(tracker.tracked_tracks) == 1

    # 10프레임 가림 (빈 detection)
    for _ in range(10):
        tracker.update(torch.empty((0, 5), device=device))

    assert len(tracker.lost_tracks) == 1
    lost_track = tracker.lost_tracks[0]

    # 이웃 없으므로 w_motion=1.0 이어야 함
    w_motion, w_neighbor = tracker.gating.get_weights(lost_track, tracker.tracked_tracks)
    assert w_motion == 1.0 and w_neighbor == 0.0, \
        f"시나리오3: w_motion={w_motion:.3f}, w_neighbor={w_neighbor:.3f}"

    # 칼만 폴백 확인: imputed_positions는 mean[:2]와 거의 같아야 함
    assert len(lost_track.imputed_positions) > 0
    last_imp = lost_track.imputed_positions[-1]
    kalman_pos = lost_track.mean[:2]
    diff = (last_imp - kalman_pos).norm().item()
    assert diff < 1e-4, f"시나리오3: 칼만 폴백 오차 {diff:.6f}"

    print(f"시나리오 3 통과: 이웃 없음, w_motion={w_motion:.3f}, 칼만 폴백 오차={diff:.6f}")


def test_scenario4():
    """시나리오 4 - 가림 첫 프레임부터 imputed_positions 누적 확인"""
    from src.tracker.tracker import Tracker
    from src.tracker.track import Track
    Track.reset_id()

    tracker = Tracker(high_thresh=0.5, max_lost=30)
    device = tracker.kf.device

    positions = [(100, 100), (200, 200), (300, 300)]

    # 5프레임 정상 추적 (warm-up)
    for frame in range(5):
        dets = torch.cat([
            make_det(x + frame * 2, y + frame * 2, device=device)
            for x, y in positions
        ])
        tracker.update(dets)

    assert len(tracker.tracked_tracks) == 3, \
        f"시나리오4 warm-up: tracked={len(tracker.tracked_tracks)}, expected 3"

    # positions[1] 트랙 제외하고 10프레임 진행, 매 프레임 imputed_positions 누적 확인
    for step in range(1, 11):
        frame = 5 + step - 1
        dets = torch.cat([
            make_det(positions[0][0] + frame * 2, positions[0][1] + frame * 2, device=device),
            make_det(positions[2][0] + frame * 2, positions[2][1] + frame * 2, device=device),
        ])
        tracker.update(dets)

        lost = tracker.lost_tracks
        assert len(lost) >= 1, f"시나리오4: step={step}, lost_tracks 비어 있음"

        imputed_track = next(
            (t for t in lost if t.state.name == 'OcclusionImputed'),
            lost[0]
        )
        assert len(imputed_track.imputed_positions) == step, (
            f"시나리오4: step={step}, imputed_positions={len(imputed_track.imputed_positions)}, "
            f"expected={step}"
        )

    final_count = len(tracker.lost_tracks[0].imputed_positions)
    print(f"시나리오 4 통과: 10프레임 가림, 매 프레임 누적 확인 (최종 imputed={final_count}개)")


def run_integration_tests():
    print("\n" + "=" * 70)
    print("Integration Tests")
    print("=" * 70)
    test_scenario1()
    test_scenario2()
    test_scenario3()
    test_scenario4()
    print("\n모든 통합 테스트 통과")


if __name__ == "__main__":
    main()
    run_integration_tests()