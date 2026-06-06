import torch

from tracker.kalman_filter import KalmanFilter


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


if __name__ == "__main__":
    main()