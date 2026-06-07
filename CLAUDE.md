# CLAUDE.md

## 프로젝트 개요

SORT 기반 Multi-Object Tracking (MOT) 구현체. ByteTrack 스타일의 2-stage association에 Adaptive Kalman Filter(R/Q), Detection-level GMC(Global Motion Compensation), 이웃 기반 가림 보완(Neighbor Imputation)을 결합한 트래커. 조류 군집 추적을 주요 타겟으로, 외형 Re-ID 없이 운동 정보만으로 동작한다.

## 디렉토리 구조

```
sort/
├── CFG/
│   └── cfg.py                       # 전역 설정 (cfg.device)
├── tracker/
│   ├── kalman_filter.py             # Adaptive R/Q Kalman Filter + 불확실성 인터페이스
│   ├── track.py                     # Track 객체 및 TrackState 관리
│   ├── association.py               # IoU cost matrix + lap.lapjv Hungarian matching
│   ├── motion_statistics.py         # Detection-level GMC
│   ├── neighbor_imputation.py       # 이웃 기반 가림 위치 보완 (GatingNetwork 연동)
│   ├── gating_network.py            # 2층 MLP 게이팅 네트워크 (w_motion / w_neighbor)
│   ├── motion_reid.py               # 운동 기반 Re-ID (imputed_positions 끝점 매칭)
│   └── tracker.py                   # 메인 Tracker 클래스
├── detector/                        # 미구현 (빈 디렉토리)
├── evaluation/                      # 미구현 (빈 디렉토리)
├── ext/                             # 미구현 (빈 디렉토리)
├── scripts/                         # 미구현 (빈 디렉토리)
├── visualization/                   # 미구현 (빈 디렉토리)
└── test_synthetic.py                # Adaptive Q + 통합 시나리오 검증 스크립트
```

## 핵심 설계

### 상태 표현

- **Kalman 상태 벡터**: `[x, y, w, h, vx, vy, vw, vh]` (8차원)
- **관측 벡터**: `[x, y, w, h]` (4차원, center 좌표 기준)
- **입력 detection**: `Tensor (N, 5)` — `[x1, y1, x2, y2, score]` (top-left/bottom-right)

### Tracker.update() 처리 흐름

```
detections (N, 5)
  → score split: high_dets (≥ high_thresh) / low_dets (≥ low_thresh)
  → KalmanFilter.predict() for all tracked_tracks
  → [GMC] tentative match → median shift → apply_global_shift
  → [1차 association] tracked_tracks ↔ high_dets (IoU + Hungarian)
      matched        → track.update() → Tracked
      unmatched track → 2차 association 대기
      unmatched high det → 신규 Track 생성
  → [2차 association] remain_tracks ↔ low_dets
      matched   → track.update() → Tracked
      unmatched → mark_lost → Lost
  → [운동 Re-ID] OcclusionImputed 트랙 ↔ 매칭 안 된 high_dets
      매칭 성공 → track.update() + clear_occlusion() → Tracked
      매칭 실패 → 기존 lost 처리 유지
  → lost_tracks 관리: max_lost 초과 시 mark_removed
  → [이웃 보완] Lost/OcclusionImputed 트랙에 NeighborImputation.impute()
      alpha = GatingNetwork.w_neighbor (이웃 없으면 칼만 폴백)
  → return tracked_tracks (TrackState.Tracked)
```

### Adaptive Kalman Filter

**Adaptive R** (`compute_r_scale`): score 낮음 + area 작음 → R 증가 → 측정 신뢰도 하락
- `score_scale = clamp(1 / score, 0.5, 5.0)`
- `area_scale  = clamp(32² / area, 0.5, 5.0)`  ← ref_area = 32×32

**Adaptive Q** (`compute_q_scale`): innovation 클수록 Q 증가; score 낮으면 팽창 억제
- `q_scale = clamp(1 + (||innovation_xy|| / 20) × score, 1.0, 10.0)`

**update 내부 순서**:
1. `project(score, area)` → Adaptive R 적용
2. innovation 계산
3. `compute_q_scale(innovation, score)` → covariance 팽창
4. 팽창된 covariance로 재투영
5. Cholesky로 Kalman gain 계산 → 상태 보정

### GMC (Detection-level)

픽셀/광학 흐름 없이 매칭된 bbox center shift의 **median**으로 카메라 움직임 추정.
- `min_pairs`(기본 2) 미만이면 GMC 비활성화
- `max_shift`(기본 50.0)로 이상치 클램핑

### Neighbor Imputation

가림 발생 시 이웃 트랙들의 운동 정보로 Lost 트랙 위치를 보완.
- `get_neighbors()`: 유클리드 거리 기준 k개(기본 k=7) 이웃 탐색
- `compute_neighbor_velocity()`: 이웃들의 평균 `(vx, vy)` 계산
- `impute(gating=...)`: `x_imputed = x_kalman + alpha * (v_neighbor - v_self)`
  - alpha = GatingNetwork.w_neighbor (gating 미제공 시 고정값 0.5)
  - 이웃이 없으면 칼만 예측값으로 자동 폴백
- 보완된 위치는 `track.imputed_positions`에 누적, 재연결 시 `clear_occlusion()` 호출

### Gating Network

가림 상태에 따라 칼만 운동과 이웃 보완의 신뢰 가중치를 동적으로 결정.
- 입력 5가지: kalman_uncertainty, detection_score, occluded_frames, neighbor_count, velocity_magnitude
- 출력: `(w_motion, w_neighbor)` — softmax 정규화, 합 = 1
- 구조: 2층 MLP (Linear → ReLU → Linear → Softmax), 입력 5 / 은닉 16 / 출력 2
- 휴리스틱 bias 초기화: 가림 없음 시 w_motion≈0.9 / 이웃 없으면 강제로 w_motion=1.0
- 추후 end-to-end 학습으로 교체 가능한 구조

### Motion Re-ID

가림 후 재등장한 검출을 보완 궤적 끝점으로 재식별 (외형 없이 운동 패턴 기반).
- `match(lost_tracks, new_detections)`:
  1. OcclusionImputed 트랙의 `imputed_positions` 마지막 위치를 끝점으로 사용
  2. 끝점 ↔ detection 중심점 유클리드 거리 행렬 계산
  3. Hungarian matching (`lap.lapjv`)
  4. `max_reid_dist`(기본 50.0 픽셀) 초과 매칭 거부
- 반환: `[(track_idx, det_idx), ...]`

## 트랙 상태 전이

```
신규 detection        → Track(Tracked)
Tracked               → (매칭 실패)              → Lost
Lost                  → (이웃 보완 적용 중)      → OcclusionImputed
OcclusionImputed      → (운동 Re-ID 매칭 성공)  → Tracked (clear_occlusion)
Lost/OcclusionImputed → (1·2차 association 성공) → Tracked (clear_occlusion)
Lost/OcclusionImputed → (max_lost 초과)          → Removed
```

## 주요 파라미터 (Tracker 기본값)

| 파라미터 | 기본값 | 역할 |
|---|---|---|
| `high_thresh` | 0.5 | 1차 association detection score 임계값 |
| `low_thresh` | 0.1 | 2차 association detection score 임계값 |
| `match_thresh` | 0.7 | IoU cost 매칭 임계값 (1-IoU 기준) |
| `max_lost` | 30 | Lost → Removed까지 허용 프레임 수 |
| `use_gmc` | True | GMC 활성화 |
| `gmc_min_pairs` | 2 | GMC 최소 매칭 쌍 수 |
| `gmc_max_shift` | 50.0 | GMC shift 클램핑 (픽셀) |
| `imputation.k` | 7 | 이웃 보완에 사용할 이웃 수 |
| `imputation.alpha` | 0.5 | 이웃 속도 보정 강도 (gating 미사용 시 fallback) |
| `motion_reid.max_reid_dist` | 50.0 | 운동 Re-ID 최대 허용 거리 (픽셀) |

## 의존성

```python
torch       # 텐서 연산, GPU 지원
numpy       # cost matrix 변환
lap         # lapjv Hungarian matching (lap.lapjv)
easydict    # cfg 관리
```

## 실행 방법

```bash
# Adaptive Q 검증 + 통합 시나리오 3종 실행
python test_synthetic.py

# pytest (테스트 파일 있을 경우)
pytest
```

## 미구현 영역

`detector/`, `evaluation/`, `ext/`, `scripts/`, `visualization/` 디렉토리는 현재 비어 있음.
- detector: 검출기 연동 (YOLO 등)
- evaluation: MOT 메트릭 (HOTA, MOTA 등)
- visualization: 트랙 시각화
- scripts: 데이터셋별 실행 스크립트

## 주의사항

- `Track._count`는 클래스 변수 — 테스트 간 격리가 필요하면 `Track.reset_id()` 호출
- `cfg.device = 'cuda'` 하드코딩 — CPU 환경에서는 `CFG/cfg.py` 수정 필요
- `association.py`의 `iou_distance`는 GPU 계산 후 `.cpu().numpy()` 반환 (lap이 numpy 요구)
- `KalmanFilter.update`에서 Q scaling 적용 후 covariance를 **in-place 대입** — 원본 covariance는 변경됨
- `track.update()` 호출 시 `clear_occlusion()`이 자동 실행 — 재연결 시 별도 호출 불필요
- `GatingNetwork.get_weights()`에서 이웃이 없으면 (w_motion, w_neighbor) = (1.0, 0.0) 강제 반환
- `MotionReID.match()`는 `imputed_positions`가 있는 트랙만 후보로 사용

## [즉시 수정] tracker.py 버그 2개

### 버그 1 — 가림 첫 프레임 보완 누락
위치: update() 내 7.5번 블록

현재 코드:
    for track in lost_tracks:
        if track.state == TrackState.Lost:
            track.start_occlusion()
        if track.state == TrackState.OcclusionImputed:
            imputed_pos = self.imputation.impute(...)
            track.add_imputed_position(imputed_pos)

문제: start_occlusion() 호출 후 상태가 OcclusionImputed로 바뀌지만
      같은 루프의 두 번째 if는 이미 평가가 끝나 실행 안 됨.
      → 가림 첫 프레임에 보완이 실행되지 않음.

수정:
    for track in lost_tracks:
        if track.state == TrackState.Lost:
            track.start_occlusion()
        # Lost/OcclusionImputed 모두 보완 실행
        imputed_pos = self.imputation.impute(
            track, activated_tracks, self.gating
        )
        track.add_imputed_position(imputed_pos)

### 버그 2 — Re-ID 소비 detection 중복 트랙 생성
위치: update() 내 6번 블록 (unmatched high-score detections → new tracks)

현재 코드 (불안정한 사후 필터링):
    for d_idx in unmatched_high_dets:
        det = high_det_tracks[d_idx]
        activated_tracks.append(det)
    ...
    activated_tracks = [
        t for t in activated_tracks
        if not (t in unmatched_new_dets and ...)
    ]

수정 (6번 블록에서 처음부터 건너뜀):
    for d_idx in unmatched_high_dets:
        if d_idx in reid_consumed_det_indices:
            continue  # Re-ID로 소비된 detection 건너뜀
        det = high_det_tracks[d_idx]
        activated_tracks.append(det)

    # 사후 필터링 블록 전체 삭제

### 완료 조건
- python test_synthetic.py 기존 테스트 모두 통과
- 버그 2개 수정 후 test_synthetic.py에 아래 시나리오 추가:
  시나리오: 트랙 3개 중 1개 가림(10프레임) →
  가림 첫 프레임부터 imputed_positions가 쌓이는지 assert로 확인

  

## [신규 구현] 평가 파이프라인 A+C

### 주의사항
- 코드 작성만 완료 (실행은 서버에서 별도 진행)
- 데이터 경로 하드코딩 없음 — 전부 argparse로 수신
- 완료 조건: 코드 작성 + import 오류 없음 확인

### scripts/gt_loader.py 신규 생성

GTLoader 클래스:

__init__(data_dir: str, img_ext=".jpg"):
    - data_dir: 이미지와 filtered/가 있는 폴더 경로
    - 이미지 파일 목록 정렬 로드
    - filtered/*.txt 경로 매핑 (이미지 stem 기준)

__len__(): 총 프레임 수 반환

__getitem__(idx) -> tuple[Path, Tensor]:
    - 반환: (img_path, gt_boxes)
    - gt_boxes: Tensor(N, 5) [x1, y1, x2, y2, score=1.0]
    - YOLO 정규화 좌표 → 픽셀 절대 좌표 변환
    - 이미지 크기: PIL로 W, H 추출
    - 객체 없는 프레임: Tensor(0, 5) 반환

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### evaluation/mot_eval.py 신규 생성

FrameEvaluator 클래스:

__init__(iou_thresh=0.5)

match_frame(gt_boxes, pred_boxes) -> dict:
    - gt_boxes:   Tensor(N, 4) [x1,y1,x2,y2]
    - pred_boxes: Tensor(M, 4) [x1,y1,x2,y2]
    - IoU 행렬 → Hungarian matching (lap.lapjv)
    - iou_thresh 미만 매칭 거부
    - 반환: {tp, fp, fn, matched_ious}

compute_metrics(results: list[dict]) -> dict:
    - 전체 TP/FP/FN 합산
    - 반환: {precision, recall, f1, mean_iou}

print_report(metrics: dict): 콘솔 출력

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### visualization/vis_tracks.py 신규 생성

TrackVisualizer 클래스:

__init__(fps=30, out_path="output.mp4"):
    - cv2.VideoWriter 초기화 (fourcc: mp4v)
    - 첫 add_frame 호출 시 해상도 자동 결정

add_frame(img_path, gt_boxes, tracked_tracks, lost_tracks):
    오버레이:
    - GT 박스:                초록색(0,255,0) 실선 두께 1
    - Tracked 트랙:           파란색(255,100,0) 실선 두께 2
                              좌상단 track_id 텍스트
    - OcclusionImputed 트랙:  빨간색(0,0,255) 점선 두께 2
                              "IMP:{id}" 텍스트
    - imputed_positions 궤적: 노란색(0,255,255) 점 연결

release(): VideoWriter 해제

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### scripts/run_eval.py 신규 생성

argparse 인자:
    --data_dir    필수, 데이터 폴더 경로
    --iou_thresh  기본 0.5
    --out_dir     기본 eval_output/
    --save_video  플래그, 시각화 영상 저장
    --max_frames  기본 None (전체)
    --device      기본 cuda

실행 흐름:
    1. GTLoader 초기화
    2. Tracker 초기화 (device 인자 반영)
    3. FrameEvaluator, TrackVisualizer 초기화
    4. 프레임별 루프:
        a. gt_boxes → Tracker.update(gt_boxes)
        b. FrameEvaluator.match_frame(gt, pred)
        c. save_video → TrackVisualizer.add_frame()
        d. 100프레임마다 중간 지표 콘솔 출력
    5. compute_metrics → print_report
    6. out_dir/metrics.txt 저장
    7. save_video → visualizer.release()

out_dir/
├── metrics.txt
└── tracks.mp4  (--save_video 시)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 완료 조건
python -c "
from scripts.gt_loader import GTLoader
from evaluation.mot_eval import FrameEvaluator
from visualization.vis_tracks import TrackVisualizer
from tracker.tracker import Tracker
print('All imports OK')
"
위 명령이 오류 없이 통과해야 함.

### 유지할 것
- tracker/ 코드 수정 없음
- test_synthetic.py 통과 유지