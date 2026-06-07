## 프로젝트 개요

SORT 기반 Multi-Object Tracking (MOT) 구현체.
ByteTrack 스타일의 2-stage association에 Adaptive Kalman Filter(R/Q),
Detection-level GMC(Global Motion Compensation),
이웃 기반 가림 보완(Neighbor Imputation)을 결합한 트래커.
조류 군집 추적을 주요 타겟으로, 외형 Re-ID 없이 운동 정보만으로 동작한다.

## 디렉토리 구조
sort/
├── CFG/
│   └── cfg.py                        # 전역 설정 (EasyDict)
├── src/
│   ├── tracker/
│   │   ├── init.py
│   │   ├── kalman_filter.py          # Adaptive R/Q Kalman Filter
│   │   ├── track.py                  # Track 객체 및 TrackState 관리
│   │   ├── association.py            # IoU cost matrix + lap.lapjv
│   │   ├── motion_statistics.py      # Detection-level GMC
│   │   ├── neighbor_imputation.py    # 이웃 기반 가림 위치 보완
│   │   ├── gating_network.py         # 2층 MLP 게이팅 네트워크
│   │   ├── motion_reid.py            # 운동 기반 Re-ID
│   │   └── tracker.py               # 메인 Tracker 클래스
│   ├── data/
│   │   ├── init.py
│   │   ├── bee24_loader.py           # BEE24 데이터 로더 + 마스킹
│   │   └── gt_loader.py             # GT 로더 (YOLO 형식)
│   ├── evaluation/
│   │   ├── init.py
│   │   └── mot_eval.py              # Precision/Recall/F1 평가
│   ├── visualization/
│   │   ├── init.py
│   │   └── vis_tracks.py            # 트랙 시각화 (mp4 저장)
│   ├── loss.py                       # OcclusionImputationLoss
│   └── trainer.py                    # Trainer 클래스
├── scripts/
│   ├── train.py                      # 학습 실행
│   ├── run_eval.py                   # 평가 실행
│   └── test_synthetic.py            # 합성 데이터 검증
├── checkpoint/                       # 학습 체크포인트 저장
└── CLAUDE.md

## 리팩토링 작업 (현재 → 목표)

### 현재 구조 → 목표 구조 매핑

| 현재 경로 | 목표 경로 | 비고 |
|---|---|---|
| tracker/*.py | src/tracker/*.py | 전체 이동 |
| data/bee24_loader.py | src/data/bee24_loader.py | 이동 |
| scripts/gt_loader.py | src/data/gt_loader.py | 구현 파일 → src로 |
| evaluation/mot_eval.py | src/evaluation/mot_eval.py | 이동 |
| visualization/vis_tracks.py | src/visualization/vis_tracks.py | 이동 |
| training/loss.py | src/loss.py | 구현 → src로 |
| training/trainer.py | src/trainer.py | 구현 → src로 |
| training/train.py | scripts/train.py | 실행 → scripts로 |
| scripts/run_eval.py | scripts/run_eval.py | 유지 |
| test_synthetic.py | scripts/test_synthetic.py | 실행 → scripts로 |
| checkpoints/ | checkpoint/ | 이름 변경 |

### 리팩토링 규칙
- 파일 이동 시 내용 수정 없이 경로만 변경
- import 경로를 새 구조에 맞게 일괄 수정
- 각 src/ 하위 디렉토리에 __init__.py 추가
- 기존 파일 삭제는 이동 완료 확인 후 수행

### import 경로 변경 규칙

```python
# 기존
from tracker.kalman_filter import KalmanFilter
from tracker.track import Track, TrackState
from tracker.association import iou_distance, linear_assignment
from tracker.motion_statistics import ...
from tracker.neighbor_imputation import NeighborImputation
from tracker.gating_network import GatingNetwork
from tracker.motion_reid import MotionReID

# 변경 후
from src.tracker.kalman_filter import KalmanFilter
from src.tracker.track import Track, TrackState
from src.tracker.association import iou_distance, linear_assignment
from src.tracker.motion_statistics import ...
from src.tracker.neighbor_imputation import NeighborImputation
from src.tracker.gating_network import GatingNetwork
from src.tracker.motion_reid import MotionReID

# 기존
from data.bee24_loader import BEE24Dataset

# 변경 후
from src.data.bee24_loader import BEE24Dataset
from src.data.gt_loader import GTLoader

# 기존
from training.loss import OcclusionImputationLoss
from training.trainer import Trainer

# 변경 후
from src.loss import OcclusionImputationLoss
from src.trainer import Trainer

# 기존
from evaluation.mot_eval import FrameEvaluator

# 변경 후
from src.evaluation.mot_eval import FrameEvaluator

# 기존
from visualization.vis_tracks import TrackVisualizer

# 변경 후
from src.visualization.vis_tracks import TrackVisualizer
```

### 완료 조건

```bash
# 1. 구조 확인
find . -name "*.py" | grep -E "^\./(src|scripts|CFG)" | sort

# 2. import 전체 확인
python -c "
from CFG.cfg import cfg
from src.tracker.tracker import Tracker
from src.tracker.track import Track, TrackState
from src.tracker.kalman_filter import KalmanFilter
from src.data.bee24_loader import BEE24Dataset
from src.data.gt_loader import GTLoader
from src.evaluation.mot_eval import FrameEvaluator
from src.visualization.vis_tracks import TrackVisualizer
from src.loss import OcclusionImputationLoss
from src.trainer import Trainer
print('All imports OK')
"

# 3. 기존 테스트 통과
python scripts/test_synthetic.py

# 4. 평가 실행 확인 (서버에서)
# python scripts/run_eval.py \
#     --data_dir /home/gpuadmin/Dataset/v2e_bird/금오도/data_1 \
#     --out_dir eval_output \
#     --device cuda
```

## 핵심 설계

### 상태 표현

- **Kalman 상태 벡터**: `[x, y, w, h, vx, vy, vw, vh]` (8차원, 유지)
- **관측 벡터**: `[x, y, w, h]` (4차원, center 좌표 기준)
- **입력 detection**: `Tensor(N, 5)` — `[x1, y1, x2, y2, score]`
- **예측 타깃**: 중심점 `(x, y)`만 — w, h는 날갯짓 노이즈로 제외
- **입력 표현**: 변위/속도 기반 (Δx, Δy) — PTZ 확장 대비

### Tracker.update() 처리 흐름
detections (N, 5)
→ score split: high_dets / low_dets
→ KalmanFilter.predict()
→ [GMC] median shift → apply_global_shift
→ [1차 association] tracked_tracks ↔ high_dets (IoU + Hungarian)
→ [2차 association] remain_tracks ↔ low_dets
→ [운동 Re-ID] OcclusionImputed ↔ unmatched high_dets
→ lost_tracks 관리
→ [이웃 보완] Lost/OcclusionImputed → NeighborImputation.impute()
→ return tracked_tracks

### 5개 모듈 구성

| 모듈 | 파일 | 딥러닝 | 역할 |
|---|---|---|---|
| ① 칼만 필터 | src/tracker/kalman_filter.py | ✗ | 정상 구간 백본 |
| ② 운동 Re-ID | src/tracker/motion_reid.py | ✓ | 재등장 ID 정합 |
| ③ 게이팅 네트워크 | src/tracker/gating_network.py | ✓ | 동적 가중치 |
| ④ GMC | src/tracker/motion_statistics.py | ✗ | 카메라 보정 (보류) |
| ⑤ 이웃 보완 | src/tracker/neighbor_imputation.py | ✓ | 가림 중 위치 생성 |

### Adaptive Kalman Filter

**Adaptive R** (`compute_r_scale`):
- `score_scale = clamp(1/score, 0.5, 5.0)`
- `area_scale = clamp(32²/area, 0.5, 5.0)`  ← 금오도 기준 수정 필요

**Adaptive Q** (`compute_q_scale`):
- `q_scale = clamp(1 + (||innovation_xy||/20) × score, 1.0, 10.0)`

### 게이팅 네트워크 (GatingNetwork)

입력 5가지 (정규화):
- kalman_uncertainty / 500.0
- detection_score (0~1)
- occluded_frames / 30.0
- neighbor_count / 8.0
- velocity_magnitude / 20.0

출력: (w_motion, w_neighbor) — softmax, 합=1
구조: Linear(5→16) → ReLU → Linear(16→2) → Softmax

### 이웃 보완 모듈 (NeighborImputation)
x_imputed = x_kalman + alpha * (v_neighbor - v_self)
alpha = GatingNetwork.w_neighbor
이웃 없으면 칼만 예측으로 폴백 (안전한 하한)

### 트랙 상태 전이
신규 detection        → Track(Tracked)
Tracked               → (매칭 실패) → Lost
Lost                  → (이웃 보완) → OcclusionImputed
OcclusionImputed      → (Re-ID 성공) → Tracked
Lost/OcclusionImputed → (max_lost 초과) → Removed

## Phase 1 학습 파이프라인 (MLP 게이팅 + k-NN)

### 목표
게이팅 네트워크(MLP)를 BEE24로 학습.
k-NN은 고정, MLP만 학습 대상.

### 검증 질문
"게이팅 학습 후 고정 alpha=0.5 대비 가림 구간 ADE가 10% 이상 감소하는가?"
→ YES → Phase 2 (GAT) 진행
→ NO  → 이웃 집계 방식 재검토

### 현재 학습 결과 (진행 중)
Kalman ADE:  21.81px
게이팅 ADE:  18~21px
개선폭:      +1~4px (6~17%)
10 에폭 연속 칼만 대비 개선 확인 ✓
Phase 1 가설 확인 중

### 실행 명령 (서버)
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --data_root /home/gpuadmin/Dataset/BEE24 \
    --out_dir checkpoint/phase1 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --device cuda
```

## 데이터셋

### 금오도 (주력)
- 경로: /home/gpuadmin/Dataset/v2e_bird/금오도/data_1, data_2
- 총 프레임: 12,837 / 총 객체: 14,080
- 프레임당 평균 객체: 1.1 (max=8)
- Width 중앙값: ~24px / Height 중앙값: ~18px
- Small(<32×32) 약 75%
- track_id 없음 → Precision/Recall/F1 평가

### BEE24 (학습용)
- 경로: /home/gpuadmin/Dataset/BEE24
- train: 31개 시퀀스 / test: 별도
- fps=25, 해상도=950×590
- gt.txt: frame_id, track_id, x, y, w, h, 1, 1, 1 (9열)

## 주요 파라미터

| 파라미터 | 값 | 역할 |
|---|---|---|
| high_thresh | 0.5 | 1차 association 임계값 |
| low_thresh | 0.1 | 2차 association 임계값 |
| match_thresh | 0.7 | IoU 매칭 임계값 |
| max_lost | 30 | Lost → Removed 허용 프레임 |
| imputation.k | 7 | 이웃 보완 이웃 수 |
| imputation.alpha | 0.5 | 이웃 속도 보정 강도 (fallback) |
| motion_reid.max_reid_dist | 50.0 | Re-ID 최대 허용 거리 (픽셀) |

## 의존성

```python
torch      # 텐서 연산, GPU 지원
numpy      # cost matrix 변환
lap        # lapjv Hungarian matching
easydict   # cfg 관리
opencv-python  # 시각화
Pillow     # 이미지 크기 확인
```

## 주의사항

- `Track._count`는 클래스 변수 — 테스트 간 격리 시 `Track.reset_id()` 호출
- `cfg.device = 'cuda'` 기본값 — CPU 환경에서는 CFG/cfg.py 수정
- `association.py`의 `iou_distance`는 `.cpu().numpy()` 반환 (lap 요구)
- `KalmanFilter.update`에서 Q scaling 후 covariance in-place 대입
- `track.update()` 호출 시 `clear_occlusion()` 자동 실행
- `GatingNetwork.get_weights()`에서 이웃 없으면 (1.0, 0.0) 강제 반환
- `MotionReID.match()`는 `imputed_positions` 있는 트랙만 후보 사용
- 수치 안정성: Cholesky 분해 전 jitter=1e-4 추가, 대칭성 강제
