# CLAUDE.md

## 프로젝트 개요

SORT 기반 Multi-Object Tracking (MOT) 구현체.
ByteTrack 스타일의 2-stage association에 Adaptive Kalman Filter(R/Q),
Detection-level GMC(Global Motion Compensation),
운동 기반 Re-ID를 결합한 트래커.
조류 추적을 주요 타겟으로, 외형 Re-ID 없이 운동 정보만으로 동작한다.

> **이웃 기반 가림 보완(Neighbor Imputation), 게이팅 네트워크(GatingNetwork),
> BEE24 학습 파이프라인은 본 버전에서 제거되었다.**

---

## 디렉토리 구조

```
sort/
├── CFG/
│   └── cfg.py                        # 전역 설정 (EasyDict)
├── src/
│   ├── tracker/
│   │   ├── __init__.py
│   │   ├── kalman_filter.py          # Adaptive R/Q Kalman Filter
│   │   ├── track.py                  # Track 객체 및 TrackState 관리
│   │   ├── association.py            # IoU cost matrix + lap.lapjv
│   │   ├── motion_statistics.py      # Detection-level GMC
│   │   ├── motion_reid.py            # 운동 기반 Re-ID
│   │   └── tracker.py               # 메인 Tracker 클래스
│   ├── data/
│   │   ├── __init__.py
│   │   └── gt_loader.py             # GT 로더 (YOLO 형식)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── mot_eval.py              # Precision/Recall/F1 평가
│   └── visualization/
│       ├── __init__.py
│       └── vis_tracks.py            # 트랙 시각화 (mp4 저장)
├── scripts/
│   └── run_eval.py                   # 평가 + 시각화 실행
└── CLAUDE.md
```

---

## 핵심 설계

### 상태 표현

- **Kalman 상태 벡터**: `[x, y, w, h, vx, vy, vw, vh]` (8차원)
- **관측 벡터**: `[x, y, w, h]` (4차원, center 좌표 기준)
- **입력 detection**: `Tensor(N, 5)` — `[x1, y1, x2, y2, score]`
- **예측 타깃**: 중심점 `(x, y)`만 — w, h는 날갯짓 노이즈로 제외
- **입력 표현**: 변위/속도 기반 (Δx, Δy)

### Tracker.update() 처리 흐름

```
detections (N, 5)
→ score split: high_dets / low_dets
→ KalmanFilter.predict()
→ [GMC] median shift → apply_global_shift
→ [1차 association] tracked_tracks ↔ high_dets (IoU + Hungarian)
→ [2차 association] remain_tracks ↔ low_dets
→ [운동 Re-ID] lost_tracks ↔ unmatched high_dets
→ lost_tracks 관리
→ return tracked_tracks
```

### 모듈 구성

| 모듈 | 파일 | 딥러닝 | 역할 |
|---|---|---|---|
| ① 칼만 필터 | src/tracker/kalman_filter.py | ✗ | 위치 예측 백본 |
| ② 운동 Re-ID | src/tracker/motion_reid.py | ✓ | 재등장 ID 정합 |
| ③ GMC | src/tracker/motion_statistics.py | ✗ | 카메라 모션 보정 |

### Adaptive Kalman Filter

**Adaptive R** (`compute_r_scale`):
- `score_scale = clamp(1/score, 0.5, 5.0)`
- `area_scale = clamp(ref_area/area, 0.5, 5.0)`

**Adaptive Q** (`compute_q_scale`):
- `q_scale = clamp(1 + (||innovation_xy||/20) × score, 1.0, 10.0)`

### 트랙 상태 전이

```
신규 detection        → Track(Tracked)
Tracked               → (매칭 실패) → Lost
Lost                  → (Re-ID 성공) → Tracked
Lost                  → (max_lost 초과) → Removed
```

---

## 평가 방식

GT가 있는 금오도 데이터를 기준으로 GT detection을 직접 입력으로 사용한다.
탐지기 없이 순수 추적 성능만 측정한다.

### 평가 메트릭
- **Precision**: 예측 트랙 중 GT와 매칭된 비율
- **Recall**: GT 트랙 중 예측과 매칭된 비율
- **F1**: Precision과 Recall의 조화 평균
- **IoU**: 매칭된 bbox 쌍의 평균 IoU

### 매칭 기준
- IoU 기반 Hungarian matching
- IoU threshold: `cfg.eval.iou_threshold` (기본 0.5)

---

## 주요 파라미터 (CFG/cfg.py)

| 파라미터 | 값 | 역할 |
|---|---|---|
| high_thresh | 0.5 | 1차 association 임계값 |
| low_thresh | 0.1 | 2차 association 임계값 |
| match_thresh | 0.7 | IoU 매칭 임계값 |
| max_lost | 30 | Lost → Removed 허용 프레임 수 |
| motion_reid.max_reid_dist | 50.0 | Re-ID 최대 허용 거리 (픽셀) |
| kalman.ref_area | 데이터에 맞게 설정 | Adaptive R area 기준값 |

---

## 실행 방법

### 평가 + 시각화

```bash
python scripts/run_eval.py \
    --data_dir /home/gpuadmin/Dataset/v2e_bird/금오도/data_1 \
    --out_dir eval_output \
    --save_video \
    --device cuda
```

출력:
- `eval_output/metrics.txt` — Precision / Recall / F1 / IoU / TP·FP·FN
- `eval_output/tracks.mp4` — 추적 결과 시각화 (`--save_video` 지정 시)

---

## 데이터셋

### 금오도 (주력 평가 데이터)
- 경로: `/home/gpuadmin/Dataset/v2e_bird/금오도/data_1`, `data_2`
- 총 프레임: 12,837 / 총 객체: 14,080
- 프레임당 평균 객체: 1.1 (max=8)
- Width 중앙값: ~24px / Height 중앙값: ~18px
- Small(<32×32) 약 75%
- track_id 없음 → Precision/Recall/F1 평가

---

## 의존성

```python
torch           # 텐서 연산, GPU 지원
numpy           # cost matrix 변환
lap             # lapjv Hungarian matching
easydict        # cfg 관리
opencv-python   # 시각화
Pillow          # 이미지 크기 확인
```

---

## 주의사항

- `Track._count`는 클래스 변수 — 테스트 간 격리 시 `Track.reset_id()` 호출
- `cfg.device = 'cuda'` 기본값 — CPU 환경에서는 `CFG/cfg.py` 수정
- `association.py`의 `iou_distance`는 `.cpu().numpy()` 반환 (lap 요구)
- `KalmanFilter.update`에서 Q scaling 후 covariance in-place 대입
- Lost 트랙도 매 프레임 `predict()` → 운동 외삽 위치로 Re-ID 후보 사용
- `MotionReID.match()`는 Lost 상태 트랙의 예측 중심점으로 ID 정합
- 수치 안정성: Cholesky 분해 전 jitter=1e-4 추가, 대칭성 강제