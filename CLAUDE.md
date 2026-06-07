# CLAUDE.md

## 프로젝트 개요

SORT 기반 Multi-Object Tracking (MOT) 구현체. ByteTrack 스타일의 2-stage association에 Adaptive Kalman Filter(R/Q)와 Detection-level GMC(Global Motion Compensation)를 결합한 트래커.

## 디렉토리 구조

```
sort/
├── CFG/
│   └── cfg.py                  # 전역 설정 (cfg.device)
├── tracker/
│   ├── kalman_filter.py        # Adaptive R/Q Kalman Filter
│   ├── track.py                # Track 객체 및 TrackState 관리
│   ├── association.py          # IoU cost matrix + lap.lapjv Hungarian matching
│   ├── motion_statistics.py    # Detection-level GMC
│   └── tracker.py              # 메인 Tracker 클래스
├── detector/                   # 미구현 (빈 디렉토리)
├── evaluation/                 # 미구현 (빈 디렉토리)
├── ext/                        # 미구현 (빈 디렉토리)
├── scripts/                    # 미구현 (빈 디렉토리)
├── visualization/              # 미구현 (빈 디렉토리)
├── test_synthetic.py           # Adaptive Q 수동 검증 스크립트
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
      matched   → track.update() → Tracked
      unmatched track → 2차 association 대기
      unmatched high det → 신규 Track 생성
  → [2차 association] remain_tracks ↔ low_dets
      matched   → track.update() → Tracked
      unmatched → mark_lost → Lost
  → lost_tracks 관리: max_lost 초과 시 mark_removed
  → return tracked_tracks (TrackState.Tracked)
```

### Adaptive Kalman Filter

**Adaptive R** (`compute_r_scale`): score 낮음 + area 작음 → R 증가 → 측정 신뢰도 하락
- `score_scale = clamp(1 / score, 0.5, 5.0)`
- `area_scale  = clamp(32²/ area, 0.5, 5.0)`  ← ref_area = 32×32

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

## 트랙 상태 전이

```
신규 detection → Track(Tracked)
Tracked → (매칭 실패) → Lost
Lost    → (max_lost 프레임 초과) → Removed
Lost    → (매칭 성공) → Tracked
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

## 의존성

```python
torch       # 텐서 연산, GPU 지원
numpy       # cost matrix 변환
lap         # lapjv Hungarian matching (lap.lapjv)
easydict    # cfg 관리
```

## 실행 방법

```bash
# Adaptive Q 동작 수동 검증
python test_synthetic.py

# pytest (테스트 파일 있을 경우)
pytest
```

## 미구현 영역

`detector/`, `evaluation/`, `ext/`, `scripts/`, `visualization/` 디렉토리는 현재 비어 있음.
향후 작업 시 채워야 할 영역:
- detector: 검출기 연동 (YOLO 등)
- evaluation: MOT 메트릭 (HOTA, MOTA 등)
- visualization: 트랙 시각화
- scripts: 데이터셋별 실행 스크립트

## 주의사항

- `Track._count`는 클래스 변수 — 테스트 간 격리가 필요하면 `Track.reset_id()` 호출
- `cfg.device = 'cuda'` 하드코딩 — CPU 환경에서는 `CFG/cfg.py` 수정 필요
- `association.py`의 `iou_distance`는 GPU 계산 후 `.cpu().numpy()` 반환 (lap이 numpy 요구)
- `KalmanFilter.update`에서 Q scaling 적용 후 covariance를 **in-place 대입** — 원본 covariance는 변경됨


## 개발 방향 (점진적 수정)

### 핵심 원칙
- 기존 동작 절대 유지 (Adaptive Kalman, 2단계 association, GMC)
- 칼만 상태벡터 [x,y,w,h,vx,vy,vw,vh] 8차원 그대로 유지
- 새 기능은 추가만, 기존 로직 수정 최소화

### 1단계 작업 목록

**track.py**
- TrackState에 OcclusionImputed 추가
- Track 필드 추가: occluded_frames, imputed_positions,
  neighbor_ids, last_observed_velocity
- 메서드 추가: start_occlusion(), add_imputed_position(), clear_occlusion()

**tracker/neighbor_imputation.py (신규)**
- NeighborImputation 클래스
- get_neighbors(): 유클리드 거리 기준 k개 이웃 탐색 (기본 k=7)
- compute_neighbor_velocity(): 이웃 평균 (vx, vy)
- impute(): 잔차 구조 — 칼만 예측 + 이웃 평균 속도 보정
  - 이웃 없으면 칼만 예측으로 자동 폴백

**tracker.py**
- Tracker.__init__에 self.imputation = NeighborImputation(k=7) 추가
- update() 내 Lost 처리 직후에 이웃 보완 블록 추가
- Lost → Tracked 재연결 시 clear_occlusion() 호출

### 배경 (연구 컨텍스트)
- 조류 군집 추적에서 외형 Re-ID 무용 → 운동 기반으로 전환
- 가림 중 이웃 새들의 변위로 가려진 새 위치 보완
- 잔차 구조: x_imputed = x_kalman + alpha*(v_neighbor - v_self)
- alpha=0.5 (초기값, 추후 게이팅 네트워크로 대체 예정)