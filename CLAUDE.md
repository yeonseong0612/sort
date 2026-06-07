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

## Phase 1 — 학습 파이프라인 (MLP 게이팅 + k-NN)

### 목표
게이팅 네트워크(MLP)를 BEE24 데이터로 학습시켜
"이웃 k-NN 보완이 칼만보다 언제 더 나은가"를 학습.
k-NN 자체는 학습 없음. MLP만 학습 대상.

### 검증 질문
"게이팅 학습 후 고정 alpha=0.5 대비 가림 구간 ADE/FDE가 줄어드는가?"
→ 줄어들면 Phase 2(GAT)로 진행
→ 줄어들지 않으면 이웃 집계 방식 재검토

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 디렉토리 추가
sort/
├── data/
│   └── bee24_loader.py        # BEE24 데이터 로더 + 마스킹
├── training/
│   ├── loss.py                # 손실 함수
│   ├── trainer.py             # 학습 루프
│   └── train.py               # 실행 스크립트
└── CFG/
    └── cfg.py                 # 기존 파일에 Phase 1 설정 추가
    (configs/phase1.yaml 사용 안 함)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### data/bee24_loader.py 신규 생성

BEE24Dataset 클래스:

BEE24 디렉토리 구조:
    BEE24/
    ├── train/                    # 31개 시퀀스
    │   ├── BEE24-01/
    │   │   ├── gt/gt.txt
    │   │   ├── img1/
    │   │   └── seqinfo.ini
    │   └── BEE24-02/ ...
    └── test/                     # 별도 시퀀스

gt.txt 형식 (9열, 콤마 구분):
    frame_id, track_id, x, y, w, h, 1, 1, 1
    - frame_id: 6자리 문자열 ("000001" → int 1)
    - x, y: top-left 픽셀 좌표 (float)
    - w, h: 픽셀 단위 (float)
    - 예: 000001,1,448.00,331.00,57.00,60.00,1,1,1

seqinfo.ini 파싱 항목:
    frameRate=25
    imWidth=950
    imHeight=590

__init__(data_root, split="train",
         mask_lengths=[5, 10, 20, 40],
         mask_ratio=0.3,
         min_track_len=30,
         k=7,
         val_seq_ids=None):
    - data_root: BEE24 루트 폴더
    - split: "train" / "val" / "test"
    - val_seq_ids: 검증 시퀀스 ID 리스트
                   None이면 train 31개 중 마지막 6개를 val로 사용
    - mask_lengths: 마스킹 길이 후보 (프레임 단위)
    - mask_ratio: 트랙당 최대 마스킹 비율
    - min_track_len: 이 길이 미만 트랙 제외
    - k: 이웃 수

데이터 로드 순서:
    1. split에 맞는 시퀀스 폴더 목록 구성
       train: 전체 31개 중 val 제외
       val:   val_seq_ids 또는 마지막 6개
       test:  BEE24/test/ 폴더
    2. 각 시퀀스별 seqinfo.ini 파싱
       → fps, imWidth, imHeight 추출
    3. gt.txt 파싱:
       - parts = line.strip().split(",")
       - frame_id = int(parts[0])
       - track_id = int(parts[1])
       - x, y, w, h = float(parts[2]), float(parts[3]),
                       float(parts[4]), float(parts[5])
       - cx = x + w/2  (top-left → center 변환)
       - cy = y + h/2
    4. {track_id: [(frame, cx, cy, w, h), ...]} 구성
       (시퀀스별로 track_id가 겹칠 수 있으므로
        키를 (seq_name, track_id)로 관리)
    5. 속도 계산 (fps 반영):
       vx = (cx[t] - cx[t-1]) * fps
       vy = (cy[t] - cy[t-1]) * fps
       첫 프레임 속도 = 0
    6. min_track_len 미만 트랙 제외
    7. 마스킹 샘플 인덱스 사전 생성

__len__(): 전체 마스킹 샘플 수 반환

__getitem__(idx) -> dict:
    반환 딕셔너리:
    {
      "target_before":  Tensor(T_obs, 6),     # 가림 전 궤적 [cx,cy,vx,vy,w,h]
      "target_gt":      Tensor(T_mask, 2),    # 가림 구간 GT [cx,cy]
      "neighbors":      Tensor(k, T_mask, 4), # 이웃 궤적 [cx,cy,vx,vy]
                                              # 이웃 없으면 zeros
      "neighbor_mask":  Tensor(k,),           # 실제 이웃 여부 (1/0)
      "kalman_pred":    Tensor(T_mask, 2),    # 칼만 예측 [cx,cy]
      "occluded_frames": int,                 # 마스킹 길이
      "neighbor_count": int,                  # 실제 이웃 수
      "kalman_uncertainty": float,            # 칼만 공분산 trace (게이팅 입력)
      "velocity_magnitude": float,            # 가림 직전 속도 크기 (게이팅 입력)
    }

마스킹 프로토콜:
    1. 트랙에서 마스킹 시작점 랜덤 선택
       유효 범위: [10, len(track) - mask_len - 10]
       (앞뒤 10프레임은 관측 확보용으로 제외)
    2. mask_lengths에서 길이 랜덤 선택
       단, 해당 트랙 길이의 mask_ratio 이하로 제한
    3. 마스킹 구간의 이웃 탐색:
       같은 시퀀스, 같은 프레임 구간에 존재하는 다른 track_id
       거리 기준 상위 k개 선택
       (거리 = 마스킹 시작 프레임에서의 중심점 유클리드 거리)
    4. 칼만 예측 계산 (등속도 외삽):
       vx_last, vy_last = 가림 직전 프레임 속도
       cx_kalman(t) = cx_last + vx_last * t  (t=1,2,...,T_mask)
       cy_kalman(t) = cy_last + vy_last * t
    5. kalman_uncertainty:
       가림 길이에 비례해 증가하는 단순 추정값 사용
       uncertainty = base_var * (1 + occluded_frames * 0.1)
       (실제 KalmanFilter 없이 근사값으로 대체)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### training/loss.py 신규 생성

OcclusionImputationLoss 클래스:

__init__(lambda_fde=2.0, alpha_gate=1.0, beta_reg=0.01):
    - lambda_fde: FDE 가중치 (재등장 위치 정합 중요)
    - alpha_gate: 게이팅 손실 가중치
    - beta_reg: 정규화 가중치

forward(pred, gt, w_neighbor, kalman_pred, neighbor_mask) -> dict:
    입력:
    - pred:          Tensor(B, T, 2) 최종 복원 위치
    - gt:            Tensor(B, T, 2) GT 위치
    - w_neighbor:    Tensor(B,)      게이팅 출력 이웃 가중치
    - kalman_pred:   Tensor(B, T, 2) 칼만 예측
    - neighbor_mask: Tensor(B,)      이웃 존재 여부

    계산:
    # ADE: 가림 구간 평균 변위 오차
    L_ADE = mean(||pred - gt||_2)  over T

    # FDE: 마지막 프레임 오차 (재등장 위치)
    L_FDE = ||pred[:,-1,:] - gt[:,-1,:]||_2

    # GNN 손실
    L_impute = L_ADE + lambda_fde * L_FDE

    # 게이팅 손실:
    # 이웃 없는 샘플에서 w_neighbor가 0에 가깝도록
    L_gate = mean(w_neighbor[neighbor_mask==0] ** 2)

    # 정규화: w_neighbor 과신뢰 방지
    L_reg = mean(w_neighbor ** 2)

    # 전체 손실
    L_total = L_impute + alpha_gate * L_gate + beta_reg * L_reg

    반환: {
        "total": L_total,
        "ade": L_ADE,
        "fde": L_FDE,
        "gate": L_gate,
        "reg": L_reg
    }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### training/trainer.py 신규 생성

Trainer 클래스:

__init__(config):
    - GatingNetwork 초기화
    - NeighborImputation(k=7) 초기화 (고정, 학습 안 함)
    - optimizer: Adam(gating.parameters(), lr=1e-3)
    - scheduler: CosineAnnealingLR
    - loss_fn: OcclusionImputationLoss

train_epoch(dataloader) -> dict:
    배치별 루프:
    1. 배치에서 데이터 로드
    2. k-NN 이웃 평균 속도 계산 (고정)
       v_neighbor = mean(neighbors velocity)
    3. 게이팅 가중치 계산
       w_motion, w_neighbor = gating.get_weights_batch(batch)
    4. 최종 복원 위치 계산
       pred = w_motion * kalman_pred +
              w_neighbor * (kalman_pred + alpha*(v_neighbor - v_self))
    5. 손실 계산 및 역전파
    6. 손실 기록

    반환: {ade, fde, total_loss} 평균

evaluate(dataloader) -> dict:
    검증 루프 (no_grad):
    가림 길이별 ADE/FDE 분리 측정
    {5프레임: {ade, fde}, 10프레임: ..., 20프레임: ..., 40프레임: ...}

save_checkpoint(epoch, path): 모델 저장
load_checkpoint(path): 모델 로드

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### gating_network.py 수정 — 배치 처리 추가

기존 get_weights()는 단일 트랙 처리.
학습용 배치 처리 메서드 추가:

get_weights_batch(batch: dict) -> tuple[Tensor, Tensor]:
    입력: __getitem__ 반환 딕셔너리의 배치
    출력: w_motion (B,), w_neighbor (B,)

    입력 벡터 구성 (배치):
    x = stack([
        batch["kalman_uncertainty"] / 500.0,
        batch["det_score"],          # 가림 중 = 0.0
        batch["occluded_frames"] / 30.0,
        batch["neighbor_count"] / 8.0,
        batch["velocity_magnitude"] / 20.0,
    ], dim=1)  # (B, 5)

    forward(x) → (B, 2) softmax
    이웃 없는 샘플은 강제로 (1.0, 0.0) 처리

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### training/train.py 신규 생성

argparse 인자:
    --data_root    BEE24 루트 경로 (필수)
    --out_dir      체크포인트 저장 경로 (기본 checkpoints/)
    --epochs       학습 에폭 (기본 50)
    --batch_size   배치 크기 (기본 64)
    --lr           학습률 (기본 1e-3)
    --k            이웃 수 (기본 7)
    --val_ratio    검증 분할 비율 (기본 0.2)
    --device       cuda / cpu (기본 cuda)
    --seed         재현성 시드 (기본 42)

실행 흐름:
    1. BEE24Dataset 초기화 (train / val 분할)
    2. DataLoader 구성 (shuffle=True, num_workers=4)
    3. Trainer 초기화
    4. 에폭 루프:
        a. train_epoch()
        b. evaluate()
        c. 10 에폭마다 체크포인트 저장
        d. val ADE 기준 best 모델 저장
        e. 콘솔 출력:
           [Epoch 10/50] Loss=0.123 ADE=12.3 FDE=18.5
           Val: ADE(5f)=8.2 ADE(10f)=12.1 ADE(20f)=19.4
    5. 학습 완료 후 best 모델 로드
    6. 가림 길이별 최종 성능 출력 및 저장

결과 저장:
    out_dir/
    ├── best_model.pth       ← val ADE 기준 최적
    ├── last_model.pth       ← 마지막 에폭
    └── train_log.csv        ← 에폭별 손실 기록

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### CFG/cfg.py 수정 — Phase 1 설정 추가

기존 cfg.py에 phase1 설정 블록을 추가한다.
yaml 미사용. EasyDict으로 기존 방식과 통일.

추가할 내용:

from easydict import EasyDict as edict

# 기존 cfg (device 등) 유지하고 아래 추가
cfg.phase1 = edict()

# 데이터
cfg.phase1.data = edict()
cfg.phase1.data.data_root       = ""          # 실행 시 argparse로 덮어씀
cfg.phase1.data.mask_lengths    = [5, 10, 20, 40]
cfg.phase1.data.mask_ratio      = 0.3
cfg.phase1.data.min_track_len   = 30
cfg.phase1.data.k               = 7
cfg.phase1.data.val_ratio       = 0.2

# 모델
cfg.phase1.model = edict()
cfg.phase1.model.input_dim      = 5
cfg.phase1.model.hidden_dim     = 16
cfg.phase1.model.output_dim     = 2
cfg.phase1.model.alpha_fallback = 0.5         # 게이팅 없을 때 fallback

# 손실
cfg.phase1.loss = edict()
cfg.phase1.loss.lambda_fde      = 2.0         # FDE 가중치
cfg.phase1.loss.alpha_gate      = 1.0         # 게이팅 손실 가중치
cfg.phase1.loss.beta_reg        = 0.01        # 정규화 가중치

# 학습
cfg.phase1.train = edict()
cfg.phase1.train.epochs         = 50
cfg.phase1.train.batch_size     = 64
cfg.phase1.train.lr             = 1e-3
cfg.phase1.train.scheduler      = "cosine"
cfg.phase1.train.seed           = 42

# training/train.py에서 사용 예시:
# from CFG.cfg import cfg
# data_root = args.data_root or cfg.phase1.data.data_root
# epochs    = args.epochs    or cfg.phase1.train.epochs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 완료 조건 (WSL — 데이터 없이 import만 확인)

# 데이터셋 없이 코드 구조만 검증
python -c "
from CFG.cfg import cfg
from data.bee24_loader import BEE24Dataset
from training.loss import OcclusionImputationLoss
from training.trainer import Trainer
print('cfg.phase1.train.epochs:', cfg.phase1.train.epochs)
print('All imports OK')
"

# 실제 학습 실행은 서버에서 BEE24 데이터 준비 후 진행
# python training/train.py \
#     --data_root /path/to/BEE24 \
#     --out_dir checkpoints/phase1 \
#     --device cuda

### 주의사항
- WSL에서는 코드 작성 + import 확인까지만
- 데이터셋(BEE24)은 서버에만 존재
- 학습 실행은 서버에서 직접 수행
- train.py의 --data_root는 argparse 필수 인자로 두되
  cfg.phase1.data.data_root를 기본값 참조로 사용