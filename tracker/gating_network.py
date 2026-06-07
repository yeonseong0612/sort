import torch
import torch.nn as nn


class GatingNetwork(nn.Module):
    """
    5가지 신호로 (w_motion, w_neighbor) 가중치를 출력하는 2층 MLP.
    학습 전에도 가림 유무 / 이웃 유무에 따라 휴리스틱한 초기값을 내도록
    bias를 사전 설정한다.

    입력 신호 (정규화 없이 raw 값을 받아 내부에서 scale):
        [kalman_uncertainty, detection_score, occluded_frames,
         neighbor_count, velocity_magnitude]
    출력:
        w_motion, w_neighbor  (softmax 정규화, 합 = 1)
    """

    INPUT_DIM = 5
    HIDDEN_DIM = 16
    OUTPUT_DIM = 2

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.INPUT_DIM, self.HIDDEN_DIM)
        self.fc2 = nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM)
        self._init_heuristic_bias()

    def _init_heuristic_bias(self):
        """
        학습 전 초기 동작 목표:
          occluded_frames=0              → w_motion≈0.9, w_neighbor≈0.1
          occluded_frames>0 + neighbor>0 → w_motion≈0.5, w_neighbor≈0.5
          occluded_frames>0 + neighbor=0 → w_motion≈0.9, w_neighbor≈0.1

        fc1 가중치는 기본 Xavier 초기화 유지,
        fc2 bias만 조정해 가림 없음 케이스에서 w_motion 우세하도록 설정.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)  # fc2도 Xavier
        with torch.no_grad():
            self.fc2.bias[0] = 2.2
            self.fc2.bias[1] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

    def get_weights_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DataLoader 배치에서 (w_motion, w_neighbor) 텐서를 반환.
        neighbor_mask == 0인 샘플은 강제로 (1.0, 0.0) 처리.

        입력: BEE24Dataset.__getitem__ 반환 딕셔너리의 배치
        출력: w_motion (B,), w_neighbor (B,)
        """
        device = next(self.parameters()).device

        kalman_uncertainty = batch["kalman_uncertainty"].to(device)   # (B,)
        det_score          = batch["det_score"].to(device)            # (B,)
        occluded_frames    = batch["occluded_frames"].to(device) if torch.is_tensor(batch["occluded_frames"]) \
                             else torch.tensor(batch["occluded_frames"], dtype=torch.float32, device=device)
        neighbor_count     = batch["neighbor_count"].to(device) if torch.is_tensor(batch["neighbor_count"]) \
                             else torch.tensor(batch["neighbor_count"], dtype=torch.float32, device=device)
        vel_mag            = batch["velocity_magnitude"].to(device)   # (B,)
        neighbor_mask      = batch["neighbor_mask"].to(device)        # (B, k)

        # neighbor_mask: 하나라도 이웃 있으면 1
        has_neighbor = (neighbor_mask.sum(dim=-1) > 0).float()        # (B,)

        x = torch.stack([
            kalman_uncertainty / 500.0,
            det_score,
            occluded_frames.float() / 30.0,
            neighbor_count.float() / 8.0,
            vel_mag / 20.0,
        ], dim=1)  # (B, 5)

        weights = self.forward(x)          # (B, 2)
        w_motion   = weights[:, 0]         # (B,)
        w_neighbor = weights[:, 1]         # (B,)

        # 이웃 없는 샘플 강제 처리
        w_motion   = torch.where(has_neighbor.bool(), w_motion,   torch.ones_like(w_motion))
        w_neighbor = torch.where(has_neighbor.bool(), w_neighbor, torch.zeros_like(w_neighbor))

        return w_motion, w_neighbor

    def get_weights(self, track, active_tracks) -> tuple[float, float]:
        """
        track과 active_tracks 정보로 입력 벡터를 구성해 (w_motion, w_neighbor) 반환.
        반환값: (w_motion, w_neighbor) — 합이 1인 float 쌍
        """
        from tracker.track import TrackState

        kf_uncertainty = track.covariance[:2, :2].trace().item()
        det_score = track.score.item() if torch.is_tensor(track.score) else float(track.score)
        occluded_frames = float(track.occluded_frames)
        neighbor_count = float(
            len([t for t in active_tracks if t.track_id != track.track_id])
        )
        vel_mag = track.mean[4:6].norm().item()

        x = torch.tensor([
            kf_uncertainty / 1000.0,   # 대략적인 스케일 맞춤
            det_score,                  # 이미 0~1
            occluded_frames / 30.0,    # max_lost 기준 정규화
            neighbor_count / 50.0,     # 예상 최대 이웃 수 기준
            vel_mag / 50.0,            # 예상 최대 속도 기준
        ], dtype=torch.float32)

        with torch.no_grad():
            weights = self.forward(x)

        w_motion = weights[0].item()
        w_neighbor = weights[1].item()

        # 이웃이 없으면 w_motion으로 모두 귀속
        if neighbor_count == 0:
            w_motion, w_neighbor = 1.0, 0.0

        return w_motion, w_neighbor
