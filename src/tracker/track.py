from enum import Enum
import torch

class TrackState(Enum):
    Tracked = 1
    Lost = 2
    Removed = 3
    OcclusionImputed = 4


class Track:
    _count = 0

    def __init__(self, mean, covariance, score, frame_id):
        self.mean = mean
        self.covariance = covariance

        self.track_id = self.next_id()
        self.score = score

        self.state = TrackState.Tracked
        self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        self.occluded_frames = 0
        self.imputed_positions = []
        self.neighbor_ids = []
        self.last_observed_velocity = torch.zeros(2, device=mean.device)

    @staticmethod
    def next_id():
        Track._count += 1
        return Track._count

    @staticmethod
    def reset_id():
        Track._count = 0

    def predict(self, kf):
        mean, covariance = kf.predict(
            self.mean.unsqueeze(0),
            self.covariance.unsqueeze(0)
        )

        self.mean = mean.squeeze(0)
        self.covariance = covariance.squeeze(0)

        self.age += 1
        self.time_since_update += 1

    def update(self, kf, measurement, score, frame_id):
        area = measurement[2] * measurement[3]

        mean, covariance = kf.update(
            self.mean.unsqueeze(0),
            self.covariance.unsqueeze(0),
            measurement.unsqueeze(0),
            score=score.unsqueeze(0) if torch.is_tensor(score) and score.dim() == 0 else score,
            area=area.unsqueeze(0) if torch.is_tensor(area) and area.dim() == 0 else area
        )

        self.mean = mean.squeeze(0)
        self.covariance = covariance.squeeze(0)

        self.score = score
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits += 1

        self.last_observed_velocity = self.mean[4:6].clone()
        self.state = TrackState.Tracked
        self.is_activated = True
        self.clear_occlusion()

    def start_occlusion(self):
        self.state = TrackState.OcclusionImputed
        self.occluded_frames = 0

    def add_imputed_position(self, pos):
        self.imputed_positions.append(pos)
        self.occluded_frames += 1

    def clear_occlusion(self):
        self.occluded_frames = 0
        self.imputed_positions = []
        self.neighbor_ids = []

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    def get_uncertainty(self) -> float:
        base = self.covariance[:2, :2].trace().item()
        if self.state in (TrackState.Lost, TrackState.OcclusionImputed):
            base *= (1.0 + self.occluded_frames * 0.1)
        return base

    def to_xywh(self):
        return self.mean[:4]

    def to_tlwh(self):
        x, y, w, h = self.mean[:4]
        return torch.stack([
            x - w / 2,
            y - h / 2,
            w,
            h
        ])

    def to_tlbr(self):
        x, y, w, h = self.mean[:4]
        return torch.stack([
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2
        ])