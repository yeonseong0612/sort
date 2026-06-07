import torch


class NeighborImputation:
    def __init__(self, k=7, alpha=0.5):
        self.k = k
        self.alpha = alpha

    def get_neighbors(self, target_track, active_tracks):
        candidates = [t for t in active_tracks if t.track_id != target_track.track_id]
        if not candidates:
            return []

        target_pos = target_track.mean[:2]
        dists = [(torch.norm(t.mean[:2] - target_pos).item(), t) for t in candidates]
        dists.sort(key=lambda x: x[0])
        return [t for _, t in dists[: self.k]]

    def compute_neighbor_velocity(self, neighbors):
        if not neighbors:
            return None
        vels = torch.stack([n.mean[4:6] for n in neighbors], dim=0)
        return vels.mean(dim=0)

    def impute(self, target_track, active_tracks):
        """
        x_imputed = x_kalman + alpha * (v_neighbor - v_self)
        이웃이 없으면 칼만 예측값(x_kalman)으로 폴백.
        target_track.neighbor_ids를 in-place 갱신함.
        """
        neighbors = self.get_neighbors(target_track, active_tracks)

        if not neighbors:
            return target_track.mean[:2].clone()

        v_neighbor = self.compute_neighbor_velocity(neighbors)
        v_self = target_track.mean[4:6]

        x_imputed = target_track.mean[:2] + self.alpha * (v_neighbor - v_self)
        target_track.neighbor_ids = [n.track_id for n in neighbors]

        return x_imputed
