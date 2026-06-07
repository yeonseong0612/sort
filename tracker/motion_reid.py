import torch
import numpy as np
import lap


class MotionReID:
    """
    가림 후 재등장한 검출을 보완 궤적 끝점으로 재식별.
    외형 Re-ID 없이 운동 패턴(imputed_positions 마지막 위치)으로 ID 정합.
    """

    def __init__(self, max_reid_dist: float = 50.0):
        self.max_reid_dist = max_reid_dist

    def match(self, lost_tracks, new_detections):
        """
        imputed_positions가 있는 lost_tracks ↔ new_detections 간 Re-ID 수행.

        Args:
            lost_tracks: imputed_positions가 쌓인 OcclusionImputed 트랙 리스트
            new_detections: 매칭 안 된 detection Track 리스트

        Returns:
            matched: [(track_idx, det_idx), ...]  — max_reid_dist 이내만 포함
        """
        candidates = [t for t in lost_tracks if t.imputed_positions]
        if not candidates or not new_detections:
            return []

        # 각 후보 트랙의 보완 궤적 끝점 (x, y)
        endpoints = torch.stack([t.imputed_positions[-1] for t in candidates])  # (M, 2)

        # 각 detection의 중심점 (x, y)
        det_centers = torch.stack([d.mean[:2] for d in new_detections])          # (K, 2)

        # 유클리드 거리 행렬 (M, K)
        dist_matrix = torch.cdist(endpoints, det_centers)                         # (M, K)
        dist_np = dist_matrix.cpu().numpy()

        # Hungarian matching
        cost, row_ind, col_ind = lap.lapjv(dist_np, extend_cost=True, cost_limit=self.max_reid_dist)

        matched = []
        for r, c in enumerate(col_ind):
            if c < 0:
                continue
            if dist_np[r, c] > self.max_reid_dist:
                continue
            # candidates[r] → original index in lost_tracks
            original_idx = lost_tracks.index(candidates[r])
            matched.append((original_idx, c))

        return matched
