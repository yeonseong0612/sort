import torch
import lap


class MotionReID:
    """
    가림 후 재등장한 검출을 Lost 트랙의 예측 위치로 재식별.
    외형 Re-ID 없이 운동 정보(칼만 예측 중심점)만으로 ID 정합.
    """

    def __init__(self, max_reid_dist: float = 50.0):
        self.max_reid_dist = max_reid_dist

    def match(self, lost_tracks, new_detections):
        """
        Lost 트랙 ↔ new_detections 간 운동 기반 Re-ID 수행.

        Args:
            lost_tracks: Lost 상태 트랙 리스트 (칼만으로 예측된 mean 보유)
            new_detections: 매칭 안 된 detection Track 리스트

        Returns:
            matched: [(track_idx, det_idx), ...]  — max_reid_dist 이내만 포함
        """
        if not lost_tracks or not new_detections:
            return []

        # 각 Lost 트랙의 예측 중심점 (x, y)
        endpoints = torch.stack([t.mean[:2] for t in lost_tracks])      # (M, 2)

        # 각 detection의 중심점 (x, y)
        det_centers = torch.stack([d.mean[:2] for d in new_detections])  # (K, 2)

        # 유클리드 거리 행렬 (M, K)
        dist_matrix = torch.cdist(endpoints, det_centers)                # (M, K)
        dist_np = dist_matrix.cpu().numpy()

        # Hungarian matching
        cost, row_ind, col_ind = lap.lapjv(dist_np, extend_cost=True, cost_limit=self.max_reid_dist)

        matched = []
        for r, c in enumerate(col_ind):
            if c < 0:
                continue
            if r >= dist_np.shape[0] or c >= dist_np.shape[1]:
                continue
            if dist_np[r, c] > self.max_reid_dist:
                continue
            matched.append((r, c))

        return matched
