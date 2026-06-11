import torch

from src.tracker.kalman_filter import KalmanFilter
from src.tracker.track import Track, TrackState
from src.tracker.association import iou_distance, linear_assignment
from src.tracker.motion_statistics import compute_bbox_shifts, robust_global_shift, apply_global_shift
from src.tracker.motion_reid import MotionReID



class Tracker:
    def __init__(
        self,
        high_thresh=0.5,
        low_thresh=0.1,
        match_thresh=0.7,
        max_lost=3,
        use_gmc=True,
        gmc_min_pairs=2,
        gmc_max_shift=50.0
    ):
        self.kf = KalmanFilter()
        self.motion_reid = MotionReID(max_reid_dist=50.0)

        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

        self.frame_id = 0

        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost = max_lost

        # GMC
        self.use_gmc = use_gmc
        self.gmc_min_pairs = gmc_min_pairs
        self.gmc_max_shift = gmc_max_shift
        self.last_global_shift = torch.zeros(2, device=self.kf.device)

    def update(self, detections):
        """
        detections: torch.Tensor, shape (N, 5)
        format: [x1, y1, x2, y2, score]
        """

        self.frame_id += 1

        if detections is None or len(detections) == 0:
            detections = torch.empty((0, 5), device=self.kf.device)

        # 1. split detections by score
        scores = detections[:, 4]

        high_mask = scores >= self.high_thresh
        low_mask = (scores >= self.low_thresh) & (scores < self.high_thresh)

        high_dets = detections[high_mask]
        low_dets = detections[low_mask]

        high_det_tracks = self._detections_to_tracks(high_dets)
        low_det_tracks = self._detections_to_tracks(low_dets)

        # 2. predict existing tracks (tracked + lost, so motion extrapolates)
        for track in self.tracked_tracks:
            track.predict(self.kf)
        for track in self.lost_tracks:
            track.predict(self.kf)

        activated_tracks = []
        lost_tracks = []
        removed_tracks = []

        # ============================================================
        # 2.5 Pixel-free GMC using detection-level motion statistics
        # ============================================================
        if self.use_gmc and len(self.tracked_tracks) > 0 and len(high_det_tracks) > 0:
            # tentative association before GMC
            tentative_cost = iou_distance(self.tracked_tracks, high_det_tracks)

            tentative_matches, _, _ = linear_assignment(
                tentative_cost,
                self.match_thresh
            )

            shifts = compute_bbox_shifts(
                self.tracked_tracks,
                high_det_tracks,
                tentative_matches
            )

            global_shift, valid_gmc = robust_global_shift(
                shifts,
                min_pairs=self.gmc_min_pairs,
                max_shift=self.gmc_max_shift
            )

            if valid_gmc:
                apply_global_shift(self.tracked_tracks, global_shift)
                self.last_global_shift = global_shift.detach()
            else:
                self.last_global_shift = torch.zeros(2, device=self.kf.device)
        else:
            self.last_global_shift = torch.zeros(2, device=self.kf.device)

        # ============================================================
        # 3. 1st association: tracked tracks <-> high score detections
        #    GMC 적용 후 다시 association
        # ============================================================
        cost_matrix = iou_distance(self.tracked_tracks, high_det_tracks)

        matches, unmatched_tracks, unmatched_high_dets = linear_assignment(
            cost_matrix,
            self.match_thresh
        )

        for t_idx, d_idx in matches:
            track = self.tracked_tracks[t_idx]
            det = high_det_tracks[d_idx]

            track.update(
                self.kf,
                det.mean[:4],
                det.score,
                self.frame_id
            )
            activated_tracks.append(track)

        # ============================================================
        # 4. 2nd association: unmatched tracks <-> low score detections
        # ============================================================
        remain_tracks = [self.tracked_tracks[i] for i in unmatched_tracks]

        cost_matrix_low = iou_distance(remain_tracks, low_det_tracks)

        matches_low, unmatched_remain_tracks, _ = linear_assignment(
            cost_matrix_low,
            self.match_thresh
        )

        for r_idx, d_idx in matches_low:
            track = remain_tracks[r_idx]
            det = low_det_tracks[d_idx]

            track.update(
                self.kf,
                det.mean[:4],
                det.score,
                self.frame_id
            )
            activated_tracks.append(track)

        # 5. tracks unmatched after both stages -> lost
        for r_idx in unmatched_remain_tracks:
            track = remain_tracks[r_idx]
            track.mark_lost()
            lost_tracks.append(track)

        # ============================================================
        # 6.5 운동 Re-ID: Lost 트랙 ↔ 매칭 안 된 high_dets
        # ============================================================
        unmatched_new_dets = [high_det_tracks[i] for i in unmatched_high_dets]

        reid_matches = self.motion_reid.match(self.lost_tracks, unmatched_new_dets)

        reid_recovered_track_ids = set()
        reid_consumed_det_indices = set()

        for t_idx, d_idx in reid_matches:
            track = self.lost_tracks[t_idx]
            det = unmatched_new_dets[d_idx]
            track.update(self.kf, det.mean[:4], det.score, self.frame_id)
            activated_tracks.append(track)
            reid_recovered_track_ids.add(track.track_id)
            reid_consumed_det_indices.add(unmatched_high_dets[d_idx])

        # 6. Re-ID로 소비되지 않은 unmatched high-score detections → 새 트랙
        for d_idx in unmatched_high_dets:
            if d_idx in reid_consumed_det_indices:
                continue
            det = high_det_tracks[d_idx]
            activated_tracks.append(det)

        # 7. previous lost track management (Re-ID로 복구된 트랙 제외)
        for track in self.lost_tracks:
            if track.track_id in reid_recovered_track_ids:
                continue
            if self.frame_id - track.frame_id > self.max_lost:
                track.mark_removed()
                removed_tracks.append(track)
            else:
                lost_tracks.append(track)

        # 8. update track pools
        self.tracked_tracks = [
            t for t in activated_tracks
            if t.state == TrackState.Tracked
        ]

        self.lost_tracks = [
            t for t in lost_tracks
            if t.state == TrackState.Lost
        ]

        self.removed_tracks.extend(removed_tracks)

        return self.tracked_tracks

    def _detections_to_tracks(self, detections):
        det_tracks = []

        for det in detections:
            x1, y1, x2, y2, score = det

            w = x2 - x1
            h = y2 - y1
            x = x1 + w / 2
            y = y1 + h / 2

            measurement = torch.stack([x, y, w, h])

            mean, covariance = self.kf.initiate(measurement)

            track = Track(
                mean=mean,
                covariance=covariance,
                score=score,
                frame_id=self.frame_id
            )

            det_tracks.append(track)

        return det_tracks