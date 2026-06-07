from pathlib import Path
import cv2
import numpy as np
import torch

from src.tracker.track import TrackState


class TrackVisualizer:
    def __init__(self, fps: int = 30, out_path: str = "output.mp4"):
        self.fps = fps
        self.out_path = out_path
        self.writer = None
        self.W = None
        self.H = None

    def _init_writer(self, W: int, H: int):
        self.W = W
        self.H = H
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (W, H))

    def add_frame(self, img_path, gt_boxes, tracked_tracks, lost_tracks):
        img = cv2.imread(str(img_path))
        if img is None:
            return

        if self.writer is None:
            H, W = img.shape[:2]
            self._init_writer(W, H)

        # GT boxes (green, thickness 1)
        if gt_boxes is not None and len(gt_boxes) > 0:
            boxes = gt_boxes[:, :4]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Tracked tracks (blue, thickness 2)
        for track in tracked_tracks:
            tlbr = track.to_tlbr().cpu().numpy()
            x1, y1, x2, y2 = map(int, tlbr)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
            cv2.putText(img, str(track.track_id), (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        # OcclusionImputed tracks (red dashed, thickness 2) + imputed trajectory (yellow dots)
        for track in lost_tracks:
            if track.state != TrackState.OcclusionImputed:
                continue

            tlbr = track.to_tlbr().cpu().numpy()
            x1, y1, x2, y2 = map(int, tlbr)
            self._draw_dashed_rect(img, x1, y1, x2, y2, (0, 0, 255), 2)
            cv2.putText(img, f"IMP:{track.track_id}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # imputed trajectory
            pts = track.imputed_positions
            if len(pts) > 1:
                for i in range(1, len(pts)):
                    p1 = pts[i - 1]
                    p2 = pts[i]
                    if isinstance(p1, torch.Tensor):
                        p1 = p1.cpu().numpy()
                    if isinstance(p2, torch.Tensor):
                        p2 = p2.cpu().numpy()
                    p1 = (int(p1[0]), int(p1[1]))
                    p2 = (int(p2[0]), int(p2[1]))
                    cv2.line(img, p1, p2, (0, 255, 255), 1)
                    cv2.circle(img, p2, 2, (0, 255, 255), -1)

        self.writer.write(img)

    def _draw_dashed_rect(self, img, x1, y1, x2, y2, color, thickness, dash=8):
        pts = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for (sx, sy), (ex, ey) in pts:
            length = int(((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5)
            if length == 0:
                continue
            steps = max(length // (dash * 2), 1)
            for k in range(steps):
                t0 = (2 * k * dash) / length
                t1 = min((2 * k + 1) * dash / length, 1.0)
                p0 = (int(sx + t0 * (ex - sx)), int(sy + t0 * (ey - sy)))
                p1 = (int(sx + t1 * (ex - sx)), int(sy + t1 * (ey - sy)))
                cv2.line(img, p0, p1, color, thickness)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
