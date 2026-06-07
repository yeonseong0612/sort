import torch
import numpy as np
import lap


def _iou_matrix(gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> np.ndarray:
    """Compute IoU matrix between gt (N,4) and pred (M,4) [x1,y1,x2,y2]."""
    if gt_boxes.shape[0] == 0 or pred_boxes.shape[0] == 0:
        return np.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), dtype=np.float32)

    lt = torch.max(gt_boxes[:, None, :2], pred_boxes[None, :, :2])
    rb = torch.min(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_gt = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * \
              (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
    area_pred = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
                (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)

    union = area_gt[:, None] + area_pred[None, :] - inter
    iou = inter / (union + 1e-7)
    return iou.cpu().numpy()


class FrameEvaluator:
    def __init__(self, iou_thresh: float = 0.5):
        self.iou_thresh = iou_thresh

    def match_frame(self, gt_boxes: torch.Tensor, pred_boxes: torch.Tensor) -> dict:
        N = gt_boxes.shape[0]
        M = pred_boxes.shape[0]

        if N == 0 and M == 0:
            return {"tp": 0, "fp": 0, "fn": 0, "matched_ious": []}

        if N == 0:
            return {"tp": 0, "fp": M, "fn": 0, "matched_ious": []}

        if M == 0:
            return {"tp": 0, "fp": 0, "fn": N, "matched_ious": []}

        iou_mat = _iou_matrix(gt_boxes, pred_boxes)
        cost_mat = (1.0 - iou_mat).astype(np.float64)

        _, x, y = lap.lapjv(cost_mat, extend_cost=True, cost_limit=1.0 - self.iou_thresh)

        matched_ious = []
        tp = 0
        for i, j in enumerate(x):
            if j >= 0 and iou_mat[i, j] >= self.iou_thresh:
                matched_ious.append(float(iou_mat[i, j]))
                tp += 1

        fp = M - tp
        fn = N - tp
        return {"tp": tp, "fp": fp, "fn": fn, "matched_ious": matched_ious}

    def compute_metrics(self, results: list) -> dict:
        total_tp = sum(r["tp"] for r in results)
        total_fp = sum(r["fp"] for r in results)
        total_fn = sum(r["fn"] for r in results)
        all_ious = [iou for r in results for iou in r["matched_ious"]]

        precision = total_tp / (total_tp + total_fp + 1e-9)
        recall = total_tp / (total_tp + total_fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        }

    def print_report(self, metrics: dict):
        print("=" * 40)
        print("Evaluation Results")
        print("=" * 40)
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  F1        : {metrics['f1']:.4f}")
        print(f"  Mean IoU  : {metrics['mean_iou']:.4f}")
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
        print("=" * 40)
