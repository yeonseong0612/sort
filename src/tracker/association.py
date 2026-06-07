import torch
import numpy as np
import lap
from CFG.cfg import cfg


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1])
        )

    cost, x, y = lap.lapjv(
        cost_matrix,
        extend_cost=True,
        cost_limit=thresh
    )

    matches = []
    for i, j in enumerate(x):
        if j >= 0:
            matches.append([i, j])

    matches = np.asarray(matches, dtype=int)

    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]

    return matches, unmatched_a, unmatched_b


def iou_distance(atracks, btracks):
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    device = cfg.device

    atlbrs = torch.stack([
        track.to_tlbr().to(device) for track in atracks
    ])

    btlbrs = torch.stack([
        track.to_tlbr().to(device) for track in btracks
    ])

    lt = torch.max(atlbrs[:, None, :2], btlbrs[None, :, :2])
    rb = torch.min(atlbrs[:, None, 2:], btlbrs[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area_a = (atlbrs[:, 2] - atlbrs[:, 0]).clamp(min=0) * \
             (atlbrs[:, 3] - atlbrs[:, 1]).clamp(min=0)

    area_b = (btlbrs[:, 2] - btlbrs[:, 0]).clamp(min=0) * \
             (btlbrs[:, 3] - btlbrs[:, 1]).clamp(min=0)

    union = area_a[:, None] + area_b[None, :] - inter
    ious = inter / (union + 1e-7)

    cost_matrix = 1.0 - ious

    return cost_matrix.detach().cpu().numpy()