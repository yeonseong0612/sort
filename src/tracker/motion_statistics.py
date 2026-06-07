import torch


def compute_bbox_shifts(tracks, detections, matches):
    """
    matched pair 기반 bbox center shift 계산

    Parameters
    ----------
    tracks : list[Track]
        Kalman predict 이후의 tracked tracks

    detections : list[Track]
        현재 프레임 detection을 Track 형태로 변환한 후보들

    matches : np.ndarray, shape (K, 2)
        Hungarian matching 결과
        each row: [track_idx, det_idx]

    Returns
    -------
    shifts : torch.Tensor, shape (K, 2)
        각 matched pair의 center shift
        shift = detection_center - predicted_track_center
    """

    if len(matches) == 0:
        if len(tracks) > 0:
            device = tracks[0].mean.device
        elif len(detections) > 0:
            device = detections[0].mean.device
        else:
            device = "cpu"

        return torch.empty((0, 2), device=device)

    shifts = []

    for t_idx, d_idx in matches:
        track = tracks[t_idx]
        det = detections[d_idx]

        track_center = track.mean[:2]
        det_center = det.mean[:2]

        shift = det_center - track_center
        shifts.append(shift)

    return torch.stack(shifts, dim=0)


def robust_global_shift(shifts, min_pairs=2, max_shift=None):
    """
    bbox shift들의 robust median을 이용해 global camera shift 추정

    Parameters
    ----------
    shifts : torch.Tensor, shape (K, 2)
        matched pair shift들

    min_pairs : int
        GMC를 적용하기 위한 최소 matched pair 수

    max_shift : float or None
        너무 큰 shift를 clamp할 때 사용.
        None이면 clamp하지 않음.

    Returns
    -------
    global_shift : torch.Tensor, shape (2,)
        추정된 global shift [dx, dy]

    valid : bool
        충분한 matched pair가 있어 GMC를 적용할 수 있는지 여부
    """

    if shifts is None or shifts.numel() == 0 or shifts.shape[0] < min_pairs:
        device = shifts.device if shifts is not None else "cpu"
        return torch.zeros(2, device=device), False

    global_shift = torch.median(shifts, dim=0).values

    if max_shift is not None:
        global_shift = torch.clamp(
            global_shift,
            min=-max_shift,
            max=max_shift
        )

    return global_shift, True


def apply_global_shift(tracks, global_shift):
    """
    모든 track의 Kalman state center에 global shift 적용

    state format:
    mean = [x, y, w, h, vx, vy, vw, vh]

    여기서는 center position x, y만 보정한다.
    """

    if len(tracks) == 0:
        return

    for track in tracks:
        track.mean[0] += global_shift[0]
        track.mean[1] += global_shift[1]