import configparser
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset

_BASE_VAR = 100.0  # kalman_uncertainty 근사용 기준 분산


class BEE24Dataset(Dataset):
    """
    BEE24 MOT 데이터셋 로더.

    BEE24 디렉토리 구조:
        BEE24/
        ├── train/   # 31개 시퀀스 (BEE24-01 ~ BEE24-31)
        │   ├── BEE24-01/
        │   │   ├── gt/gt.txt
        │   │   ├── img1/
        │   │   └── seqinfo.ini
        │   └── ...
        └── test/    # 별도 시퀀스

    gt.txt 형식 (9열, 콤마 구분):
        frame_id, track_id, x, y, w, h, 1, 1, 1
        예: 000001,1,448.00,331.00,57.00,60.00,1,1,1
        - frame_id: "000001" → int(1)
        - x, y: top-left → cx, cy center 변환
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        mask_lengths: list = None,
        mask_ratio: float = 0.3,
        min_track_len: int = 30,
        k: int = 7,
        val_seq_ids: list = None,
    ):
        self.data_root = data_root
        self.split = split
        self.mask_lengths = [10]
        self.mask_ratio = mask_ratio
        self.min_track_len = min_track_len
        self.k = k
        self.val_seq_ids = val_seq_ids

        # track_uid → {"dets": [(frame, cx, cy, w, h), ...],
        #               "vels": [(vx, vy), ...],   # fps 반영, 첫 = (0,0)
        #               "seq_name": str}
        self.tracks: dict[int, dict] = {}

        # (seq_name, frame) → [(track_uid, cx, cy), ...]  — 이웃 탐색용
        self.frame_index: dict[tuple, list] = defaultdict(list)

        self._load()

        # 마스킹 샘플: [(track_uid, mask_start_idx, mask_len), ...]
        self.samples = self._build_samples()
        

    # ── 시퀀스 목록 구성 ──────────────────────────────────────────────────

    def _get_train_dir(self) -> str:
        return os.path.join(self.data_root, "train")

    def _all_train_seqs(self) -> list[str]:
        train_dir = self._get_train_dir()
        if not os.path.isdir(train_dir):
            return []
        entries = sorted(
            e for e in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, e))
        )
        return entries  # e.g. ["BEE24-01", ..., "BEE24-31"]

    def _resolve_sequences(self) -> list[tuple[str, str]]:
        """(seq_name, seq_dir) 목록 반환."""
        if self.split == "test":
            test_dir = os.path.join(self.data_root, "test")
            if not os.path.isdir(test_dir):
                return []
            entries = sorted(
                e for e in os.listdir(test_dir)
                if os.path.isdir(os.path.join(test_dir, e))
            )
            return [(e, os.path.join(test_dir, e)) for e in entries]

        all_seqs = self._all_train_seqs()
        train_dir = self._get_train_dir()

        if self.val_seq_ids is not None:
            val_names = set(self.val_seq_ids)
        else:
            # 마지막 6개를 val로 사용
            val_names = set(all_seqs[-6:]) if len(all_seqs) >= 6 else set(all_seqs)

        if self.split == "val":
            selected = [s for s in all_seqs if s in val_names]
        else:  # train
            selected = [s for s in all_seqs if s not in val_names]

        return [(s, os.path.join(train_dir, s)) for s in selected]

    # ── seqinfo.ini 파싱 ──────────────────────────────────────────────────

    @staticmethod
    def _parse_seqinfo(seq_dir: str) -> float:
        ini_path = os.path.join(seq_dir, "seqinfo.ini")
        fps = 25.0  # 기본값
        if os.path.exists(ini_path):
            cfg = configparser.ConfigParser()
            cfg.read(ini_path)
            try:
                fps = float(cfg["Sequence"]["frameRate"])
            except (KeyError, ValueError):
                pass
        return fps

    # ── 데이터 로드 ───────────────────────────────────────────────────────

    def _load(self):
        seq_list = self._resolve_sequences()
        uid = 0

        for seq_name, seq_dir in seq_list:
            gt_path = os.path.join(seq_dir, "gt", "gt.txt")
            if not os.path.exists(gt_path):
                continue

            fps = self._parse_seqinfo(seq_dir)

            # gt.txt 파싱: (seq_name, track_id) → [(frame, cx, cy, w, h)]
            raw: dict[int, list] = defaultdict(list)
            with open(gt_path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 6:
                        continue
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    raw[track_id].append((frame_id, cx, cy, w, h))

            for track_id, dets in raw.items():
                dets.sort(key=lambda r: r[0])
                if len(dets) < self.min_track_len:
                    continue

                # 속도 계산 (fps 반영): vx = (cx[t] - cx[t-1]) * fps
                vels = [(0.0, 0.0)]  # 첫 프레임 속도 = 0
                for i in range(1, len(dets)):
                    dt = dets[i][0] - dets[i - 1][0]
                    if dt == 0:
                        dt = 1
                    vx = (dets[i][1] - dets[i - 1][1]) * fps / dt
                    vy = (dets[i][2] - dets[i - 1][2]) * fps / dt
                    vels.append((vx, vy))

                self.tracks[uid] = {
                    "dets":     dets,
                    "vels":     vels,
                    "seq_name": seq_name,
                }

                # 프레임 인덱스 (같은 시퀀스 내 이웃 탐색용)
                for frame_id, cx, cy, w, h in dets:
                    self.frame_index[(seq_name, frame_id)].append((uid, cx, cy))

                uid += 1

    # ── 샘플 빌드 ────────────────────────────────────────────────────────

    def _build_samples(self) -> list:
        samples = []
        for track_uid, info in self.tracks.items():
            n = len(info["dets"])
            max_mask_frames = int(n * self.mask_ratio)
            for mask_len in self.mask_lengths:
                if mask_len > max_mask_frames:
                    continue
                lo = 10
                hi = n - mask_len - 10
                if lo >= hi:
                    continue
                for start in range(lo, hi):
                    samples.append((track_uid, start, mask_len))
        return samples

    # ── Dataset 인터페이스 ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        track_uid, mask_start, mask_len = self.samples[idx]
        info = self.tracks[track_uid]
        dets: list = info["dets"]
        vels: list = info["vels"]
        seq_name: str = info["seq_name"]

        obs      = dets[:mask_start]
        masked   = dets[mask_start: mask_start + mask_len]
        obs_vels = vels[:mask_start]

        # 마지막 관측 속도
        vx_last, vy_last = obs_vels[-1] if obs_vels else (0.0, 0.0)

        # ── target_before: 고정 길이 obs_len으로 자르기 ──────────────
        OBS_LEN = 10  # 가림 직전 고정 관측 길이
        pairs = list(zip(obs, obs_vels))

        if len(pairs) >= OBS_LEN:
            pairs = pairs[-OBS_LEN:]          # 마지막 OBS_LEN개만
        else:
            # 부족하면 앞을 0으로 패딩
            pad_n = OBS_LEN - len(pairs)
            dummy_det = (0, 0.0, 0.0, 0.0, 0.0)
            dummy_vel = (0.0, 0.0)
            pairs = [(dummy_det, dummy_vel)] * pad_n + pairs

        target_before = torch.tensor(
            [[cx, cy, vx, vy, w, h]
            for (_, cx, cy, w, h), (vx, vy) in pairs],
            dtype=torch.float32,
        )  # → 항상 (OBS_LEN, 6) = (10, 6)

        # ── target_gt: (T_mask, 2) ───────────────────────────────────
        target_gt = torch.tensor(
            [[cx, cy] for _, cx, cy, _, _ in masked],
            dtype=torch.float32,
        )

        # ── 칼만 예측: 등속도 외삽 ───────────────────────────────────
        cx_last, cy_last = obs[-1][1], obs[-1][2]
        fps = self._get_fps_for_track(track_uid)
        kalman_pred = torch.tensor(
            [[cx_last + vx_last * t / fps,
            cy_last + vy_last * t / fps]
            for t in range(1, mask_len + 1)],
            dtype=torch.float32,
        )  # (T_mask, 2)

        # ── kalman_uncertainty ───────────────────────────────────────
        kalman_uncertainty = _BASE_VAR * (1.0 + mask_len * 0.1)

        # ── 이웃 탐색 ────────────────────────────────────────────────
        frame_start = masked[0][0]
        target_cx, target_cy = masked[0][1], masked[0][2]

        candidates = [
            (n_uid, ncx, ncy)
            for n_uid, ncx, ncy in self.frame_index.get((seq_name, frame_start), [])
            if n_uid != track_uid
        ]
        candidates.sort(
            key=lambda c: (c[1] - target_cx) ** 2 + (c[2] - target_cy) ** 2
        )
        candidates = candidates[: self.k]

        # ── neighbors: (k, T_mask, 4) ────────────────────────────────
        neighbors     = torch.zeros(self.k, mask_len, 4, dtype=torch.float32)
        neighbor_mask = torch.zeros(self.k,           dtype=torch.float32)

        for ni, (n_uid, _, _) in enumerate(candidates):
            n_info = self.tracks[n_uid]
            n_dets = n_info["dets"]
            n_vels = n_info["vels"]

            n_frame_map = {f: (cx, cy, w, h) for f, cx, cy, w, h in n_dets}
            n_vel_map   = {f: vv for (f, *_), vv in zip(n_dets, n_vels)}

            valid = False
            for ti, (f, *_) in enumerate(masked):
                if f in n_frame_map:
                    ncx, ncy, *_ = n_frame_map[f]
                    nvx, nvy = n_vel_map.get(f, (0.0, 0.0))
                    neighbors[ni, ti] = torch.tensor([ncx, ncy, nvx, nvy])
                    valid = True
            if valid:
                neighbor_mask[ni] = 1.0

        vel_magnitude = float((vx_last ** 2 + vy_last ** 2) ** 0.5)

        return {
            "target_before":      target_before,          # (10, 6)  ← 고정
            "target_gt":          target_gt,               # (T_mask, 2)
            "neighbors":          neighbors,               # (k, T_mask, 4)
            "neighbor_mask":      neighbor_mask,           # (k,)
            "kalman_pred":        kalman_pred,             # (T_mask, 2)
            "occluded_frames":    mask_len,
            "neighbor_count":     int(neighbor_mask.sum().item()),
            "kalman_uncertainty": torch.tensor(kalman_uncertainty, dtype=torch.float32),
            "det_score":          torch.tensor(0.0),
            "velocity_magnitude": torch.tensor(vel_magnitude, dtype=torch.float32),
        }

    # ── 헬퍼 ──────────────────────────────────────────────────────────────

    def _get_fps_for_track(self, track_uid: int) -> float:
        """track_uid가 속한 시퀀스의 fps 반환. seqinfo 없으면 25.0."""
        # fps는 로드 시 track에 저장해두지 않았으므로 seqinfo 재파싱 없이
        # seq_name → fps 캐시를 통해 조회한다.
        if not hasattr(self, "_fps_cache"):
            self._fps_cache: dict[str, float] = {}
        seq_name = self.tracks[track_uid]["seq_name"]
        if seq_name not in self._fps_cache:
            # seq_dir 재구성
            if self.split in ("train", "val"):
                seq_dir = os.path.join(self.data_root, "train", seq_name)
            else:
                seq_dir = os.path.join(self.data_root, "test", seq_name)
            self._fps_cache[seq_name] = self._parse_seqinfo(seq_dir)
        return self._fps_cache[seq_name]
