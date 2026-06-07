from pathlib import Path
from PIL import Image
import torch


class GTLoader:
    def __init__(self, data_dir: str, img_ext: str = ".jpg"):
        self.data_dir = Path(data_dir)
        self.filtered_dir = self.data_dir / "filtered"

        self.img_paths = sorted(self.data_dir.glob(f"*{img_ext}"))

        self.gt_map = {}
        if self.filtered_dir.exists():
            for txt_path in self.filtered_dir.glob("*.txt"):
                self.gt_map[txt_path.stem] = txt_path

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        stem = img_path.stem

        if stem not in self.gt_map:
            return img_path, torch.zeros((0, 5), dtype=torch.float32)

        img = Image.open(img_path)
        W, H = img.size

        rows = []
        with open(self.gt_map[stem]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                # YOLO format: class cx cy w h (normalized)
                _, cx, cy, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cx_px = cx * W
                cy_px = cy * H
                w_px = w * W
                h_px = h * H
                x1 = cx_px - w_px / 2
                y1 = cy_px - h_px / 2
                x2 = cx_px + w_px / 2
                y2 = cy_px + h_px / 2
                rows.append([x1, y1, x2, y2, 1.0])

        if not rows:
            return img_path, torch.zeros((0, 5), dtype=torch.float32)

        gt_boxes = torch.tensor(rows, dtype=torch.float32)
        return img_path, gt_boxes
