import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from CFG.cfg import cfg
from data.bee24_loader import BEE24Dataset
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: GatingNetwork 학습")
    p.add_argument("--data_root",  default=cfg.phase1.data.data_root or None, required=False)
    p.add_argument("--out_dir",    default="checkpoints/phase1")
    p.add_argument("--epochs",     type=int,   default=cfg.phase1.train.epochs)
    p.add_argument("--batch_size", type=int,   default=cfg.phase1.train.batch_size)
    p.add_argument("--lr",         type=float, default=cfg.phase1.train.lr)
    p.add_argument("--k",          type=int,   default=cfg.phase1.data.k)
    p.add_argument("--val_ratio",  type=float, default=cfg.phase1.data.val_ratio)
    p.add_argument("--device",     default=cfg.device)
    p.add_argument("--seed",       type=int,   default=cfg.phase1.train.seed)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """가변 길이 target_before를 패딩해서 배치로 합침."""
    import torch.nn.functional as F

    max_obs = max(b["target_before"].shape[0] for b in batch)
    for b in batch:
        pad = max_obs - b["target_before"].shape[0]
        if pad > 0:
            b["target_before"] = F.pad(b["target_before"], (0, 0, 0, pad))

    keys_tensor = ["target_before", "target_gt", "neighbors", "neighbor_mask",
                   "kalman_pred", "kalman_uncertainty", "det_score", "velocity_magnitude"]
    keys_scalar = ["occluded_frames", "neighbor_count"]

    out = {}
    for k in keys_tensor:
        out[k] = torch.stack([b[k] for b in batch])
    for k in keys_scalar:
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
    return out


def main():
    args = parse_args()

    if not args.data_root:
        raise ValueError("--data_root 경로를 지정하세요 (BEE24 루트 폴더).")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 데이터셋
    dataset = BEE24Dataset(
        data_root=args.data_root,
        split="train",
        mask_lengths=cfg.phase1.data.mask_lengths,
        mask_ratio=cfg.phase1.data.mask_ratio,
        min_track_len=cfg.phase1.data.min_track_len,
        k=args.k,
    )

    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, collate_fn=collate_fn)

    # Trainer
    trainer_cfg = {
        "device":     args.device,
        "k":          args.k,
        "lr":         args.lr,
        "epochs":     args.epochs,
        "lambda_fde": cfg.phase1.loss.lambda_fde,
        "alpha_gate": cfg.phase1.loss.alpha_gate,
        "beta_reg":   cfg.phase1.loss.beta_reg,
    }
    trainer = Trainer(trainer_cfg)

    log_path = os.path.join(args.out_dir, "train_log.csv")
    best_val_ade = float("inf")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_ade", "train_fde",
                         "val_ade_5", "val_ade_10", "val_ade_20", "val_ade_40"])

        for epoch in range(1, args.epochs + 1):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics   = trainer.evaluate(val_loader)

            val_ades = {ml: val_metrics.get(ml, {}).get("ade", float("nan"))
                        for ml in [5, 10, 20, 40]}
            val_ade_mean = sum(v for v in val_ades.values() if not np.isnan(v))

            print(
                f"[Epoch {epoch:3d}/{args.epochs}] "
                f"Loss={train_metrics['total_loss']:.3f} "
                f"ADE={train_metrics['ade']:.1f} "
                f"FDE={train_metrics['fde']:.1f}"
            )
            val_str = "  ".join(
                f"ADE({ml}f)={val_ades[ml]:.1f}" for ml in [5, 10, 20, 40]
            )
            print(f"  Val: {val_str}")

            writer.writerow([epoch, train_metrics["total_loss"],
                             train_metrics["ade"], train_metrics["fde"],
                             val_ades[5], val_ades[10], val_ades[20], val_ades[40]])

            # 체크포인트
            if epoch % 10 == 0:
                trainer.save_checkpoint(epoch, os.path.join(args.out_dir, f"ckpt_epoch{epoch}.pth"))
            if val_ade_mean < best_val_ade:
                best_val_ade = val_ade_mean
                trainer.save_checkpoint(epoch, os.path.join(args.out_dir, "best_model.pth"))

    trainer.save_checkpoint(args.epochs, os.path.join(args.out_dir, "last_model.pth"))

    # 최종 성능 출력
    print("\n=== 최종 평가 (best model) ===")
    trainer.load_checkpoint(os.path.join(args.out_dir, "best_model.pth"))
    final = trainer.evaluate(val_loader)
    for ml, m in sorted(final.items()):
        print(f"  {ml:2d}프레임 가림: ADE={m['ade']:.2f}  FDE={m['fde']:.2f}")


if __name__ == "__main__":
    main()
