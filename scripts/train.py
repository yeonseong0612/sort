import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from CFG.cfg import cfg
from src.data.bee24_loader import BEE24Dataset
from src.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: GatingNetwork 학습")
    p.add_argument("--data_root",  default=cfg.phase1.data.data_root or None, required=False)
    p.add_argument("--out_dir",    default="checkpoint/phase1")
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
    out = {}
    for k in batch[0].keys():
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = torch.tensor([b[k] for b in batch])
    return out


def main():
    args = parse_args()

    if not args.data_root:
        raise ValueError("--data_root 경로를 지정하세요 (BEE24 루트 폴더).")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    dataset = BEE24Dataset(
        data_root=args.data_root,
        split="train",
        mask_lengths=cfg.phase1.data.mask_lengths,
        mask_ratio=cfg.phase1.data.mask_ratio,
        min_track_len=cfg.phase1.data.min_track_len,
        k=args.k,
    )

    n_val   = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    print(f"Train samples: {n_train}  Val samples: {n_val}")

    # ── Trainer ───────────────────────────────────────────────────────────
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

    # ── CSV 로그 ──────────────────────────────────────────────────────────
    log_path = os.path.join(args.out_dir, "train_log.csv")
    best_val_ade = float("inf")

    csv_header = [
        "epoch",
        "train_loss", "train_ade", "train_fde",
        "train_gate", "train_reg", "train_kalman_ade",
        "val_ade_5",  "val_ade_10",  "val_ade_20",  "val_ade_40",
        "val_kalman_5", "val_kalman_10", "val_kalman_20", "val_kalman_40",
        "val_improve_5", "val_improve_10", "val_improve_20", "val_improve_40",
    ]

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

        for epoch in range(1, args.epochs + 1):
            train_m = trainer.train_epoch(train_loader)
            val_m   = trainer.evaluate(val_loader)

            # 가림 길이별 검증 지표
            ML = [5, 10, 20, 40]
            val_ade     = {ml: val_m.get(ml, {}).get("ade",        float("nan")) for ml in ML}
            val_kalman  = {ml: val_m.get(ml, {}).get("kalman_ade", float("nan")) for ml in ML}
            val_improve = {ml: val_kalman[ml] - val_ade[ml]                       for ml in ML}

            # 유효한 val ADE 평균 (best model 기준)
            valid_ades = [val_ade[ml] for ml in ML if not np.isnan(val_ade[ml])]
            val_ade_mean = sum(valid_ades) / len(valid_ades) if valid_ades else float("inf")

            # ── 콘솔 출력 ──────────────────────────────────────────────
            print(
                f"[Epoch {epoch:3d}/{args.epochs}] "
                f"Loss={train_m['total_loss']:.3f}  "
                f"ADE={train_m['ade']:.2f}  "
                f"FDE={train_m['fde']:.2f}  "
                f"Gate={train_m['gate']:.4f}  "
                f"Reg={train_m['reg']:.4f}  "
                f"Kalman={train_m['kalman_ade']:.2f}"
            )

            # 검증: 게이팅 ADE / 칼만 ADE / 개선량 나란히 출력
            for ml in ML:
                a  = val_ade[ml]
                k  = val_kalman[ml]
                im = val_improve[ml]
                if np.isnan(a):
                    print(f"  Val({ml:2d}f): N/A (샘플 없음)")
                else:
                    print(
                        f"  Val({ml:2d}f): "
                        f"Gate={a:.2f}  Kalman={k:.2f}  "
                        f"개선={im:+.2f}px"
                        f"{'  ✓' if im > 0 else '  ✗'}"
                    )

            # ── CSV 기록 ───────────────────────────────────────────────
            writer.writerow([
                epoch,
                train_m["total_loss"], train_m["ade"], train_m["fde"],
                train_m["gate"],       train_m["reg"], train_m["kalman_ade"],
                val_ade[5],    val_ade[10],    val_ade[20],    val_ade[40],
                val_kalman[5], val_kalman[10], val_kalman[20], val_kalman[40],
                val_improve[5], val_improve[10], val_improve[20], val_improve[40],
            ])
            f.flush()

            # ── 체크포인트 ─────────────────────────────────────────────
            if epoch % 10 == 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch}.pth")
                trainer.save_checkpoint(epoch, ckpt_path)
                print(f"  → 체크포인트 저장: {ckpt_path}")

            if val_ade_mean < best_val_ade:
                best_val_ade = val_ade_mean
                trainer.save_checkpoint(
                    epoch, os.path.join(args.out_dir, "best_model.pth")
                )
                print(f"  → Best model 갱신 (val ADE={val_ade_mean:.2f})")

    trainer.save_checkpoint(
        args.epochs, os.path.join(args.out_dir, "last_model.pth")
    )

    # ── 최종 평가 ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("최종 평가 (best model)")
    print("="*60)
    trainer.load_checkpoint(os.path.join(args.out_dir, "best_model.pth"))
    final = trainer.evaluate(val_loader)

    for ml, m in sorted(final.items()):
        im = m["kalman_ade"] - m["ade"]
        print(
            f"  {ml:2d}프레임 가림: "
            f"Gate ADE={m['ade']:.2f}  "
            f"Kalman ADE={m['kalman_ade']:.2f}  "
            f"개선={im:+.2f}px"
            f"{'  ✓' if im > 0 else '  ✗'}"
        )

    print(f"\n로그 저장: {log_path}")


if __name__ == "__main__":
    main()