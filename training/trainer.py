import os
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tracker.gating_network import GatingNetwork
from tracker.neighbor_imputation import NeighborImputation
from training.loss import OcclusionImputationLoss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))

        self.gating = GatingNetwork().to(self.device)
        self.neighbor_imputation = NeighborImputation(k=config.get("k", 7))

        self.optimizer = optim.Adam(
            self.gating.parameters(),
            lr=config.get("lr", 1e-4),  # 1e-3 → 1e-4 로 낮춤
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 50),
        )
        self.loss_fn = OcclusionImputationLoss(
            lambda_fde=config.get("lambda_fde", 2.0),
            alpha_gate=config.get("alpha_gate", 1.0),
            beta_reg=config.get("beta_reg", 0.01),
        )

    # ── 학습 ─────────────────────────────────────────────────────────────

    def train_epoch(self, dataloader) -> dict:
        self.gating.train()
        total_loss = ade_sum = fde_sum = gate_sum = reg_sum = 0.0
        kalman_ade_sum = 0.0
        n = 0
        kalman_ade_logged = False

        for batch in dataloader:
            kalman_pred     = batch["kalman_pred"].to(self.device)      # (B, T, 2)
            target_gt       = batch["target_gt"].to(self.device)        # (B, T, 2)
            neighbors       = batch["neighbors"].to(self.device)        # (B, k, T, 4)
            neighbor_mask_k = batch["neighbor_mask"].to(self.device)    # (B, k)
            target_before   = batch["target_before"].to(self.device)    # (B, T_obs, 6)

            B, T = kalman_pred.shape[:2]

            # ── 칼만 베이스라인 ADE (첫 배치만 출력, 전체는 누적) ──
            with torch.no_grad():
                k_ade = (kalman_pred - target_gt).norm(dim=-1).mean().item()
                kalman_ade_sum += k_ade * B
                if not kalman_ade_logged:
                    print(f"  [Debug] Kalman-only ADE (first batch): {k_ade:.2f}px")
                    kalman_ade_logged = True

            # ── 게이팅 가중치 ──
            w_motion, w_neighbor = self.gating.get_weights_batch(batch)  # (B,)

            # ── 이웃 평균 속도 ──
            mask_expand     = neighbor_mask_k.unsqueeze(-1).unsqueeze(-1)  # (B, k, 1, 1)
            v_neighbor_full = (neighbors[:, :, :, 2:4] * mask_expand).sum(dim=1)  # (B, T, 2)
            n_count         = neighbor_mask_k.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
            v_neighbor      = v_neighbor_full / n_count                    # (B, T, 2)

            # ── 자신의 마지막 속도 ──
            v_self = target_before[:, -1, 2:4].unsqueeze(1)               # (B, 1, 2)

            # ── 최종 예측 ──
            neighbor_correction = kalman_pred + (v_neighbor - v_self)
            wm   = w_motion.view(B, 1, 1)
            wn   = w_neighbor.view(B, 1, 1)
            pred = wm * kalman_pred + wn * neighbor_correction            # (B, T, 2)

            # ── 손실 ──
            has_neighbor = (neighbor_mask_k.sum(dim=1) > 0).float()
            losses = self.loss_fn(pred, target_gt, w_neighbor, kalman_pred, has_neighbor)

            self.optimizer.zero_grad()
            losses["total"].backward()

            # gradient clipping — 안정적 수렴을 위해 추가
            torch.nn.utils.clip_grad_norm_(self.gating.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += losses["total"].item() * B
            ade_sum    += losses["ade"].item()   * B
            fde_sum    += losses["fde"].item()   * B
            gate_sum   += losses["gate"].item()  * B
            reg_sum    += losses["reg"].item()   * B
            n          += B

        self.scheduler.step()

        denom = max(n, 1)
        return {
            "total_loss":  total_loss      / denom,
            "ade":         ade_sum         / denom,
            "fde":         fde_sum         / denom,
            "gate":        gate_sum        / denom,
            "reg":         reg_sum         / denom,
            "kalman_ade":  kalman_ade_sum  / denom,  # 에폭 전체 칼만 ADE
        }

    # ── 검증 ─────────────────────────────────────────────────────────────

    def evaluate(self, dataloader) -> dict:
        self.gating.eval()
        stats = defaultdict(lambda: {"ade": 0.0, "fde": 0.0,
                                     "kalman_ade": 0.0, "n": 0})

        with torch.no_grad():
            for batch in dataloader:
                kalman_pred     = batch["kalman_pred"].to(self.device)
                target_gt       = batch["target_gt"].to(self.device)
                neighbors       = batch["neighbors"].to(self.device)
                neighbor_mask_k = batch["neighbor_mask"].to(self.device)
                target_before   = batch["target_before"].to(self.device)

                B, T = kalman_pred.shape[:2]

                w_motion, w_neighbor = self.gating.get_weights_batch(batch)

                mask_expand     = neighbor_mask_k.unsqueeze(-1).unsqueeze(-1)
                v_neighbor_full = (neighbors[:, :, :, 2:4] * mask_expand).sum(dim=1)
                n_count         = neighbor_mask_k.sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)
                v_neighbor      = v_neighbor_full / n_count

                v_self = target_before[:, -1, 2:4].unsqueeze(1)

                neighbor_correction = kalman_pred + (v_neighbor - v_self)
                wm   = w_motion.view(B, 1, 1)
                wn   = w_neighbor.view(B, 1, 1)
                pred = wm * kalman_pred + wn * neighbor_correction

                ade_per    = (pred - target_gt).norm(dim=-1).mean(dim=1)       # (B,)
                fde_per    = (pred[:, -1] - target_gt[:, -1]).norm(dim=-1)     # (B,)
                k_ade_per  = (kalman_pred - target_gt).norm(dim=-1).mean(dim=1) # (B,)

                occluded_frames = batch["occluded_frames"]
                occ_list = (occluded_frames.tolist()
                            if torch.is_tensor(occluded_frames)
                            else list(occluded_frames))

                for i, occ in enumerate(occ_list):
                    key = int(occ)
                    stats[key]["ade"]        += ade_per[i].item()
                    stats[key]["fde"]        += fde_per[i].item()
                    stats[key]["kalman_ade"] += k_ade_per[i].item()
                    stats[key]["n"]          += 1

        result = {}
        for mask_len, s in sorted(stats.items()):
            nn = max(s["n"], 1)
            result[mask_len] = {
                "ade":        s["ade"]        / nn,
                "fde":        s["fde"]        / nn,
                "kalman_ade": s["kalman_ade"] / nn,  # 칼만 베이스라인도 함께 출력
            }
        return result

    # ── 체크포인트 ────────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "epoch":     epoch,
            "model":     self.gating.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.gating.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt.get("epoch", 0)