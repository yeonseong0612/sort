import torch
import torch.nn as nn


class OcclusionImputationLoss(nn.Module):
    def __init__(self, lambda_fde: float = 2.0, alpha_gate: float = 1.0, beta_reg: float = 0.01):
        super().__init__()
        self.lambda_fde = lambda_fde
        self.alpha_gate = alpha_gate
        self.beta_reg   = beta_reg

    def forward(
        self,
        pred: torch.Tensor,          # (B, T, 2)
        gt: torch.Tensor,            # (B, T, 2)
        w_neighbor: torch.Tensor,    # (B,)
        kalman_pred: torch.Tensor,   # (B, T, 2)
        neighbor_mask: torch.Tensor, # (B,)  1=이웃 있음
    ) -> dict:
        # ADE
        L_ADE = (pred - gt).norm(dim=-1).mean()

        # FDE
        L_FDE = (pred[:, -1, :] - gt[:, -1, :]).norm(dim=-1).mean()

        L_impute = L_ADE + self.lambda_fde * L_FDE

        # 이웃 없는 샘플에서 w_neighbor → 0 유도
        no_neighbor = neighbor_mask == 0
        if no_neighbor.any():
            L_gate = (w_neighbor[no_neighbor] ** 2).mean()
        else:
            L_gate = torch.tensor(0.0, device=pred.device)

        L_reg = (w_neighbor ** 2).mean()

        L_total = L_impute + self.alpha_gate * L_gate + self.beta_reg * L_reg

        return {
            "total": L_total,
            "ade":   L_ADE,
            "fde":   L_FDE,
            "gate":  L_gate,
            "reg":   L_reg,
        }
