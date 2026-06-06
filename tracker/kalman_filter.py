import torch
from CFG.cfg import cfg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter:
    def __init__(self):
        self.device = cfg.device
        ndim, dt = 4, 1

        self._motion_mat = torch.eye(2 * ndim, 2 * ndim, device=self.device)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = torch.eye(ndim, 2 * ndim, device=self.device)
        
        self._std_weight_position = torch.tensor(1. / 40, device=self.device)
        self._std_weight_velocity = torch.tensor(1. / 160, device=self.device)

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = torch.zeros_like(mean_pos)
        mean = torch.cat([mean_pos, mean_vel])
        
        std = torch.cat([
            2 * self._std_weight_position * measurement[[2, 3, 2, 3]],
            10 * self._std_weight_velocity * measurement[[2, 3, 2, 3]]
        ])
        covariance = torch.diag(torch.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        std_pos = self._std_weight_position * mean[:, [2, 3, 2, 3]]
        std_vel = self._std_weight_velocity * mean[:, [2, 3, 2, 3]]
        
        std_all = torch.cat([std_pos, std_vel], dim=1) # (N, 8)
        motion_cov = torch.diag_embed(torch.square(std_all)) # (N, 8, 8)

        mean = mean @ self._motion_mat.T
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance
    
    def project(self, mean, covariance, score=None, area=None):
        std = self._std_weight_position * mean[:, [2, 3, 2, 3]]
        innovation_cov = torch.diag_embed(torch.square(std))

        r_scale = self.compute_r_scale(score, area)
        innovation_cov = innovation_cov * r_scale[:, None, None]

        mean = mean @ self._update_mat.T
        covariance = self._update_mat @ covariance @ self._update_mat.T
    
        return mean, covariance + innovation_cov
    
    def multi_predict(self, mean, covariance):
        std_pos = self._std_weight_position * mean[:, [2, 3, 2, 3]]
        std_vel = self._std_weight_velocity * mean[:, [2, 3, 2, 3]]

        std_all = torch.cat([std_pos, std_vel], dim=1)

        motion_cov = torch.diag_embed(torch.square(std_all))
    
        mean = mean @ self._motion_mat.T
    
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        
        return mean, covariance
    def compute_r_scale(self, score=None, area=None):
        """
        score와 bbox area를 기반으로 observation noise R scaling 계산.

        score 낮음 -> R 증가
        area 작음  -> R 증가
        """

        # score/area가 없으면 기본 R 사용
        if score is None and area is None:
            return torch.ones(1, device=self.device)

        if score is not None:
            score = torch.as_tensor(score, device=self.device).float()
            score_scale = 1.0 / (score + 1e-6)
            score_scale = torch.clamp(score_scale, min=0.5, max=5.0)
        else:
            score_scale = 1.0

        if area is not None:
            area = torch.as_tensor(area, device=self.device).float()

            # 기준 면적. 일단 synthetic bbox 기준으로 32*32 정도 사용
            ref_area = torch.tensor(32.0 * 32.0, device=self.device)

            area_scale = ref_area / (area + 1e-6)
            area_scale = torch.clamp(area_scale, min=0.5, max=5.0)
        else:
            area_scale = 1.0

        r_scale = score_scale * area_scale

        if not torch.is_tensor(r_scale):
            r_scale = torch.tensor([r_scale], device=self.device).float()

        if r_scale.dim() == 0:
            r_scale = r_scale.unsqueeze(0)

        return r_scale
    
    def compute_q_scale(
        self,
        innovation,
        score=None,
        min_scale=1.0,
        max_scale=10.0,
        residual_ref=20.0
    ):
        """
        innovation residual 기반 process noise Q scaling 계산.

        residual 큼 + score 높음 -> Q 증가
        residual 큼 + score 낮음 -> Q 증가 억제
        """

        # innovation: (N, 4)
        residual_norm = torch.norm(innovation[:, :2], dim=1)

        residual_scale = residual_norm / (residual_ref + 1e-6)

        if score is not None:
            score = torch.as_tensor(score, device=self.device).float()

            if score.dim() == 0:
                score = score.unsqueeze(0)

            score_weight = torch.clamp(score, min=0.0, max=1.0)
        else:
            score_weight = 1.0

        q_scale = 1.0 + residual_scale * score_weight
        q_scale = torch.clamp(q_scale, min=min_scale, max=max_scale)

        return q_scale
        
    def update(self, mean, covariance, measurement, score=None, area=None):
        # 1. adaptive R이 적용된 projection
        projected_mean, projected_cov = self.project(
            mean,
            covariance,
            score=score,
            area=area
        )

        # 2. innovation residual
        innovation = measurement - projected_mean

        # 3. Adaptive Q scaling
        q_scale = self.compute_q_scale(
            innovation,
            score=score
        )

        # covariance를 innovation이 클수록 팽창시킴
        covariance = covariance * q_scale[:, None, None]

        # Q scaling 후 다시 projection
        projected_mean, projected_cov = self.project(
            mean,
            covariance,
            score=score,
            area=area
        )

        B = covariance @ self._update_mat.T

        L = torch.linalg.cholesky(projected_cov)

        kalman_gain = torch.cholesky_solve(
            B.transpose(1, 2), L
        ).transpose(1, 2)

        innovation = measurement - projected_mean

        new_mean = mean + torch.bmm(
            innovation.unsqueeze(1),
            kalman_gain.transpose(1, 2)
        ).squeeze(1)

        new_covariance = covariance - (
            kalman_gain @ projected_cov @ kalman_gain.transpose(1, 2)
        )

        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements):
        projected_mean, projected_cov = self.project(mean, covariance)

        diff = measurements.unsqueeze(0) - projected_mean.unsqueeze(1)
        # diff: (N, M, 4)

        L = torch.linalg.cholesky(projected_cov)

        z = torch.cholesky_solve(diff.transpose(1, 2), L)
        # z: (N, 4, M)

        distance = torch.sum(diff.transpose(1, 2) * z, dim=1)
        # distance: (N, M)

        return distance