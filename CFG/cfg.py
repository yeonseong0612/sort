from easydict import EasyDict

cfg = EasyDict()

cfg.device = 'cuda'

# ── Phase 1: MLP 게이팅 학습 파이프라인 ──────────────────────────────────
cfg.phase1 = EasyDict()

cfg.phase1.data = EasyDict()
cfg.phase1.data.data_root       = ""
cfg.phase1.data.mask_lengths    = [5, 10, 20, 40]
cfg.phase1.data.mask_ratio      = 0.3
cfg.phase1.data.min_track_len   = 30
cfg.phase1.data.k               = 7
cfg.phase1.data.val_ratio       = 0.2

cfg.phase1.model = EasyDict()
cfg.phase1.model.input_dim      = 5
cfg.phase1.model.hidden_dim     = 16
cfg.phase1.model.output_dim     = 2
cfg.phase1.model.alpha_fallback = 0.5

cfg.phase1.loss = EasyDict()
cfg.phase1.loss.lambda_fde      = 2.0
cfg.phase1.loss.alpha_gate      = 1.0
cfg.phase1.loss.beta_reg        = 0.01

cfg.phase1.train = EasyDict()
cfg.phase1.train.epochs         = 50
cfg.phase1.train.batch_size     = 64
cfg.phase1.train.lr             = 1e-3
cfg.phase1.train.scheduler      = "cosine"
cfg.phase1.train.seed           = 42
