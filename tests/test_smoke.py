import unittest

import torch

from src.models import LatentDiffusionTransformer, TransformerBackbone
from src.training.config import TrainingConfig


class CPDiTSmokeTests(unittest.TestCase):
    def test_training_config_loads_nested_yaml(self):
        cfg = TrainingConfig.from_yaml("configs/train_config.yaml")
        self.assertEqual(cfg.latent_dim, 256)
        self.assertEqual(cfg.context_length, 12)
        self.assertEqual(cfg.forecast_length, 6)
        self.assertEqual(cfg.stage, 2)
        self.assertEqual(cfg.freeze_vae, True)
        self.assertEqual(cfg.image_channels, 5)

    def test_transformer_forward_shape(self):
        model = TransformerBackbone(latent_dim=16, num_layers=2, num_heads=4, feedforward_dim=32, max_seq_len=32)
        x = torch.randn(2, 8, 16)
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_model_forward(self):
        model = LatentDiffusionTransformer(image_channels=3, image_size=64, latent_dim=16, num_transformer_layers=2, num_heads=4, feedforward_dim=32, num_diffusion_steps=10, denoiser_hidden_dim=32)
        context = torch.randn(1, 12, 3, 64, 64)
        target = torch.randn(1, 6, 3, 64, 64)
        loss, _ = model(context, target)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
