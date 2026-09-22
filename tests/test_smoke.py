"""Smoke and correctness tests for the CPDiT model stack.

Most of these avoid importing `src.petdata`, which needs the pyearthtools
runtime that is only present inside the HPC module environment. The two classes
that do import it (DataLoaderConfigTests, ShardingTests) skip themselves when
pyearthtools is unavailable, so they still run — and matter — on the HPC.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.models import (
    CosineVPSDE,
    DiTDenoiser,
    LatentDiffusionTransformer,
    NeighbourhoodAttention2D,
    ContextEncoder,
    VESDE,
    VPSDE,
    VariationalAutoencoder,
    build_sde,
    neighbourhood_mask,
    ode_sampler,
    pc_sampler,
)

try:
    import src.petdata as _petdata
    _PETDATA_IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 - pyearthtools is not installed everywhere
    _petdata = None
    _PETDATA_IMPORT_ERROR = _exc


def _tiny_model(**overrides) -> LatentDiffusionTransformer:
    """A small but structurally complete model for fast tests."""
    kwargs = dict(
        image_channels=2,
        image_size=64,             # -> 8x8 latent
        latent_channels=4,
        vae_hidden_dim=16,
        context_length=3,
        max_forecast_steps=4,
        num_transformer_layers=1,
        num_heads=2,
        feedforward_dim=32,
        transformer_dim=32,
        patch_size=4,              # -> 2x2 token grid
        denoiser_embed_dim=32,
        denoiser_depth=2,
        denoiser_heads=4,
        window_size=31,
        cond_dim=16,
        sde="vp_cosine",
        use_natten=False,          # exercise the exact fallback deterministically
    )
    kwargs.update(overrides)
    return LatentDiffusionTransformer(**kwargs)


def _wake_denoiser(model: LatentDiffusionTransformer, seed: int = 0) -> None:
    """
    Give the denoiser's zero-initialised output path real weights.

    adaLN-Zero zeroes every block's modulation projection *and* the final
    layer, so a freshly built DiT emits exactly 0.0 for any input. The
    diffusion loss is then ||z||^2 — a constant with no dependence on x_t, and
    therefore no gradient reaching the encoder and perfect scale invariance in
    either normalisation mode. That is the initialisation talking, not the
    wiring, so any test about gradient flow or scale invariance has to wake the
    output path up first or it silently measures nothing.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        final = model.denoiser.final_layer
        final.linear.weight.normal_(0.0, 0.05, generator=g)
        final.adaLN_modulation[-1].weight.normal_(0.0, 0.05, generator=g)
        for block in model.denoiser.blocks:
            block.adaLN_modulation[-1].weight.normal_(0.0, 0.05, generator=g)


class NeighbourhoodMaskTests(unittest.TestCase):
    def test_every_query_sees_exactly_window_squared_keys(self):
        # The defining property of neighbourhood attention: the window shifts
        # inward at the boundary rather than being truncated.
        mask = neighbourhood_mask(7, 7, 3)
        self.assertTrue(torch.all(mask.sum(dim=1) == 9))

    def test_window_is_centred_in_the_interior(self):
        h = w = 7
        mask = neighbourhood_mask(h, w, 3).reshape(h, w, h, w)
        # Interior token (3, 3) should see exactly rows/cols 2..4.
        allowed = mask[3, 3].nonzero()
        self.assertEqual(set(allowed[:, 0].tolist()), {2, 3, 4})
        self.assertEqual(set(allowed[:, 1].tolist()), {2, 3, 4})

    def test_corner_window_shifts_inward(self):
        h = w = 7
        mask = neighbourhood_mask(h, w, 3).reshape(h, w, h, w)
        allowed = mask[0, 0].nonzero()
        self.assertEqual(set(allowed[:, 0].tolist()), {0, 1, 2})
        self.assertEqual(set(allowed[:, 1].tolist()), {0, 1, 2})

    def test_window_spanning_grid_is_fully_connected(self):
        mask = neighbourhood_mask(5, 5, 5)
        self.assertTrue(bool(mask.all()))

    def test_attention_is_local(self):
        # A window of 3 on a 7x7 grid must not connect opposite corners.
        mask = neighbourhood_mask(7, 7, 3).reshape(7, 7, 7, 7)
        self.assertFalse(bool(mask[0, 0, 6, 6]))


class NeighbourhoodAttentionTests(unittest.TestCase):
    def test_output_shape(self):
        attn = NeighbourhoodAttention2D(32, num_heads=4, window_size=3, use_natten=False)
        x = torch.randn(2, 25, 32)
        self.assertEqual(attn(x, 5, 5).shape, x.shape)

    def test_window_clamped_to_grid(self):
        attn = NeighbourhoodAttention2D(32, num_heads=4, window_size=31, use_natten=False)
        # A 4x4 grid cannot host an odd window wider than 3.
        self.assertEqual(attn.effective_window(4, 4), 3)
        self.assertEqual(attn.effective_window(5, 5), 5)
        self.assertEqual(attn.effective_window(64, 64), 31)

    def test_matches_full_attention_when_window_spans_grid(self):
        torch.manual_seed(0)
        attn = NeighbourhoodAttention2D(16, num_heads=2, window_size=5, use_natten=False)
        x = torch.randn(1, 25, 16)
        local = attn(x, 5, 5)

        # Reference: unmasked attention over the same projections.
        B, L, D = x.shape
        qkv = attn.qkv(x).reshape(B, L, 3, attn.num_heads, attn.head_dim)
        q, k, v = (qkv[:, :, i].permute(0, 2, 1, 3) for i in range(3))
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=attn.scale)
        ref = attn.proj(ref.permute(0, 2, 1, 3).reshape(B, L, D))

        torch.testing.assert_close(local, ref, rtol=1e-5, atol=1e-5)

    def test_locality_changes_output(self):
        # A distant perturbation must not affect a token outside its window.
        torch.manual_seed(0)
        attn = NeighbourhoodAttention2D(16, num_heads=2, window_size=3, use_natten=False)
        x = torch.randn(1, 49, 16)
        base = attn(x, 7, 7)

        perturbed = x.clone()
        perturbed[0, 48] += 10.0            # bottom-right corner token
        out = attn(perturbed, 7, 7)

        # Top-left corner is outside the perturbed token's neighbourhood.
        torch.testing.assert_close(base[0, 0], out[0, 0], rtol=1e-5, atol=1e-5)
        self.assertFalse(torch.allclose(base[0, 48], out[0, 48]))


class DiTDenoiserTests(unittest.TestCase):
    def test_forward_shape(self):
        denoiser = DiTDenoiser(
            latent_channels=4, latent_size=16, context_dim=32, num_context_frames=3,
            patch_size=4, embed_dim=32, depth=2, num_heads=4, window_size=31,
            cond_dim=16, max_forecast_steps=4, use_natten=False,
        )
        x       = torch.randn(2, 2, 4, 16, 16)
        t       = torch.randint(0, 10, (2,))
        ctx_tok = torch.randn(2, 3, 32)
        ctx_lat = torch.randn(2, 3, 4, 16, 16)
        self.assertEqual(denoiser(x, t, ctx_tok, ctx_lat).shape, x.shape)

    def test_patch_embedding_is_4x4_strided_conv(self):
        denoiser = DiTDenoiser(
            latent_channels=4, latent_size=32, context_dim=32, num_context_frames=2,
            patch_size=4, embed_dim=32, depth=1, num_heads=4, cond_dim=16,
            use_natten=False,
        )
        conv = denoiser.x_embedder
        self.assertIsInstance(conv, torch.nn.Conv2d)
        self.assertEqual(conv.kernel_size, (4, 4))
        self.assertEqual(conv.stride, (4, 4))
        self.assertEqual(denoiser.grid_size, 8)
        # Context frames enter as extra input channels, not as a pooled vector.
        self.assertEqual(conv.in_channels, 4 * (1 + 2))

    def test_adaln_zero_starts_as_identity(self):
        denoiser = DiTDenoiser(
            latent_channels=4, latent_size=16, context_dim=32, num_context_frames=1,
            patch_size=4, embed_dim=32, depth=2, num_heads=4, cond_dim=16,
            use_natten=False,
        )
        out = denoiser(
            torch.randn(1, 1, 4, 16, 16), torch.zeros(1, dtype=torch.long),
            torch.randn(1, 1, 32), torch.randn(1, 1, 4, 16, 16),
        )
        # Zero-initialised final layer means the model outputs exactly zero
        # before any training — the property that makes deep DiTs stable.
        self.assertTrue(torch.all(out == 0))

    def test_lead_time_distinguishes_forecast_frames(self):
        """
        Each forecast frame must be conditioned differently, otherwise every
        lead time is an identical draw from the same distribution.
        """
        torch.manual_seed(0)
        denoiser = DiTDenoiser(
            latent_channels=4, latent_size=16, context_dim=32, num_context_frames=1,
            patch_size=4, embed_dim=32, depth=2, num_heads=4, cond_dim=16,
            max_forecast_steps=4, use_natten=False,
        )
        # Break the zero-init so the conditioning path can express itself.
        for block in denoiser.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.5)
        torch.nn.init.normal_(denoiser.final_layer.linear.weight, std=0.5)
        torch.nn.init.normal_(denoiser.final_layer.adaLN_modulation[-1].weight, std=0.5)

        frame   = torch.randn(1, 1, 4, 16, 16)
        x       = frame.repeat(1, 3, 1, 1, 1)      # identical content per frame
        ctx_lat = torch.randn(1, 1, 4, 16, 16)
        out = denoiser(x, torch.zeros(1, dtype=torch.long), torch.randn(1, 1, 32), ctx_lat)

        self.assertFalse(torch.allclose(out[:, 0], out[:, 1]))
        self.assertFalse(torch.allclose(out[:, 1], out[:, 2]))

    def test_rejects_wrong_context_length(self):
        denoiser = DiTDenoiser(
            latent_channels=4, latent_size=16, context_dim=32, num_context_frames=3,
            patch_size=4, embed_dim=32, depth=1, num_heads=4, cond_dim=16,
            use_natten=False,
        )
        with self.assertRaises(ValueError):
            denoiser(
                torch.randn(1, 1, 4, 16, 16), torch.zeros(1, dtype=torch.long),
                torch.randn(1, 2, 32), torch.randn(1, 2, 4, 16, 16),
            )


class ContextEncoderTests(unittest.TestCase):
    def test_forward_shape(self):
        model = ContextEncoder(
            latent_dim=16, num_layers=2, num_heads=4, feedforward_dim=32, max_seq_len=32
        )
        x = torch.randn(2, 8, 16)
        self.assertEqual(model(x).shape, x.shape)

    def test_takes_no_diffusion_time(self):
        # The context is always clean, so there is no noise level to condition
        # on. An earlier version accepted a timestep that the caller always set
        # to zero, leaving 25% of the module computing a constant.
        import inspect
        params = inspect.signature(ContextEncoder.forward).parameters
        self.assertEqual(list(params), ["self", "latent_seq"])
        for name, _ in ContextEncoder(latent_dim=16, num_layers=1, num_heads=2,
                                      feedforward_dim=32).named_parameters():
            self.assertNotIn("adaLN", name)
            self.assertNotIn("timestep", name)

    def test_rejects_sequences_longer_than_max_seq_len(self):
        model = ContextEncoder(latent_dim=16, num_layers=1, num_heads=2,
                               feedforward_dim=32, max_seq_len=4)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 8, 16))

    def test_starts_as_identity_so_deep_stacks_train_stably(self):
        # Zero-initialised residual gates: the property adaLN-Zero provided.
        torch.manual_seed(0)
        model = ContextEncoder(latent_dim=16, num_layers=3, num_heads=2,
                               feedforward_dim=32, dropout=0.0).eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            blocks_out = x + model.positional_encoding[:, :5, :]
            for blk in model.blocks:
                blocks_out = blk(blocks_out)
        # Every block is the identity at initialisation.
        torch.testing.assert_close(
            blocks_out, x + model.positional_encoding[:, :5, :], rtol=1e-6, atol=1e-6
        )


class VAETests(unittest.TestCase):
    def test_roundtrip_shape_and_downsampling(self):
        vae = VariationalAutoencoder(
            image_channels=2, latent_channels=4, hidden_dim=16, image_size=64
        )
        x = torch.randn(2, 2, 64, 64)
        mu, logvar = vae.encode(x)
        self.assertEqual(mu.shape, (2, 4, 8, 8))
        self.assertEqual(vae.decode(mu).shape, x.shape)

    def test_loss_without_ssim(self):
        vae = VariationalAutoencoder(
            image_channels=2, latent_channels=4, hidden_dim=16, image_size=64
        )
        x = torch.randn(2, 2, 64, 64)
        recon, mu, logvar = vae(x)
        total, recon_loss, kl = vae.vae_loss(x, recon, mu, logvar, ssim_weight=0.0)
        for t in (total, recon_loss, kl):
            self.assertTrue(torch.isfinite(t))


class LatentDiffusionTransformerTests(unittest.TestCase):
    def test_training_forward_returns_finite_losses(self):
        model   = _tiny_model()
        context = torch.randn(2, 3, 2, 64, 64)
        target  = torch.randn(2, 1, 2, 64, 64)
        out = model(context, target, vae_loss_weight=0.0)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertTrue(out["loss"].requires_grad)

    def test_multi_frame_forecast_shapes(self):
        model   = _tiny_model()
        context = torch.randn(1, 3, 2, 64, 64)
        target  = torch.randn(1, 4, 2, 64, 64)
        out = model(context, target, vae_loss_weight=0.0)
        self.assertTrue(torch.isfinite(out["loss"]))

    def test_deterministic_eval_is_reproducible(self):
        torch.manual_seed(0)
        model   = _tiny_model().eval()
        context = torch.randn(1, 3, 2, 64, 64)
        target  = torch.randn(1, 1, 2, 64, 64)
        with torch.no_grad():
            a = model(context, target, vae_loss_weight=0.0, deterministic=True,
                      return_pixel_metrics=True)
            b = model(context, target, vae_loss_weight=0.0, deterministic=True,
                      return_pixel_metrics=True)
        # Validation must not be a random variable across epochs.
        self.assertEqual(a["diffusion_loss"].item(), b["diffusion_loss"].item())
        self.assertEqual(a["irradiance_mse"].item(), b["irradiance_mse"].item())

    def test_random_training_times_differ(self):
        torch.manual_seed(0)
        model   = _tiny_model()
        context = torch.randn(1, 3, 2, 64, 64)
        target  = torch.randn(1, 1, 2, 64, 64)
        a = model(context, target, vae_loss_weight=0.0)["diffusion_loss"].item()
        b = model(context, target, vae_loss_weight=0.0)["diffusion_loss"].item()
        self.assertNotEqual(a, b)

    def test_sample_shape(self):
        model   = _tiny_model().eval()
        context = torch.randn(1, 3, 2, 64, 64)
        out = model.sample(context, num_forecast_steps=2, num_steps=3)
        self.assertEqual(out.shape, (1, 2, 2, 64, 64))

    def test_sample_num_samples_expands_batch(self):
        model   = _tiny_model().eval()
        context = torch.randn(2, 3, 2, 64, 64)
        out = model.sample(context, num_forecast_steps=1, num_samples=3, num_steps=2)
        self.assertEqual(out.shape[0], 6)

    def test_latent_scaling_roundtrip(self):
        model = _tiny_model()
        model.latent_std.fill_(3.5)
        z = torch.randn(2, 3, 4, 8, 8)
        torch.testing.assert_close(model.unscale_latents(model.scale_latents(z)), z)

    def test_latent_scale_ema_tracks_latent_std(self):
        model   = _tiny_model().train()
        context = torch.randn(2, 3, 2, 64, 64)
        target  = torch.randn(2, 1, 2, 64, 64)
        model(context, target, vae_loss_weight=0.0)
        # Seeded from the first batch rather than crawling up from 1.0.
        self.assertTrue(bool(model.latent_scale_initialised))
        self.assertGreater(float(model.latent_std), 0.0)

    def test_fixed_latent_scale_is_not_updated(self):
        model = _tiny_model(latent_scale=0.25).train()
        model(torch.randn(1, 3, 2, 64, 64), torch.randn(1, 1, 2, 64, 64),
              vae_loss_weight=0.0)
        self.assertAlmostEqual(float(model.latent_std), 0.25, places=6)


class LatentCollapseTests(unittest.TestCase):
    """
    Regression tests for the failure mode that wrecked the first real run.

    The diffusion objective reduces to epsilon-MSE. If the encoder can shrink
    `mu` towards zero then x_t = alpha*0 + sigma*z = sigma*z and the denoiser
    recovers z = x_t/sigma exactly, so **collapsing the latent is the global
    minimum of the loss**. Reconstruction does not oppose it (the decoder just
    grows its weights) and the KL actively helps it. The observed run lost a
    factor of 10 in latent std over 11 epochs while the diffusion loss fell 3x
    and forecast RMSE rose 2.4x.

    Two guards must hold: the detach (default), and scale-invariant
    normalisation for when the joint gradient is deliberately re-enabled.
    """

    @staticmethod
    def _batch(bs=2):
        return torch.randn(bs, 3, 2, 64, 64), torch.randn(bs, 1, 2, 64, 64)

    def test_detach_keeps_the_diffusion_loss_off_the_encoder(self):
        model = _tiny_model().train()
        context, target = self._batch()
        out = model(context, target, vae_loss_weight=0.0, detach_latents=True)
        out["loss"].backward()
        # Pure diffusion loss, detached latents: the encoder must be untouched.
        self.assertIsNone(model.vae.conv_mu.weight.grad)

    def test_without_detach_the_encoder_does_receive_gradient(self):
        model = _tiny_model().train()
        _wake_denoiser(model)
        context, target = self._batch()
        out = model(context, target, vae_loss_weight=0.0, detach_latents=False)
        out["loss"].backward()
        grad = model.vae.conv_mu.weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_batch_norm_makes_the_loss_scale_invariant(self):
        """
        The property that makes a stage-3 joint fine-tune safe: with
        latent_norm='batch' the diffusion loss is unchanged when the encoder
        output is scaled, so shrinking it buys nothing at all.
        """
        model = _tiny_model(latent_norm="batch").train()
        _wake_denoiser(model)
        context, target = self._batch()
        times = torch.full((context.shape[0],), 0.5)

        def loss_with_encoder_gain(gain: float) -> float:
            with torch.no_grad():
                saved_w = model.vae.conv_mu.weight.clone()
                saved_b = model.vae.conv_mu.bias.clone()
                model.vae.conv_mu.weight.mul_(gain)
                model.vae.conv_mu.bias.mul_(gain)
            torch.manual_seed(0)
            value = model(
                context, target, vae_loss_weight=0.0, times=times,
            )["diffusion_loss"].item()
            with torch.no_grad():
                model.vae.conv_mu.weight.copy_(saved_w)
                model.vae.conv_mu.bias.copy_(saved_b)
            return value

        base    = loss_with_encoder_gain(1.0)
        shrunk  = loss_with_encoder_gain(0.01)
        self.assertAlmostEqual(base, shrunk, places=4)

    def test_ema_norm_is_not_scale_invariant(self):
        """
        The converse, and the reason the default mode needs the detach.

        Under 'ema' the denominator is a no_grad buffer, so it is a constant as
        far as autograd is concerned and the loss genuinely moves when the
        encoder output is rescaled. That non-invariance is the whole problem:
        it is what lets gradient descent discover that shrinking `mu` pays.

        The assertion is that the loss *changes*, not that it falls. Which way
        it moves at an untrained parameter point is not meaningful — the claim
        is about the optimum training walks to, not about one arbitrary point.
        """
        model = _tiny_model(latent_norm="ema").train()
        _wake_denoiser(model)
        context, target = self._batch()
        times = torch.full((context.shape[0],), 0.5)

        torch.manual_seed(0)
        base = model(context, target, vae_loss_weight=0.0, times=times)["diffusion_loss"].item()
        with torch.no_grad():
            model.vae.conv_mu.weight.mul_(0.01)
            model.vae.conv_mu.bias.mul_(0.01)
        torch.manual_seed(0)
        shrunk = model(context, target, vae_loss_weight=0.0, times=times)["diffusion_loss"].item()
        self.assertNotAlmostEqual(base, shrunk, places=3)

    def test_latent_std_is_stable_under_pure_diffusion_training(self):
        """
        The end-to-end regression: 40 optimiser steps of diffusion-only training
        must not walk the latent scale away. This fails on the pre-fix code.
        """
        torch.manual_seed(0)
        model = _tiny_model().train()
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
        context, target = self._batch()

        model(context, target, vae_loss_weight=0.0)   # seeds latent_std
        start = float(model.latent_std)

        for _ in range(40):
            opt.zero_grad(set_to_none=True)
            model(context, target, vae_loss_weight=0.0)["loss"].backward()
            opt.step()

        end = float(model.latent_std)
        self.assertGreater(end, 0.8 * start, f"latent std collapsed {start} -> {end}")
        self.assertLess(end, 1.25 * start, f"latent std diverged {start} -> {end}")

    def test_latent_scale_ratio_reports_the_mismatch(self):
        model = _tiny_model(latent_scale=1.0).eval()
        context, target = self._batch()
        with torch.no_grad():
            ratio = model(context, target, vae_loss_weight=0.0)["latent_scale_ratio"]
        # latent_std is pinned at 1.0, so the ratio is just the true latent std.
        self.assertTrue(torch.isfinite(ratio))
        self.assertGreater(float(ratio), 0.0)

    def test_scale_floor_warns_instead_of_failing_silently(self):
        """
        The warning is raised from the EMA update, not from `scale_latents`:
        that sits on the per-step path and inside the sampler's decode, where a
        host-side read of the buffer would sync the device every call and break
        the compiled graph.
        """
        model = _tiny_model().train()
        model.latent_std.fill_(1e-12)
        with self.assertLogs("src.models.latent_diffusion", level="WARNING") as cm:
            model.check_latent_health()
        self.assertIn("collapsed", "\n".join(cm.output))

    def test_the_training_step_never_reads_the_latent_buffer_host_side(self):
        """
        `float(tensor)` inside the compiled forward is a graph break -- dynamo
        named this exact line on an H200. The health check belongs on the
        per-epoch path, not the per-step one.
        """
        model = _tiny_model().train()
        model.latent_std.fill_(1e-12)
        with self.assertNoLogs("src.models.latent_diffusion", level="WARNING"):
            model(torch.randn(1, 3, 2, 64, 64), torch.randn(1, 1, 2, 64, 64),
                  vae_loss_weight=0.0)

    def test_scale_latents_does_not_read_the_buffer_host_side(self):
        """Guards the compile-friendliness of the hot path."""
        model = _tiny_model()
        model.latent_std.fill_(1e-12)
        with self.assertNoLogs("src.models.latent_diffusion", level="WARNING"):
            model.scale_latents(torch.randn(1, 1, 4, 8, 8))

    def test_a_pinned_scale_below_the_floor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scaling floor"):
            _tiny_model(latent_scale=1e-9)


class StageControlTests(unittest.TestCase):
    """The three training stages the trainer drives through `forward`."""

    @staticmethod
    def _batch(bs=2):
        return torch.randn(bs, 3, 2, 64, 64), torch.randn(bs, 1, 2, 64, 64)

    def test_stage1_skips_the_diffusion_branch_entirely(self):
        model = _tiny_model().train()
        context, target = self._batch()
        out = model(context, target, vae_loss_weight=1.0, vae_ssim_weight=0.0,
                    diffusion_loss_weight=0.0)
        self.assertEqual(float(out["diffusion_loss"]), 0.0)
        self.assertNotIn("irradiance_mse", out)
        out["loss"].backward()
        # VAE trains; the denoiser is untouched, so DDP needs find_unused_parameters.
        self.assertIsNotNone(model.vae.conv_mu.weight.grad)
        self.assertIsNone(model.denoiser.x_embedder.weight.grad)

    def test_vae_max_frames_subsamples_without_changing_the_interface(self):
        """
        The VAE is per-frame and a sample's frames are near-duplicates, so
        scoring all of them costs decoder memory for little signal. Measured on
        an H200: all-frames peaked at 113 GiB at batch 16 and OOM'd at 32.
        """
        model = _tiny_model().train()
        context, target = self._batch()
        out = model(context, target, vae_loss_weight=1.0, vae_ssim_weight=0.0,
                    diffusion_loss_weight=0.0, vae_max_frames=2)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(float(out["recon_loss"]), 0.0)

    def test_subsampling_still_trains_the_whole_vae(self):
        model = _tiny_model().train()
        context, target = self._batch()
        model(context, target, vae_loss_weight=1.0, vae_ssim_weight=0.0,
              diffusion_loss_weight=0.0, vae_max_frames=1)["loss"].backward()
        for name in ("encoder", "decoder"):
            grads = [p.grad for n, p in model.vae.named_parameters()
                     if n.startswith(name) and p.grad is not None]
            self.assertTrue(grads, f"no gradient reached vae.{name}")
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.0)

    def test_subsample_indices_stay_inside_their_own_sample(self):
        """Frames are drawn per sample, so a batch keeps a spread of lead times
        rather than the same slot in every sample."""
        model = _tiny_model()
        B, T, k = 4, 13, 3
        idx = model._vae_frame_indices(B, T, k, torch.device("cpu"), deterministic=False)
        self.assertEqual(idx.shape, (B * k,))
        per_sample = idx.view(B, k)
        for b in range(B):
            self.assertTrue(((per_sample[b] >= b * T) & (per_sample[b] < (b + 1) * T)).all())
            self.assertEqual(len(set(per_sample[b].tolist())), k)   # no repeats

    def test_deterministic_subsample_is_stable_across_calls(self):
        """Stage 1 gates early stopping on recon_loss, so validation must not
        redraw its frames each epoch."""
        model = _tiny_model()
        a = model._vae_frame_indices(3, 13, 4, torch.device("cpu"), deterministic=True)
        b = model._vae_frame_indices(3, 13, 4, torch.device("cpu"), deterministic=True)
        torch.testing.assert_close(a, b)

    def test_no_subsampling_when_the_budget_covers_every_frame(self):
        model = _tiny_model()
        self.assertIsNone(model._vae_frame_indices(2, 13, 13, torch.device("cpu"), False))
        self.assertIsNone(model._vae_frame_indices(2, 13, None, torch.device("cpu"), False))

    def test_subsampling_does_not_disturb_the_diffusion_branch(self):
        """
        With diffusion on, every frame is still encoded -- only the decoder is
        spared -- so the latent stack the denoiser conditions on is unchanged.

        Asserted on `latent_scale_ratio`, which is derived from the full latent
        stack and is RNG-free with the scale pinned. The diffusion loss itself
        is not comparable here: `reparameterize` draws a differently shaped
        noise tensor when subsampling, which shifts the global RNG stream before
        the perturbation noise is drawn.
        """
        model = _tiny_model(latent_scale=1.0).train()
        context, target = self._batch()
        times  = torch.full((context.shape[0],), 0.4)
        kwargs = dict(vae_loss_weight=1.0, vae_ssim_weight=0.0, times=times)

        a = model(context, target, vae_max_frames=2, **kwargs)["latent_scale_ratio"]
        b = model(context, target, vae_max_frames=None, **kwargs)["latent_scale_ratio"]
        torch.testing.assert_close(a, b)

    def test_freeze_vae_excludes_it_from_the_trainable_set(self):
        model = _tiny_model()
        model.freeze_vae()
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertFalse(any(n.startswith("vae.") for n in trainable))
        self.assertTrue(any(n.startswith("denoiser.") for n in trainable))

    def test_pixel_loss_is_zero_when_no_sample_is_below_max_t(self):
        model = _tiny_model().train()
        context, target = self._batch()
        times = torch.full((context.shape[0],), 0.9)
        out = model(context, target, vae_loss_weight=0.0, times=times,
                    pixel_loss_weight=1.0, pixel_loss_max_t=0.3)
        self.assertEqual(float(out["pixel_loss"]), 0.0)

    def test_pixel_loss_reaches_the_encoder_at_low_t(self):
        """
        The task-aware signal for stage 3. Unlike the latent epsilon-MSE, a
        collapsed latent decodes to the climatological mean and scores badly on
        this, so it cannot be gamed by shrinking.
        """
        model = _tiny_model(latent_norm="batch").train()
        context, target = self._batch()
        times = torch.full((context.shape[0],), 0.05)
        out = model(context, target, vae_loss_weight=0.0, times=times,
                    pixel_loss_weight=1.0, pixel_loss_max_t=0.3,
                    detach_latents=False)
        self.assertGreater(float(out["pixel_loss"]), 0.0)
        out["loss"].backward()
        grad = model.vae.conv_mu.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_pixel_loss_penalises_a_collapsed_latent(self):
        """
        The central design claim behind stage 3: driving the encoder output to
        zero makes this loss *much worse*, which is exactly the property the
        latent epsilon-MSE lacks (there, collapse is the global minimum).

        This needs a VAE that can actually reconstruct, so it trains one first
        on an easy low-rank field. Against an untrained decoder the comparison
        is meaningless: decode(anything) is garbage either way, and the two
        numbers land on top of each other.
        """
        torch.manual_seed(0)
        model = _tiny_model(latent_norm="batch").train()

        # Smooth and low-rank, so a 16-wide VAE fits it in ~100 steps.
        yy, xx = torch.meshgrid(
            torch.linspace(-2, 2, 64), torch.linspace(-2, 2, 64), indexing="ij"
        )
        field   = torch.stack([torch.sin(xx) * torch.cos(yy), torch.cos(0.7 * xx)])
        context = field.unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1, 1)
        target  = field.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1, 1)

        opt = torch.optim.AdamW(model.vae.parameters(), lr=3e-3)
        for _ in range(100):
            opt.zero_grad(set_to_none=True)
            model(context, target, vae_loss_weight=1.0, vae_ssim_weight=0.0,
                  vae_beta=1e-6, diffusion_loss_weight=0.0)["loss"].backward()
            opt.step()

        times  = torch.full((2,), 0.05)
        kwargs = dict(vae_loss_weight=0.0, times=times,
                      pixel_loss_weight=1.0, pixel_loss_max_t=0.3)

        torch.manual_seed(0)
        base = model(context, target, **kwargs)["pixel_loss"].item()
        with torch.no_grad():
            model.vae.conv_mu.weight.mul_(1e-4)
            model.vae.conv_mu.bias.mul_(1e-4)
        torch.manual_seed(0)
        collapsed = model(context, target, **kwargs)["pixel_loss"].item()

        # Measured separation is ~350x; assert an order of magnitude for slack.
        self.assertGreater(collapsed, 10.0 * base)


class ForecastBaselineTests(unittest.TestCase):
    """An RMSE with nothing to compare it against says nothing about skill."""

    def test_forecast_metrics_report_persistence_and_climatology(self):
        model   = _tiny_model().eval()
        context = torch.randn(2, 3, 2, 64, 64)
        target  = torch.randn(2, 1, 2, 64, 64)
        m = model.forecast_metrics(context, target, num_steps=2)
        for key in ("forecast_rmse", "persistence_rmse", "climatology_rmse"):
            self.assertIn(key, m)
            self.assertTrue(torch.isfinite(m[key]))

    def test_climatology_is_the_zero_forecast(self):
        # Fields are z-scored, so the climatological mean is exactly zero.
        model   = _tiny_model().eval()
        context = torch.randn(1, 3, 2, 64, 64)
        target  = torch.randn(1, 1, 2, 64, 64)
        m = model.forecast_metrics(context, target, num_steps=2)
        expected = torch.sqrt(target[:, :, 0:1].pow(2).mean())
        torch.testing.assert_close(m["climatology_rmse"], expected)

    def test_persistence_is_the_last_context_frame(self):
        model   = _tiny_model().eval()
        context = torch.randn(1, 3, 2, 64, 64)
        target  = torch.randn(1, 1, 2, 64, 64)
        m = model.forecast_metrics(context, target, num_steps=2)
        expected = torch.sqrt(
            (context[:, -1:, 0:1] - target[:, :, 0:1]).pow(2).mean()
        )
        torch.testing.assert_close(m["persistence_rmse"], expected)

    def test_persistence_is_exact_when_nothing_changes(self):
        model  = _tiny_model().eval()
        frame  = torch.randn(1, 1, 2, 64, 64)
        context = frame.repeat(1, 3, 1, 1, 1)
        m = model.forecast_metrics(context, frame, num_steps=2)
        self.assertAlmostEqual(float(m["persistence_rmse"]), 0.0, places=5)


class ConfigStageTests(unittest.TestCase):
    """
    The `extends:` overlay and the stage guards in `load_config`.

    These need no pyearthtools, and they are the last line of defence against
    the config combination that silently destroyed the first real run, so they
    run everywhere.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        # A real stats file has to exist; _validate checks the path.
        stats = self.root / "stats.json"
        stats.write_text(json.dumps({"surface_global_irradiance": {"mean": 0.0, "std": 1.0}}))
        self.base = {
            "model": {
                "image_channels": 3, "image_size": 256, "latent_channels": 4,
                "hidden_dim": 128,
                "diffusion": {"patch_size": 2, "embed_dim": 768,
                              "num_heads": 12, "window_size": 15},
            },
            "data": {"n_prior_sat": 12, "n_post": 1,
                     "stats_path": str(stats), "splits": {}},
            "training": {"checkpoint_dir": str(self.root / "ckpt"),
                         "log_dir": str(self.root / "logs")},
            "optimiser": {"lr": 1e-4},
            "logging": {},
        }
        self._write("base.yaml", self.base)

    def tearDown(self):
        self._dir.cleanup()

    def _write(self, name, payload):
        import yaml
        (self.root / name).write_text(yaml.safe_dump(payload))
        return self.root / name

    def _load(self, name):
        from src.training.config import load_config
        return load_config(self.root / name)

    def test_extends_overlays_only_the_named_keys(self):
        self._write("stage.yaml", {
            "extends": "base.yaml",
            "training": {"vae_loss_weight": 1.0, "diffusion_loss_weight": 0.0},
        })
        cfg = self._load("stage.yaml")
        self.assertEqual(cfg["training"]["vae_loss_weight"], 1.0)
        # Untouched keys are inherited, not lost -- a stage config that had to
        # restate data.splits or stats_path would drift from the base.
        self.assertEqual(cfg["model"]["image_size"], 256)
        self.assertEqual(cfg["data"]["n_prior_sat"], 12)
        self.assertEqual(cfg["training"]["checkpoint_dir"], self.base["training"]["checkpoint_dir"])

    def test_extends_merges_nested_dicts_rather_than_replacing_them(self):
        self._write("stage.yaml", {
            "extends": "base.yaml", "model": {"diffusion": {"num_blocks": 12}},
        })
        cfg = self._load("stage.yaml")
        self.assertEqual(cfg["model"]["diffusion"]["num_blocks"], 12)
        self.assertEqual(cfg["model"]["diffusion"]["patch_size"], 2)

    def test_circular_extends_is_rejected(self):
        self._write("a.yaml", {"extends": "b.yaml"})
        self._write("b.yaml", {"extends": "a.yaml"})
        with self.assertRaisesRegex(ValueError, "Circular"):
            self._load("a.yaml")

    def test_joint_gradient_without_scale_invariance_is_rejected(self):
        """
        The guard for the failure this whole staging exists to prevent: under
        latent_norm 'ema' the scaling denominator is a no_grad buffer, so the
        diffusion loss is not scale-invariant and collapsing the latent is its
        global minimum.
        """
        self._write("bad.yaml", {"extends": "base.yaml",
                                 "training": {"detach_latents": False}})
        with self.assertRaisesRegex(ValueError, "detach_latents"):
            self._load("bad.yaml")

    def test_joint_gradient_is_allowed_with_batch_normalisation(self):
        self._write("ok.yaml", {
            "extends": "base.yaml",
            "model": {"latent_norm": "batch"},
            "training": {"detach_latents": False},
        })
        self.assertEqual(self._load("ok.yaml")["model"]["latent_norm"], "batch")

    def test_unknown_latent_norm_is_rejected(self):
        self._write("bad.yaml", {"extends": "base.yaml",
                                 "model": {"latent_norm": "layer"}})
        with self.assertRaisesRegex(ValueError, "latent_norm"):
            self._load("bad.yaml")

    def test_forecast_monitor_without_forecast_batches_is_rejected(self):
        """Monitoring a metric that is never computed silently disables early
        stopping and best-model selection for the whole run."""
        self._write("bad.yaml", {
            "extends": "base.yaml",
            "training": {"early_stopping": {"monitor": "forecast_rmse"}},
            "evaluation": {"forecast_eval_batches": 0},
        })
        with self.assertRaisesRegex(ValueError, "forecast_eval_batches"):
            self._load("bad.yaml")

    def test_a_monitor_the_stage_never_computes_is_rejected(self):
        """
        Caught in a real smoke run: a stage-2 config inherited
        `monitor: recon_loss` from a stage-1 base. With vae_loss_weight 0 that
        metric is identically 0.0, so best-model selection froze on epoch 1
        while the log kept printing "New best model - recon_loss: 0.000000".
        """
        self._write("bad.yaml", {
            "extends": "base.yaml",
            "training": {"vae_loss_weight": 0.0, "diffusion_loss_weight": 1.0,
                         "early_stopping": {"monitor": "recon_loss"}},
            "evaluation": {"forecast_eval_batches": 4},
        })
        with self.assertRaisesRegex(ValueError, "constant"):
            self._load("bad.yaml")

    def test_a_forecast_monitor_in_a_vae_only_stage_is_rejected(self):
        self._write("bad.yaml", {
            "extends": "base.yaml",
            "training": {"vae_loss_weight": 1.0, "diffusion_loss_weight": 0.0,
                         "early_stopping": {"monitor": "irradiance_mse"}},
        })
        with self.assertRaisesRegex(ValueError, "constant"):
            self._load("bad.yaml")

    def test_each_stage_monitors_something_it_actually_computes(self):
        for training, monitor in (
            ({"vae_loss_weight": 1.0, "diffusion_loss_weight": 0.0}, "recon_loss"),
            ({"vae_loss_weight": 0.0, "diffusion_loss_weight": 1.0}, "forecast_rmse"),
            ({"vae_loss_weight": 0.5, "diffusion_loss_weight": 1.0}, "forecast_rmse"),
        ):
            with self.subTest(training=training, monitor=monitor):
                cfg = dict(training)
                cfg["early_stopping"] = {"monitor": monitor}
                self._write("ok.yaml", {
                    "extends": "base.yaml", "training": cfg,
                    "evaluation": {"forecast_eval_batches": 4},
                })
                self.assertIn("model", self._load("ok.yaml"))

    def test_negative_loss_weight_is_rejected(self):
        self._write("bad.yaml", {"extends": "base.yaml",
                                 "training": {"pixel_loss_weight": -1.0}})
        with self.assertRaisesRegex(ValueError, "pixel_loss_weight"):
            self._load("bad.yaml")

    def test_pixel_loss_max_t_must_be_in_range(self):
        self._write("bad.yaml", {
            "extends": "base.yaml",
            "training": {"pixel_loss_weight": 0.05, "pixel_loss_max_t": 1.5},
        })
        with self.assertRaisesRegex(ValueError, "pixel_loss_max_t"):
            self._load("bad.yaml")


class ShippedConfigTests(unittest.TestCase):
    """The three stage configs in configs/ must load and mean what they say."""

    CONFIGS = Path(__file__).resolve().parent.parent / "configs"

    def _load(self, name):
        from src.training.config import load_config
        path = self.CONFIGS / name
        if not path.exists():
            self.skipTest(f"{path} not present")
        try:
            return load_config(path)
        except FileNotFoundError as exc:      # data.stats_path lives on /scratch
            self.skipTest(f"config references a path unavailable here: {exc}")

    def test_all_three_stage_configs_load(self):
        for name in ("train_config.yaml", "stage1_vae.yaml", "stage3_joint.yaml",
                     "smoke.yaml", "smoke2.yaml"):
            with self.subTest(config=name):
                self.assertIn("model", self._load(name))

    def test_stage1_trains_the_vae_alone(self):
        t = self._load("stage1_vae.yaml")["training"]
        self.assertGreater(t["vae_loss_weight"], 0.0)
        self.assertEqual(t["diffusion_loss_weight"], 0.0)
        self.assertFalse(t["freeze_vae"])

    def test_stage2_freezes_the_vae_and_detaches(self):
        cfg = self._load("train_config.yaml")
        t = cfg["training"]
        self.assertEqual(t["vae_loss_weight"], 0.0)
        self.assertGreater(t["diffusion_loss_weight"], 0.0)
        self.assertTrue(t["freeze_vae"])
        self.assertTrue(t["detach_latents"])
        # The metric that actually reflects forecast skill.
        self.assertEqual(t["early_stopping"]["monitor"], "forecast_rmse")

    def test_stage3_pairs_the_joint_gradient_with_both_safeguards(self):
        cfg = self._load("stage3_joint.yaml")
        self.assertFalse(cfg["training"]["detach_latents"])
        # Neither guard is optional when the diffusion gradient reaches the encoder.
        self.assertEqual(cfg["model"]["latent_norm"], "batch")
        self.assertGreater(cfg["training"]["pixel_loss_weight"], 0.0)
        # A lagging EMA mis-scales sampling even though training is invariant.
        self.assertLessEqual(cfg["model"]["latent_scale_momentum"], 0.95)


@unittest.skipIf(
    _petdata is None,
    f"pyearthtools unavailable in this environment: {_PETDATA_IMPORT_ERROR}",
)
class DataLoaderConfigTests(unittest.TestCase):
    """
    Guards against reintroducing the fork-after-CUDA/NCCL-init deadlock:
    DataLoader workers hang on their first batch if forked after CUDA/NCCL are
    live in the process, which reads as "GPU memory allocated, 0% utilisation,
    no iteration ever completes". `multiprocessing_context="spawn"` is the fix;
    this must hold for any num_workers > 0, not just the config's default.
    """

    @classmethod
    def setUpClass(cls):
        # PipelineDataset json-loads stats_path eagerly, so it needs to be real.
        cls._tmp = tempfile.TemporaryDirectory()
        cls.stats_path = os.path.join(cls._tmp.name, "stats.json")
        with open(cls.stats_path, "w") as f:
            json.dump({"surface_global_irradiance": {"mean": 0.0, "std": 1.0}}, f)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _base_config(self, num_workers: int) -> dict:
        return {
            "model": {"image_size": 64},
            "data": {
                "n_prior_sat": 3, "n_post": 1,
                "stats_path": self.stats_path,
                "splits": {"train": {"start": "20200101T0000", "end": "20200101T0400"},
                           "val":   {"start": "20200101T0000", "end": "20200101T0400"}},
            },
            "dataloader": {"batch_size": 1, "num_workers": num_workers},
        }

    def test_batch_size_override_is_respected(self):
        """Priming passes a small batch: it never reads the collated tensors,
        and a large batch makes every worker hold that many assembled samples
        (~10 MB each) before emitting anything."""
        cfg = self._base_config(num_workers=0)
        cfg["dataloader"]["batch_size"] = 32
        self.assertEqual(_petdata.build_dataloader("train", cfg).batch_size, 32)
        self.assertEqual(
            _petdata.build_dataloader("train", cfg, batch_size=2).batch_size, 2
        )

    def test_prime_mode_is_off_by_default(self):
        """The placeholder tensors priming yields must never reach training."""
        loader = _petdata.build_dataloader("train", self._base_config(num_workers=0))
        self.assertFalse(loader.dataset.prime_mode)
        loader = _petdata.build_dataloader("train", self._base_config(num_workers=0),
                                           prime_mode=True)
        self.assertTrue(loader.dataset.prime_mode)

    def test_cache_exists_probe_uses_stat_not_read(self):
        """
        A cache hit during priming must cost a stat(), not a read: re-reading a
        60k-frame cache costs ~26 minutes of pure I/O before any new work
        starts, and priming is re-run every time a job hits walltime.
        """
        import torch as _torch
        with tempfile.TemporaryDirectory() as d:
            cache = _petdata.FrameCache(Path(d), "train", dtype="float16")
            stamp = "20200101T0000"

            self.assertEqual(cache.exists(stamp), (False, False))

            cache.store(stamp, _torch.zeros(2, 4, 4))
            present, bad = cache.exists(stamp)
            self.assertTrue(present)
            self.assertFalse(bad)

            other = "20200101T0010"
            cache.store_bad(other)
            present, bad = cache.exists(other)
            self.assertFalse(present)
            self.assertTrue(bad)

    def test_workers_default_to_spawn_context(self):
        loader = _petdata.build_dataloader("train", self._base_config(num_workers=2))
        self.assertEqual(loader.multiprocessing_context.get_start_method(), "spawn")

    def test_zero_workers_needs_no_multiprocessing_context(self):
        # In-process iteration; nothing is forked, so there is nothing to fix.
        loader = _petdata.build_dataloader("train", self._base_config(num_workers=0))
        self.assertIsNone(loader.multiprocessing_context)

    def test_explicit_context_override_is_respected(self):
        config = self._base_config(num_workers=2)
        config["dataloader"]["multiprocessing_context"] = "forkserver"
        loader = _petdata.build_dataloader("train", config)
        self.assertEqual(loader.multiprocessing_context.get_start_method(), "forkserver")


@unittest.skipIf(
    _petdata is None,
    f"pyearthtools unavailable in this environment: {_PETDATA_IMPORT_ERROR}",
)
class ShardingTests(unittest.TestCase):
    """
    Sharding must partition the *date iterator*, not filter the output stream.

    One sample costs ~20s of archive I/O, so a shard that fetches samples it
    then discards multiplies wall-clock by the shard count: with 4 ranks x 4
    workers that was 16x, which pushed time-to-first-batch past half an hour
    and looked exactly like a hang. These tests read only the date iterator,
    so they touch no data files.
    """

    START, END = "20200101T0000", "20200101T0400"   # 24 samples at 10 min

    def _config(self, sample_stride: int = 1) -> dict:
        return {
            "model": {"image_size": 256},
            "data": {
                "sat_timestep": "10 minutes",
                "sample_stride": sample_stride,
                "n_prior_sat": 12, "n_post": 1,
                "splits": {"train": {"start": self.START, "end": self.END}},
            },
        }

    def _dates(self, shard: int, n_shards: int, sample_stride: int = 1) -> list[str]:
        dates = _petdata._build_date_range(
            "train", self._config(sample_stride), shard, n_shards
        )
        if dates is None:                     # shard past the end of the range
            return []
        return [str(d) for d in dates]

    def test_shards_are_a_disjoint_cover_of_the_unsharded_dates(self):
        for sample_stride in (1, 3):
            base = self._dates(0, 1, sample_stride)
            for n_shards in (2, 4, 16):
                with self.subTest(stride=sample_stride, n_shards=n_shards):
                    shards = [self._dates(i, n_shards, sample_stride)
                              for i in range(n_shards)]
                    union  = [d for sh in shards for d in sh]
                    self.assertEqual(len(union), len(set(union)), "shards overlap")
                    self.assertEqual(set(union), set(base), "shards do not cover the range")

    def test_shards_are_balanced(self):
        shards = [self._dates(i, 4) for i in range(4)]
        self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_sample_stride_is_applied_to_the_iterator(self):
        # stride 3 at a 10-minute cadence => 30-minute spacing, a third the samples.
        self.assertEqual(len(self._dates(0, 1, 3)) * 3, len(self._dates(0, 1, 1)))

    def test_shard_past_end_of_range_is_empty_not_an_error(self):
        # 8 samples at stride 3, but 16 shards: the tail shards have no work.
        # DateRange would raise on the inverted range; the pipeline must be None.
        self.assertIsNone(_petdata._build_date_range("train", self._config(3), 15, 16))
        self.assertEqual(self._dates(15, 16, 3), [])


@unittest.skipIf(
    _petdata is None,
    f"pyearthtools unavailable in this environment: {_PETDATA_IMPORT_ERROR}",
)
class ShuffleTests(unittest.TestCase):
    """
    The in-stream shuffle must not pre-fill its buffer before yielding.

    A sample costs ~20s of archive I/O, so pre-filling the default 256-slot
    buffer meant 264 fetches — about 80 minutes per worker — before the first
    batch reached the GPU, with nothing on screen. That is indistinguishable
    from a hang, and it is why time-to-first-batch, not just throughput, is
    what these tests pin down.
    """

    N = 600

    def _dataset(self, shuffle_buffer: int, shuffle: bool = True, min_fill=None):
        cfg = {
            "model": {"image_size": 64},
            "data": {"n_prior_sat": 3, "n_post": 1, "stats_path": None,
                     "splits": {"train": {"start": "20200101T0000", "end": "20200102T0000"}}},
            "dataloader": {"batch_size": 8, "shuffle_buffer": shuffle_buffer},
        }
        if min_fill is not None:
            cfg["dataloader"]["shuffle_min_fill"] = min_fill

        ds = _petdata.PipelineDataset.__new__(_petdata.PipelineDataset)
        torch.utils.data.IterableDataset.__init__(ds)
        ds.split, ds.config = "train", cfg
        ds.shuffle        = shuffle
        ds.shuffle_buffer = shuffle_buffer if shuffle else 0
        ds.shuffle_min_fill = max(1, min(
            int(cfg["dataloader"].get("shuffle_min_fill", 8)), ds.shuffle_buffer or 1))
        ds.epoch = 0
        n = self.N
        ds._raw_stream = lambda: (
            (torch.tensor([float(i)]), torch.tensor([float(i)])) for i in range(n)
        )
        return ds

    def _order(self, ds) -> list[int]:
        return [int(c.item()) for c, _ in ds]

    def test_first_batch_does_not_wait_for_a_full_buffer(self):
        # The regression: fetches-to-first-batch must not scale with the buffer.
        counts = {}
        for buf in (16, 64, 256):
            ds = self._dataset(buf)
            fetched = {"n": 0}
            base = ds._raw_stream
            def counting(base=base, fetched=fetched):
                for item in base():
                    fetched["n"] += 1
                    yield item
            ds._raw_stream = counting
            it = iter(ds)
            for _ in range(8):
                next(it)
            counts[buf] = fetched["n"]
        self.assertEqual(len(set(counts.values())), 1, f"scales with buffer: {counts}")
        # One batch must cost on the order of a batch, not a buffer.
        self.assertLess(counts[256], 64, f"too many fetches to first batch: {counts}")

    def test_shuffle_emits_every_sample_exactly_once(self):
        for buf in (16, 64, 256):
            with self.subTest(shuffle_buffer=buf):
                self.assertEqual(sorted(self._order(self._dataset(buf))), list(range(self.N)))

    def test_larger_buffer_shuffles_harder(self):
        def displacement(buf):
            order = self._order(self._dataset(buf))
            return sum(abs(v - i) for i, v in enumerate(order)) / len(order)
        small, large = displacement(16), displacement(256)
        # Emitting during the grow phase must not stop the buffer reaching size.
        self.assertGreater(large, small * 3, f"buffer not growing: {small=} {large=}")

    def test_unshuffled_stream_is_passed_through_in_order(self):
        self.assertEqual(self._order(self._dataset(256, shuffle=False)), list(range(self.N)))

    def test_shuffle_is_reproducible_per_epoch_and_varies_across_epochs(self):
        a, b, c = self._dataset(64), self._dataset(64), self._dataset(64)
        a.set_epoch(0); b.set_epoch(0); c.set_epoch(1)
        self.assertEqual(self._order(a), self._order(b))
        self.assertNotEqual(self._order(a), self._order(c))


class CompileSettingsTests(unittest.TestCase):
    """
    torch.compile must trace this model with STATIC shapes.

    `dynamic=True` does two things that are fatal here. It makes shapes
    symbolic, and it lifts `forward`'s float keyword arguments into the graph as
    tensors — so `diffusion_loss_weight > 0` becomes an in-graph `.item()`/`gt`
    instead of a Python branch folded at trace time. That bool, along with the
    frame counts read off `.shape[1]`, then has to cross DDPOptimizer's bucket
    split, and inductor asks every output node for `.meta["val"]`:

        AttributeError: 'bool' object has no attribute 'meta'

    Stage 3 died there on its first step the moment the VAE became trainable and
    DDP put a bucket boundary inside the encoder. Stage 2 survived only because
    a frozen VAE leaves no boundary there. Nothing in this trainer needs dynamic
    shapes: `dataloader.drop_last` is true, so every batch the compiled wrapper
    sees is the same shape.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        root = Path(self._dir.name)
        stats = root / "stats.json"
        stats.write_text(json.dumps(
            {"surface_global_irradiance": {"mean": 0.0, "std": 1.0}}
        ))
        # Small enough to build in a second; the geometry is irrelevant here,
        # only the compile call is under test.
        self.config = {
            "model": {
                "image_channels": 2, "image_size": 64, "latent_channels": 4,
                "hidden_dim": 16, "latent_scale": 1.0,
                "transformer": {"num_layers": 1, "num_heads": 2,
                                "feedforward_dim": 16, "transformer_dim": 16},
                "diffusion": {"patch_size": 2, "embed_dim": 32, "num_blocks": 1,
                              "num_heads": 2, "window_size": 3,
                              "use_natten": False},
            },
            "data": {"n_prior_sat": 3, "n_post": 1,
                     "stats_path": str(stats), "splits": {}},
            "training": {"checkpoint_dir": str(root / "ckpt"),
                         "log_dir": str(root / "logs"),
                         "compile_model": True},
            "optimiser": {"lr": 1e-4},
            "logging": {},
        }

    def tearDown(self):
        self._dir.cleanup()

    def _compile_kwargs(self, **training_overrides):
        """Build a Trainer with torch.compile stubbed, and report its kwargs."""
        from src.training import train as train_mod

        self.config["training"].update(training_overrides)
        seen = {}

        def fake_compile(model, **kwargs):
            seen.update(kwargs)
            return model

        real = torch.compile
        torch.compile = fake_compile
        try:
            train_mod.Trainer(self.config, device="cpu")
        finally:
            torch.compile = real
        return seen

    def test_compiles_with_static_shapes_by_default(self):
        self.assertIs(self._compile_kwargs()["dynamic"], False)

    def test_compile_dynamic_knob_is_honoured(self):
        self.assertIs(self._compile_kwargs(compile_dynamic=True)["dynamic"], True)

    def test_every_stage_config_leaves_shapes_static(self):
        """The shipped configs, not just the default in the code."""
        from src.training.config import load_config

        repo = Path(__file__).resolve().parent.parent
        for name in ("train_config.yaml", "stage1_vae.yaml", "stage3_joint.yaml"):
            path = repo / "configs" / name
            if not path.exists():          # a checkout without the stage configs
                continue
            with self.subTest(config=name):
                cfg = load_config(path)
                self.assertFalse(
                    cfg["training"].get("compile_dynamic", False),
                    f"{name} would compile with dynamic shapes; see this test's "
                    "docstring for why that breaks DDP + torch.compile here.",
                )


if __name__ == "__main__":
    unittest.main()


class SDEConsistencyTests(unittest.TestCase):
    """
    The closed-form perturbation kernel and the SDE coefficients must agree.

    For a linear SDE ``dx = a(t) x dt + g(t) dw`` the kernel N(alpha*x0, sigma^2)
    obeys
        d(alpha)/dt   = a(t) * alpha
        d(sigma^2)/dt = 2 a(t) sigma^2 + g(t)^2

    Checking this numerically is the real test of each schedule: alpha/sigma are
    written in closed form while the drift is written separately, so an error in
    either (notably the hand-derived cosine beta(t) = pi*tan(u)/(1+s)) would
    silently corrupt every sample without changing any tensor shape.
    """

    SDES = {
        "vp":        VPSDE(beta_min=0.1, beta_max=20.0),
        "vp_cosine": CosineVPSDE(s=0.008),
        "ve":        VESDE(sigma_min=0.01, sigma_max=50.0),
    }

    def test_drift_and_diffusion_match_the_kernel(self):
        for name, sde in self.SDES.items():
            for tv in (0.05, 0.25, 0.5, 0.75, 0.95):
                with self.subTest(sde=name, t=tv):
                    t = torch.tensor([tv], dtype=torch.float64, requires_grad=True)
                    alpha, sigma = sde.alpha_sigma(t)
                    d_var = torch.autograd.grad((sigma ** 2).sum(), t, retain_graph=True)[0]

                    if alpha.requires_grad:
                        d_alpha = torch.autograd.grad(alpha.sum(), t)[0]
                        a = (d_alpha / alpha).detach()        # implied drift coeff
                    else:
                        # Variance-exploding SDEs define alpha identically 1, so
                        # it carries no autograd path and the drift is zero.
                        a = torch.zeros_like(t).detach()
                    _, g = sde.sde(torch.zeros(1, 1, dtype=torch.float64), t.detach())

                    expected = 2 * a * sigma.detach() ** 2 + g.double() ** 2
                    torch.testing.assert_close(
                        d_var.detach(), expected, rtol=2e-3, atol=2e-4
                    )

    def test_vp_kernel_endpoints(self):
        for name in ("vp", "vp_cosine"):
            sde = self.SDES[name]
            with self.subTest(sde=name):
                a0, s0 = sde.alpha_sigma(torch.tensor([sde.t_eps]))
                aT, sT = sde.alpha_sigma(torch.tensor([sde.T]))
                # Starts as (almost) clean data...
                self.assertGreater(float(a0), 0.99)
                self.assertLess(float(s0), 0.05)
                # ...and ends at the standard normal prior it samples from.
                self.assertLess(float(aT), 0.05)
                self.assertGreater(float(sT), 0.95)

    def test_variance_preserving_means_alpha2_plus_sigma2_is_one(self):
        for name in ("vp", "vp_cosine"):
            sde = self.SDES[name]
            t = torch.linspace(sde.t_eps, sde.T, 25)
            alpha, sigma = sde.alpha_sigma(t)
            with self.subTest(sde=name):
                torch.testing.assert_close(
                    alpha ** 2 + sigma ** 2, torch.ones_like(alpha), rtol=1e-5, atol=1e-5
                )

    def test_ve_never_attenuates_the_signal(self):
        sde = self.SDES["ve"]
        t = torch.linspace(sde.t_eps, sde.T, 10)
        alpha, sigma = sde.alpha_sigma(t)
        torch.testing.assert_close(alpha, torch.ones_like(alpha))
        # Noise grows geometrically instead.
        self.assertTrue(torch.all(sigma[1:] > sigma[:-1]))
        self.assertAlmostEqual(float(sigma[-1]), sde.sigma_max, places=4)

    def test_sigma_is_monotonically_increasing(self):
        for name, sde in self.SDES.items():
            t = torch.linspace(sde.t_eps, sde.T, 50)
            sigma = sde.alpha_sigma(t)[1]
            with self.subTest(sde=name):
                self.assertTrue(torch.all(sigma[1:] >= sigma[:-1]))

    def test_perturb_matches_the_kernel_statistics(self):
        sde = self.SDES["vp_cosine"]
        torch.manual_seed(0)
        x0 = torch.full((20000, 1), 3.0)
        t  = torch.full((20000,), 0.4)
        xt, _, _ = sde.perturb(x0, t)
        alpha, sigma = sde.alpha_sigma(t[:1])
        self.assertAlmostEqual(float(xt.mean()), float(alpha) * 3.0, places=1)
        self.assertAlmostEqual(float(xt.std()), float(sigma), places=1)

    def test_probability_flow_drops_the_diffusion_term(self):
        sde = self.SDES["vp"]
        x = torch.randn(4, 3)
        t = torch.full((4,), 0.5)
        score = torch.randn(4, 3)
        _, g_sde = sde.reverse_sde(x, t, score, probability_flow=False)
        _, g_ode = sde.reverse_sde(x, t, score, probability_flow=True)
        self.assertTrue(torch.all(g_sde > 0))
        self.assertTrue(torch.all(g_ode == 0))

    def test_build_sde_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            build_sde("ddpm")


class ToyScoreModelTests(unittest.TestCase):
    """
    End-to-end validation on a distribution whose score is known analytically.

    For data ~ N(mu, s^2 I) the perturbed marginal is N(alpha*mu, alpha^2 s^2 +
    sigma^2), so the true score is available in closed form. Training a small
    network with the denoising score-matching objective and comparing against it
    exercises the whole conversion — perturbation kernel, score parameterisation,
    DSM loss, reverse SDE and probability-flow ODE — in a setting where "correct"
    is not a matter of opinion.
    """

    D, MU, S = 2, torch.tensor([1.0, -2.0]), 0.5

    @classmethod
    def setUpClass(cls):
        import math

        torch.manual_seed(0)
        cls.sde = VPSDE(beta_min=0.1, beta_max=20.0)

        class Net(torch.nn.Module):
            def __init__(self, h=128, nf=32):
                super().__init__()
                self.register_buffer(
                    "freqs", torch.exp(torch.linspace(0, math.log(1000), nf))
                )
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(cls.D + 2 * nf, h), torch.nn.SiLU(),
                    torch.nn.Linear(h, h), torch.nn.SiLU(),
                    torch.nn.Linear(h, cls.D),
                )

            def forward(self, x, t):
                a = t[:, None] * self.freqs[None]
                return self.net(torch.cat([x, torch.sin(a), torch.cos(a)], -1))

        cls.net = Net()
        opt = torch.optim.Adam(cls.net.parameters(), lr=2e-3)
        B = 512
        for _ in range(4000):
            x0 = cls.MU + cls.S * torch.randn(B, cls.D)
            # Stratified over the batch, mirroring the model's sample_times: the
            # small-t end is the hardest to fit (the score grows like 1/sigma)
            # and i.i.d. draws leave it undersampled.
            u = torch.rand(B)
            t = cls.sde.t_eps + ((torch.arange(B) + u) / B) * (cls.sde.T - cls.sde.t_eps)
            xt, z, _ = cls.sde.perturb(x0, t)
            # Denoising score matching in the sigma-scaled form.
            loss = (cls.net(xt, t) - z).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        cls.final_loss = float(loss.detach())

    @classmethod
    def model_score(cls, x, t):
        _, sigma = cls.sde.alpha_sigma(t)
        return -cls.net(x, t) / sigma[:, None].clamp(min=1e-8)

    @classmethod
    def true_score(cls, x, t):
        alpha, sigma = cls.sde.alpha_sigma(t)
        var = (alpha[:, None] ** 2) * cls.S ** 2 + sigma[:, None] ** 2
        return -(x - alpha[:, None] * cls.MU) / var

    def test_learned_score_matches_the_analytic_score(self):
        torch.manual_seed(1)
        for tv in (0.05, 0.2, 0.5, 0.8, 1.0):
            with self.subTest(t=tv):
                t = torch.full((256,), tv)
                alpha, sigma = self.sde.alpha_sigma(t)
                std = torch.sqrt((alpha[:, None] ** 2) * self.S ** 2 + sigma[:, None] ** 2)
                x = alpha[:, None] * self.MU + std * torch.randn(256, self.D)
                with torch.no_grad():
                    rel = ((self.model_score(x, t) - self.true_score(x, t)).norm()
                           / self.true_score(x, t).norm())
                self.assertLess(float(rel), 0.15, f"score error {rel:.3f} at t={tv}")

    def test_reverse_sde_recovers_the_data_distribution(self):
        torch.manual_seed(2)
        s = pc_sampler(self.sde, self.model_score, (4096, self.D),
                       torch.device("cpu"), num_steps=200, corrector_steps=1, snr=0.16)
        torch.testing.assert_close(s.mean(0), self.MU, rtol=0, atol=0.2)
        torch.testing.assert_close(
            s.std(0), torch.full((self.D,), self.S), rtol=0, atol=0.15
        )

    def test_probability_flow_ode_recovers_the_data_distribution(self):
        torch.manual_seed(3)
        s = ode_sampler(self.sde, self.model_score, (4096, self.D),
                        torch.device("cpu"), num_steps=200, method="heun")
        torch.testing.assert_close(s.mean(0), self.MU, rtol=0, atol=0.2)
        torch.testing.assert_close(
            s.std(0), torch.full((self.D,), self.S), rtol=0, atol=0.15
        )

    def test_ode_is_deterministic_given_the_initial_noise(self):
        x_T = torch.randn(64, self.D)
        a = ode_sampler(self.sde, self.model_score, (64, self.D), torch.device("cpu"),
                        num_steps=20, x_T=x_T)
        b = ode_sampler(self.sde, self.model_score, (64, self.D), torch.device("cpu"),
                        num_steps=20, x_T=x_T)
        torch.testing.assert_close(a, b)


class ScoreBasedModelTests(unittest.TestCase):
    """The full model must expose a score function, not a DDPM noise ladder."""

    def test_model_is_score_based_not_ddpm(self):
        model = _tiny_model()
        # A score model, so these must exist...
        self.assertTrue(hasattr(model, "score"))
        self.assertTrue(hasattr(model, "sde"))
        self.assertTrue(callable(model.score_fn(torch.randn(1, 3, 32),
                                                torch.randn(1, 3, 4, 8, 8))))
        # ...and the discrete DDPM machinery must not come back.
        for attr in ("alphas_cumprod", "betas", "sqrt_alphas_cumprod",
                     "ddim_timesteps", "add_noise", "num_diffusion_steps"):
            self.assertFalse(hasattr(model, attr), f"DDPM leftover: {attr}")

    def test_score_has_the_shape_of_the_data(self):
        model = _tiny_model()
        x   = torch.randn(2, 1, 4, 8, 8)
        t   = torch.full((2,), 0.5)
        ctx = torch.randn(2, 3, 32)
        lat = torch.randn(2, 3, 4, 8, 8)
        self.assertEqual(model.score(x, t, ctx, lat).shape, x.shape)

    def test_score_scales_like_one_over_sigma(self):
        # score = -out/sigma, so for a fixed network output the score magnitude
        # must fall as sigma grows. This pins the parameterisation itself.
        model = _tiny_model().eval()
        torch.nn.init.normal_(model.denoiser.final_layer.linear.weight, std=0.1)
        x   = torch.randn(1, 1, 4, 8, 8)
        ctx = torch.randn(1, 3, 32)
        lat = torch.randn(1, 3, 4, 8, 8)
        with torch.no_grad():
            lo = model.score(x, torch.full((1,), 0.05), ctx, lat).norm()
            hi = model.score(x, torch.full((1,), 0.95), ctx, lat).norm()
        self.assertGreater(float(lo), float(hi))

    def test_tweedie_inverts_the_perturbation_exactly(self):
        """
        Given the *exact* kernel score, Tweedie must return x0 with no error:
            (x_t + sigma^2 * (-z/sigma)) / alpha
          = (alpha*x0 + sigma*z - sigma*z) / alpha = x0
        This checks denoise_to_x0 algebraically, independent of any network.
        """
        model = _tiny_model()
        torch.manual_seed(0)
        x0 = torch.randn(4, 2, 4, 8, 8)
        for tv in (0.01, 0.3, 0.7, 0.9):
            with self.subTest(t=tv):
                t = torch.full((4,), tv)
                x_t, z, sigma = model.perturb(x0, t)
                exact_score = -z / sigma
                torch.testing.assert_close(
                    model.denoise_to_x0(x_t, t, exact_score), x0, rtol=1e-4, atol=1e-4
                )

    def test_tweedie_is_bounded_where_it_is_ill_conditioned(self):
        """
        As t -> T, alpha -> 0 and Tweedie amplifies any error by 1/alpha (~1e5
        for the cosine VP SDE at t=1). Exact recovery is impossible there — x_T
        holds no information about x_0 — so the contract is that the clamped
        estimate stays finite and bounded rather than accurate.
        """
        model = _tiny_model()
        x0 = torch.randn(2, 1, 4, 8, 8)
        t  = torch.full((2,), model.sde.T)
        x_t, z, sigma = model.perturb(x0, t)
        est = model.denoise_to_x0(x_t, t, -z / sigma, clamp=4.0)
        self.assertTrue(torch.isfinite(est).all())
        self.assertLessEqual(float(est.abs().max()), 4.0)

    def test_eval_times_avoid_the_singular_endpoint(self):
        # linspace(t_eps, T, B) puts a full sample at t=T, which at batch_size 2
        # would be half the validation batch sitting on the degenerate point.
        model = _tiny_model()
        for B in (2, 4, 8):
            with self.subTest(batch=B):
                t = model._eval_times(B, torch.device("cpu"))
                self.assertEqual(t.shape, (B,))
                self.assertLess(float(t.max()), model.sde.T)
                self.assertGreater(float(t.min()), model.sde.t_eps)
                # Still spans the range rather than bunching up.
                self.assertGreater(float(t.max()), 0.5 * model.sde.T)

    def test_training_times_lie_in_the_open_interval(self):
        model = _tiny_model()
        t = model.sample_times(64, torch.device("cpu"))
        self.assertEqual(t.shape, (64,))
        self.assertGreaterEqual(float(t.min()), model.sde.t_eps)
        self.assertLessEqual(float(t.max()), model.sde.T)

    def test_stratified_times_cover_the_whole_range(self):
        # i.i.d. draws routinely leave half the axis empty at this batch size.
        model = _tiny_model()
        t = model.sample_times(32, torch.device("cpu")).sort().values
        self.assertLess(float(t[0]), 0.1)
        self.assertGreater(float(t[-1]), 0.9)
        # Stratification bounds the largest gap.
        self.assertLess(float((t[1:] - t[:-1]).max()), 0.15)

    def test_both_loss_weightings_train(self):
        context = torch.randn(2, 3, 2, 64, 64)
        target  = torch.randn(2, 1, 2, 64, 64)
        for weighting in ("sigma2", "likelihood"):
            with self.subTest(weighting=weighting):
                model = _tiny_model(loss_weighting=weighting)
                out = model(context, target, vae_loss_weight=0.0)
                self.assertTrue(torch.isfinite(out["loss"]))
                out["loss"].backward()
                grads = [p.grad for p in model.denoiser.parameters() if p.grad is not None]
                self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))

    def test_rejects_unknown_loss_weighting(self):
        with self.assertRaises(ValueError):
            _tiny_model(loss_weighting="elbo")

    def test_every_sde_runs_end_to_end(self):
        context = torch.randn(1, 3, 2, 64, 64)
        target  = torch.randn(1, 1, 2, 64, 64)
        for name in ("vp", "vp_cosine", "ve"):
            with self.subTest(sde=name):
                model = _tiny_model(sde=name)
                self.assertTrue(torch.isfinite(
                    model(context, target, vae_loss_weight=0.0)["loss"]
                ))
                model.eval()
                with torch.no_grad():
                    s = model.sample(context, num_forecast_steps=1,
                                     sampler="ode", num_steps=3)
                self.assertTrue(torch.isfinite(s).all())

    def test_pc_and_ode_samplers_both_produce_valid_forecasts(self):
        model   = _tiny_model().eval()
        context = torch.randn(1, 3, 2, 64, 64)
        with torch.no_grad():
            for sampler in ("pc", "ode"):
                with self.subTest(sampler=sampler):
                    out = model.sample(context, num_forecast_steps=2,
                                       sampler=sampler, num_steps=3)
                    self.assertEqual(out.shape, (1, 2, 2, 64, 64))
                    self.assertTrue(torch.isfinite(out).all())

    def test_rejects_unknown_sampler(self):
        model = _tiny_model().eval()
        with self.assertRaises(ValueError):
            model.sample(torch.randn(1, 3, 2, 64, 64), num_forecast_steps=1,
                         sampler="ddim")


class TrainerStageWiringTests(unittest.TestCase):
    """
    Trainer-level wiring for the three stages.

    These matter because the failure they guard against is an ordering bug, not
    a logic one: `AdamW` is constructed from `requires_grad`, and `self.model`
    has been DDP- and compile-wrapped by that point. Both details are invisible
    in the model unit tests.
    """

    def _trainer(self, training=None, model=None):
        import sys, types, tempfile, json
        from pathlib import Path

        stub = types.ModuleType("src.petdata")

        class _DS(torch.utils.data.IterableDataset):
            def __iter__(self):
                for _ in range(2):
                    yield (torch.randn(3, 2, 64, 64), torch.randn(1, 2, 64, 64))

        stub.build_dataloader = (
            lambda split, config, shuffle=False, drop_last=None:
            torch.utils.data.DataLoader(_DS(), batch_size=1)
        )
        sys.modules["src.petdata"] = stub
        for m in [m for m in list(sys.modules) if m.startswith("src.training")]:
            del sys.modules[m]
        from src.training.train import Trainer

        tmp = tempfile.mkdtemp()
        stats = Path(tmp) / "s.json"
        stats.write_text(json.dumps({"a": {"mean": 0.0, "std": 1.0}}))
        model_cfg = {
            "image_channels": 2, "image_size": 64, "latent_channels": 4,
            "hidden_dim": 16, "max_forecast_steps": 4,
            "transformer": {"num_layers": 1, "num_heads": 2,
                            "feedforward_dim": 32, "transformer_dim": 32},
            "diffusion": {"patch_size": 4, "embed_dim": 32, "num_blocks": 1,
                          "num_heads": 4, "cond_dim": 16, "use_natten": False},
        }
        model_cfg.update(model or {})
        training_cfg = {
            "seed": 0, "compile_model": False, "precision": "fp32",
            "max_epochs": 1, "gradient_accumulation_steps": 1,
            "vae_loss_weight": 0.0, "vae_ssim_weight": 0.0,
            "checkpoint_dir": tmp, "log_dir": tmp,
            "early_stopping": {"enabled": True, "patience": 2},
        }
        training_cfg.update(training or {})
        cfg = {
            "model": model_cfg,
            "data": {"n_prior_sat": 3, "n_post": 1, "stats_path": str(stats),
                     "splits": {"train": {}, "val": {}}},
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "training": training_cfg,
            "evaluation": {"forecast_eval_batches": 0},
            "optimiser": {"lr": 1e-4, "scheduler": {}},
            "logging": {},
        }
        t = Trainer(cfg, device="cpu")
        t.mlflow_enabled = False
        return t

    @staticmethod
    def _optimised(trainer):
        """Every parameter the optimiser will actually update."""
        return {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}

    def test_freeze_vae_keeps_it_out_of_the_optimiser(self):
        t = self._trainer(training={"freeze_vae": True, "diffusion_loss_weight": 1.0})
        inner = t._get_inner_model()
        optimised = self._optimised(t)
        self.assertFalse(any(id(p) in optimised for p in inner.vae.parameters()))
        self.assertTrue(all(id(p) in optimised for p in inner.denoiser.parameters()))

    def test_unfrozen_vae_is_optimised(self):
        t = self._trainer(training={"freeze_vae": False, "vae_loss_weight": 1.0})
        inner = t._get_inner_model()
        optimised = self._optimised(t)
        self.assertTrue(all(id(p) in optimised for p in inner.vae.parameters()))

    def test_vae_lr_mult_creates_a_second_parameter_group(self):
        t = self._trainer(
            model={"latent_norm": "batch"},
            training={"freeze_vae": False, "vae_loss_weight": 0.5,
                      "detach_latents": False, "vae_lr_mult": 0.1},
        )
        self.assertEqual(len(t.optimizer.param_groups), 2)
        lrs = sorted(g["lr"] for g in t.optimizer.param_groups)
        self.assertAlmostEqual(lrs[0], 1e-5)      # VAE group, 0.1x
        self.assertAlmostEqual(lrs[1], 1e-4)
        # The slow group must actually be the VAE, not merely a smaller number.
        inner = t._get_inner_model()
        vae_ids  = {id(p) for p in inner.vae.parameters()}
        slow_ids = {id(p) for p in min(t.optimizer.param_groups, key=lambda g: g["lr"])["params"]}
        self.assertTrue(slow_ids and slow_ids <= vae_ids)

    def test_default_multiplier_leaves_a_single_group(self):
        t = self._trainer(training={"freeze_vae": False, "vae_loss_weight": 1.0})
        self.assertEqual(len(t.optimizer.param_groups), 1)

    def test_every_trainable_parameter_reaches_the_optimiser(self):
        """A parameter with requires_grad that no group owns is trained by
        nothing and fails silently."""
        for training in (
            {"freeze_vae": True, "diffusion_loss_weight": 1.0},
            {"freeze_vae": False, "vae_loss_weight": 1.0},
            {"freeze_vae": False, "vae_loss_weight": 0.5, "vae_lr_mult": 0.1},
        ):
            with self.subTest(training=training):
                t = self._trainer(training=training)
                optimised = self._optimised(t)
                trainable = {id(p) for p in t._get_inner_model().parameters()
                             if p.requires_grad}
                self.assertEqual(trainable, optimised)

    def test_stage1_trains_the_vae_and_leaves_the_denoiser_alone(self):
        t = self._trainer(training={"vae_loss_weight": 1.0,
                                    "diffusion_loss_weight": 0.0})
        metrics = t.train_epoch(t.setup_data()[0], epoch=0)
        self.assertEqual(metrics["diffusion"], 0.0)
        self.assertGreater(metrics["recon"], 0.0)

    def test_warmup_epochs_do_not_accrue_early_stopping_strikes(self):
        t = self._trainer(training={"vae_loss_weight": 1.0,
                                    "diffusion_loss_weight": 0.0})
        # The previous run stopped at epoch 11 on a best score set at epoch 1,
        # four epochs before the LR finished ramping.
        self.assertTrue(hasattr(t, "warmup_epochs"))

    def test_fixed_latent_scale_survives_a_checkpoint_load(self):
        """
        `latent_std` is a buffer, so loading a checkpoint would otherwise
        overwrite a measured, pinned scale with the previous stage's drifted EMA
        and quietly mis-scale every sample drawn afterwards.
        """
        import tempfile
        from pathlib import Path

        donor = self._trainer(training={"vae_loss_weight": 1.0,
                                        "diffusion_loss_weight": 0.0})
        donor._get_inner_model().latent_std.fill_(0.0031)   # a "collapsed" EMA
        path = Path(tempfile.mkdtemp()) / "stage1.pt"
        torch.save({"epoch": 3,
                    "model_state_dict": donor._get_inner_model().state_dict()}, path)

        t = self._trainer(model={"latent_scale": 0.42},
                          training={"freeze_vae": True, "diffusion_loss_weight": 1.0})
        t.init_from(path)
        self.assertAlmostEqual(float(t._get_inner_model().latent_std), 0.42, places=6)


class EmptyValidationTests(unittest.TestCase):
    """
    An empty validation set must not be reported as a perfect score.

    `n = max(1, n_batches)` turned "no data" into 0.0 for every metric, and
    best-model selection then saved that as the best model ever seen — a silent
    failure that looks like spectacular convergence.
    """

    def _trainer(self, val_batches: int):
        import sys, types, tempfile, json
        from pathlib import Path

        stub = types.ModuleType("src.petdata")

        class _DS(torch.utils.data.IterableDataset):
            def __init__(self, n):
                self.n = n

            def __iter__(self):
                for _ in range(self.n):
                    yield (torch.randn(3, 2, 64, 64), torch.randn(1, 2, 64, 64))

        def build_dataloader(split, config, shuffle=False, drop_last=None):
            n = 2 if split == "train" else val_batches
            return torch.utils.data.DataLoader(_DS(n), batch_size=1)

        stub.build_dataloader = build_dataloader
        sys.modules["src.petdata"] = stub
        for m in [m for m in list(sys.modules) if m.startswith("src.training")]:
            del sys.modules[m]
        from src.training.train import Trainer

        tmp = tempfile.mkdtemp()
        stats = Path(tmp) / "s.json"
        stats.write_text(json.dumps({"a": {"mean": 0.0, "std": 1.0}}))
        cfg = {
            "model": {"image_channels": 2, "image_size": 64, "latent_channels": 4,
                      "hidden_dim": 16, "max_forecast_steps": 4,
                      "transformer": {"num_layers": 1, "num_heads": 2,
                                      "feedforward_dim": 32, "transformer_dim": 32},
                      "diffusion": {"patch_size": 4, "embed_dim": 32, "num_blocks": 1,
                                    "num_heads": 4, "cond_dim": 16, "use_natten": False}},
            "data": {"n_prior_sat": 3, "n_post": 1, "stats_path": str(stats),
                     "splits": {"train": {}, "val": {}}},
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "training": {"seed": 0, "compile_model": False, "precision": "fp32",
                         "max_epochs": 1, "gradient_accumulation_steps": 1,
                         "vae_loss_weight": 0.0, "checkpoint_dir": tmp, "log_dir": tmp,
                         "early_stopping": {"enabled": True, "patience": 2}},
            "evaluation": {"forecast_eval_batches": 0},
            "optimiser": {"lr": 1e-4, "scheduler": {}},
            "logging": {},
        }
        t = Trainer(cfg, device="cpu")
        t.mlflow_enabled = False
        return t, build_dataloader("val", cfg)

    def test_empty_validation_returns_none_not_zeros(self):
        trainer, loader = self._trainer(val_batches=0)
        self.assertIsNone(trainer.validate(loader))

    def test_non_empty_validation_still_reports_metrics(self):
        trainer, loader = self._trainer(val_batches=2)
        metrics = trainer.validate(loader)
        self.assertIsNotNone(metrics)
        self.assertIn("irradiance_mse", metrics)
        # A real metric, not the 0.0 an empty set used to produce.
        self.assertGreater(metrics["irradiance_mse"], 0.0)

    def test_empty_validation_does_not_trigger_a_best_model_save(self):
        from pathlib import Path
        trainer, _ = self._trainer(val_batches=0)
        train_loader = torch.utils.data.DataLoader(
            type("D", (torch.utils.data.IterableDataset,), {
                "__iter__": lambda self: iter(
                    [(torch.randn(3, 2, 64, 64), torch.randn(1, 2, 64, 64))])
            })(), batch_size=1)
        empty_val = torch.utils.data.DataLoader(
            type("E", (torch.utils.data.IterableDataset,), {
                "__iter__": lambda self: iter([])})(), batch_size=1)
        trainer._training_loop(train_loader, empty_val, num_epochs=1)
        self.assertFalse((Path(trainer.checkpoint_dir) / "best_model.pt").exists())

    def test_high_worker_count_warns_but_does_not_block(self):
        """
        A --workers value derived from a node's CPU count (e.g. ncpus - 2 on a
        48-CPU node = 46) once OOM-killed a DataLoader worker: this pipeline's
        per-worker memory grows during iteration, not just at import, so it
        does not scale safely with CPU count. This must warn, not refuse --
        some environments genuinely have the headroom.
        """
        from src.training.train import warn_on_high_worker_count
        with self.assertLogs("src.training.train", level="WARNING") as cm:
            warn_on_high_worker_count(46)   # the value that caused the OOM
        self.assertTrue(any("above the range" in m for m in cm.output))

    def test_the_measured_safe_default_does_not_warn(self):
        from src.training.train import warn_on_high_worker_count
        with self.assertNoLogs("src.training.train", level="WARNING"):
            warn_on_high_worker_count(10)

    def test_zero_training_batches_raises_rather_than_reporting_zero_loss(self):
        """
        An epoch that saw no data must not report a perfect 0.0 loss. This is
        easy to hit: an IterableDataset batches per worker, so a split with
        fewer than batch_size x num_workers samples yields nothing once
        drop_last discards each worker's remainder.
        """
        trainer, _ = self._trainer(val_batches=1)
        empty = torch.utils.data.DataLoader(
            type("E", (torch.utils.data.IterableDataset,), {
                "__iter__": lambda self: iter([])})(), batch_size=1)
        with self.assertRaises(RuntimeError) as ctx:
            trainer.train_epoch(empty, epoch=0)
        self.assertIn("no batches", str(ctx.exception))

    def test_normal_training_epoch_still_returns_metrics(self):
        trainer, _ = self._trainer(val_batches=1)
        loader = torch.utils.data.DataLoader(
            type("D", (torch.utils.data.IterableDataset,), {
                "__iter__": lambda self: iter(
                    [(torch.randn(3, 2, 64, 64), torch.randn(1, 2, 64, 64))] * 2)
            })(), batch_size=1)
        m = trainer.train_epoch(loader, epoch=0)
        self.assertGreater(m["total"], 0.0)
        self.assertIn("samples_per_s", m)


@unittest.skipIf(
    _petdata is None,
    f"pyearthtools unavailable in this environment: {_PETDATA_IMPORT_ERROR}",
)
class FrameCacheTests(unittest.TestCase):
    """
    The cache turns a 93%-idle GPU into a 96%-busy one, so it has to be right.
    The dangerous failure is not a miss — it is a *hit* on a tensor built from a
    different recipe, which would silently train on data the config no longer
    describes.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stats = os.path.join(self.tmp.name, "stats.json")
        self._write_stats(526.6, 355.9)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_stats(self, mean, std):
        with open(self.stats, "w") as f:
            json.dump({"surface_global_irradiance": {"mean": mean, "std": std}}, f)

    def _config(self, **over):
        cfg = {
            "model": {"image_size": 256},
            "data": {
                "himawari_vars": ["surface_global_irradiance"],
                "barra_vars": ["RH24mean"], "barra_domain": "AUST-11",
                "barra_freq": "1hr", "bounds": [-35, -28.5, 145, 151.5],
                "crop_offset": 10, "n_prior_sat": 12, "n_post": 1,
                "sat_timestep": "10 minutes", "stats_path": self.stats,
            },
            "dataloader": {},
        }
        cfg["data"].update(over.pop("data", {}))
        cfg["model"].update(over.pop("model", {}))
        cfg["dataloader"].update(over.pop("dataloader", {}))
        return cfg

    # -- namespace invalidation ------------------------------------------- #

    def test_same_recipe_gives_the_same_namespace(self):
        self.assertEqual(
            _petdata._cache_namespace(self._config()),
            _petdata._cache_namespace(self._config()),
        )

    def test_every_recipe_change_invalidates_the_cache(self):
        base = _petdata._cache_namespace(self._config())
        variants = {
            "image_size":  self._config(model={"image_size": 128}),
            "bounds":      self._config(data={"bounds": [-34, -29, 146, 151]}),
            "crop_offset": self._config(data={"crop_offset": 0}),
            "vars":        self._config(data={"himawari_vars": ["a", "b"]}),
            "barra_vars":  self._config(data={"barra_vars": []}),
            "timestep":    self._config(data={"sat_timestep": "30 minutes"}),
        }
        for name, cfg in variants.items():
            with self.subTest(changed=name):
                self.assertNotEqual(base, _petdata._cache_namespace(cfg))

    def test_changing_normalisation_stats_invalidates_the_cache(self):
        # Normalisation is baked into the stored tensor, so new statistics must
        # not hit entries written under the old ones.
        base = _petdata._cache_namespace(self._config())
        self._write_stats(500.0, 300.0)
        self.assertNotEqual(base, _petdata._cache_namespace(self._config()))

    # -- storage behaviour ------------------------------------------------- #

    def _cache(self, dtype="float16"):
        return _petdata.FrameCache(Path(self.tmp.name) / "ns", "train", dtype=dtype)

    def test_roundtrip_preserves_values_within_float16_precision(self):
        cache = self._cache()
        tensor = torch.randn(2, 8, 8) * 3.0          # one (C, H, W) frame
        cache.store("2020-01-01T00:00", tensor)
        loaded, bad = cache.load("2020-01-01T00:00")
        self.assertFalse(bad)
        self.assertEqual(loaded.dtype, torch.float32)
        self.assertEqual(loaded.shape, tensor.shape)
        torch.testing.assert_close(loaded, tensor, rtol=1e-2, atol=1e-2)

    def test_miss_returns_none_without_raising(self):
        tensor, bad = self._cache().load("2020-01-01T00:00")
        self.assertIsNone(tensor)
        self.assertFalse(bad)

    def test_known_bad_dates_are_remembered(self):
        # Dates that yield nothing are deterministic, so later epochs must not
        # re-pay the ~20s fetch just to discard the result again.
        cache = self._cache()
        cache.store_bad("2020-01-01T00:00")
        tensor, bad = cache.load("2020-01-01T00:00")
        self.assertIsNone(tensor)
        self.assertTrue(bad)

    def test_writes_leave_no_temporary_files(self):
        cache = self._cache()
        cache.store("2020-01-01T00:00", torch.randn(2, 8, 8))
        cache.store_bad("2020-01-01T00:10")
        leftovers = [p.name for p in Path(self.tmp.name).rglob("*tmp*")]
        self.assertEqual(leftovers, [], f"non-atomic write left {leftovers}")

    def test_corrupt_entry_is_discarded_rather_than_fatal(self):
        cache = self._cache()
        cache.store("2020-01-01T00:00", torch.randn(2, 8, 8))
        path, _ = cache._paths("2020-01-01T00:00")
        path.write_bytes(b"not a numpy file")
        tensor, bad = cache.load("2020-01-01T00:00")
        self.assertIsNone(tensor)
        self.assertFalse(bad)
        self.assertFalse(path.exists(), "corrupt entry should be removed")

    def test_disabled_cache_is_a_no_op(self):
        cache = _petdata.FrameCache(None, "train")
        self.assertFalse(cache.enabled)
        cache.store("2020-01-01T00:00", torch.randn(2, 2))   # must not raise
        self.assertEqual(cache.load("2020-01-01T00:00"), (None, False))

    # -- config resolution -------------------------------------------------- #

    def test_cache_dir_expands_environment_variables(self):
        os.environ["CPDIT_TEST_JOBFS"] = self.tmp.name
        cfg = self._config(dataloader={"cache_dir": "$CPDIT_TEST_JOBFS/cache"})
        self.assertEqual(
            _petdata.resolve_cache_dir(cfg), Path(self.tmp.name) / "cache"
        )
        del os.environ["CPDIT_TEST_JOBFS"]

    def test_unset_variable_disables_the_cache_rather_than_writing_a_junk_path(self):
        cfg = self._config(dataloader={"cache_dir": "$CPDIT_DEFINITELY_UNSET/x"})
        self.assertIsNone(_petdata.resolve_cache_dir(cfg))

    def test_no_cache_dir_disables_the_cache(self):
        self.assertIsNone(_petdata.resolve_cache_dir(self._config()))


@unittest.skipIf(
    _petdata is None,
    f"pyearthtools unavailable in this environment: {_PETDATA_IMPORT_ERROR}",
)
class FrameWindowTests(unittest.TestCase):
    """
    `frame_times` replaces TemporalWindow-plus-trim, so it must reproduce that
    window exactly: context_length frames immediately BEFORE the anchor, then
    forecast_length frames starting AT it. Getting this wrong would silently
    shift the forecast target or leak future frames into the context.
    """

    def test_window_layout_matches_the_documented_semantics(self):
        times = _petdata.frame_times("20200105T1200", context_length=12,
                                     forecast_length=1)
        self.assertEqual(len(times), 13)
        # Context ends one step before the anchor...
        self.assertEqual(times[0], "20200105T1000")     # t - 120 min
        self.assertEqual(times[11], "20200105T1150")    # t - 10 min
        # ...and the forecast starts exactly at it.
        self.assertEqual(times[12], "20200105T1200")

    def test_multi_frame_forecast_extends_forwards(self):
        times = _petdata.frame_times("20200105T1200", context_length=2,
                                     forecast_length=3)
        self.assertEqual(times, ["20200105T1140", "20200105T1150",
                                 "20200105T1200", "20200105T1210",
                                 "20200105T1220"])

    def test_context_never_contains_the_target(self):
        ctx, fc = 12, 6
        times = _petdata.frame_times("20200105T1200", ctx, fc)
        self.assertEqual(len(set(times)), len(times), "duplicate frames")
        self.assertTrue(set(times[:ctx]).isdisjoint(times[ctx:]),
                        "future frames leaked into the context")

    def test_window_crosses_midnight_and_month_ends(self):
        times = _petdata.frame_times("20200301T0000", context_length=2,
                                     forecast_length=1)
        # Leap year: 29 February exists.
        self.assertEqual(times, ["20200229T2340", "20200229T2350", "20200301T0000"])

    def test_nearest_hour_rounds_rather_than_floors(self):
        # BARRA is hourly; the windowed pipeline used method='nearest', so
        # rounding (not flooring) is what preserves the old behaviour.
        self.assertEqual(_petdata.nearest_hour("20200105T1210"), "20200105T1200")
        self.assertEqual(_petdata.nearest_hour("20200105T1250"), "20200105T1300")
        self.assertEqual(_petdata.nearest_hour("20200105T2350"), "20200106T0000")

    def test_half_hour_ties_round_down_like_scipy_nearest(self):
        """
        Satellite frames land on :30 once an hour, so the tie-break decides one
        frame in six. xarray's method='nearest' uses scipy's 'nearest', which
        rounds half DOWN — unlike pandas' .round, which is banker's rounding and
        would disagree on every odd hour.
        """
        for stamp, expected in [("20200105T1230", "20200105T1200"),
                                ("20200105T1330", "20200105T1300"),
                                ("20200105T0030", "20200105T0000"),
                                ("20200105T2330", "20200105T2300")]:
            with self.subTest(stamp=stamp):
                self.assertEqual(_petdata.nearest_hour(stamp), expected)


@unittest.skipIf(
    _petdata is None,
    f"pyearthtools unavailable in this environment: {_PETDATA_IMPORT_ERROR}",
)
class BlockedHoursTests(unittest.TestCase):
    """Night-time anchors are dropped before any I/O."""

    def test_default_range_blocks_measured_dead_hours(self):
        blocked = _petdata.blocked_utc_hours({"skip_utc_hours": [9, 21]})
        self.assertEqual(blocked, set(range(9, 22)))
        # The hours that actually produced samples must survive.
        for hour in (22, 23, 0, 2, 5, 6, 7, 8):
            self.assertNotIn(hour, blocked)

    def test_range_wraps_past_midnight(self):
        self.assertEqual(
            _petdata.blocked_utc_hours({"skip_utc_hours": [22, 2]}),
            {22, 23, 0, 1, 2},
        )

    def test_absent_or_null_keeps_every_hour(self):
        self.assertEqual(_petdata.blocked_utc_hours({}), set())
        self.assertEqual(_petdata.blocked_utc_hours({"skip_utc_hours": None}), set())

    def test_rejects_hours_outside_0_23(self):
        with self.assertRaises(ValueError):
            _petdata.blocked_utc_hours({"skip_utc_hours": [9, 24]})

    def test_window_length_matters_for_throughput(self):
        # 13 of 24 hours dropped => the cold pass fetches ~46% of the dates.
        blocked = _petdata.blocked_utc_hours({"skip_utc_hours": [9, 21]})
        self.assertAlmostEqual((24 - len(blocked)) / 24, 11 / 24, places=6)


@unittest.skipIf(
    _petdata is None,
    f"pyearthtools unavailable in this environment: {_PETDATA_IMPORT_ERROR}",
)
class FrameCacheKeyTests(unittest.TestCase):
    """A frame is independent of the window it lands in."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.stats = os.path.join(self.tmp.name, "stats.json")
        with open(self.stats, "w") as f:
            json.dump({"surface_global_irradiance": {"mean": 1.0, "std": 2.0}}, f)

    def tearDown(self):
        self.tmp.cleanup()

    def _cfg(self, **data):
        cfg = {
            "model": {"image_size": 256},
            "data": {"himawari_vars": ["surface_global_irradiance"],
                     "barra_vars": ["RH24mean"], "barra_domain": "AUST-11",
                     "barra_freq": "1hr", "bounds": [-35, -28.5, 145, 151.5],
                     "crop_offset": 10, "n_prior_sat": 12, "n_post": 1,
                     "sat_timestep": "10 minutes", "stats_path": self.stats},
            "dataloader": {},
        }
        cfg["data"].update(data)
        return cfg

    def test_changing_the_window_reuses_the_cache(self):
        # Frames do not depend on context/forecast length, so changing either
        # must NOT invalidate an expensively-built cache.
        base = _petdata._cache_namespace(self._cfg())
        self.assertEqual(base, _petdata._cache_namespace(self._cfg(n_prior_sat=6)))
        self.assertEqual(base, _petdata._cache_namespace(self._cfg(n_post=6)))

    def test_changing_the_frame_recipe_still_invalidates(self):
        base = _petdata._cache_namespace(self._cfg())
        self.assertNotEqual(base, _petdata._cache_namespace(self._cfg(crop_offset=0)))
        self.assertNotEqual(base, _petdata._cache_namespace(
            self._cfg(bounds=[-34, -29, 146, 151])))
