# CPDiT — Latent Diffusion Transformer for Solar Irradiance Nowcasting

Probabilistic nowcasting of surface solar irradiance from Himawari-8/9 satellite
imagery, conditioned on BARRA-R2 reanalysis fields.

```
Himawari + BARRA  →  VAE encoder  →  latent maps  →  temporal transformer
                                          ↓                    ↓
                                   DiT denoiser (adaLN-Zero, neighbourhood attn)
                                          ↓
                        reverse-SDE / PF-ODE sampling → VAE decoder → forecast
```

## Architecture

| Component | File | Role |
|---|---|---|
| VAE | [src/models/vae.py](src/models/vae.py) | `(C, 256, 256) → (4, 32, 32)` spatial latents |
| Networks | [src/models/networks.py](src/models/networks.py) | `ContextEncoder` (temporal) + `DiTDenoiser` (score network) |
| Diffusion process | [src/models/diffusion.py](src/models/diffusion.py) | VP / cosine-VP / VE SDEs + PC and PF-ODE samplers; no parameters |
| Full model | [src/models/latent_diffusion.py](src/models/latent_diffusion.py) | Assembly: `score()`, the DSM objective, `sample()` |

Four modules, split by role: `vae.py` moves between pixels and latents,
`networks.py` holds everything with learned weights, `diffusion.py` holds the
maths with none, and `latent_diffusion.py` is the only class training and
inference touch.
| Data pipeline | [src/petdata/\_\_init\_\_.py](src/petdata/__init__.py) | pyearthtools → `(context, forecast)` tensors |

### Score-based, not DDPM

The generative model is **score-based** in the SDE framework of
[Song et al. 2021](https://arxiv.org/abs/2011.13456). It learns
`s_theta(x, t) ≈ ∇_x log p_t(x)` over **continuous** time under a forward SDE,
rather than epsilon-prediction over a discrete `betas` / `alphas_cumprod` ladder.

| | |
|---|---|
| Forward | `dx = f(x,t) dt + g(t) dw`, kernel `N(alpha(t) x0, sigma(t)^2 I)` |
| Objective | denoising score matching against the exact kernel score `-z/sigma` |
| Sampling | reverse-time SDE + Langevin corrector, or probability-flow ODE |
| SDEs | `vp_cosine` (default), `vp`, `ve` — see `model.diffusion.sde` |

The network emits `out` and the score is **defined** as `-out/sigma(t)`. That
keeps the regression target unit-variance at every noise level instead of asking
the network to learn an O(1/sigma) blow-up. A consequence worth being explicit
about: under the default `sigma^2` weighting this objective is numerically the
same as epsilon-prediction MSE. That equivalence is a known property of the VP
family, not a sign the change is cosmetic — what it buys is everything built on
having a score: continuous time, a choice of SDE, the Langevin corrector, the
deterministic PF-ODE, and `loss_weighting: likelihood` (`lambda = g(t)^2`), which
is *not* equivalent to epsilon-MSE and targets the log-likelihood bound.

The corrector is the practical payoff: it re-equilibrates onto `p_t` at each
noise level rather than letting discretisation error accumulate down the chain,
so the sampler tolerates far fewer steps than an ancestral chain would.

Checkpoints from the DDPM version will not load — the noise-schedule buffers are
gone and `model.diffusion` keys changed.

### The DiT denoiser

The denoiser is a Diffusion Transformer, not a U-Net. Per forecast frame:

1. The noisy latent is concatenated with **all context latent frames along the
   channel axis**, giving `latent_channels × (1 + context_length)` input channels.
2. A **4×4 strided convolution** patch-embeds that stack into tokens.
3. Fixed 2-D sin-cos positional embeddings are added.
4. **`num_blocks` transformer blocks** apply **2-D neighbourhood attention** (local window)
   followed by an MLP, each modulated by **adaLN-Zero**.
5. A zero-initialised final layer projects tokens back to patch pixels.

Conditioning enters two ways, deliberately:

- **Spatially**, through the channel concatenation, so the denoiser keeps the
  full spatial structure of the recent past. A single pooled context vector
  cannot tell a convolutional or attention stack *where* the clouds are.
- **Globally**, through adaLN-Zero, from the diffusion timestep, a pooled
  summary of the temporal transformer output, and a **lead-time embedding**.

The lead-time embedding is what makes different forecast frames distinguishable.
Without it, every frame of the horizon is an identical draw from the same
conditional distribution.

### Neighbourhood attention window sizing

Neighbourhood attention only helps when the window is *smaller* than the token
grid. The grid is `image_size / 8 / patch_size`, so:

| `image_size` | `patch_size` | Token grid | Largest usable window | Tokens/frame |
|---:|---:|---:|---:|---:|
| 256 | 4 | 8×8 | **7** (31 clamps down) | 64 |
| 256 | 2 | 16×16 | 15 | 256 |
| 256 | 1 | 32×32 | **31** | 1024 |
| 512 | 4 | 16×16 | 15 | 256 |
| 1024 | 4 | 32×32 | **31** | 1024 |

At the default 256 px with patch size 4 the grid is only 8×8, so a 31×31 window
is clamped to 7 and attention is effectively global. The model logs a warning
when this happens. This is correct but wasteful — the sparsity buys nothing at
this resolution. To make the 31×31 window meaningful, either raise `image_size`
to 1024 or drop `patch_size` to 1.

`NATTEN`'s fused CUDA kernels are used automatically when importable; otherwise
an exact masked-softmax fallback runs, which is mathematically identical and
perfectly adequate at these grid sizes.

## Data

Each sample is `n_prior_sat` context frames immediately before time *t*, and
`n_post` forecast frames starting at *t*. With the defaults (12 context, 1
forecast, 10-minute cadence) that is a 10-minute-ahead nowcast from two hours
of history.

Channels follow `xr.merge` order, so the Himawari variables come first and
`irradiance_channel: 0` is `himawari_vars[0]`.

> **Normalisation.** `stats_path` holds means and standard deviations in **raw
> physical units** (W/m², %). The pipeline must therefore apply *no* other
> scaling before the z-score. Composing a second scaling with these statistics
> collapses the data to a near-constant field, and training will appear to
> converge beautifully while learning nothing.

Regenerate statistics with:

```bash
python scripts/calc_norm_stats.py
```

## Setup

```bash
source hpc_setup.sh        # loads the pet/0.6.2 module on NCI Gadi
```

## Training

Single GPU or CPU:

```bash
python -m src.training.train --config configs/train_config.yaml
```

Multi-GPU:

```bash
torchrun --nproc_per_node=4 -m src.training.train --config configs/train_config.yaml
```

Resume:

```bash
python -m src.training.train --resume /path/to/checkpoint_epoch_050.pt
```

Training is single-stage: the VAE and the diffusion model are optimised
together, with `training.vae_loss_weight` balancing the two. Set it to `0` to
train the diffusion model against a frozen-in-practice VAE.

### Checkpoints

The best model is written to a stable `best_model.pt`; periodic checkpoints are
epoch-numbered and rotated to the last `keep_last_n_checkpoints`. Every
checkpoint embeds its config, so inference reconstructs the architecture without
a second config file.

### Latent scaling

Diffusion assumes roughly unit-variance data, but VAE latents have an arbitrary
scale. The model tracks an EMA of the latent standard deviation and divides by
it — the running-statistics analogue of Stable Diffusion's fixed `0.18215`. The
value is logged each epoch as `Latent scale (EMA std)`. Once the VAE has
converged, freeze it by setting `model.latent_scale` to that number.

## Evaluation and inference

```bash
python scripts/evaluate.py  --checkpoint .../best_model.pt --split val
python scripts/inference.py --checkpoint .../best_model.pt --num-samples 8
```

`evaluate.py` integrates the full reverse process, reports metrics in physical units, and
prints a **persistence baseline** with a skill score. A nowcast that does not
beat persistence is not a nowcast.

During training two validation signals are logged:

- `irradiance_mse` / `irradiance_mae` — a cheap single-step x̂₀ estimate at fixed
  times with a fixed noise draw, so it is reproducible across epochs.
- `forecast_rmse` / `forecast_mae` — the full reverse process (PF-ODE), on the first
  `evaluation.forecast_eval_batches` validation batches only, because it is
  expensive.

## Configuration

[configs/train_config.yaml](configs/train_config.yaml) is the single source of
truth, and every key in it is read by the code. Notable ones:

```yaml
model:
  image_size: 256
  latent_scale: null              # null = EMA-tracked; a float freezes it
  diffusion:
    patch_size: 4                 # 4x4 strided-conv patch embedding
    embed_dim: 768
    num_blocks: 4
    num_heads: 12
    window_size: 31               # neighbourhood window (odd; clamped to the grid)

data:
  n_prior_sat: 12                 # context frames
  n_post: 1                       # forecast frames
  sample_stride: 1                # skip overlapping windows

dataloader:
  batch_size: 1
  num_workers: 10                 # pipeline is built per worker, so >0 is safe
  shuffle_buffer: 256             # reservoir shuffle over the date-ordered stream
  shuffle_min_fill: 8             # emitted from here; the buffer is not pre-filled

training:
  precision: bf16                 # auto-falls back to fp16 on pre-Ampere GPUs
  gradient_accumulation_steps: 4
```

Values above track [configs/train_config.yaml](configs/train_config.yaml); it is
the source of truth if they ever drift.

## Testing

```bash
python -m pytest tests/ -q
```

The tests cover neighbourhood-attention semantics (window centring, boundary
shifting, locality, and equivalence with full attention when the window spans
the grid), adaLN-Zero initialisation, lead-time conditioning, SDE consistency
construction, and latent scaling. They do not import `src.petdata`, so they run
without the pyearthtools runtime.

## Performance notes

- **The dataloader is the usual bottleneck.** The pipeline is built inside each
  worker, so `num_workers > 0` is safe and important; the GPU otherwise waits on
  xarray I/O and interpolation.
- **bf16 needs Ampere. The gpuvolta queue is V100 (sm_70).** Measured on a
  V100-SXM2-32GB: fp16 **92.1** TFLOP/s, fp32 13.8, bf16 **10.4** — bf16 is
  8.8x slower than fp16 and slower than plain fp32, because it has no
  tensor-core path before sm_80. `torch.cuda.is_bf16_supported()` returns True
  on a V100; it means "runs", not "is fast". End to end this was **3.8 s/it
  against 1.2 s/it** on 4xV100. The trainer now detects a pre-Ampere card and
  falls back to fp16 (which the GradScaler already covers), so `precision:
  bf16` stays correct on newer hardware.
- **The cache stores frames, not samples.** Samples overlap heavily — the window
  is `n_prior_sat + n_post` frames but advances only `sample_stride` — so
  caching assembled samples would store and decode each frame ~4 times. Caching
  frames stores each timestamp once: ~11 GB per year instead of ~56 GB, and a
  correspondingly cheaper cold pass. `scripts/verify_frame_equivalence.py`
  confirms per-frame assembly reproduces the old windowed pipeline bit-for-bit.
  `n_prior_sat`/`n_post` are deliberately *not* part of the cache key, so
  changing the window reuses the frames instead of rebuilding.
- **Night-time anchors are skipped before any I/O.** A sample needs its whole
  context window in daylight; outside that the irradiance field is absent or
  NaN. Measured over this domain (145-151.5E, UTC+10), UTC hours 09-21 produced
  0 usable samples out of 26 attempts. `data.skip_utc_hours: [9, 21]` drops them
  without a fetch, removing 13 of 24 hours from the cold pass. It is a fixed
  clock window, not a solar calculation, so it is conservative in summer.
- **The sample cache is what makes the GPUs usable.** Building one sample from
  the archive costs ~16-20s; reading it back from `/scratch` costs ~3 ms. The
  recipe is deterministic, so `dataloader.cache_dir` stores each processed
  sample as one `.npy` keyed by timestamp, under a namespace hashing the
  variables, bounds, window, crop and normalisation statistics — change any of
  them and you get a fresh namespace, so stale tensors can never be served.
  Measured on one H200: epoch 1 (cold) 0.36 samples/s with the GPU 93% idle;
  epoch 2 (warm) **7.89 samples/s with the GPU 96% busy** — a 22x speedup that
  turns an I/O-bound run into a compute-bound one. Prime it with
  `--prime-cache` on a CPU-only job first; the cache outlives the job.
- **Never block the loader to fill a buffer.** A sample costs ~20s of archive
  I/O, so anything that holds samples back before the first yield is billed
  directly as GPU idle time. The in-stream shuffle therefore emits from
  `shuffle_min_fill` (~one batch) onward and grows its buffer while running,
  instead of pre-filling `shuffle_buffer` slots — which cost 264 fetches, about
  80 minutes per worker, before the first batch reached the GPU.
- **Shard the date iterator, never the output stream.** One sample costs
  ~20s of archive I/O (measured), so a shard must only *request* the samples it
  will actually yield. `_build_pipeline` therefore takes `shard_index` /
  `num_shards` and offsets and strides its `DateRange`. Filtering the output
  stream instead (`itertools.islice`) still pulls every discarded sample
  through the pipeline: with 4 ranks x 4 workers that threw away 15 of every 16
  fetches and pushed time-to-first-batch past half an hour, which is
  indistinguishable from a hang (GPU memory allocated, 0% SM utilisation).
- **DataLoader workers are spawned, not forked** — a precaution, since
  `_build_pipeline` is not fork-safe and CUDA/NCCL are live by then. This was
  not the cause of any observed hang.
- **Windows overlap heavily.** At a 10-minute cadence with a 12-frame window,
  consecutive samples share 11 of 12 frames. `sample_stride` and
  `shuffle_buffer` exist to decorrelate them.
- **Memory scales with `batch_size × (n_prior_sat + n_post)`**, since every
  frame passes through the VAE. Prefer gradient accumulation over a large batch.
- **`bf16` needs no `GradScaler`**; the scaler is enabled only for `fp16`.

## References

- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Rombach et al. 2022
- [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) — Peebles & Xie 2023
- [Neighborhood Attention Transformer](https://arxiv.org/abs/2204.07143) — Hassani et al. 2023
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al. 2020
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al. 2020
- [Improved DDPM](https://arxiv.org/abs/2102.09672) — Nichol & Dhariwal 2021

## License

MIT — see [LICENSE](LICENSE).
