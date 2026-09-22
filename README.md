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
| VAE | [src/models/vae.py](src/models/vae.py) | `(3, 256, 256) → (4, 32, 32)` spatial latents |
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
2. A **`patch_size`×`patch_size` strided convolution** patch-embeds that stack
   into tokens (2×2 by default, giving a 16×16 grid at 256 px).
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

The defaults are `patch_size: 2` and `window_size: 15` — a 16×16 grid with a
window that spans it, i.e. global attention, stated explicitly rather than
arrived at by clamping. At 256 tokens per frame that costs almost nothing, so
locality buys nothing here; it starts to matter at `patch_size: 1`.

The earlier `patch_size: 4` was a real modelling error, not just a wasted
optimisation: an 8×8 grid puts one token per 32×32 pixel block, and cloud
advection over a 10-minute lead is a few pixels. The thing being forecast was
invisible at that resolution. The model logs a warning whenever the configured
window is clamped, which is the symptom to watch for.

`NATTEN`'s fused CUDA kernels are used automatically when importable; otherwise
an exact masked-softmax fallback runs, which is mathematically identical and
perfectly adequate at these grid sizes.

## Data

Each sample is `n_prior_sat` context frames immediately before time *t*, and
`n_post` forecast frames starting at *t*. With the defaults (12 context, 1
forecast, 10-minute cadence) that is a 10-minute-ahead nowcast from two hours
of history.

Channels follow `xr.merge` order, so the Himawari variables come first and
`irradiance_channel: 0` is `himawari_vars[0]`. The default three are
`surface_global_irradiance`, `solar_elevation` (both Himawari) and `RH24mean`
(BARRA).

`solar_elevation` earns its channel: irradiance is z-scored globally, so the
diurnal cycle dominates its variance, and without this the model has to infer
time of day from the cloud field alone. It comes from the same heliosat product
and already has statistics in `combined_stats.json`.

> **The archive starts on 2019-04-01.** `petdata.archive.Himawari` reads
> `/g/data/rv74/.../himawari-ahi/solar/p1s/latest/`, and `latest` is a symlink
> to `v1.1`, whose first day is 2019-04-01. The `v1.0` tree does hold 2015 to
> 2019-03 and does contain `solar_elevation`, but the accessor cannot see it, so
> every request before that date raises `DataNotFoundError`.
>
> This is easy to lose hours to, because it does not look like an error. A
> priming run with `train.start: 2016-01-01` spent 3.2 of its 3.4 minutes
> writing 26,232 "known bad" markers, produced nothing, and showed zero batches
> on the progress bar throughout. The dataset now warns after 200 consecutive
> unusable anchors, and `--prime-cache` logs an error if a split yields nothing
> at all. `scripts/probe_data.py` answers the same question in seconds.

> **Changing the variable list changes the frame cache namespace.** The cache
> key hashes the full recipe, so new entries land in a fresh directory rather
> than serving stale tensors — but the cache must then be re-primed, which needs
> no GPU:
>
> ```bash
> python -m src.training.train --config configs/train_config.yaml --prime-cache
> ```
>
> Watch the summary line for a jump in `skipped: nan`. `solar_elevation` is
> negative at night and any NaN rejects the whole frame.

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

Resume (restores optimiser, scheduler and epoch):

```bash
python -m src.training.train --resume /path/to/checkpoint_epoch_050.pt
```

### Training is staged, and the staging is not optional

Training runs in two required stages plus an optional third. Each starts from
the previous one's weights via `--init-from`, which takes **weights only** and
begins a fresh run at epoch 0 — unlike `--resume`, which also restores optimiser
moments and the LR schedule belonging to the run that produced the checkpoint.

```bash
# 0. Is a latent model viable on this data at all? (see below)
python scripts/check_vae_floor.py --config configs/train_config.yaml

# 1. Train the VAE alone.
python -m src.training.train --config configs/stage1_vae.yaml

# ...then measure the latent scale and pin it in train_config.yaml.
python scripts/measure_latent_scale.py --checkpoint .../stage1_checkpoints/best_model.pt

# 2. Train the diffusion model against the frozen VAE. This is the deliverable.
python -m src.training.train --config configs/train_config.yaml \
    --init-from .../stage1_checkpoints/best_model.pt

# 3. OPTIONAL: fine-tune the VAE jointly with the nowcast. Ships only if it
#    beats stage 2's forecast_rmse under an identical sampler and val subset.
python -m src.training.train --config configs/stage3_joint.yaml \
    --init-from .../testing_checkpoints/best_model.pt
```

Stage configs use `extends:` to overlay a handful of keys onto
`train_config.yaml`, so the data pipeline, geometry and HPC block cannot drift
between stages.

### Why the VAE is pretrained rather than trained jointly

**Collapsing the latent is the global minimum of the diffusion loss.** The
objective reduces to ε-MSE, so if the encoder can drive `mu → 0` then
`x_t = α·0 + σ·z = σ·z` and the denoiser recovers `z = x_t/σ` exactly. Nothing
opposes it: reconstruction is scale-free (the decoder simply grows its weights)
and the KL pushes the same way.

A single-stage run did exactly this. Over 11 epochs the latent std fell 10×
(0.0238 → 0.0023) and the diffusion loss fell 3× — while forecast RMSE got 2.4×
*worse* and the model ended up worse than predicting the climatological mean.
The reverse process starts from `N(0, I)`, and the score network had only ever
seen latents with a fraction of that variance.

Two mechanisms guard against it:

| Guard | Key | What it does |
|---|---|---|
| Detach | `training.detach_latents: true` | Cuts the diffusion gradient at the encoder. Default, and what stages 1-2 rely on. |
| Scale invariance | `model.latent_norm: batch` | Divides by a std computed *with* gradient, so the loss is exactly invariant to the latent scale and shrinking buys nothing. |
| Task-aware signal | `training.pixel_loss_weight` | Scores the *decoded* x̂₀ against truth at low `t`. A collapsed latent decodes to climatology and scores terribly, so it cannot be gamed by shrinking. |

`load_config` **rejects** `detach_latents: false` unless `latent_norm: batch` is
also set. Stage 3 is the only configuration that turns the detach off, and it
turns on both other guards at the same time.

Watch `latent_scale_ratio` in the epoch log: it is the true latent std over the
one the sampler's prior assumes, and it must stay near 1.0.

### Checkpoints

The best model is written to a stable `best_model.pt`; periodic checkpoints are
epoch-numbered and rotated to the last `keep_last_n_checkpoints`. Every
checkpoint embeds its config, so inference reconstructs the architecture without
a second config file.

### Latent scaling

Diffusion assumes roughly unit-variance data, but VAE latents have an arbitrary
scale. The model tracks an EMA of the latent standard deviation and divides by
it — the running-statistics analogue of Stable Diffusion's fixed `0.18215`. The
value is logged each epoch as `Latent scale (EMA std)`, alongside the
`true/assumed std ratio` that says whether it is actually keeping up.

For stage 2 this must be **pinned, not tracked**: run
`scripts/measure_latent_scale.py` on the stage-1 checkpoint and put the result in
`model.latent_scale`. An EMA is a moving target, and while the VAE is frozen
there is nothing left to track anyway.

`latent_scale_momentum` matters whenever the encoder is trainable. Measured on a
moving encoder: **0.99 left the buffer 15-40% behind the true std, while 0.9
tracked to within 1%.** Training under `latent_norm: batch` is invariant to that,
but *sampling* is not — the reverse chain unscales with this buffer — so stages 1
and 3 use 0.9.

A checkpoint carries `latent_std` as a buffer, so loading one would otherwise
overwrite a pinned value with the previous stage's drifted EMA. Both `--resume`
and `--init-from` re-apply `model.latent_scale` after the load when it is set.

## Evaluation and inference

```bash
# VAE round-trip error against persistence -- run this before training anything
python scripts/check_vae_floor.py --checkpoint .../best_model.pt --split val

# Qualitative: context / truth / forecast panel
python scripts/plot_forecast.py --checkpoint .../best_model.pt --split val --sample-index 0
```

`LatentDiffusionTransformer.forecast_metrics()` integrates the full reverse
process and returns `forecast_rmse` alongside `persistence_rmse` and
`climatology_rmse`, so every score arrives with its baselines attached. **A
nowcast that does not beat persistence is not a nowcast.**

Measured on a validation subset, in z-scored units and W/m² (irradiance std is
356 W/m²):

| Forecast | RMSE (z) | RMSE (W/m²) |
|---|---:|---:|
| persistence — last observed frame | **0.155** | **55** |
| climatology — the dataset mean | 0.499 | 178 |

At a 10-minute lead persistence is a *very* strong baseline: it is 3.2x better
than climatology. That is the bar, and it is worth knowing before reading any
model number as a success.

`scripts/check_vae_floor.py` answers a prior question: the VAE round trip is
lossy, so its reconstruction error is an irreducible floor under any forecast
the latent model can produce. If that floor is not comfortably below
persistence, no amount of denoiser capacity will help and the compression itself
is what needs to change. It reports per channel in both z-scored and physical
units, and prints a PASS / MARGINAL / FAIL verdict on the scored channel.

For programmatic use, `src.inference` exposes `Forecaster` and
`load_model_from_checkpoint`; note these run in fp32, which is what you want for
a sampler that takes 100+ sequential solver steps.

During training these validation signals are logged:

- `forecast_rmse` / `forecast_mae` — the full reverse process (PF-ODE), on the
  first `evaluation.forecast_eval_batches` validation batches only, because it is
  expensive. **This is the early-stopping monitor**, and it is run outside
  autocast: 100+ sequential solver steps accumulate real error in bf16.
- `persistence_rmse` / `climatology_rmse` — logged beside it, with the ratio to
  persistence and a plain `BEATS IT` / `WORSE` verdict. A nowcast that does not
  beat copying the last observed frame is not a nowcast, and an RMSE with nothing
  to compare it against says nothing at all.
- `irradiance_mse` / `irradiance_mae` — a cheap single-step x̂₀ estimate at fixed
  times with a fixed noise draw. Useful as a trace, but **not** a skill metric:
  it is evaluated at only `batch_size` fixed times, half of them where `α` is
  small enough that the x̂₀ estimate is meaningless, and it improved steadily
  through the run that destroyed forecast skill. It is no longer the default
  monitor for that reason.
- `latent_scale_ratio` — the latent health check described above.

`load_config` rejects a `forecast_*` monitor when
`evaluation.forecast_eval_batches` is 0, since the metric would never be computed
and early stopping would silently never fire.

## Configuration

[configs/train_config.yaml](configs/train_config.yaml) is the single source of
truth, and every key in it is read by the code. The stage configs overlay it:

| Config | Stage | Overrides |
|---|---|---|
| [train_config.yaml](configs/train_config.yaml) | 2 — diffusion, frozen VAE | the base; everything else `extends:` it |
| [stage1_vae.yaml](configs/stage1_vae.yaml) | 1 — VAE alone | `diffusion_loss_weight: 0`, weaker KL, `monitor: recon_loss` |
| [stage3_joint.yaml](configs/stage3_joint.yaml) | 3 — joint fine-tune | `detach_latents: false` + both guards, `vae_lr_mult: 0.1` |
| [smoke.yaml](configs/smoke.yaml) / [smoke2.yaml](configs/smoke2.yaml) / [smoke3.yaml](configs/smoke3.yaml) | — | the same three stages over ~19 cached days, for exercising the whole loop in minutes |

The smoke chain is the fastest way to confirm a change has not broken anything
end to end — real data, real optimiser, real forecast metrics, a couple of
minutes per stage:

```bash
CK=/scratch/er8/cd3022/CPDiT_smoke/checkpoints
python -m src.training.train --config configs/smoke.yaml  --workers 4
python -m src.training.train --config configs/smoke2.yaml --workers 4 --init-from $CK/best_model.pt
python -m src.training.train --config configs/smoke3.yaml --workers 4 --init-from ${CK}_stage2/best_model.pt
```

Notable keys:

```yaml
model:
  image_size: 256
  latent_scale: null              # null = EMA-tracked; a float pins it
  latent_norm: ema                # ema | batch (batch = scale-invariant loss)
  latent_scale_momentum: 0.99     # use 0.9 whenever the encoder is trainable
  diffusion:
    patch_size: 2                 # 16x16 token grid at 256 px
    embed_dim: 768
    num_blocks: 12                # DiT-B depth
    num_heads: 12
    window_size: 15               # neighbourhood window (odd; clamped to the grid)

data:
  n_prior_sat: 12                 # context frames
  n_post: 1                       # forecast frames
  sample_stride: 3                # skip overlapping windows

dataloader:
  batch_size: 8                   # re-measure with --benchmark after any resize
  num_workers: 10                 # pipeline is built per worker, so >0 is safe
  shuffle_buffer: 256             # reservoir shuffle over the date-ordered stream
  shuffle_min_fill: 8             # emitted from here; the buffer is not pre-filled

training:
  precision: bf16                 # auto-falls back to fp16 on pre-Ampere GPUs
  gradient_accumulation_steps: 4
  vae_loss_weight: 0.0            # stage weights; see "Training is staged"
  diffusion_loss_weight: 1.0
  freeze_vae: true
  detach_latents: true            # load_config rejects false unless latent_norm: batch

evaluation:
  forecast_eval_batches: 16       # this metric gates early stopping
  max_val_batches: 200            # the val split is otherwise as large as train
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
construction, and latent scaling. Most do not import `src.petdata`, so they run
without the pyearthtools runtime; the data-pipeline tests skip themselves.

Three groups exist specifically to keep the collapse from coming back:

- `LatentCollapseTests` — the detach cuts the encoder gradient; `latent_norm:
  batch` makes the loss exactly scale-invariant while `ema` does not; and 40
  steps of diffusion-only training leave the latent std within 20% of where it
  started. **That last one fails on the pre-fix code**, which is the point.
- `StageControlTests` — each stage wires up the parameters it should and no
  others, and the pixel loss actually penalises a collapsed latent (measured
  separation: ~350x, once the VAE can reconstruct at all).
- `ConfigStageTests` / `ShippedConfigTests` — `extends:` overlays correctly, and
  the shipped configs still mean what they claim.

One subtlety worth knowing when writing tests here: **adaLN-Zero makes a fresh
denoiser emit exactly 0.0 for any input.** The loss is then `||z||²`, a constant
with no dependence on `x_t`, so any test of gradient flow or scale invariance
silently measures nothing. `_wake_denoiser()` gives the output path real weights
first.

## Performance notes

### Measured throughput, 1x H200, 256 px, 3 channels

From `--benchmark`, at the geometry in the configs. Per rank; the H200 has
143 GiB.

| Stage | batch | peak mem | samples/s | note |
|---|---:|---:|---:|---|
| 1 — VAE alone | 16 | 27.2 GiB | 34.3 | with `vae_max_frames: 3` |
| 1 — VAE alone | 32 | 53.2 GiB | 35.3 | |
| 1 — *all 13 frames* | 16 | *113.4 GiB* | *7.8* | and OOM at batch 32 |
| 2 — diffusion, frozen VAE | 16 | 17.9 GiB | 50.0 | |
| **2 — diffusion, frozen VAE** | **32** | **29.3 GiB** | **54.1** | **the knee** |
| 2 — diffusion, frozen VAE | 96 | 68.9 GiB | 47.7 | past the peak |
| 3 — joint fine-tune | 8 | 32.5 GiB | 13.1 | with `vae_max_frames: 3` |
| **3 — joint fine-tune** | **16** | **62.8 GiB** | **13.3** | |
| 3 — joint fine-tune | 32 | 123.3 GiB | 13.4 | no headroom |

Three things to take from this:

- **Throughput saturates well before memory does.** Stage 2 peaks at batch 32
  and is *slower* at 96. Filling the card is not the goal.
- **`vae_max_frames` is the difference between fitting and not.** The VAE is a
  per-frame model and a sample's 13 frames are 10 minutes apart, so
  reconstructing all of them costs 13x the decoder memory for near-duplicate
  signal. Scoring 3 gave stage 1 **4.4x the throughput at a quarter of the
  memory**, and lifted an OOM.
- **Stage 3 is 4x slower than stage 2** — the whole autoencoder is in the
  autograd graph and the decoder runs twice. Budget for it before starting.

For reference, the run that motivated all of this managed **6.75 samples/s** at
batch 2 with `gradient_accumulation_steps: 2`, on the same hardware.

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
- **`--prime-cache` worker count does not scale with CPU count — 10 is both the
  safe and the fast default.** Measured on uncached 20-day ranges, cold archive
  fetch, on a 48-CPU normal-queue node:

  | workers | frames/s |
  |---:|---:|
  | 6 | 4.8 |
  | **10** | **8.4** |
  | 16 | 7.4 |
  | 24 | 3.5 |

  It anti-scales past ~10-16, and it is memory-bound at the top end: 12 workers
  alone reached ~35 GiB resident and was still rising after 2 minutes, and a
  `--workers` value derived from `ncpus - 2` (46) OOM-killed a DataLoader
  worker on a 188 GiB node. `--workers` above 20 warns rather than refuses,
  since some environments have the headroom; raise it by measurement, never by
  CPU count.
- **94% of a cold frame fetch is the satellite read**, and most of that is gzip
  decompression of data we then throw away. Per-frame profile: satellite open
  152 ms + lazy resolve 247 ms, against BARRA 12 ms, regrid 7 ms, normalise
  7 ms. The Himawari files are chunked `(1, 20, 2214)` — full longitude width —
  so extracting our 326-column window still decompresses whole rows, ~6 MB per
  variable for the 0.85 MB we keep. Adding `solar_elevation` as a second
  satellite variable therefore roughly doubles priming cost; it is a one-time
  cost paid into a persistent cache, but it is the reason priming is not faster.
- **Priming stats the cache instead of reading it.** `--prime-cache` only needs
  each frame to *be* on disk, so it yields placeholders rather than decoding and
  collating tensors it will never look at. Re-walking a 60k-frame cache to reach
  the uncached frontier went from **23 to 128 anchors/s** — 26 minutes of pure
  re-reading down to under 5. This matters because priming is re-run every time
  a job hits walltime, and each re-run starts from the beginning of the range.
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
