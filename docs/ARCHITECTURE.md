# CPDiT architecture

A conceptual walkthrough of the latent diffusion transformer used for solar
irradiance nowcasting, with Mermaid diagrams of the training step, the sampling
loop, the denoiser internals, and the training stages. Numbers throughout are
the defaults in `configs/train_config.yaml`.

The diagrams render on GitHub, or paste a block into https://mermaid.live.

Where the pieces live:
- `src/models/vae.py` — pixel <-> latent compression
- `src/models/networks.py` — `ContextEncoder` + `DiTDenoiser` (the two learned nets)
- `src/models/diffusion.py` — SDEs and reverse-time samplers (no parameters)
- `src/models/latent_diffusion.py` — assembly, score function, loss, sampling
- `src/training/train.py`, `src/training/config.py`, `configs/*.yaml` — staging
- `src/petdata/__init__.py` — pyearthtools data pipeline

## The one-paragraph version

CPDiT is a **score-based latent diffusion model** conditioned on the recent past.
A VAE squashes each 256x256x3 satellite frame to a 4x32x32 latent map. The 12
context frames are encoded temporally by a small transformer; the forecast
frames are corrupted by a continuous-time SDE and a DiT is trained to predict
the noise. At inference you start from pure Gaussian noise in latent space,
integrate the reverse SDE (or the probability-flow ODE) down to t=0 while
calling the DiT at each step, and decode the result to pixels.

Three ideas carry most of the weight:

1. **Latent, not pixel, diffusion.** 256x256x3 -> 4x32x32 is a ~48x reduction in
   what the diffusion model must model (`vae.py`).
2. **The context enters twice, on purpose.** *Spatially* — the 12 context latent
   maps are concatenated channel-wise to the noisy latent before patch embedding,
   so the denoiser knows *where* the clouds are. *Globally* — a pooled summary
   from the temporal transformer, plus diffusion time and lead time, drives
   adaLN-Zero modulation in every block (`networks.py:530-537`).
3. **Score parameterisation.** The net emits `eps`, the score is `-eps/sigma`.
   Under the default `sigma2` weighting the loss is numerically identical to
   epsilon-MSE, but everything built on top — continuous time, choice of SDE,
   PC/ODE samplers, likelihood weighting — is genuinely score-based.

## Diagram 1 — training step (end to end)

```mermaid
flowchart TD
    subgraph DATA["1 · Data — src/petdata"]
        A["Himawari + BARRA archive<br/>3 ch: irradiance, solar elevation, RH24<br/>z-scored, 256x256 crop, 10-min cadence"]
        A --> B["context_images<br/>12 frames, t-120min .. t-10min<br/>B x 12 x 3 x 256 x 256"]
        A --> C["target_images<br/>n_post frames from t<br/>B x Tf x 3 x 256 x 256"]
    end

    B --> E
    C --> E

    subgraph VAE["2 · VAE encoder — vae.py"]
        E["encode every frame once<br/>conv + 3 strided downsamples, /8"]
        E --> F["mu, logvar<br/>4 x 32 x 32 per frame"]
        F --> G["detach mu<br/>stage 1 and 2 only"]
        G --> H["divide by latent_std<br/>EMA buffer, or pinned 2.06<br/>-> roughly unit variance"]
    end

    F -.-> V1["VAE branch: reparameterise,<br/>decode a few frames,<br/>MSE + SSIM + beta*KL"]

    H --> I["context_latents<br/>B x 12 x 4 x 32 x 32"]
    H --> J["target_latents x0<br/>B x Tf x 4 x 32 x 32"]

    subgraph CTX["3 · Temporal context — networks.py"]
        I --> K["flatten 4x32x32 = 4096<br/>Linear -> 512-d token per frame"]
        K --> L["ContextEncoder<br/>4 pre-norm blocks, 8 heads<br/>bidirectional, sinusoidal pos enc<br/>no diffusion time: context is clean"]
        L --> M["encoded_context<br/>B x 12 x 512"]
    end

    subgraph SDE["4 · Forward SDE — diffusion.py"]
        J --> N["sample t, stratified over batch<br/>t in [1e-3, 1]"]
        N --> O["perturb: x_t = alpha(t)*x0 + sigma(t)*z<br/>CosineVPSDE closed-form kernel"]
        O --> P["x_t, z, sigma"]
    end

    P --> Q
    M --> Q
    I --> Q

    subgraph DIT["5 · DiT denoiser — networks.py"]
        Q["DiTDenoiser<br/>see diagram 3"]
        Q --> R["eps_pred<br/>B x Tf x 4 x 32 x 32"]
    end

    R --> S["score = -eps_pred / sigma"]
    S --> T["residual = sigma*score + z = z - eps_pred"]
    T --> U["diffusion_loss<br/>lambda(t) * mean of residual^2<br/>lambda = sigma^2 or g(t)^2"]

    S --> W["Tweedie x0 = (x_t + sigma^2*score)/alpha"]
    W --> X["decode to pixels<br/>pixel_loss on irradiance ch,<br/>only for t <= 0.3"]

    U --> Z["total loss"]
    V1 --> Z
    X --> Z
    Z --> ZZ["AdamW + warmup/cosine, AMP, DDP"]
```

## Diagram 2 — inference / sampling

```mermaid
flowchart TD
    A["context_images<br/>B x 12 x 3 x 256 x 256"] --> B["VAE encode, deterministic mu<br/>scale by latent_std"]
    B --> C["context_latents<br/>B x 12 x 4 x 32 x 32"]
    C --> D["ContextEncoder -> encoded_context"]

    C --> E["score_fn = closure binding<br/>encoded_context + context_latents"]
    D --> E

    F["x_T ~ prior<br/>N(0, I) for VP, sigma_max*N(0,I) for VE<br/>shape B x Tf x 4 x 32 x 32"] --> G

    subgraph LOOP["Reverse integration, t: 1 -> 1e-3, num_steps = 100"]
        G{"sampler"}
        G -->|"pc — stochastic, for ensembles"| H["Langevin corrector x corrector_steps<br/>step size from snr * noise_norm / grad_norm<br/>x += step*score + sqrt(2*step)*noise"]
        H --> I["Predictor: Euler-Maruyama on reverse SDE<br/>drift = f - g^2*score,  plus g*sqrt(-dt)*noise"]
        G -->|"ode — deterministic, for scoring"| J["Probability-flow ODE, Heun<br/>drift = f - 0.5*g^2*score<br/>2 score evals per step"]
        I --> K["clamp to +/- 8 as a divergence guard"]
        J --> K
        K --> L{"t > t_eps ?"}
        L -->|yes| G
    end

    E -.->|"called at every score evaluation"| G

    L -->|no| M["final Tweedie denoise at t_eps<br/>x0 = (x + sigma^2*score)/alpha"]
    M --> N["multiply by latent_std"]
    N --> O["VAE decode -> B x Tf x 3 x 256 x 256"]
    O --> P["forecast_rmse vs target"]
    O --> Q["compare against persistence<br/>last context frame held constant"]
    O --> R["compare against climatology<br/>zero, since fields are z-scored"]
```

## Diagram 3 — inside the DiT denoiser

```mermaid
flowchart TD
    subgraph IN["Input assembly"]
        A["x_t: noisy latent, 4 ch<br/>per forecast frame"]
        B["context_latents: 12 x 4 = 48 ch<br/>broadcast across the horizon"]
        A --> C["concat on channel axis<br/>52 x 32 x 32"]
        B --> C
        C --> D["patch embed: Conv2d 2x2 stride 2<br/>-> 16x16 = 256 tokens, dim 768"]
        D --> E["+ 2-D sin-cos positional embedding"]
    end

    subgraph COND["Conditioning vector, dim 256"]
        F["diffusion time t<br/>scaled by 1000, sinusoidal + MLP"]
        G["context_tokens.mean over T<br/>Linear 512 -> 256"]
        H["lead-time embedding<br/>nn.Embedding, which frame of the horizon"]
        F --> I["sum"]
        G --> I
        H --> I
        I --> J["SiLU + Linear -> cond<br/>per forecast frame"]
    end

    E --> K

    subgraph BLOCKS["12 x DiTBlock"]
        K["LayerNorm, no affine"]
        J -.->|"6 modulation vectors<br/>shift/scale/gate x2"| K
        K --> L["adaLN modulate: x*(1+scale)+shift"]
        L --> M["2-D neighbourhood attention<br/>window 15 on a 16x16 grid,<br/>i.e. effectively global<br/>NATTEN kernel or exact masked softmax"]
        M --> N["residual, gated<br/>gate is zero at init: block = identity"]
        N --> O["LayerNorm + modulate + MLP, ratio 4"]
        O --> P["residual, gated"]
    end

    P -->|"x12"| K
    P --> Q["FinalLayer<br/>adaLN modulate + Linear -> p*p*C<br/>zero-initialised"]
    Q --> R["unpatchify -> 4 x 32 x 32"]
    R --> S["eps_pred per forecast frame"]
```

## Diagram 4 — why training is staged

```mermaid
flowchart LR
    A["Stage 1 — stage1_vae.yaml<br/>vae_loss_weight 1, diffusion 0<br/>VAE trainable<br/>monitor recon_loss"] --> B["measure_latent_scale.py<br/>pin model.latent_scale"]
    B --> C["Stage 2 — train_config.yaml<br/>vae_loss_weight 0, diffusion 1<br/>freeze_vae true, detach_latents true<br/>monitor forecast_rmse"]
    C --> D["Stage 3 — stage3_joint.yaml<br/>joint fine-tune, detach_latents false<br/>guards: latent_norm batch<br/>+ pixel_loss_weight"]

    E["The failure this prevents:<br/>the diffusion loss is epsilon-MSE,<br/>so shrinking mu to zero makes<br/>x_t = sigma*z and eps = x_t/sigma exactly.<br/>Collapsing the latent is its global minimum.<br/>Observed: latent std fell 10x, diffusion loss fell 3x,<br/>forecast skill got 2.4x worse."]
    E -.-> C
    E -.-> D

    F["Guard A: latent_norm=batch divides by a std<br/>computed WITH gradient -> loss is scale-invariant"] -.-> D
    G["Guard B: pixel_loss decodes the Tweedie x0<br/>-> a collapsed latent scores badly, cannot be gamed"] -.-> D
```

## Shapes, at the configured defaults

| Stage | Tensor | Shape |
|---|---|---|
| input | context / target frames | `B x 12 x 3 x 256 x 256` / `B x 1 x 3 x 256 x 256` |
| VAE latent | per frame | `4 x 32 x 32` |
| context token | per frame | `512` |
| DiT input | per forecast frame | `52 x 32 x 32` |
| DiT tokens | patch 2, grid 16x16 | `256 x 768` |
| cond vector | per forecast frame | `256` |
| output | eps_pred | `B x Tf x 4 x 32 x 32` |

## Things worth knowing that the diagrams flatten

- **`t` is continuous, not a step index.** No discrete timestep ladder anywhere;
  `sample_times` draws stratified across the batch so the whole time axis is
  visited each step.
- **The ContextEncoder is not conditioned on diffusion time** — it only ever sees
  clean latents, so there is no noise level to tell it about.
- **`window_size: 15` on a 16x16 token grid means attention is global.** The
  neighbourhood machinery only bites if `patch_size` drops or `image_size` grows.
- **The lead-time embedding is load-bearing** for multi-step forecasts: without
  it every frame of the horizon would be an identical draw from the same
  conditional distribution. With `n_post: 1` it is currently near-inert.
- **Validation is deterministic by construction** — fixed times via `_eval_times`
  and a seeded noise draw — so early stopping is not watching a random variable.
- **`forecast_metrics` runs the real reverse process** on a small fixed subset,
  and always reports persistence and climatology alongside, because an RMSE with
  no baseline says nothing about nowcasting skill.

## Further reading

- Song et al. 2021 — https://arxiv.org/abs/2011.13456 — score-based SDEs
- Rombach et al. 2022 — https://arxiv.org/abs/2112.10752 — latent diffusion
- Peebles & Xie 2023 — https://arxiv.org/abs/2212.09748 — DiT and adaLN-Zero
- Hassani et al. 2023 — https://arxiv.org/abs/2204.07143 — neighbourhood attention
