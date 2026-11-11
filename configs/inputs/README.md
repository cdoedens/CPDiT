# Model variants

Each file here is one model variant: the input variables it is trained on, and
nothing else. Everything else comes from the stage config it is combined with,
so variants differ only in their inputs. The file name is the variant name.

```
python -m src.training.train --config configs/train_vae_extension.yaml --inputs configs/inputs/<variant>.yaml
python -m src.training.train --config configs/train_base.yaml          --inputs configs/inputs/<variant>.yaml --init-from stage1_vae
```

An inputs file may set only `data.himawari_vars`, `data.himawari_bands` and
`data.barra_vars`, and always states the complete set: keys it leaves out are
empty, not inherited. `model.image_channels` is derived from the list.

Weights land in `<runs_root>/<variant>/<stage>/<run_id>/`, with a `latest` link
per stage; `--init-from stage1_vae` resolves to that variant's latest stage-1
`best_model.pt`. `python scripts/list_runs.py` tabulates every run.

A new variant needs, before training:

1. Statistics for any variable not yet in `data.stats_path`:
   `python scripts/calc_norm_stats.py --inputs <file> --only-missing`
   (without arguments it recomputes every variable of every variant here, which
   starts a new frame cache for all of them). `load_config` refuses a config
   whose channels have no statistics.
2. A primed frame cache (optional but ~20x faster): the input list is part of
   the cache key, so each variant has its own.
   `INPUTS=configs/inputs/<variant>.yaml bash scripts/prime_cache.sh`

`barra_vars` takes any BARRA-R2 hourly variable, from either the AUST-11
(convective diagnostics) or AUS-11 (pressure levels, e.g. `hus500`) domain, and
the derived variables `KI` and `TCD`, which the dataloader computes from their
archive fields (see `BARRA_DERIVED` in `src/petdata/__init__.py`).
