"""
Pull a few batches from a split and report shapes, timing and health.

The quickest way to answer "is the data pipeline actually working?" without
starting a training run. Reports tensor shapes, NaN presence and the achieved
samples/s, so a slow first epoch can be attributed to archive I/O rather than
guessed at.

If it hangs or yields nothing, the split almost certainly starts before the
archive does -- see the ARCHIVE RANGE note in configs/train_config.yaml.

Usage:
    python scripts/probe_data.py [config] [split] [workers] [batch] [n_batches]
    python scripts/probe_data.py configs/train_config.yaml train 4 2 6

The __main__ guard is required, not decorative: workers are spawned, so without
it each worker re-executes this module and builds its own DataLoader.
"""
import logging, sys, time


def main():
    logging.basicConfig(level=logging.INFO)
    import torch
    from src.training.config import load_config
    from src.petdata import build_dataloader

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/train_config.yaml")
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    cfg["dataloader"]["num_workers"] = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    cfg["dataloader"]["batch_size"] = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    n_batches = int(sys.argv[5]) if len(sys.argv) > 5 else 3

    t0 = time.perf_counter()
    loader = build_dataloader(split, cfg, shuffle=False, drop_last=False)
    print(f"[{split}] {cfg['data']['splits'][split]} "
          f"workers={loader.num_workers} batch={loader.batch_size}", flush=True)
    n = 0
    for i, (ctx, tgt) in enumerate(loader):
        n += ctx.shape[0]
        dt = time.perf_counter() - t0
        print(f"batch {i}: ctx={tuple(ctx.shape)} tgt={tuple(tgt.shape)} t={dt:.1f}s "
              f"nan={bool(torch.isnan(ctx).any() or torch.isnan(tgt).any())} "
              f"| {n/dt:.2f} samples/s", flush=True)
        if i + 1 >= n_batches:
            break
    print(f"TOTAL {n} samples in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
