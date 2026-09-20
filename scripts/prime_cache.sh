#!/bin/bash
# Populate the frame cache from the archive. Needs NO GPU -- run it on the
# normal queue with plenty of CPUs, then the GPU job is compute-bound from its
# first epoch. Safe to re-run and safe to interrupt: entries are written
# atomically and a partly-populated cache is simply a partly-warm one.
#
#   qsub -q normal -l ncpus=48,mem=190GB,walltime=4:00:00 scripts/prime_cache.sh
#
# WORKER COUNT IS MEMORY-BOUND HERE, NOT CPU-BOUND. Each worker re-imports the
# full pyearthtools/dask/xarray stack (measured: ~700 MB at import) and then
# keeps growing while it iterates -- MEASURED on a 188 GiB normal-queue node:
# 12 workers alone climbed to ~35 GiB used and was STILL RISING after 2 minutes
# of real fetching, not plateaued. Scaling that to (ncpus - 2) = 46 workers, as
# an earlier version of this script did, exhausted the node and got a worker
# SIGKILL'd by the OOM killer -- a RuntimeError from the DataLoader, but the
# real cause is one level down.
#
# This is consistent with the config's own long-standing warning
# (dataloader.num_workers in train_config.yaml): 24 workers "never finished
# starting" on this pipeline before. That was attributed to import time; the
# measurement here shows sustained memory growth is also part of it.
#
# MORE WORKERS ARE ALSO SLOWER, not just riskier. Measured end-to-end on
# uncached 20-day ranges on a 48-CPU normal-queue node, cold archive fetch:
#     6 workers ... 4.8 frames/s
#    10 workers ... 8.4 frames/s   <- best
#    16 workers ... 7.4 frames/s
#    24 workers ... 3.5 frames/s
# It anti-scales past ~10-16. The work is dominated by gzip decompression of
# over-read HDF5 chunks (the Himawari chunk layout is (1, 20, 2214) -- full
# longitude width -- so reading our 326-column window still decompresses whole
# rows), plus Lustre metadata ops: three day-directory globs per frame inside
# the archive accessor. Neither parallelises the way raw CPU count suggests.
#
# 10 is therefore both the safe AND the fast default. Override with WORKERS=N
# only after measuring -- do not scale this to the node's CPU count.
echo "Running prime_cache, make sure environment is set up beforehand"

WORKERS="${WORKERS:-10}"
CONFIG="${CONFIG:-configs/train_config.yaml}"

echo "Priming $CONFIG with $WORKERS workers..."

# If this sits at zero batches for minutes it is not being slow -- it is being
# asked for dates the archive does not have. The dataset warns after 200
# consecutive unusable anchors. See the ARCHIVE RANGE note in the config:
# nothing exists before 2019-04-01.
python -m src.training.train --config "$CONFIG" --prime-cache --workers "$WORKERS"

echo "Priming completed!"
