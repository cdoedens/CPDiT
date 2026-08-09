import subprocess
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys

for split in ["train", "val", "test"]:
            
    # Generate a unique file name based on iteration
    joboutdir = '/home/548/cd3022/repos/CPDiT/scripts/data_prep/'
    job_script_filename = joboutdir + f'prepare_data_{split}.qsub'
    
    # Open the file for writing
    with open(job_script_filename, "w") as f3:
        f3.write('#!/bin/bash \n')
        f3.write('#PBS -l walltime=4:00:00 \n')
        f3.write('#PBS -l mem=96GB \n')
        f3.write('#PBS -l ncpus=48 \n')
        f3.write('#PBS -l jobfs=10GB \n')
        f3.write('#PBS -l storage=gdata/dk92+gdata/rt52+scratch/nf33+gdata/rv74+gdata/rq0+gdata/ra22+gdata/xp65+gdata/er8+scratch/er8+gdata/ob53 \n')
        f3.write('#PBS -l other=hyperthread \n')
        f3.write('#PBS -q normal \n')
        f3.write('#PBS -P er8 \n')
        f3.write(f'#PBS -o /home/548/cd3022/repos/CPDiT/outputs/logs/data_prep/prepare_data_{split}.log \n')
        f3.write('#PBS -j oe \n')
        f3.write('cd /home/548/cd3022/repos/CPDiT \n') 
        f3.write('source hpc_setup.sh \n')
        f3.write(f'python3 scripts/prepare-data.py {split}\n')


    # Submit the generated script to the job scheduler (PBS) using qsub
    try:
        # Run the qsub command and submit the script
        subprocess.run(['qsub', job_script_filename], check=True)
        print(f"Job script {job_script_filename} submitted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job script {job_script_filename}: {e}")