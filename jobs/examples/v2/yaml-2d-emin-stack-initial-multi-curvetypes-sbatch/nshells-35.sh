#!/bin/bash
#SBATCH -N 1 # Number of Nodes
#SBATCH -n 1 # Number of Tasks (cpus)
#SBATCH --mem=5G # Requested Memory
#SBATCH -p cpu-preempt # Partition
#SBATCH --constraint=mpi
#SBATCH -t 01:00:00 # Job time limit
#SBATCH -o /work/pi_grason_umass_edu/kboisjolie/cgshells/jobs/examples/v2/yaml-2d-emin-stack-initial-multi-curvetypes-sbatch/nshells-35-%j.out # %j = job ID
#SBATCH --account=pi_grason_umass_edu

### load modules
module load python/3.12.3
module load conda/latest
conda activate /work/pi_grason_umass_edu/PACKAGES/curvsim3/curvsim3
module load openmpi/5.0.3


### go to project_root directory
cd /work/pi_grason_umass_edu/kboisjolie/cgshells

python3 -u << 'EOF' 

### python code

### import modules

# standard python packages
import os
import sys
import numpy as np
import subprocess
import importlib
import pathlib
#import shutil
#import yaml
#import datetime
#import pytz
import time

# utils packages and useful paths
import utils.run_manager as rm
from utils.run_manager import PROJECT_ROOT, lmpunity, lmplocal
from utils.readsim import ReadSim
#JOBDIR = pathlib.Path(__file__).resolve().parents[0]    # absolute path to directory that holds this file
#JOB = os.path.splitext(os.path.basename(sys.argv[0]))[0]    # name of this file without extension
version = "v2"    # select which version of curvsim to use
curvsim = importlib.import_module(f"utils.curvsim.v2")
Curvamer2D = rm.load_class(version, "curvamer2d", "Curvamer2D")
Curvamer3D = rm.load_class(version, "curvamer3d", "Curvamer3D")
versionpath = "/".join(curvsim.__name__.split("."))
DATASCRIPTS = f"{versionpath}/DataScripts"    # location of compatible data scripts (relative to PROJECT_ROOT)
INPUTSCRIPTS = f"{versionpath}/InputScripts"    # location of compatible data scripts

simtype = "emin"

# dependent variables to be run in series
series_simpaths = ["data/examples/2d/emin/wx-10.000-t0-0.600-Nbeads-30/fraction-0.333/species-1/r0-{'A': 6.5}/sigma-0.25000/kh-10.00000/nshells-35/cg-12"]
rc_max = [1]

simpaths_torun = []    # list of simulations that need to be run/restarted
rcmax_torun = []
# sweep through variable values and find those that need to run/restart
for i in np.arange(len(series_simpaths)):
    simpath = series_simpaths[i]
    rcmax = rc_max[i]

    # check to see if this value has already run to completion 
    try:
        restartjob = rm.check_restart(f"{PROJECT_ROOT}/{simpath}")    
            # returns True if in need of restart; False if completed; error if not run yet
    except:
        restartjob = True

    if restartjob:    # True if value needs to be run or restarted
        simpaths_torun.append(simpath)
        rcmax_torun.append(rcmax)

if len(simpaths_torun) > 0:
    # For this job, select first variable value that needs to be run/restarted
    simpath = simpaths_torun[0]
    rcmax = rcmax_torun[0]

    meta = rm.read_metadata(f"{PROJECT_ROOT}/{simpath}")
    if meta['logistics']['run_counter'] == 0:
        rc = 1
    else: 
        rc = meta['logistics']['run_counter']

    rm.print_header(version)
    print(f"Running {simpath}")
    print(f"Run number {rc}")

    # Make datafile if needed (on first run)
    if meta['logistics']['run_counter'] == 0:
        make_data = rm.load_class('DataScripts',f'stack-1species-mixed-curves','make_data',base_module=f"utils.curvsim.{version}")
        t1 = time.time()
        make_data(simpath)
        t2 = time.time()
        rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",increment_run=True,sub='logistics',params={'datatime':t2-t1})
        print(f"Compute time to create data file: {rm.convert_time(t2-t1)}")


    # Run LAMMPS
    print("Updating metadata with start time...")
    rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",start_times=True)
    print("Executing LAMMPS...")
    ncpus = 1
    meta = rm.read_metadata(f"{PROJECT_ROOT}/{simpath}")    # check what stage to use
    rm.run_lmp(simpath,"unity",ncpus,True,stage = meta['logistics']['run_counter'])
    print("Updating metadata with run info ...")
    rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",walltimes=True,steps=True,energies=True)
    if simtype == "emin":
        result = ReadSim(f"{PROJECT_ROOT}/{simpath}")
        result.read_log()
        rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",sub="logistics",params = {'stopcriterion':result.why_stop})

    restartjob = rm.check_restart(f"{PROJECT_ROOT}/{simpath}")  

    # Submit sbatch again if run hasn't completed
    if restartjob:
        if (rc < rcmax):
            print("Job incomplete.  Resubmitting...")
            sbatch = subprocess.run(['sbatch','/work/pi_grason_umass_edu/kboisjolie/cgshells/jobs/examples/v2/yaml-2d-emin-stack-initial-multi-curvetypes-sbatch/nshells-35.sh'],
                                stdout=subprocess.PIPE,universal_newlines = True)

            jobid = int(sbatch.stdout[len('Submitted batch job '):])
            rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",increment_run=True,submit_times=True,jobids = jobid)
        else:
            print("Error:  Exceeded max number of restarts (2).")

    # If job has finished, submit next job
    if restartjob == False:  
        if len(simpaths_torun) > 1:    # resubmit this .sh file only if there are more calculations to do
            print("Submitting next job.")
            simpath = simpaths_torun[1]
            sbatch = subprocess.run(['sbatch','/work/pi_grason_umass_edu/kboisjolie/cgshells/jobs/examples/v2/yaml-2d-emin-stack-initial-multi-curvetypes-sbatch/nshells-35.sh'],
                                    stdout=subprocess.PIPE,universal_newlines = True)

            jobid = int(sbatch.stdout[len('Submitted batch job '):])
            rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",increment_run=False,submit_times=True,jobids = jobid)
        
    print("Done.")



else:
    print("All variable values completed.  No runs left to do for this job.")


EOF
