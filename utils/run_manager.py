# run_manager.py module

import time
import pathlib
import importlib
import yaml
import subprocess
import datetime
import pytz
import os
import shutil
import subprocess

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
lmplocal = "/Users/kyle/Documents/Code/lammps-23Jun2022/src/lmp_mpi" 
lmpunity = "/home/kyltsullivan_umass_edu/lammps-23Jun2022/src/lmp_mpi"

from utils.readsim import ReadSim

# self.cwd = os.getcwd()    # current working directory where curvsim code is being executed

def cluster_modules(computer):
    
    if computer == "unity":
        mods = """module load python/3.12.3
module load conda/latest
conda activate curvsim3
module load openmpi/5.0.3
"""
        
    return mods

def load_class(version: str, module: str, class_name: str, base_module="utils.curvsim") -> type:
    """
    Dynamically import a class from a versioned module.

    Args:
        version: str, e.g., "v1" or "v2"
        module: str, e.g. "curvamer2d" for curvamer.py
        class_name: str, e.g., "Curvamer2D"; not limited to classes, can be anything in the module (e.g functions)
        base_module: str, root module path

    Returns:
        The class object.
    """
    module_name = f"{base_module}.{version}.{module}"
    mymodule = importlib.import_module(module_name)
    return getattr(mymodule, class_name)

def convert_time(time):
    days = int(time//(3600*24))
    hrs = int((time-(days*3600*24))//3600)
    mins = int((time - (days*3600*24) - (hrs*3600))//60)
    secs = time - (days*3600*24) - (hrs*3600) - (mins*60)
    return f"{days}d - {hrs}hrs {mins}min {secs}sec"

def write_metadata(simpath,params):
    """
    Write simulation metadata.
    
    Args:
        simpath: str, location to write yaml file
        params: dict, parameters to write to file
        
    Returns:
        None.
    """
    
    #if (os.path.exists(f"{simpath}/metadata.yaml") == False):

    with open(f"{simpath}/metadata.yaml", "w") as f:
        yaml.dump(params, f, sort_keys=False)
        
def read_metadata(simpath):
    """
    Read simulation metadata.
    
    Args:
        simpath: str, path to simulation directory containing yaml file
        
    Returns:
        metadata dictionary
    """
    with open(f"{simpath}/metadata.yaml") as f:
        meta = yaml.safe_load(f)
    return meta
        
def update_metadata(simpath,increment_run=False,sub=None,params=None,submit_times=False,jobids=None,
                    start_times=False,walltimes=False,steps=False,energies=False):
    """
    Update simulation metadata file.
    
    Args:
        simpath: str, location to write yaml file
        params: dict, parameters to update file with (new or overwrite, cannot append to lists)
        
    Returns:
        None.
    """
    
    current_time = datetime.datetime.now(pytz.timezone("America/New_York"))
    
    meta = read_metadata(simpath)
        
    if increment_run:
        meta['logistics']['run_counter']+=1
        
    if params != None:
        if sub == None: 
            meta.update(params)
        else:
            try:
                meta[sub].update(params)
            except:
                meta.setdefault(sub,{})
                meta[sub].update(params)
                
    if submit_times:
        meta['logistics'].setdefault('submit_times',[]).append(f"{current_time}") 
        
    if jobids != None:
        meta['logistics'].setdefault('jobids',[]).append(f"{jobids}") 
        
    if start_times:
        meta['logistics'].setdefault('start_times',[]).append(f"{current_time}") 
    
    if walltimes or steps or energies:
        result = ReadSim(simpath)
        result.read_log()
        walltime = result.walltime
        step = result.thermo_Step[-1]
        energy = result.thermo_TotEng[-1]
        if walltimes:
            meta['logistics'].setdefault('wall_times',[]).append(f"{walltime}")
        if steps:
            meta['logistics'].setdefault('steps',[]).append(f"{step}")
        if energies:
            meta['logistics'].setdefault('energies',[]).append(f"{energy}")
    
    write_metadata(simpath,meta)
        
def make_simpaths_file(JOBDIR,JOB):
    """
    Make empty status file for job.
    
    Args:
        JOBDIR: str, absolute path to directory containing job file
        JOB: str, name of job file
    """
    with open(f"{JOBDIR}/{JOB}-simpaths.txt","w") as f:
        pass
    
def update_simpaths_file(JOBDIR,JOB,simpath):
    """
    Append 'simpaths' file with paths to simulation directories for this job.
    
    Args:
        JOBDIR: str, absolute path to directory containing job file
        JOB: str, name of job file
        simpath: str, relative path from PROJECT_ROOT to simulation directory
    """
        
    with open(f"{JOBDIR}/{JOB}-simpaths.txt","a") as f:
        
        f.write(f"{simpath}\n")
        
def check_restart(simpath):
    
    meta = read_metadata(simpath)
    simtype = meta['simulation']['simtype']
    
    if simtype == "emin":
        try:
            if meta['logistics']['stopcriterion'] == 'walltime limit reached':
                return True
            else:
                return False
        except:
            return True
        
    if simtype == "md":
        try:
            stepsrun = float(meta['logistics']['steps'][-1])
            if stepsrun == float(meta['simulation']['runsteps']):
                return False
            elif stepsrun < float(meta['simulation']['runsteps']):
                return True
        except:
            return True

        
def print_header(version):
    print(f"""
######################################
######################################

    Coarse-Grained Shell Simulator   
    
        Kyle Thomas Sullivan
       kyltsullivan@umass.edu
       
######################################
######################################

Using curvsim {version}

######################################""")
        
def run_lmp(simpath,computer,ncpus,screen,stage=1):
    
    """
    Run LAMMPS executable.
    
    Args:
        simpath: str, relative path from PROJECT_ROOT to simulation directory
        computer: str, The machine this code will be run on.  
                        Tells location of executable file with lmp variables at top of page.
        ncpus: int, number of processors to use when running in parallel.
        screen: bool, whether to print LAMMPS output to screen while running in addition to a log file.
        stage: int, run counter
    """
    
    if computer == "local":
        lmpmpi = lmplocal
    elif computer =="unity":
        lmpmpi = lmpunity

    t1 = time.time()
    
    if screen==False:
        run = subprocess.run(["mpirun","-np","{}".format(ncpus),lmpmpi,"-var","stage",f"{stage}","-in","in.lammps","-l",f"log{stage}.lammps","-screen","none"],cwd=f"{PROJECT_ROOT}/{simpath}",check=True)
    else:
        run = subprocess.run(["mpirun","-np","{}".format(ncpus),lmpmpi,"-var","stage",f"{stage}","-in","in.lammps","-l",f"log{stage}.lammps"],cwd=f"{PROJECT_ROOT}/{simpath}",check=True)
    
    t2 = time.time()
    telapsed = t2-t1
    hrs = telapsed//3600
    mins = (telapsed%3600)//60
    secs = (telapsed%3600)%60
    print(f"LAMMPS Runtime: {hrs}hrs {mins}min {secs:.1f}sec")
    
def run_job_local(simpath,maxrestarts,jobcounter,simcounter,ncpus,screen,version,datascript,simtype):
    
    """
    Run simulation job locally.
    
    Args:
        simpath: str, relative path from PROJECT_ROOT to simulation directory
        maxrestarts: int, maximum number of resubmissions to finish job if not finished.
        jobcounter: int, job number to be run (outer loop).
        simcounter: int, simulation number within a job (inner loop).
        ncpus: int, number of processors to use when running in parallel.
        screen: bool, whether to print LAMMPS output to screen while running in addition to a log file.
        version: int, version of Curvsim module to use.
        datascript: str, name of data script to use to make data file (no .py), located in utils/curvsim/version/DataScripts folder.
    """
    
    computer = "local"
    nrestarts = 1
    restartjob = True
    while (restartjob==True) and (nrestarts<=maxrestarts):
        print(f"Running job {jobcounter} sim {simcounter}.{nrestarts} on local machine...")

        meta = read_metadata(f"{PROJECT_ROOT}/{simpath}")
        # on first run create data file
        if meta['logistics']['run_counter'] == 0:
            make_data = load_class('DataScripts',f'{datascript}','make_data',base_module=f"utils.curvsim.{version}")
            t1 = time.time()
            make_data(simpath)
            t2 = time.time()
            update_metadata(f"{PROJECT_ROOT}/{simpath}",increment_run=True,sub='logistics',params={'datatime':t2-t1})
            print(f"Time to create data file: {convert_time(t2-t1)}")

        print("Updating metadata with start time...")
        update_metadata(f"{PROJECT_ROOT}/{simpath}",start_times=True)
        print("Executing LAMMPS...")
        meta = read_metadata(f"{PROJECT_ROOT}/{simpath}")    # check what stage to use
        run_lmp(simpath,computer,ncpus,screen,stage = meta['logistics']['run_counter'])
        print("Updating metadata with run info ...")
        update_metadata(f"{PROJECT_ROOT}/{simpath}",walltimes=True,steps=True,energies=True)
        if simtype == "emin":
            result = ReadSim(f"{PROJECT_ROOT}/{simpath}")
            result.read_log()
            update_metadata(f"{PROJECT_ROOT}/{simpath}",sub="logistics",params = {'stopcriterion':result.why_stop})
        restartjob = check_restart(f"{PROJECT_ROOT}/{simpath}")
        if restartjob:
            print("Reached walltime limit.  Restarting new run...")
            nrestarts += 1
            if nrestarts <= maxrestarts:
                update_metadata(f"{PROJECT_ROOT}/{simpath}",increment_run=True)
        print("\n#####################################")

    if nrestarts > maxrestarts:
        print(f"Max number of restarts exceeded ({maxrestarts}).")
        print("#####################################")
            
def run_job_cluster(computer,series_simpaths,jobcounter,nnodes,ncpus,mem,partition,tlim_hrs,tlim_min,JOBDIR,JOB,shname,version,datascript,simtype,screen,maxrestarts,delete_existing):
    
    # find current run counts at time when job is submitted
    # make list of max run counts for each simpath in this job
    rc_max = []
    for s in range(len(series_simpaths)):
        simpath = series_simpaths[s]
        meta = read_metadata(f"{PROJECT_ROOT}/{simpath}")
        rc_max.append( meta['logistics']['run_counter'] + maxrestarts-1 )
    
    ##################################
    ##### SBATCH SUBMISSION FILE #####
    ##################################

    print(f"----Writing sbatch submission file for job {jobcounter}...")
    
    if computer == "unity":
    
        sbatchcontents = f"""#!/bin/bash
#SBATCH -N {nnodes} # Number of Nodes
#SBATCH -n {ncpus} # Number of Tasks (cpus)
#SBATCH --mem={mem}G # Requested Memory
#SBATCH -p {partition} # Partition
#SBATCH --constraint=mpi
#SBATCH -t {tlim_hrs:02d}:{tlim_min:02d}:00 # Job time limit
#SBATCH -o {JOBDIR}/{JOB}-sbatch/{shname}-%j.out # %j = job ID

### load modules
{cluster_modules(computer)}

### go to project_root directory
cd {PROJECT_ROOT}

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
version = "{version}"    # select which version of curvsim to use
curvsim = importlib.import_module(f"utils.curvsim.{version}")
Curvamer2D = rm.load_class(version, "curvamer2d", "Curvamer2D")
Curvamer3D = rm.load_class(version, "curvamer3d", "Curvamer3D")
versionpath = "/".join(curvsim.__name__.split("."))
DATASCRIPTS = f"{{versionpath}}/DataScripts"    # location of compatible data scripts (relative to PROJECT_ROOT)
INPUTSCRIPTS = f"{{versionpath}}/InputScripts"    # location of compatible data scripts

simtype = "{simtype}"

# dependent variables to be run in series
series_simpaths = {series_simpaths}
rc_max = {rc_max}

simpaths_torun = []    # list of simulations that need to be run/restarted
rcmax_torun = []
# sweep through variable values and find those that need to run/restart
for i in np.arange(len(series_simpaths)):
    simpath = series_simpaths[i]
    rcmax = rc_max[i]

    # check to see if this value has already run to completion 
    try:
        restartjob = rm.check_restart(f"{{PROJECT_ROOT}}/{{simpath}}")    
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

    meta = rm.read_metadata(f"{{PROJECT_ROOT}}/{{simpath}}")
    if meta['logistics']['run_counter'] == 0:
        rc = 1
    else: 
        rc = meta['logistics']['run_counter']

    rm.print_header(version)
    print(f"Running {{simpath}}")
    print(f"Run number {{rc}}")

    # Make datafile if needed (on first run)
    if meta['logistics']['run_counter'] == 0:
        make_data = rm.load_class('DataScripts',f'{datascript}','make_data',base_module=f"utils.curvsim.{{version}}")
        t1 = time.time()
        make_data(simpath)
        t2 = time.time()
        rm.update_metadata(f"{{PROJECT_ROOT}}/{{simpath}}",increment_run=True,sub='logistics',params={{'datatime':t2-t1}})
        print(f"Compute time to create data file: {{rm.convert_time(t2-t1)}}")


    # Run LAMMPS
    print("Updating metadata with start time...")
    rm.update_metadata(f"{{PROJECT_ROOT}}/{{simpath}}",start_times=True)
    print("Executing LAMMPS...")
    ncpus = {ncpus}
    meta = rm.read_metadata(f"{{PROJECT_ROOT}}/{{simpath}}")    # check what stage to use
    rm.run_lmp(simpath,"{computer}",ncpus,{screen},stage = meta['logistics']['run_counter'])
    print("Updating metadata with run info ...")
    rm.update_metadata(f"{{PROJECT_ROOT}}/{{simpath}}",walltimes=True,steps=True,energies=True)
    if simtype == "emin":
        result = ReadSim(f"{{PROJECT_ROOT}}/{{simpath}}")
        result.read_log()
        rm.update_metadata(f"{{PROJECT_ROOT}}/{{simpath}}",sub="logistics",params = {{'stopcriterion':result.why_stop}})

    restartjob = rm.check_restart(f"{{PROJECT_ROOT}}/{{simpath}}")  

    # Submit sbatch again if run hasn't completed
    if restartjob:
        if (rc < rcmax):
            print("Job incomplete.  Resubmitting...")
            sbatch = subprocess.run(['sbatch','{JOBDIR}/{JOB}-sbatch/{shname}.sh'],
                                stdout=subprocess.PIPE,universal_newlines = True)

            jobid = int(sbatch.stdout[len('Submitted batch job '):])
            rm.update_metadata(f"{{PROJECT_ROOT}}/{{simpath}}",increment_run=True,submit_times=True,jobids = jobid)
        else:
            print("Error:  Exceeded max number of restarts ({maxrestarts}).")

    # If job has finished, submit next job
    if restartjob == False:  
        if len(simpaths_torun) > 1:    # resubmit this .sh file only if there are more calculations to do
            print("Submitting next job.")
            simpath = simpaths_torun[1]
            sbatch = subprocess.run(['sbatch','{JOBDIR}/{JOB}-sbatch/{shname}.sh'],
                                    stdout=subprocess.PIPE,universal_newlines = True)

            jobid = int(sbatch.stdout[len('Submitted batch job '):])
            rm.update_metadata(f"{{PROJECT_ROOT}}/{{simpath}}",increment_run=False,submit_times=True,jobids = jobid)
        
    print("Done.")



else:
    print("All variable values completed.  No runs left to do for this job.")


EOF
"""

    # WRITE SBATCHCONTENTS TO JOBS/*.SH (or some subdirectory)
    if delete_existing == True:
        if jobcounter == 1:
            if os.path.isdir(f"{JOBDIR}/{JOB}-sbatch"):
                shutil.rmtree(f"{JOBDIR}/{JOB}-sbatch")
    os.makedirs(f"{JOBDIR}/{JOB}-sbatch", exist_ok=True)
    with open(f"{JOBDIR}/{JOB}-sbatch/{shname}.sh", "w") as f:
        f.write(sbatchcontents)

    ###############################
    ##### SUBMIT JOB TO QUEUE #####
    ###############################

    print(f"----Submitting job {jobcounter} to queue...")

    sbatch = subprocess.run(['sbatch',f"{JOBDIR}/{JOB}-sbatch/{shname}.sh"],
                            stdout=subprocess.PIPE,universal_newlines = True)

    jobid = int(sbatch.stdout[len('Submitted batch job '):])


    ###########################
    ##### UPDATE METADATA #####
    ###########################

    print(f"----Updating metadata file...")
    update_metadata(f"{PROJECT_ROOT}/{series_simpaths[0]}",submit_times=True,jobids = jobid)

    print(f"Job {jobcounter} fully prepared and submitted.\n")
    print("#####################################")    
        
        
