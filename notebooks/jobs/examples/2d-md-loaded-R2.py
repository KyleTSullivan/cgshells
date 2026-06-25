# example job submission script
# molecular dynamics job
#     initial configuration loaded from final state of previous job
#     will submit md runs at various temps and concentrations to run in parallel

###########################
##### IMPORT PACKAGES #####
###########################

### standard python packages
import os
import sys
import numpy as np
import subprocess
import importlib
import pathlib
import shutil
import yaml
import datetime
import pytz
import time

### append project_root directory to path
    # needed if executing submission script from cgshells/jobs/path/to/submit.py
    # can comment out if submitting from cgshells/ using "python3 -m jobs.path.to.submit" 
    # see jobs/ReadMe.txt for difference in output when using os.getcwd() for both methods
cwd = os.getcwd()
itop=cwd.find("cgshells/")+len("cgshells")
PROJECT_ROOT = cwd[:itop]
sys.path.insert(0, PROJECT_ROOT )

### utils packages and useful paths
import utils.run_manager as rm
from utils.run_manager import PROJECT_ROOT, lmpunity, lmplocal
from utils.readsim import ReadSim
JOBDIR = pathlib.Path(__file__).resolve().parents[0]    # absolute path to directory that holds this file
JOB = os.path.splitext(os.path.basename(sys.argv[0]))[0]    # name of this file without extension
version = "v1"    # select which version of curvsim to use
curvsim = importlib.import_module(f"utils.curvsim.{version}")
Curvamer2D = rm.load_class(version, "curvamer2d", "Curvamer2D")
Curvamer3D = rm.load_class(version, "curvamer3d", "Curvamer3D")
versionpath = "/".join(curvsim.__name__.split("."))
DATASCRIPTS = f"{versionpath}/DataScripts"    # location of compatible data scripts (relative to PROJECT_ROOT)

rm.print_header(version)
rm.make_simpaths_file(JOBDIR,JOB)     # make empty status file for this job

################################
##### SIMULATION VARIABLES #####
################################

phi_list = [0.05,0.10]    # independent jobs that run in parallel (must be list)
temp_list = [0.025,0.1]    # more independent jobs that run in parallel (must be list)
kh_list = [1000,500]    # jobs that will run in series (must be list)


jobcounter = 0
for i in range(len(phi_list)):
    for j in range(len(temp_list)):
    
        simcounter = 0
        jobcounter += 1

        print("#####################################\n")
        print(f"Setting up job {jobcounter}...")

        series_simpaths = [] # append simpaths that will run in series for each job (i.e. different kh values)

        for k in range(len(kh_list)):  

            simcounter += 1

            print(f"----Preparing simulation {simcounter}...")

            ##### PARTICLE #####
            ### Geometry
            dimension = 2
            dcore = 1.0    # hard core diameter of beads (dcore approx thickness of one DNA helix 3.5nm)
            t0 = 1.0 * dcore    # structural thickness
            wx = 56 * dcore    # shell width (arclength along midline)
            r0 = 34 * dcore   # set to "flat" for particles with zero curvature
            Nbeads = 150    # number of beads per layer (2Nbeads is beads per curvamer)
            fraction = 1/3    # middle patch of beads has width = fraction * wx

            if r0 == "flat":
                k_0 = 0
            else:
                k_0 = 1/r0

            ### Elasticity
            kh = kh_list[k]
            nu = 0.3
            d = wx/(Nbeads-1)   # bead spacing
            kvkh = 1 #(12 * t0**2 * (1-nu)) / (3 * d**2 * (1-nu) - 4 * t0**2)
            kckh = 1 #(3 * (d**2 + t0**2) * (1-nu))/(2 * t0**2)

            ### Interactions
            pair_ints = "1patch" #"none", "repulsive", "1patch", "patchy", "attractive", or "2attractive"
            soft_ints = False
            sigma = 1*dcore
            epsilon = 0.0044
            shift = dcore - 2**(1/6)*sigma     # shift factor to make sure lj minimum is at dcore
            ljcut = 5*sigma #t0 + 2*dcore               # cutoff distance for attractive lj potential
            wcacut = dcore    # cutoff distance for repulsive wca potential
            softsigma = 5*sigma
            softepsilon = 5e-8 * epsilon
            softshift = 0 #softcore - 2**(1/6)*softsigma
            softcut = 2**(1/6) * softsigma

            ##### SIMULATION #####
            config = "dispersed" #"dispersed", "lattice", or "stacked"
            simtype = "md"
            datascript = "load"    # script to make data file with, NO .py EXTENSION, "stack", "load", or "lattice"
            nshells = 40
            datagz = True
            trajgz = True
            dumpbonds = False    # whether to calculate and dump bond data
            screen = True    # output lammps log to screen


            ### Stacked config settings
    #         k_i = 1.25 * k_0    # curvature of bottom shell in stack
    #         xlo = -2*wx
    #         xhi = 2*wx
    #         ylo = -4*r0
    #         yhi = nshells*r0 + 4*r0
    #         zlo = -0.5
    #         zhi = 0.5

            ### Lattice config settings
    #         Nx = 2    # number of particle columns for initial config 
    #         Ny = int(nshells/Nx)
    #         nshells = int(Nx*Ny)   # true number of shells in simulation
    #         k_i = 0    # initial curvature of shells in lattice (need flat for high concentrations)
    #         theta = "random"   # orientation of shells in lattice (0 = concave down, np.pi = concave up, "random" = randomly up or down)

            ### Dispersed config settings
            phi = phi_list[i]    # concentration of molecules (area fraction) - only for MD
            v0 = wx * (t0 + dcore)    # approx area of monomer
            lbox = np.sqrt(nshells * v0 / phi)    # side length of (square) sim box to give proper concentration
            xlo = -lbox/2
            xhi = lbox/2
            ylo = -lbox/2
            yhi = lbox/2
            zlo = -0.5
            zhi = 0.5

            ### Dynamics/Minimization Settings
    #         minstyle = "cg"
    #         etol = 1e-12
    #         maxiter = 100000

            Tstart = temp_list[j]
            Tstop = Tstart
            Tdamp = 1
            seed = 15298
            timestep = 0.001
            runsteps = 100000

            dumpfreq = 5000 #maxiter
            thermofreq = 1000

    #         force  = 0.1

            ##### LOGISTICS #####

            ### Simulation Directories
            delete_existing = True    # if True, deletes simulation directory (and .sh files) if it exists before creating again
            simpath = f"data/examples/{int(dimension)}d/md/kh-{kh:0.3f}/phi-{phi:0.3f}/kT-{Tstart:0.3f}" # path to simulation directory (relative to PROJECT_ROOT)
    #         load_simpath = False # location of simulation to load in (set to False if not loading in state)
            load_simpath = f"data/examples/{int(dimension)}d/md/phi-{phi:0.3f}/initial" # location of simulation to load in (set to False if not loading in state)
            load_dumpname = -1 # name of trajectory file to load (-1 selects highest integer found, e.g trajectory2.dump)
            load_frame = -1    # frame number to load in (not timestep! 0 is initial state; -1 is last completed dump)

            ### Computation
            computer = "local"
    #         computer = "unity"
            nnodes = 1
            mem = 1 #GB
            tlim_hrs = 1
            tlim_min = 0
            partition = "cpu-preempt"    # requested partition
            jobname = os.path.splitext(os.path.basename(sys.argv[0]))[0]
            requested_walltime = f'{tlim_hrs:02d}:{tlim_min:02d}:00'
            tbuffer = 5 # stop lammps tbuffer minutes before walltime is exceeded
                            # any non-zero value means script will auto resubmit until done
                            # zero means no auto resubmission - job stops when done or if time is exceeded
            px = 1    # number of cpus along x
            py = 1
            pz = 1
            gridfreq = 10000    # check cpu partitioning of simbox every gridfreq steps
            thresh = 1.01    # threshold imbalance to repartition simbox
            maxrestarts = 10    # max number of runs for one job (cluster only, local machine limit is set below to 5)

            # add tstep start variable?


            #################
            ##### SETUP #####
            #################

            print("--------Creating simulation directory...")

            ### Make simulation directory

            if delete_existing == True:
                if os.path.isdir(PROJECT_ROOT / simpath):
                    print("""----------Simulation directory already exists.
----------Deleting...""")
                    shutil.rmtree(PROJECT_ROOT / simpath)
                    print("----------Creating new simulation directory...")

            os.makedirs(PROJECT_ROOT / simpath, exist_ok=True)

            ### Append simpaths to job status file and to series_simpaths
            print("--------Adding simulation directory to job simpaths file...")
            rm.update_simpaths_file(JOBDIR,JOB,simpath)
            series_simpaths.append(simpath)

            ###############################
            ##### LAMMPS INPUT SCRIPT #####
            ###############################

            print("--------Writing LAMMPS input file...")

            ### Header
            inputcontents = f"""# Coarse-grained shell model - LAMMPS input file

units lj
dimension {dimension}
boundary p p p
atom_style molecular
bond_style harmonic
angle_style none
dihedral_style none
improper_style none

processors {px} {py} {pz} grid onelevel

comm_style tiled
"""

            if datagz==True:
                inputcontents +="""
variable restart_exists equal is_file(restart.final)
if "${restart_exists}" then "read_restart restart.final" else "read_data data.lammps.gz"

"""
            else:
                inputcontents +="""
variable restart_exists equal is_file(restart.final)
if "${restart_exists}" then "read_restart restart.final" else "read_data data.lammps"

"""

            ### Interactions
            if pair_ints == "none":
                inputcontents += "pair_style none"
            elif pair_ints == "repulsive":
                inputcontents += f"""
pair_style hybrid lj/expand {ljcut}
# upside down bonding 1
pair_coeff 1 1 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
# upside down bonding 2
pair_coeff 2 2 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
# correct bonding
pair_coeff 1 2 lj/expand {epsilon} {sigma} {shift} {wcacut-shift} # edge-edge
pair_coeff 3 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift} # center-center
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes
\n
"""

            else:
                if soft_ints == True:
                    inputcontents += f"""
pair_style hybrid lj/expand {ljcut}
# upside down bonding 1
pair_coeff 1 1 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 3 3 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 1 3 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
# upside down bonding 2
pair_coeff 2 2 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 4 4 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 2 4 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
"""
                else:
                    inputcontents += f"""
pair_style hybrid lj/expand {ljcut}
# upside down bonding 1
pair_coeff 1 1 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
# upside down bonding 2
pair_coeff 2 2 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
"""


                if pair_ints == "1patch":
                    inputcontents += f"""
# correct bonding
pair_coeff 1 2 lj/expand {epsilon} {sigma} {shift} {wcacut-shift} # edge-edge
pair_coeff 3 4 lj/expand {epsilon} {sigma} {shift} {ljcut-shift} # center-center
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes
"""
                elif pair_ints == "patchy":
                    inputcontents += f"""
# correct bonding
pair_coeff 1 2 lj/expand {epsilon} {sigma} {shift} {ljcut-shift} # edge-edge
pair_coeff 3 4 lj/expand {epsilon} {sigma} {shift} {ljcut-shift} # center-center
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes
\n
"""

                elif pair_ints == "attractive":
                    inputcontents += f"""
# correct bonding
pair_coeff 1 2 lj/expand {epsilon} {sigma} {shift} {ljcut-shift} # edge-edge
pair_coeff 3 4 lj/expand {epsilon} {sigma} {shift} {ljcut-shift} # center-center
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {ljcut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {ljcut-shift}
pair_modify shift yes
\n
"""


            # Fixes
            if dimension == 2:
                inputcontents += "fix 0 all enforce2d"
            if simtype == "md":
                inputcontents += f"""
fix 1 all langevin {Tstart} {Tstop} {Tdamp} {seed}
fix 2 all nve
"""
    #         inputcontents += f"""
    # fix 3 all rigid molecule
    # fix 4 all setforce NULL 0.0 NULL
    # fix 4 all setforce 0.0 0.0 0.0

    # group redge id {Nbeads:g} {2*Nbeads:g}
    # fix 5 redge setforce {force:f} 0.0 0.0
    # fix 5 redge addforce {force:f} 0.0 0.0
    # group ledge id {1:g} {Nbeads+1:g}
    # fix 6 ledge setforce {-force:f} 0.0 0.0
    # fix 6 ledge addforce {force:f} 0.0 0.0"""

            inputcontents += f"""
fix 7 all balance {gridfreq} {thresh} rcb

neigh_modify exclude molecule/intra all every 5 delay 0 check yes

comm_modify cutoff {t0 + 2*dcore}

thermo_style custom step etotal ke pe epair ebond temp press
thermo {thermofreq}
thermo_modify norm no
"""

            if dumpbonds == True:
                inputcontents += f"""
compute btype all property/local btype
compute batom1 all property/local batom1
compute batom2 all property/local batom2
compute bdist all bond/local dist
compute bpot all bond/local engpot
dump 2 all local {dumpfreq} bonds.dump.gz index c_btype c_batom1 c_batom2 c_bdist c_bpot
"""
    #     fdata.write("\ndump 1 all custom {} {}/trajectory.dump mol id type x y z".format(dumpfreq,simpath))

            if trajgz==True:
                inputcontents += f"""
dump 1 all custom/gz {dumpfreq} trajectory${{stage}}.dump.gz mol id type x y z
dump_modify 1 append no sort id
"""
            else:
                inputcontents += f"""
dump 1 all custom {dumpfreq} trajectory${{stage}}.dump mol id type x y z
dump_modify 1 append no sort id
"""

            if simtype == "md":
                inputcontents += f"""
timer timeout {int(tlim_hrs*3600 + tlim_min*60 - tbuffer*60)}
timestep {timestep}
run {runsteps} upto
write_restart restart.final
"""
            elif simtype == "emin":
                inputcontents += f"""
timer timeout {int(tlim_hrs*3600 + tlim_min*60 - tbuffer*60)}
min_style {minstyle}
minimize {etol} 0.0 {maxiter} {10*maxiter}
write_restart restart.final
"""        

            # Write LAMMPS input file
            with open(PROJECT_ROOT / f"{simpath}/in.lammps", "w") as f:
                f.write(inputcontents)



            ###############################
            ##### WRITE METADATA FILE #####
            ###############################

            print("--------Writing metadata file...")

            params = {

                'particle':{
                    'geometry':{
                        'dimension':dimension,
                        'dcore':dcore,
                        't0':t0,
                        'wx':wx,
                        'r0':r0,
                        'Nbeads':Nbeads,
                        'fraction':fraction
                    },
                    'elasticity':{
                        'nu':nu,
                        'kh':kh,
                        'kckh':kckh,
                        'kvkh':kvkh
                    },
                    'interactions':{
                        'pair_ints':pair_ints,
                        'soft_ints':int(soft_ints),
                        'sigma':sigma,
                        'epsilon':epsilon,
                        'shift':shift,
                        'ljcut':ljcut,
                        'wcacut':wcacut,

                    },
                },

                'simulation':{
                    'simtype':simtype,
                    'config':config,
                    'nshells':nshells,
                    'datascript':datascript,
                    'xlo':float(xlo),
                    'xhi':float(xhi),
                    'ylo':float(ylo),
                    'yhi':float(yhi),
                    'zlo':float(zlo),
                    'zhi':float(zhi),
                    'simbox_x':float(xhi-xlo),
                    'simbox_y':float(yhi-ylo),
                    'simbox_z':float(zhi-zlo),
                    'thermofreq':thermofreq,
                    'dumpfreq':dumpfreq,
                    'datagz':datagz,
                    'trajgz':trajgz,
                    'dumpbonds':dumpbonds


                },

                'logistics':{
                    'computer':computer,
                    'jobname':jobname,
                    'simpath':simpath,
                    'tbuffer':tbuffer,
                    'run_counter':0

                }
            }

            if soft_ints == True:
                softparams = {'softsigma':softsigma,'softepsilon':softepsilon,
                        'softshift':softshift,'softcut':softcut}
                params['particle']['interactions'].update(softparams)

            if simtype == "emin":
                eminparams = {'minstyle':minstyle,'etol':etol,'maxiter':maxiter}
                params['simulation'].update(eminparams)

            if simtype == "md":
                mdparams = {'Tstart':Tstart,'Tstop':Tstop,'Tdamp':Tdamp,'seed':seed,
                            'timestep':timestep,'runsteps':runsteps}
                params['simulation'].update(mdparams)

            if config == "dispersed":
                disp_params = {'phi':phi}
                params['simulation'].update(disp_params)

            if config == "lattice":
                disp_params = {'phi':phi,"Nx":Nx,"Ny":Ny,'k_i':k_i,"theta":theta}
                params['simulation'].update(disp_params) 

            if config == "stacked":
                stack_params = {'k_i':k_i}
                params['simulation'].update(stack_params)

            if computer != 'local':
                clusterparams = {'nnodes':nnodes,'cpus':px*py*pz,'mem':mem,
                                 'partition':partition,'requested_walltime':requested_walltime}
                params['logistics'].update(clusterparams)

            if load_simpath == False:
                loadparams = {'load_simpath':int(load_simpath)}
            else:
                loadparams = {'load_simpath':load_simpath,'load_dumpname':load_dumpname,'load_frame':int(load_frame)}
            params['logistics'].update(loadparams)


            # Write YAML metadata
            rm.write_metadata(f"{PROJECT_ROOT}/{simpath}",params)


            print(f"----Simulation {simcounter} prepared.")

            ################################
            ##### RUN ON LOCAL MACHINE #####
            ################################

            if computer == "local":
                nrestarts = 1
                maxrestarts = 5    # max number of runs to try to finish job
                restartjob = True
                while (restartjob==True) and (nrestarts<=maxrestarts):
                    print(f"Running job {jobcounter} sim {simcounter}.{nrestarts} on local machine...")

                    meta = rm.read_metadata(f"{PROJECT_ROOT}/{simpath}")
                    # on first run create data file
                    if meta['logistics']['run_counter'] == 0:
                        make_data = rm.load_class('DataScripts',f'{datascript}','make_data',base_module=f"utils.curvsim.{version}")
                        t1 = time.time()
                        make_data(simpath)
                        t2 = time.time()
                        rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",increment_run=True,sub='logistics',params={'datatime':t2-t1})
                        print(f"Time to create data file: {rm.convert_time(t2-t1)}")

                    print("Updating metadata with start time...")
                    rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",start_times=True)
                    print("Executing LAMMPS...")
                    ncpus = px*py*pz
                    meta = rm.read_metadata(f"{PROJECT_ROOT}/{simpath}")    # check what stage to use
                    rm.run_lmp(simpath,computer,ncpus,screen,stage = meta['logistics']['run_counter'])
                    print("Updating metadata with run info ...")
                    rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",walltimes=True,steps=True,energies=True)
                    if simtype == "emin":
                        result = ReadSim(f"{PROJECT_ROOT}/{simpath}")
                        result.read_log()
                        rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",sub="logistics",params = {'stopcriterion':result.why_stop})
                    restartjob = rm.check_restart(f"{PROJECT_ROOT}/{simpath}")
                    if restartjob:
                        print("Reached walltime limit.  Restarting new run...")
                        nrestarts += 1
                        if nrestarts <= maxrestarts:
                            rm.update_metadata(f"{PROJECT_ROOT}/{simpath}",increment_run=True)
                    print("\n#####################################")

                if nrestarts > maxrestarts:
                    print(f"Max number of restarts exceeded ({maxrestarts}).")
                    print("#####################################")
    
        #################################
        ##### RUN ON REMOTE CLUSTER #####
        #################################

        if computer != "local":


            ##################################
            ##### SBATCH SUBMISSION FILE #####
            ##################################

            print(f"----Writing sbatch submission file for job {jobcounter}...")
            # name of sbatch .sh file (no extension)
            if simtype == "emin":
                shname = f"nshells-{nshells}"    
            elif simtype == "md":
                shname = f"nshells-{nshells}-phi-{phi:0.5f}-kT-{Tstart:0.5f}"  
            ncpus = px*py*pz
            sbatchcontents = f"""#!/bin/bash
#SBATCH -N {nnodes} # Number of Nodes
#SBATCH -n {ncpus} # Number of Tasks (cpus)
#SBATCH --mem={mem}G # Requested Memory
#SBATCH -p {partition} # Partition
#SBATCH --constraint=mpi
#SBATCH -t {tlim_hrs:02d}:{tlim_min:02d}:00 # Job time limit
#SBATCH -o {JOBDIR}/{JOB}-sbatch/{shname}-%j.out # %j = job ID

### load modules
{rm.cluster_modules(computer)}

### go to project_root directory
cd {PROJECT_ROOT}

python3 -u << 'EOF' 

### python code
"""

            sbatchcontents += f"""
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

simtype = "{simtype}"

# dependent variables to be run in series
series_simpaths = {series_simpaths}


simpaths_torun = []    # list of simulations that need to be run/restarted

# sweep through variable values and find those that need to run/restart
for i in np.arange(len(series_simpaths)):
    simpath = series_simpaths[i]

    # check to see if this value has already run to completion 
    try:
        restartjob = rm.check_restart(f"{{PROJECT_ROOT}}/{{simpath}}")    
            # returns True if in need of restart; False if completed; error if not run yet
    except:
        restartjob = True

    if restartjob:    # True if value needs to be run or restarted
        simpaths_torun.append(simpath)

if len(simpaths_torun) > 0:
    # For this job, select first variable value that needs to be run/restarted
    simpath = simpaths_torun[0]

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
        if (rc < {maxrestarts}):
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
            rm.update_metadata(f"{PROJECT_ROOT}/{series_simpaths[0]}",submit_times=True,jobids = jobid)

            print(f"Job {jobcounter} fully prepared and submitted.\n")
            print("#####################################")
    


            
