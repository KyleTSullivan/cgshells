# example job submission script
# molecular dynamics job
#     lattice of repulsive, flat monomers 
#     used to created random initial configurations for further md jobs
#     will submit md jobs of various concentrations to run in parallel

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
version = "v2"    # select which version of curvsim to use
curvsim = importlib.import_module(f"utils.curvsim.{version}")
Curvamer2D = rm.load_class(version, "curvamer2d", "Curvamer2D")
Curvamer3D = rm.load_class(version, "curvamer3d", "Curvamer3D")
versionpath = "/".join(curvsim.__name__.split("."))
DATASCRIPTS = f"{versionpath}/DataScripts"    # location of compatible data scripts (relative to PROJECT_ROOT)
INPUTSCRIPTS = f"{versionpath}/InputScripts"    # location of compatible data scripts


rm.print_header(version)
rm.make_simpaths_file(JOBDIR,JOB)     # make empty status file for this job

################################
##### SIMULATION VARIABLES #####
################################

phi_list = [0.25]    # independent jobs that run in parallel (must be list)
temp_list = [0.1]    # more independent jobs that run in parallel (must be list)
kh_list = [5000]    # jobs that will run in series (must be list)

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
            t0 = 0.6 * dcore    # structural thickness
            wx = 4.9 * dcore    # shell width (arclength along midline)
            r0 = [3.0 * dcore, 7.5 * dcore, 'flat']   # set to "flat" for particles with zero curvature
            weightedaveragecurves = [0.375, 0.375, 0.25] # weighted average between listed curves in r0. Must add up to one.
            labels = ['A', 'B', 'C']
            allowedattractions = ['A-B', 'A-C', 'B-C']
            Nbeads = 15    # number of beads per layer (2Nbeads is beads per curvamer)
            fraction = 1/3    # middle patch of beads has width = fraction * wx

            if r0 == "flat":
                k_0 = 0
                r0string = r0
            else:
                #k_0 = 1/r0
                r0string = f"{r0}"

            ### Elasticity
            kh = kh_list[k]
            nu = 0.3
            d = wx/(Nbeads-1)   # bead spacing
            alpha = t0/d
            kvkh = 2*(1-alpha**2 * nu)/(alpha**2 - nu)
            kckh = nu*(1 + alpha**2)/(alpha**2 - nu)

            ### Interactions
            nspecies = len(labels) # same as len(labels)
            pair_ints = "repulsive" #"none", "repulsive", "beta"
            beta = fraction # fraction of total attractive energy that comes from mid patch;
                                # 0 = outer flanks only; 1 = mid patch only (previous "1patch"); 
                                # fraction = evenly split between mid and flanks (previous "patchy")
            soft_ints = False
            sigma = 0.25*dcore
            epsilon = 0.11208258168520176
            shift = dcore - 2**(1/6)*sigma     # shift factor to make sure lj minimum is at dcore
            ljcut = 5*sigma #t0 + 2*dcore               # cutoff distance for attractive lj potential
            wcacut = dcore    # cutoff distance for repulsive wca potential
            softsigma = 5*sigma
            softepsilon = 5e-8 * epsilon
            softshift = 0 #softcore - 2**(1/6)*softsigma
            softcut = 2**(1/6) * softsigma

            ##### SIMULATION #####
            config = "lattice" #"dispersed", "lattice", or "stacked"
            simtype = "md"
            datascript = "lattice-multispecies-mixed-curves"    # script to make data file with, NO .py EXTENSION
            inputscript = "2d-multispecies-mixed-curves"    # script to make lammps input file, NO .py EXTENSION   
            nshells = 80
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
            Nx = 5    # number of particle columns for initial config 
            Ny = int(nshells/Nx)
            nshells = int(Nx*Ny)   # true number of shells in simulation
            k_i = 0    # initial curvature of shells in lattice (need flat for high concentrations)
            theta = "random"   # orientation of shells in lattice (0 = concave down, np.pi = concave up, "random" = randomly up or down)

            ### Dispersed config settings
            phi = phi_list[i]    # concentration of molecules (area fraction) - only for MD
            v0 = (wx + dcore) * (t0 + dcore)    # approx area of monomer
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
            damp = 10
            seed = 15298
            timestep = 0.0001
            runsteps = 1*1000000

            dumpfreq = 100000 #maxiter
            thermofreq = 10000#dumpfreq

            ##### LOGISTICS #####

            ### Simulation Directories
            delete_existing = True    # if True, deletes simulation directory (and .sh files) if it exists before creating again
            simpath = f"data/examples/{int(dimension)}d/md/wx-{wx:0.3f}-t0-{t0:0.3f}-Nbeads-{Nbeads}/fraction-{fraction:0.3f}/nshells-{nshells}/species-{int(nspecies)}/r0-{r0string}/phi-{phi:0.3f}/initial" # path to simulation directory (relative to PROJECT_ROOT)
            load_simpath = False # location of simulation to load in (set to False if not loading in state)
    #         load_simpath = f"data/examples/{int(dimension)}d/emin/kh-{kh_load:0.2f}/nshells-{nshells}" # location of simulation to load in (set to False if not loading in state)
    #         load_dumpname = -1 # name of trajectory file to load (-1 selects highest integer found, e.g trajectory2.dump)
    #         load_frame = -1    # frame number to load in (not timestep! 0 is initial state; -1 is last completed dump)

            ### Computation
            computer = "local"
            #computer = "unity"
            nnodes = 1
            mem = 5 #GB
            tlim_hrs = 1 
            tlim_min = 0
            partition = "cpu-preempt"    # requested partition
            jobname = os.path.splitext(os.path.basename(sys.argv[0]))[0]
            requested_walltime = f'{tlim_hrs:02d}:{tlim_min:02d}:00'
            tbuffer = 30 # stop lammps tbuffer minutes before walltime is exceeded
                            # any non-zero value means script will auto resubmit until done
                            # zero means no auto resubmission - job stops when done or if time is exceeded
            px = 1    # number of cpus along x
            py = 1
            pz = 1
            gridfreq = 10000    # check cpu partitioning of simbox every gridfreq steps
            thresh = 1.01    # threshold imbalance to repartition simbox
            maxrestarts = 2    # max number of runs for one job 

            #################
            ##### SETUP #####
            #################

            print("--------Creating simulation directory...")

            ### Make simulation directory

            if delete_existing == True:
                if os.path.isdir(f"{PROJECT_ROOT}/{simpath}"):
                    print("""----------Simulation directory already exists.
----------Deleting...""")
                    shutil.rmtree(f"{PROJECT_ROOT}/{simpath}")
                    print("----------Creating new simulation directory...")

            os.makedirs(f"{PROJECT_ROOT}/{simpath}", exist_ok=True)

            ### Append simpaths to job status file and to series_simpaths
            print("--------Adding simulation directory to job simpaths file...")
            rm.update_simpaths_file(JOBDIR,JOB,simpath)
            series_simpaths.append(simpath)
            
            
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
                        'wavgcurves':weightedaveragecurves,
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
                        'nspecies':nspecies,
                        'pair_ints':pair_ints,
                        'soft_ints':int(soft_ints),
                        'sigma':sigma,
                        'epsilon':epsilon,
                        'shift':shift,
                        'ljcut':ljcut,
                        'wcacut':wcacut,
                        'beta':beta,
                        'labels':labels,
                        'allowedattractions':allowedattractions
                    },
                },

                'simulation':{
                    'simtype':simtype,
                    'config':config,
                    'nshells':nshells,
                    'datascript':datascript,
                    'inputscript':inputscript,
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
                    'dumpbonds':dumpbonds,
                    'gridfreq':gridfreq,
                    'thresh':thresh


                },

                'logistics':{
                    'computer':computer,
                    'jobname':jobname,
                    'simpath':simpath,
                    'tbuffer':tbuffer,
                    'tlim_hrs':tlim_hrs,
                    'tlim_min':tlim_min,
                    'run_counter':0,
                    'px':px,
                    'py':py,
                    'pz':pz

                }
            }

            if soft_ints == True:
                softparams = {'softsigma':softsigma,'softepsilon':softepsilon,
                        'softshift':softshift,'softcut':softcut}
                params['particle']['interactions'].update(softparams)
                
            if pair_ints == "beta":
                betaparams = {'beta':beta}
                params['particle']['interactions'].update(betaparams)

            if simtype == "emin":
                eminparams = {'minstyle':minstyle,'etol':etol,'maxiter':maxiter}
                params['simulation'].update(eminparams)

            if simtype == "md":
                mdparams = {'Tstart':Tstart,'Tstop':Tstop,'damp':damp,'seed':seed,
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

            existsflag = os.path.isfile(f"{PROJECT_ROOT}/{simpath}/metadata.yaml")
            if existsflag==True:
                meta = rm.read_metadata(f"{PROJECT_ROOT}/{simpath}") # read old metadata
                try:    # see if old metadata file has new style
                    test = meta['simulation']['inputscript']
                except:     # add new style variables
                    meta['logistics'].update({'px':px,'py':py,'pz':pz,'tlim_hrs':tlim_hrs,'tlim_min':tlim_min})
                    meta['simulation'].update({'gridfreq':gridfreq,'thresh':thresh,'inputscript':inputscript,'damp':damp})
                meta['simulation']['runsteps'] = params['simulation']['runsteps']  # update with new number of steps to run to
                meta['logistics']['tlim_hrs'] = params['logistics']['tlim_hrs']
                meta['logistics']['tlim_min'] = params['logistics']['tlim_min']
                meta['logistics']['tbuffer'] = params['logistics']['tbuffer']
                meta['logistics']['px'] = params['logistics']['px']
                meta['logistics']['py'] = params['logistics']['py']
                meta['logistics']['pz'] = params['logistics']['pz']
                meta['simulation']['gridfreq'] = params['simulation']['gridfreq']
                meta['simulation']['thresh'] = params['simulation']['thresh']
                if computer != 'local':
                    meta['logistics']['requested_walltime'] = params['logistics']['requested_walltime']
                    meta['logistics']['nnodes'] = params['logistics']['nnodes']
                    meta['logistics']['cpus'] = params['logistics']['cpus']
                    meta['logistics']['partition'] = params['logistics']['partition']
                    meta['logistics']['mem'] = params['logistics']['mem']
                params = meta # replace new metadata with updated old metadata
                params['logistics']['run_counter'] += 1 # increment run counter

            # Write YAML metadata
            rm.write_metadata(f"{PROJECT_ROOT}/{simpath}",params)


            print(f"--------Done.")
            
            
            ###############################
            ##### LAMMPS INPUT SCRIPT #####
            ###############################

            print("--------Writing LAMMPS input file...")
            
            make_input = rm.load_class('InputScripts',f'{inputscript}','make_input',base_module=f"utils.curvsim.{version}")
            make_input(simpath)
            
            print(f"----Simulation {simcounter} prepared.")
            
            
            ################################
            ##### RUN ON LOCAL MACHINE #####
            ################################

            if computer == "local":
                nrestarts = 1
                restartjob = True
                ncpus = int(px*py*pz)
                rm.run_job_local(simpath,maxrestarts,jobcounter,simcounter,ncpus,screen,version,datascript,simtype)
                
        #################################
        ##### RUN ON REMOTE CLUSTER #####
        #################################

        if computer != "local":


            ##################################
            ##### SBATCH SUBMISSION FILE #####
            ##################################

            # name of sbatch .sh file (no extension)
            if simtype == "emin":
                shname = f"nshells-{nshells}"    
            elif simtype == "md":
                shname = f"nshells-{nshells}-phi-{phi:0.5f}-kT-{Tstart:0.5f}"  
            ncpus = int(px*py*pz)
            rm.run_job_cluster(computer,series_simpaths,jobcounter,nnodes,ncpus,
                               mem,partition,tlim_hrs,tlim_min,JOBDIR,JOB,shname,version,
                               datascript,simtype,screen,maxrestarts,delete_existing)
            
