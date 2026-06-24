# get_energies_gaps.py

# searches through data directories, reads final state energy and copies to a txt file to copy for easy transfering/reading on local machine
# also can calculate and record inter-particle gaps (needs to be added)

get_energies = True
get_gaps = True

### Import modules
import numpy as np
import os
import sys
import gzip
import subprocess
import importlib
import pathlib
import shutil
import yaml
import datetime
import pytz
import time
from scipy.optimize import curve_fit
cwd = os.getcwd()
itop=cwd.find("cgshells/")+len("cgshells")
PROJECT_ROOT = cwd[:itop]
sys.path.insert(0, PROJECT_ROOT )

from utils.readsim import ReadSim
# from utils.curvsim.v1.curvamer2d import Curvamer2D
# from utils.curvsim.v1.curvamer3d import Curvamer3D
import utils.run_manager as rm
# from utils.run_manager import PROJECT_ROOT, lmpunity, lmplocal
version = "v1"    # select which version of curvsim to use
curvsim = importlib.import_module(f"utils.curvsim.{version}")
Curvamer2D = rm.load_class(version, "curvamer2d", "Curvamer2D")
Curvamer3D = rm.load_class(version, "curvamer3d", "Curvamer3D")
versionpath = "/".join(curvsim.__name__.split("."))
DATASCRIPTS = f"{versionpath}/DataScripts"    # location of compatible data scripts (relative to PROJECT_ROOT)


def find_midlayer_pt_3d(mesh_dir,simdir,dumpname,ptx,pty,frame):
    ### load mesh data
    xpos = np.loadtxt("{}/xpos".format(mesh_dir))
    ypos = np.loadtxt("{}/ypos".format(mesh_dir))
    zpos = np.loadtxt("{}/zpos".format(mesh_dir))
    # nlist = np.loadtxt("{}/nlist".format(mesh_dir),dtype=int)
    # indexlist = np.loadtxt("{}/indexlist".format(mesh_dir),dtype=int)
    # npairs = np.loadtxt("{}/npairs".format(mesh_dir),dtype=int)
    # ipairs = np.loadtxt("{}/ipairs".format(mesh_dir),dtype=int)
    atomids = np.loadtxt("{}/atomids".format(mesh_dir),dtype=int)
    # atomtypes = np.loadtxt("{}/atomtypes".format(mesh_dir),dtype=int)
    # bidlist = np.loadtxt("{}/bidlist".format(mesh_dir),dtype=int)
    # btypelist = np.loadtxt("{}/btypelist".format(mesh_dir),dtype=int)

    # wx = np.max(xpos)-np.min(xpos)
    # wy = np.max(ypos)-np.min(ypos)
    natoms = np.size(atomids)
    
    ### read dump file
    result = ReadSim(simdir)
    if (frame==0) or (frame==-1):
        result.read_dump(dumpname=dumpname,readall=False)
    else:
        result.read_dump(dumpname=dumpname,readall=True)
#     result.read_dump(dumpname=dumpname,auto_labels="yes",gz=True)

    ### locate atom nearest to ptx,pty in flat mesh
    dist2 = (xpos-ptx)**2+(ypos-pty)**2
    top = zpos>0
    bot = zpos<0
    iclosest_top = np.argmin(dist2[top])
    iclosest_bot = np.argmin(dist2[bot])
    idtop = atomids[top][iclosest_top]
    idbot = atomids[bot][iclosest_bot]

    ### calculate mid-layer points

    tops = result.dump_id[frame]%natoms == (idtop+1)%natoms
    bots = result.dump_id[frame]%natoms == (idbot+1)%natoms

    dx = result.dump_x[frame][tops]-result.dump_x[frame][bots]
    dy = result.dump_y[frame][tops]-result.dump_y[frame][bots]
    dz = result.dump_z[frame][tops]-result.dump_z[frame][bots]
    rmx = result.dump_x[frame][bots] + 0.5 * dx
    rmy = result.dump_y[frame][bots] + 0.5 * dy
    rmz = result.dump_z[frame][bots] + 0.5 * dz
    dists = np.sqrt(np.diff(rmx)**2 + np.diff(rmy)**2 + np.diff(rmz)**2)

    return (rmx,rmy,rmz,dists)

def latest_traj(files):
    # collect trajectory files
    trajectories = []
    for f in files:
        if f[:10] == "trajectory":
            trajectories.append(f)

    # find latest trajectory file
    tlast = 0
    restart = 0
    for i in np.arange(len(trajectories)):
        traj = trajectories[i]
        p = traj.index(".")
        n = int(traj[10:p])
        if n > restart:
            tlast = i
            restart = n
    dumpname = trajectories[tlast]
    return dumpname

#sigma_list = [0.75,0.7,0.6367897,0.6,0.55,0.5,0.45,0.410829,0.35,0.3,0.25]
sigma_list = [0.5]
kh_list = [1.6]
#kh_list = "all"
nshells_list = np.arange(2,61,1)
#nshells_list = [8]

#alpha = 0 * np.pi/180    # shape 'angle' (0 = cylinder w/ curvature in x; pi/4 = sphere; -pi/4 = saddle) 
alpha = 45 * np.pi/180
#alpha = -45 * np.pi/180

meshdesigns_dir = f"{PROJECT_ROOT}/data/3d/MeshDesigns"
ptx, pty = 0, 0    # find atom closest to this point
frame = -1
gaptxt_dir = f"{PROJECT_ROOT}/jobs/paper3D/gaps"
txt_dir = f"{PROJECT_ROOT}/jobs/paper3D/energies"

for s in np.arange(len(sigma_list)):
    sigma = sigma_list[s]


    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    ##### PARTICLE #####
    ### Geometry
    dimension = 3
    dcore = 1.0    # hard core diameter of beads (dcore approx thickness of one DNA helix 3.5nm)
    wx = 30 * dcore    # mesh width in x 
    wy = 30 * dcore    # mesh width in y 
    a = 0.316 * dcore    # lattice constant of mesh
    mesh_name = f"a-{a:.3f}-wx-{wx:.2f}-wy-{wy:.2f}"    # name of bead-spring mesh to use for shell
  
    

    #sigma = 0.5
    #sigma = 0.45
    #sigma = 0.4
    #sigma = 0.39
    #sigma = 0.38
    #sigma = 0.37
    #sigma = 0.36
    #sigma = 0.35
    #sigma = 0.34
    #sigma = 0.33
    #sigma = 0.32
    #sigma = 0.315
    #sigma = 0.31
    #sigma = 0.305
    #sigma = 0.3
    #sigma = 0.295
    #sigma = 0.29
    #sigma = 0.285
    #sigma = 0.28
    #sigma = 0.275
    #sigma = 0.27
    #sigma = 0.265
    #sigma = 0.26
    #sigma = 0.255
    #sigma = 0.25
    #sigma = 0.1 

    #sigma = 0.25
    #sigma = 0.35
    #sigma = 0.4108
    #sigma = 0.5
    #t0 = 4.434 * dcore
    #r0 = 54.053 * dcore

    #sigma = 0.25
    #sigma = 0.4108
    #t0 = 2.35 * dcore    # structural thickness
    #r0 = 32.893 * dcore   # RMS radius of curvature; set to "flat" for planar plates 
    
    #sigma = 0.25
    #sigma = 0.6368 
    #t0 = 1.2 * dcore    
    #r0 = 21.221 * dcore

    #sigma = 0.25 
    #sigma = 0.4108 
    #t0 = 0.8 * dcore
    #r0 = 21.221 * dcore
    
    #sigma = 0.75
    #sigma = 0.7
    #sigma = 0.6367897
    #sigma = 0.6
    #sigma = 0.55
    #sigma = 0.5
    #sigma = 0.45
    #sigma = 0.410829
    #sigma = 0.35
    #sigma = 0.3
    #sigma = 0.25 
    #sigma = 0.6368 
    #sigma = 0.25 
    #sigma = 0.4108 
    t0 = 0.6 * dcore    
    r0 = 21.221 * dcore

    #sigma = 0.25
    #sigma = 0.8999 
    #sigma = 0.6368 
    #sigma = 0.5
    #sigma = 0.410829
    #t0 = 0.6 * dcore    
    #r0 = 15.015 * dcore

    #alpha = 0 * np.pi/180    # shape 'angle' (0 = cylinder w/ curvature in x; pi/4 = sphere; -pi/4 = saddle) 
    #alpha = 45 * np.pi/180
    #alpha = -45 * np.pi/180
    theta = 0    # angle principal direction 1 makes with material x-axis

    if r0 == "flat":
        k_0 = 0
    else:
        k_0 = 1/r0    # RMS curvature

    # preferred curvatures
    k01 = k_0 * np.cos(alpha)    # principal curvatures
    k02 = k_0 * np.sin(alpha)
    kx_0 = k01 * np.cos(theta)**2 + k02 * np.sin(theta)**2  # curvatures in material directions x, y, xy
    ky_0 = k01 * np.sin(theta)**2 + k02 * np.cos(theta)**2
    kxy_0 = (k01-k02) * np.sin(theta) * np.cos(theta)


    ### Elasticity
    # kh = kh_list[j]
    nuxy = 0.333
    nuz = 0.001
    kvkh = (3*(1-3*nuxy)*(1-nuxy-2*nuz*(t0/a)**2))/(2*nuz*(4*nuz*(t0/a)**2+3*nuxy-1))
    kckh = ((1-3*nuxy)*(1+(t0/a)**2))/(4*nuz*(t0/a)**2+3*nuxy-1)

    #         kvkh = 1 
    #         kckh = 1 

    ### Interactions
    pair_ints = "patchy" #"none", "repulsive", "1patch", "patchy", "attractive", or "2attractive"
    soft_ints = False
    #sigma = 0.4108*dcore
    # epsilon = 0
    shift = dcore - 2**(1/6)*sigma     # shift factor to make sure lj minimum is at dcore
    ljcut = 5*sigma #t0 + 2*dcore               # cutoff distance for attractive lj potential
    wcacut = dcore    # cutoff distance for repulsive wca potential
    # softsigma = 5*sigma
    # softepsilon = 5e-8 * epsilon
    # softshift = 0 #softcore - 2**(1/6)*softsigma
    # softcut = 2**(1/6) * softsigma

    ### Dynamics/Minimization Settings
    minstyle = "cg"
    etol = 1e-10
    #etol = 1e-12
    #etol = 1e-14
    maxiter = 100000


    #parentdir = f"data2/paper3D/{int(dimension)}d/emin/{mesh_name}/t0-{t0:0.3f}-r0-{r0:0.3f}/alpha_{alpha*180/np.pi:0.3f}/sigma-{sigma:0.5f}" 
    parentdir = f"data/paper3D/{int(dimension)}d/emin/{mesh_name}/t0-{t0:0.3f}-r0-{r0:0.3f}/alpha_{alpha*180/np.pi:0.3f}/sigma-{sigma:0.5f}" 
    
    etotal_list = []
    ebond_list = []
    epair_list = []
    nshells_good = []
    stop_list = []
    energy2_list = []
    gaps_list = []

    if kh_list != "all": 
        khlist = kh_list
    else:
        khlist = []
        try:
            result = ReadSim(f"{PROJECT_ROOT}/{parentdir}")
            files = result.files
            for f in files:
                if f[:2] == "kh":
                    khlist.append(float(f[3:]))        
        except:
            pass

    for i in np.arange(len(khlist)):
        kh = khlist[i]
        etotal_i = []
        ebond_i = []
        epair_i = []
        nshells_i = []
        stop_i = []
        energy2_i = []
        gaps_i = []
        for j in np.arange(len(nshells_list)):
            try:
                nshells = nshells_list[j]
                      #simpath = f"data/examples/{int(dimension)}d/emin/{mesh_name}/t0-{t0:0.3f}-r0-{r0:0.3f}/alpha_{alpha*180/np.pi:0.3f}/sigma-{sigma:0.5f}/kh-{kh:0.2f}/nshells-{nshells}/{minstyle}-{int(np.abs(np.log10(etol)))}"
                simpath = f"{parentdir}/kh-{kh:0.5f}/nshells-{nshells}/{minstyle}-{int(np.abs(np.log10(etol)))}" 
                logname = "log.lammps"
                result = ReadSim(f"{PROJECT_ROOT}/{simpath}")
                
                if get_energies == True:
                    result.read_log(logname=logname)
                    etotal_i.append(result.energyf)
                    ebond_i.append(result.thermo_E_bond[-1])
                    epair_i.append(result.thermo_E_pair[-1])
                    nshells_i.append(nshells)
                    stop_i.append(result.why_stop)
                    energy2_i.append(result.energy2)    # second to last energy
                if get_gaps == True:
                    files = result.files
                    dumpname = latest_traj(files)
                    mesh_dir = "{}/{}".format(meshdesigns_dir,mesh_name)
                    rmx,rmy,rmz,dists = find_midlayer_pt_3d(mesh_dir,f"{PROJECT_ROOT}/{simpath}",dumpname,ptx,pty,frame)
                    gaps = -1 * np.ones(np.max(nshells_list)-1)
                    gaps[:len(dists)] = dists
                    gaps_i.append(gaps.tolist())
            except:
                pass

        etotal_list.append(etotal_i)
        ebond_list.append(ebond_i)
        epair_list.append(epair_i)
        nshells_good.append(nshells_i)
        stop_list.append(stop_i)
        energy2_list.append(energy2_i)
        gaps_list.append(gaps_i)

        if len(nshells_good[-1]) > 0:
            if get_energies == True:
                econtents = "# nshells TotEng E_bond E_pair second-to-last_TotEng stopping_criterion\n"
                for j in np.arange(len(nshells_good[-1])):
                    econtents += f'{nshells_good[-1][j]} {etotal_list[-1][j]} {ebond_list[-1][j]} {epair_list[-1][j]} {energy2_list[-1][j]} "{stop_list[-1][j]}"\n'
                    
                with open(f"{txt_dir}/{mesh_name}-t0-{t0:0.3f}-r0-{r0:0.3f}-alpha_{alpha*180/np.pi:0.3f}-sigma-{sigma:0.5f}-kh-{kh:0.5f}-{minstyle}-{int(np.abs(np.log10(etol)))}.txt", "w") as fe:
                    fe.write(econtents)

            if get_gaps == True:
                gcontents = "# nshells gap_between_midsurfaces (-1 = does not exist)\n"
                for j in np.arange(len(nshells_good[-1])):
                    gapinfo = " ".join(np.array(gaps_list[-1][j],dtype=str))
                    gcontents += f'{nshells_good[-1][j]} {gapinfo}\n'

                with open(f"{gaptxt_dir}/{mesh_name}-t0-{t0:0.3f}-r0-{r0:0.3f}-alpha_{alpha*180/np.pi:0.3f}-sigma-{sigma:0.5f}-kh-{kh:0.5f}-{minstyle}-{int(np.abs(np.log10(etol)))}.txt", "w") as fg:
                    fg.write(gcontents) 

                    
