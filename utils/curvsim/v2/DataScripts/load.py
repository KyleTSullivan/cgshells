# load.py

import os
import sys
import numpy as np
import importlib
import pathlib
import yaml
import gzip

# # import custom python classes for my sims
# parentdir = sys.argv[1]
# sys.path.insert(0, parentdir)
# from curvsim.readsim import ReadSim
# from curvsim.curvamer2d import Curvamer2D

### utils packages and useful paths
import utils.run_manager as rm
from utils.run_manager import PROJECT_ROOT, lmpunity, lmplocal
from utils.readsim import ReadSim

version = "v2"    # select which version of curvsim to use
curvsim = importlib.import_module(f"utils.curvsim.{version}")
Curvamer2D = rm.load_class(version, "curvamer2d", "Curvamer2D")
Curvamer3D = rm.load_class(version, "curvamer3d", "Curvamer3D")
versionpath = "/".join(curvsim.__name__.split("."))
DATASCRIPTS = f"{PROJECT_ROOT}/{versionpath}/DataScripts"    # location of compatible data scripts
INPUTSCRIPTS = f"{PROJECT_ROOT}/{versionpath}/InputScripts"    # location of compatible data scripts


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

def make_data(simpath):

    # ### Import Variables
    metadatafile = f"{PROJECT_ROOT}/{simpath}/metadata.yaml"   #sys.argv[1]
    with open(metadatafile) as f:
        meta = yaml.safe_load(f)

    simpath = meta['logistics']['simpath']
    load_simpath = meta['logistics']['load_simpath']
    load_dumpname = meta['logistics']['load_dumpname']
    load_frame = meta['logistics']['load_frame']
    nshells = meta['simulation']['nshells']
    dcore = meta['particle']['geometry']['dcore']
    wx = meta['particle']['geometry']['wx']
    Nbeads = meta['particle']['geometry']['Nbeads']
    fraction = meta['particle']['geometry']['fraction']
    t0 = meta['particle']['geometry']['t0']
#     r0 = meta['particle']['geometry']['r0']
#     if r0 == "flat":
#         k_0 = 0
#     else:
#         k_0 = 1/r0
    kh = meta['particle']['elasticity']['kh']
    kckh = meta['particle']['elasticity']['kckh']
    kvkh = meta['particle']['elasticity']['kvkh']
    datagz = meta['simulation']['datagz']
    xlo = meta['simulation']['xlo']
    xhi = meta['simulation']['xhi']
    ylo = meta['simulation']['ylo']
    yhi = meta['simulation']['yhi']
    
    ##### LOAD OLD DATA #####
    print("Using load.py to create data file...")
    print("nshells = {}".format(nshells))
    
    old = ReadSim(f"{PROJECT_ROOT}/{load_simpath}")
    
    if load_dumpname == -1:    # choose latest trajectory file
        files = old.files
        load_dumpname = latest_traj(files)
    
    # test whether to read all frames from trajectory file or just first/last
    if (load_frame==-1)or(load_frame==0):
        readall=False
    else:
        readall=True
        
    # gather old position data
    print("...fetching old simulation data")
    old.read_dump(dumpname=load_dumpname,readall=readall)
    data_atoms_old = []
    nmol_old = []
    curv_old = []    # preferred curvature
    moltype_old = []
    for i in np.arange(old.natoms):
        data_atoms_old.append("{} {} {} {} {} {}".format(
            int(old.dump_id[load_frame][i]),int(old.dump_mol[load_frame][i]),int(old.dump_type[load_frame][i]),
            old.dump_x[load_frame][i],old.dump_y[load_frame][i],old.dump_z[load_frame][i]))
        if old.dump_mol[load_frame][i] in nmol_old:
            pass
        else:
            nmol_old.append(int(old.dump_mol[load_frame][i]))
            curv_old.append(old.dump_d_curv[load_frame][i])
            moltype_old.append(int(old.dump_i_moltype[load_frame][i]))
    
    ##### MAKE NEW CURVAMERS #####


    print("...preparing new simulation data")
    theta = 0    # orientation of stack (0 = concave down)
    x_i = 0    # location of bottom molecule
    y_i = 0
    sim = Curvamer2D(directory=f"{PROJECT_ROOT}/{simpath}")
    for n in range(int(nshells)):
        # place holder values to be replaced with loaded sim
        rx_n = 0    # x pos of nth molecule
        ry_n = 0    # y pos of nth molecule
        k_n = 0    # initial curvature
        k_0 = curv_old[n]
        moltype = moltype_old[n]
        sim.make_curvamer(moltype,rx_n,ry_n,theta,wx,Nbeads,fraction,t0,k_0,k_n,kh,kckh,kvkh)

    ##### REPLACE POSITION DATA #####
    print("...updating atom positions")
    sim.data_atoms = data_atoms_old  
        
    ##### MAKE DATA FILE #####
    print("...making data file")
    sim.make_datafile(xlo,xhi,ylo,yhi,zlo=-0.5,zhi=0.5,datadir="simdir",gz=datagz)
          
if __name__ == "__main__":
    simpath = sys.argv[1]
    make_data(simpath)
