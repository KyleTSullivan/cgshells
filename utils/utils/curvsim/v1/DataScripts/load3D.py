# load3D.py

import os
import sys
import numpy as np
import importlib
import pathlib
import yaml
import gzip

### utils packages and useful paths
import utils.run_manager as rm
from utils.run_manager import PROJECT_ROOT
from utils.readsim import ReadSim

version = "v1"    # select which version of curvsim to use
curvsim = importlib.import_module(f"utils.curvsim.{version}")
Curvamer2D = rm.load_class(version, "curvamer2d", "Curvamer2D")
Curvamer3D = rm.load_class(version, "curvamer3d", "Curvamer3D")
versionpath = "/".join(curvsim.__name__.split("."))
DATASCRIPTS = f"{PROJECT_ROOT}/{versionpath}/DataScripts"    # location of compatible data scripts

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
    mesh_name = meta['particle']['geometry']['mesh_name']
    mesh_dir = f"{PROJECT_ROOT}/data/3d/MeshDesigns/{mesh_name}"    # path to mesh directory
    t0 = meta['particle']['geometry']['t0']
    dcore = meta['particle']['geometry']['dcore']
    ki1 = meta['simulation']['ki1']
    ki2 = meta['simulation']['ki2']
    theta_i = meta['simulation']['theta_i']
    kx_0 = meta['particle']['geometry']['kx_0']
    ky_0 = meta['particle']['geometry']['ky_0']
    kxy_0 = meta['particle']['geometry']['kxy_0']
    kh = meta['particle']['elasticity']['kh']
    kckh = meta['particle']['elasticity']['kckh']
    kvkh = meta['particle']['elasticity']['kvkh']
    xlo = meta['simulation']['xlo']
    xhi = meta['simulation']['xhi']
    ylo = meta['simulation']['ylo']
    yhi = meta['simulation']['yhi']
    zlo = meta['simulation']['zlo']
    zhi = meta['simulation']['zhi']
    datagz = meta['simulation']['datagz']
    
    ##### LOAD OLD DATA #####

    print("Using load3D.py to create data file...")
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
    for i in np.arange(old.natoms):
        data_atoms_old.append("{} {} {} {} {} {}".format(
            int(old.dump_id[load_frame][i]),int(old.dump_mol[load_frame][i]),int(old.dump_type[load_frame][i]),
            old.dump_x[load_frame][i],old.dump_y[load_frame][i],old.dump_z[load_frame][i]))
    
    ##### MAKE CURVAMERS #####
    print("...preparing new simulation data")
    x_i = 0    # location of bottom molecule
    y_i = 0
    z_i = 0
    sim = Curvamer3D(directory=f"{PROJECT_ROOT}/{simpath}")

    for n in range(int(nshells)):
        rx_n = x_i    # x pos of nth molecule
        ry_n = y_i    # y pos of nth molecule
        rz_n = z_i #+ n*(t0+dcore)
        theta_zn = 0   # rigid body rotation about z axis for nth molecule
        
        # principal curvatures of nth shell
        if ki1 == 0:
            kn1 = 0
        else:
            kn1 = 1 / ( (1/ki1) + n * (t0+dcore) )  
        if ki2 == 0:
            kn2 = 0
        else:
            kn2 = 1 / ( (1/ki2) + n * (t0+dcore) )

        theta_n = theta_i
        kx_n = 0 #kn1 * np.cos(theta_n)**2 + kn2 * np.sin(theta_n)**2  # curvatures in material directions x, y, xy
        ky_n = 0 #kn1 * np.sin(theta_n)**2 + kn2 * np.cos(theta_n)**2
        kxy_n = 0 #(kn1-kn2) * np.sin(theta_n) * np.cos(theta_n)

        sim.make_curvamer(mesh_dir,rx_n,ry_n,rz_n,theta_zn,t0,kx_0,ky_0,kxy_0,kx_n,ky_n,kxy_n,kh,kckh,kvkh,dcore)
        
    ##### REPLACE POSITION DATA #####
    print("...updating atom positions")
    sim.data_atoms = data_atoms_old 

    ##### MAKE DATA FILE #####
    print("...making data file")
    # Make 1 data file in sim directory containing sim dims and atom, bond, bondcoeff, mass info
    sim.make_datafile(xlo=xlo,xhi=xhi,ylo=ylo,yhi=yhi,zlo=zlo,zhi=zhi,name="data.lammps",
                      datadir="simdir",atoms=True,bonds=True,bondcoeffs=True,masses=True,gz=datagz)
 
          
if __name__ == "__main__":
    simpath = sys.argv[1]
    make_data(simpath)
