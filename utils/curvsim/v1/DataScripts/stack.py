# stack.py

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

def make_data(simpath):

    # ### Import Variables
    metadatafile = f"{PROJECT_ROOT}/{simpath}/metadata.yaml"   #sys.argv[1]
    with open(metadatafile) as f:
        meta = yaml.safe_load(f)

    simpath = meta['logistics']['simpath']
    nshells = meta['simulation']['nshells']
    dcore = meta['particle']['geometry']['dcore']
    wx = meta['particle']['geometry']['wx']
    Nbeads = meta['particle']['geometry']['Nbeads']
    fraction = meta['particle']['geometry']['fraction']
    t0 = meta['particle']['geometry']['t0']
    r0 = meta['particle']['geometry']['r0']
    if r0 == "flat":
        k_0 = 0
    else:
        k_0 = 1/r0
    kh = meta['particle']['elasticity']['kh']
    kckh = meta['particle']['elasticity']['kckh']
    kvkh = meta['particle']['elasticity']['kvkh']
    datagz = meta['simulation']['datagz']
    xlo = meta['simulation']['xlo']
    xhi = meta['simulation']['xhi']
    ylo = meta['simulation']['ylo']
    yhi = meta['simulation']['yhi']
    k_i = meta['simulation']['k_i']

    ##### MAKE CURVAMERS #####

    print("Using stack.py to create data file...")
    print("nshells = {}".format(nshells))
    print("...preparing new simulation data")
    theta = 0    # orientation of stack (0 = concave down)
    x_i = 0    # location of bottom molecule
    y_i = 0
    sim = Curvamer2D(directory=f"{PROJECT_ROOT}/{simpath}")
    for n in range(int(nshells)):

        rx_n = x_i    # x pos of nth molecule
        ry_n = y_i + n * (t0+dcore)    # y pos of nth molecule

        if k_i == 0:    # stack of flat molecules
            k_n = 0    
        else:    # curvature focused stack
            r_i = 1/k_i
            r_n = r_i + n * (t0+dcore)    # radius of curvature of nth molecule
            k_n = 1/r_n

        sim.make_curvamer(rx_n,ry_n,theta,wx,Nbeads,fraction,t0,k_0,k_n,kh,kckh,kvkh)

    ##### MAKE DATA FILE #####
    print("...making data file")
    sim.make_datafile(xlo,xhi,ylo,yhi,zlo=-0.5,zhi=0.5,datadir="simdir",gz=datagz)
          
if __name__ == "__main__":
    simpath = sys.argv[1]
    make_data(simpath)
