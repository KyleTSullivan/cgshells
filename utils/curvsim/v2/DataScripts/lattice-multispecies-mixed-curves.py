# lattice.py

import os
import sys
import numpy as np
import importlib
import pathlib
import yaml
import gzip
import random

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


def make_data(simpath):

    # ### Import Variables
    metadatafile = f"{PROJECT_ROOT}/{simpath}/metadata.yaml"   #sys.argv[1]
    with open(metadatafile) as f:
        meta = yaml.safe_load(f)

    simpath = meta['logistics']['simpath']
    nshells = meta['simulation']['nshells']
    Nx = meta['simulation']['Nx']
    Ny = meta['simulation']['Ny']
    phi = meta['simulation']['phi']
    dcore = meta['particle']['geometry']['dcore']
    wx = meta['particle']['geometry']['wx']
    Nbeads = meta['particle']['geometry']['Nbeads']
    fraction = meta['particle']['geometry']['fraction']
    t0 = meta['particle']['geometry']['t0']
    r0 = meta['particle']['geometry']['r0']
    allowedattractions = meta['particle']['interactions']['allowedattractions']
    labels = meta['particle']['interactions']['labels']
    w_average_curves = meta['particle']['geometry']['wavgcurves']
    if r0 == "flat":
        k_0 = 0
    kh = meta['particle']['elasticity']['kh']
    kckh = meta['particle']['elasticity']['kckh']
    kvkh = meta['particle']['elasticity']['kvkh']
    datagz = meta['simulation']['datagz']
    xlo = meta['simulation']['xlo']
    xhi = meta['simulation']['xhi']
    ylo = meta['simulation']['ylo']
    yhi = meta['simulation']['yhi']
    k_i = meta['simulation']['k_i']
    theta = meta['simulation']['theta']
    
    lbox_x = xhi-xlo
    lbox_y = yhi-ylo

    ##### MAKE CURVAMERS #####
    
    print("Using lattice.py to create data file...")
    
    ### Set up initial configuration for melting
    a_x = lbox_x/Nx    # lattice spacing of molecules along x
    a_y = lbox_y/Ny   # lattice spacing of molecules along y
    print("Setting up initial configuration lattice of shells")
    print("nshells = {}".format(nshells))
    print("phi = {}".format(phi))
    print("Sim box dimensions: {:.2f}dcore x {:.2f}dcore".format(lbox_x,lbox_y))
    rxlist = []
    rylist = []
    for j in range(int(Ny)):
        for i in range(int(Nx)):
            xi = xlo + a_x/2 + i * a_x
            yi = yhi - a_y/2 - j * a_y
            rxlist.append(xi)
            rylist.append(yi)
    rxlist = np.array(rxlist)
    rylist = np.array(rylist)
    print("nshells check: {} = {}?".format(Nx*Ny,nshells))
    print("    {}".format(Nx*Ny==nshells))
    print("x-spacing check: {:.2f} > {:.2f}?".format(a_x,wx))   # want greater than particle width
    print("    {}".format(a_x>wx))
    print("y-spacing check: {:.2f} > {:.2f}?".format(a_y,t0+dcore))   # want greater than particle thickness
    print("    {}".format(a_y>t0+dcore))

    #     # visualize lattice
    #     fig, ax = plt.subplots(1,1,figsize=(6,6))
    #     ax.plot(rxlist,rylist,".")
    #     ax.vlines(xlo,ylo,yhi)
    #     ax.vlines(xhi,ylo,yhi)
    #     ax.hlines(ylo,xlo,xhi)
    #     ax.hlines(yhi,xlo,xhi)

    print("...preparing new simulation data")
    sim = Curvamer2D(directory=f"{PROJECT_ROOT}/{simpath}")
    if theta == "random":
        theta = np.pi    # randomnly orient up or down
        for i in range(int(nshells)):
            indexlist = [i for i in range(len(r0))]
            chosen_index = random.choices(indexlist, weights=w_average_curves, k=1)
            chosen_r0 = r0[chosen_index[0]]
            molindex = chosen_index[0] + 1
            moltype_i = molindex
            theta_i = np.random.randint(0,2)*theta
            if chosen_r0 == 'flat':
                k_0 = 0
            else:
                k_0 = 1 / chosen_r0
            
            sim.make_curvamer(moltype_i,rxlist[i],rylist[i],theta_i,wx,Nbeads,fraction,t0,k_0,k_i,kh,kckh,kvkh)
    else:
        for i in range(int(nshells)):
            sim.make_curvamer(moltype_i,rxlist[i],rylist[i],theta,wx,Nbeads,fraction,t0,k_0,k_i,kh,kckh,kvkh)

    ##### MAKE DATA FILE #####
    print("...making data file")
    sim.make_datafile(xlo,xhi,ylo,yhi,zlo=-0.5,zhi=0.5,datadir="simdir",gz=datagz)
          
if __name__ == "__main__":
    simpath = sys.argv[1]
    make_data(simpath)
