### 3D Curvamer Simulation
# Make stack data file
# Load atom positions 

import numpy as np
import os
import sys
import subprocess
import time
from scipy.optimize import curve_fit
import gzip

# import custom python classes for my sims
from curvsim3d.readsim import ReadSim
from curvsim3d.curvamer3d import Curvamer3D


### Mesh Variables
dcore = float(sys.argv[1])
a = float(sys.argv[2])
wx = float(sys.argv[3])
wy = float(sys.argv[4])
# mesh_name = "a-{:.3f}-wx-{:.2f}-wy-{:.2f}".format(a,wx,wy)
# meshdesigns_dir = "./MeshDesigns"
mesh_dir = sys.argv[5]#"{}/{}".format(meshdesigns_dir,mesh_name)    

### Curvamer Variables
nmols = int(sys.argv[6])
kh = float(sys.argv[7])
t0 = float(sys.argv[8])
# r0 = wx
# geometry = "flat"
# geometry = "cylinder"
# geometry = "sphere"
# geometry = "saddle"
kx_0 = float(sys.argv[9])
ky_0 = float(sys.argv[10])
kxy_0 = float(sys.argv[11])
kx_i = 0    # initial positions don't matter.  Will update with loaddirectory positions.
ky_i = 0
kxy_i = 0
nu = float(sys.argv[12])
davg = a
kckh = float(sys.argv[13]) #3*(t0**2 + davg**2)*(1 - nu) / (2 * t0**2)
kvkh = float(sys.argv[14]) #(12 * t0**2)*(1 - nu) / (4*t0**2 - 3*davg**2 * (1 - nu))
rx = 0
ry = 0
rz = 0
theta_z = 0

### Sim Box
xlo = -1.5*wx
xhi = 1.5*wx
ylo = -1.5*wy
yhi = 1.5*wy
zbuffer = 10*(t0+dcore)
zlo = -nmols*(t0+dcore) - zbuffer
zhi = nmols*(t0+dcore) + zbuffer

### Run Data
px = int(sys.argv[15])
py = int(sys.argv[16])
pz = int(sys.argv[17])
nproc = [px,py,pz]
sigma = float(sys.argv[18])
epsilon = float(sys.argv[19])
thermofreq = int(sys.argv[20])
etol = float(sys.argv[21])
emax = int(sys.argv[22])
dumpfreq = emax
minstyle = sys.argv[23] #cg hftn sd quickmin fire
simpath = sys.argv[24]
loaddirectory = sys.argv[25]
shift = dcore - 2**(1/6)*sigma
ljcut = shift + 5*sigma #shift + 5*sigma #t0 + 2*dcore
wcacut = dcore
setforce=False # False or [0.0,0.0,0.0] or ["NULL","NULL",0.0]

### Make Curvamers
sim = Curvamer3D(directory = simpath)

for n in range(nmols):
    rx_n = rx
    ry_n = ry
    rz_n = rz + n*(t0+dcore)
    if kx_i != 0:
        kx_n = 1/((1/kx_i) + n*(t0+dcore))
    else:
        kx_n = 0

    if ky_i != 0:
        ky_n = 1/((1/ky_i) + n*(t0+dcore))
    else:
        ky_n = 0
        
    kxy_n = kxy_i
    ### NEED kxy_n!!!
        
    sim.make_curvamer(mesh_dir,rx_n,ry_n,rz_n,theta_z,t0,kx_0,ky_0,kxy_0,kx_n,ky_n,kxy_n,kh,kckh,kvkh,dcore=1.0)

### Load starting configuration from previous calculation

# read old dumpfile for atom positions
old = ReadSim(loaddirectory)
old.read_dump(dumpname="dump.lammps",auto_labels="yes",gz=True)
# reformat
data_atoms_old = []
for i in np.arange(sim.natoms):
    data_atoms_old.append("{} {} {} {} {} {}".format(
        int(old.id_f[i]), int(old.mol_f[i]), int(old.type_f[i]),
        old.x_f[i], old.y_f[i], old.z_f[i]))
# replace new positions with old
sim.data_atoms = data_atoms_old

### Make Data Files

# Make 1 data file in sim directory containing sim dims and atom, bond, bondcoeff, mass info
sim.make_datafile(xlo=xlo,xhi=xhi,ylo=ylo,yhi=yhi,zlo=zlo,zhi=zhi,name="data.lammps",
                      datadir="simdir",atoms=True,bonds=True,bondcoeffs=True,masses=True,gz=True)

# Make data files in different locations
# sim.make_datafile(xlo=xlo,xhi=xhi,ylo=ylo,yhi=yhi,zlo=zlo,zhi=zhi,name="atoms.lammps",
#                   atoms=True,bonds=False,bondcoeffs=False,masses=True) # atom positions + masses
# sim.make_datafile(xlo=xlo,xhi=xhi,ylo=ylo,yhi=yhi,zlo=zlo,zhi=zhi,name="N{}Data.lammps".format(nmols),
#                   datadir="BondTopologies",atoms=True,bonds=True,bondcoeffs=True,masses=True) # bond topology
# sim.make_datafile(xlo=xlo,xhi=xhi,ylo=ylo,yhi=yhi,zlo=zlo,zhi=zhi,name="bondcoeffs.lammps",
#                   atoms=False,bonds=False,bondcoeffs=True,masses=False) # bond coeffs

### Run Simulation

## Run locally
#sim.run_minimization(nproc,sigma,epsilon,shift,ljcut,wcacut,dumpfreq,etol,emax,minstyle=minstyle,lmpmpi="local",pairints=True,setforce=setforce,thermofreq=thermofreq,datafile="data.lammps.gz",gz=True)

### Run on cluster
#sim.run_minimization(nproc,sigma,epsilon,shift,ljcut,wcacut,dumpfreq,etol,emax,minstyle=minstyle,lmpmpi="cluster",pairints=True,setforce=setforce,thermofreq=thermofreq,datafile="data.lammps.gz",gz=True)
#
