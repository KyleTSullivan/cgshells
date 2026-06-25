# size_dists.py
# goes through trajectory file and calculates number of occurences of each stack size
# writes to .txt file for reading in locally


import numpy as np
import os
import sys
cwd = os.getcwd()
itop=cwd.find("cgshells/")+len("cgshells")
PROJECT_ROOT = cwd[:itop]
sys.path.insert(0, PROJECT_ROOT )

import gzip
from scipy.spatial import KDTree
from utils.readsim import ReadSim
# from utils.curvsim.v1.curvamer2d import Curvamer2D
# from utils.curvsim.v1.curvamer3d import Curvamer3D

import matplotlib as mpl
import matplotlib.pyplot as plt


def find_midlayer_pt(result,frame):
    nshells = np.max(result.dump_mol)    # number of molecules in simulation
    molatoms = result.natoms/nshells    # number of atoms per molecule
    
    ri = []
    vi = []
    for i in np.arange(nshells):
        mol = i+1
        aid1 = int(molatoms/4 + (mol - 1)*molatoms)    # id of middle-most atom in bottom layer 
        aid2 = int(3*molatoms/4 + (mol - 1)*molatoms)    # id of middle-most atom in bottom layer 
        mask = (result.dump_mol[frame]==mol)    # molecule mask
        a1mask = (result.dump_id[frame][mask]==aid1)
        a2mask = (result.dump_id[frame][mask]==aid2)
        r1 = np.array([result.dump_x[frame][mask][a1mask][0], # position of bottom atom
                       result.dump_y[frame][mask][a1mask][0]])
        r2 = np.array([result.dump_x[frame][mask][a2mask][0],    # position of top atom
               result.dump_y[frame][mask][a2mask][0]])
        zdist = np.sqrt(np.sum((r2-r1)**2))
        vmol = (r2-r1)/zdist    # molecule orientation vector
        rmol = r1 + 0.5*zdist*vmol    # position of molecule center
        ri.append(rmol.tolist())
        vi.append(vmol.tolist())
    return np.array(ri), np.array(vi)

def find_stacks(result,frame,rcut):
    # returns list of aggregates 
    # result must have read dumpfile already

    nshells = np.max(result.dump_mol)
    box_x = result.xhi - result.xlo
    box_y = result.yhi - result.ylo
    rshift = 0.5*np.array(box_x,box_y)    # shift points so that box starts at (0,0)
    rm, vm = find_midlayer_pt(result,frame)
    tree = KDTree(rm+rshift,boxsize=2*rshift)
    
    done = []
    aggs = []
    for i in np.arange(nshells,dtype=int):    # Note: rm[0] is mol = 1 in LAMMPS
        if i not in done:
            done.append(i)
            agg_i = [i]    # indeces of molecules in same aggregate as mol i (i = index, not molid)
            ni_indeces = tree.query_ball_point((rm+rshift)[i],rcut)
            for j in ni_indeces:
                if (j != i)and(np.sum(vm[i]*vm[j])>0):                          
                    agg_i.append(j)
                    done.append(j)
                    nj_indeces = tree.query_ball_point((rm+rshift)[j],rcut)
                    for k in nj_indeces:
                        if (k not in ni_indeces)and(np.sum(vm[k]*vm[j])>0):
                            ni_indeces.append(k)
            aggs.append(agg_i)
            
    return aggs

txtdir = "jobs/dynamics2D/wx-18.67-t0-1-Nbeads-20/Nmols-400/sizes"
rcut = 3
lastf = 500

phi_list =[0.4]
temp_list = [0.06]
kh_list = [5000]   

if not os.path.exists(f"{PROJECT_ROOT}/{txtdir}"):
    os.makedirs(f"{PROJECT_ROOT}/{txtdir}")

### Collect list of stack sizes for each frame
for i in range(len(phi_list)):
    for j in range(len(temp_list)):

        for k in range(len(kh_list)):
            dimension = 2
            dcore = 1.0    # hard core diameter of beads (dcore approx thickness of one DNA helix 3.5nm)
            t0 = 1.0 * dcore    # structural thickness
            wx = 18.67 * dcore    # shell width (arclength along midline)
            r0 = "flat" # 15 * dcore   # set to "flat" for particles with zero curvature
            Nbeads = 20    # number of beads per layer (2Nbeads is beads per curvamer)
            fraction = 1/3    # middle patch of beads has width = fraction * wx

            if r0 == "flat":
                k_0 = 0
            else:
                k_0 = 1/r0

            kh = kh_list[k]
            nu = 0.3
            d = wx/(Nbeads-1)   # bead spacing
            alpha = t0/d
            kvkh = 2*(1-alpha**2 * nu)/(alpha**2 - nu)
            kckh = nu*(1 + alpha**2)/(alpha**2 - nu)

            pair_ints = "1patch" #"none", "repulsive", "1patch", "patchy", "attractive", or "2attractive"
            soft_ints = True
            sigma = 1*dcore
            epsilon = 0.11782843040702126
            shift = dcore - 2**(1/6)*sigma     # shift factor to make sure lj minimum is at dcore
            ljcut = 5*sigma #t0 + 2*dcore               # cutoff distance for attractive lj potential
            wcacut = dcore    # cutoff distance for repulsive wca potential
            softsigma = 5*sigma
            softepsilon = 5e-8 * epsilon
            softshift = 0 #softcore - 2**(1/6)*softsigma
            softcut = 2**(1/6) * softsigma
            
            nshells = 400
            phi = phi_list[i]
            Tstart = temp_list[j]

            ### Read trajectory file
            simdir = f"data/dynamics2D/{int(dimension)}d/md/wx-{wx:0.3f}-t0-{t0:0.3f}-Nbeads-{Nbeads}/nshells-{nshells}/r0-{r0}/sigma-{sigma:0.5f}-kh-{kh:0.3f}/phi-{phi:0.3f}/kT-{Tstart:0.3f}"
            dumpname = f"trajectory1.dump.gz"
            readall = True
            result = ReadSim(f"{PROJECT_ROOT}/{simdir}")
            result.read_dump(dumpname=dumpname,readall=readall)
            
            ### Find occurences of stack sizes in each of the last [lastf] frames
            sizes_list = []
            maxsize = 0    # max stack size seen in last [lastf] frames

            if lastf > result.nframes:
                f1 = 0
            else:
                f1 = lastf

            for f in np.arange(result.nframes)[-f1:]:
                aggs = find_stacks(result,f,rcut)
                sizes = []
                for s in aggs:
                    sizes.append(len(s))
                sizes_list.append(sizes)
                if np.max(sizes)>maxsize:
                    maxsize = np.max(sizes)
            
            # histogram sizes for each frame checked
            contents = ""
            stacks = np.arange(maxsize)+1    # list of stack sizes seen; 1 2 3 ... maxsize
            for sj in stacks:
                contents += f"{sj} "
            for sk in np.arange(len(sizes_list)):
                contents += "\n"
                s = plt.hist(sizes_list[sk],bins=np.arange(0.5,maxsize+1.5,1))
                for si in s[0]:
                    contents += f"{int(si)} "
            
            if r0 == "flat":
                fname = f"{PROJECT_ROOT}/{txtdir}/r0-{r0}-sigma-{sigma:0.5f}-kh-{kh:0.5f}-phi-{phi:0.5f}-kT-{Tstart:0.5f}.txt"
            else:
                fname = f"{PROJECT_ROOT}/{txtdir}/r0-{r0:0.3f}-sigma-{sigma:0.5f}-kh-{kh:0.5f}-phi-{phi:0.5f}-kT-{Tstart:0.5f}.txt"

            with open(fname, "w") as fe:
                fe.write(contents)
                    
# FIRST LINE DEFINES BIN EDGES (use maxsize to determine number of bins needed)
# FOLLOWING LINE LISTS SIZE OF EACH BIN (ONE LINE = ONE FRAME)
                               
