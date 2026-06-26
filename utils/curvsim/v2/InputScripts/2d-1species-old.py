# 2d-1species.py

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

def make_input(simpath):

    ### Import variables from metadata file
    meta = rm.read_metadata(f"{PROJECT_ROOT}/{simpath}") # read metadata
    simpath = meta['logistics']['simpath']
    dimension = meta['particle']['geometry']['dimension']
    px = meta['logistics']['px']
    py = meta['logistics']['py']
    pz = meta['logistics']['pz']
    datagz = meta['simulation']['datagz']
    pair_ints = meta['particle']['interactions']['pair_ints']
    beta = meta['particle']['interactions']['beta']
    ljcut = meta['particle']['interactions']['ljcut']
    epsilon = meta['particle']['interactions']['epsilon']
    sigma = meta['particle']['interactions']['sigma']
    wcacut = meta['particle']['interactions']['wcacut']
    shift = meta['particle']['interactions']['shift']
    soft_ints = meta['particle']['interactions']['soft_ints']
    if soft_ints:
        softepsilon = meta['particle']['interactions']['softepsilon']
        softsigma = meta['particle']['interactions']['softsigma']
        softcut = meta['particle']['interactions']['softcut']
        softshift = meta['particle']['interactions']['softshift']
    simtype = meta['simulation']['simtype']
    if simtype == "md":
        Tstart = meta['simulation']['Tstart']
        Tstop = meta['simulation']['Tstop']
        damp = meta['simulation']['damp']
        seed = meta['simulation']['seed']
        runsteps = meta['simulation']['runsteps']
        timestep = meta['simulation']['timestep']
    if simtype == "emin":
        etol = meta['simulation']['etol']
        maxiter = meta['simulation']['maxiter']
        minstyle = meta['simulation']['minstyle']

    t0 = meta['particle']['geometry']['t0']
    dcore = meta['particle']['geometry']['dcore']
    fraction = meta['particle']['geometry']['fraction']
    thermofreq = meta['simulation']['thermofreq']
    dumpfreq = meta['simulation']['dumpfreq']
    trajgz = meta['simulation']['trajgz']   
    dumpbonds = meta['simulation']['dumpbonds']
    tlim_hrs = meta['logistics']['tlim_hrs']
    tlim_min = meta['logistics']['tlim_min']
    tbuffer = meta['logistics']['tbuffer']
    gridfreq = meta['simulation']['gridfreq']
    thresh = meta['simulation']['thresh']

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

fix prop all property/atom i_mol i_moltype d_curv
"""

    if datagz==True:
        inputcontents +="""
variable restart_exists equal is_file(restart.final)
if "${restart_exists}" then "read_restart restart.final" else "read_data data.lammps.gz fix prop NULL Molecules"

"""
    else:
        inputcontents +="""
variable restart_exists equal is_file(restart.final)
if "${restart_exists}" then "read_restart restart.final" else "read_data data.lammps fix prop NULL Molecules"

"""

    ### Interactions
    if pair_ints == "none":
        inputcontents += "pair_style none\n"
    elif pair_ints == "repulsive":
        inputcontents += f"""
pair_style hybrid lj/expand {ljcut}
pair_coeff * * lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes
\n
"""

    else:
        if soft_ints == True:
            inputcontents += f"""
pair_style hybrid lj/expand {ljcut}
# upside down bonding 1
pair_coeff 1 5 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 3 3 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 1 3 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 1 1 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 3 5 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 5 5 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}

# upside down bonding 2
pair_coeff 2 6 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 4 6 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 6 6 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 2 4 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 4 4 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}
pair_coeff 2 2 lj/expand {softepsilon} {softsigma} {softshift} {softcut-softshift}

"""
        else:
            inputcontents += f"""
pair_style hybrid lj/expand {ljcut}
# upside down bonding 1
pair_coeff 1 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 1 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 5 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}

# upside down bonding 2
pair_coeff 2 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 6 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 2 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}

"""


        if pair_ints == "beta":
            if fraction == 0:
#                 print(f"ERROR: code does not currently allow for fraction = {fraction}.")
#                 print("    to update, must re-assign atom types to be contiguous (start from 1 and can't skip atom types 3 and 4).")  
                inputcontents += f"""
# correct bonding
pair_coeff 1 2 lj/expand {epsilon} {sigma} {shift} {ljcut-shift} 
pair_coeff 3 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift} 
pair_coeff 5 6 lj/expand {epsilon} {sigma} {shift} {ljcut-shift}
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes

"""
    
            elif fraction == 1:
#                 print(f"ERROR: code does not currently allow for fraction = {fraction}.")
#                 print("    to update, must re-assign atom types to be contiguous (start from 1 can't skip atom types 1, 5, 2, 6).")
                inputcontents += f"""
# correct bonding
pair_coeff 1 2 lj/expand {epsilon} {sigma} {shift} {wcacut-shift} 
pair_coeff 3 4 lj/expand {epsilon} {sigma} {shift} {ljcut-shift} 
pair_coeff 5 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes

"""

            else:
                if beta == 0:
                    epsilon_flanks = epsilon / (1-fraction)
                    sigma_flanks = sigma / np.sqrt(1-fraction)
                    shift_flanks = dcore - 2**(1/6)*sigma_flanks  
                    ljcut_flanks = 5*sigma_flanks              
                    
                    inputcontents += f"""
# correct bonding
pair_coeff 1 2 lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks} 
pair_coeff 3 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift} 
pair_coeff 5 6 lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks}
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes

"""
                elif beta == 1:
                    epsilon_mid = epsilon / (fraction)
                    sigma_mid = sigma / np.sqrt(fraction)
                    shift_mid = dcore - 2**(1/6)*sigma_mid  
                    ljcut_mid = 5*sigma_mid 
                    
                    inputcontents += f"""
# correct bonding
pair_coeff 1 2 lj/expand {epsilon} {sigma} {shift} {wcacut-shift} 
pair_coeff 3 4 lj/expand {epsilon_mid} {sigma_mid} {shift_mid} {ljcut_mid-shift_mid} 
pair_coeff 5 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes

"""
                else:
                    epsilon_mid = epsilon * beta / (fraction)
                    sigma_mid = sigma * np.sqrt(beta / fraction)
                    shift_mid = dcore - 2**(1/6)*sigma_mid  
                    ljcut_mid = 5*sigma_mid 
                    
                    epsilon_flanks = epsilon * (1-beta) / (1-fraction)
                    sigma_flanks = sigma * np.sqrt( (1-beta) / (1-fraction) )
                    shift_flanks = dcore - 2**(1/6)*sigma_flanks  
                    ljcut_flanks = 5*sigma_flanks
                    
                    inputcontents += f"""
# correct bonding
pair_coeff 1 2 lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks} 
pair_coeff 3 4 lj/expand {epsilon_mid} {sigma_mid} {shift_mid} {ljcut_mid-shift_mid} 
pair_coeff 5 6 lj/expand {epsilon_flanks} {sigma_flanks} {shift_flanks} {ljcut_flanks-shift_flanks}
pair_coeff 1 4 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 1 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 3 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 2 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 3 6 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_coeff 4 5 lj/expand {epsilon} {sigma} {shift} {wcacut-shift}
pair_modify shift yes

"""
                
    # Fixes
    if dimension == 2:
        inputcontents += "fix 0 all enforce2d\n"
    if simtype == "md":
        inputcontents += f"""
fix 1 all langevin {Tstart} {Tstop} {damp} {seed}
fix 2 all nve

"""
#     inputcontents += f"""
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
thermo_modify norm no flush yes

variable energies_exists equal is_file(energies.txt)
if "${{energies_exists}} == 0" then 'fix energyOut all print {thermofreq} "$(step) $(etotal) $(pe) $(ke) $(epair) $(ebond) $(temp)" append energies.txt screen no title "step etotal pe ke epair ebond temp"' else 'fix energyOut all print {thermofreq} "$(step) $(etotal) $(pe) $(ke) $(epair) $(ebond) $(temp)" append energies.txt screen no'

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

    if trajgz==True:
        inputcontents += f"""
dump 1 all custom/gz {dumpfreq} trajectory${{stage}}.dump.gz mol id type x y z i_moltype d_curv
dump_modify 1 append no sort id
"""
    else:
        inputcontents += f"""
dump 1 all custom {dumpfreq} trajectory${{stage}}.dump mol id type x y z i_moltype d_curv
dump_modify 1 append no sort id
"""

    if simtype == "md":
        inputcontents += f"""
timer timeout {int(tlim_hrs*3600 + tlim_min*60 - tbuffer*60)}
timestep {timestep}
print "STARTING MD"
run {runsteps} upto
print "MD FINISHED"
write_restart restart.final
"""
    elif simtype == "emin":
        inputcontents += f"""
timer timeout {int(tlim_hrs*3600 + tlim_min*60 - tbuffer*60)}
print "STARTING MINIMIZATION"
min_style {minstyle}
print "MINIMIZATION FINISHED"
minimize {etol} 0.0 {maxiter} {10*maxiter}
write_restart restart.final
"""           

    # Write LAMMPS input file
    with open(f"{PROJECT_ROOT}/{simpath}/in.lammps", "w") as f:
        f.write(inputcontents)