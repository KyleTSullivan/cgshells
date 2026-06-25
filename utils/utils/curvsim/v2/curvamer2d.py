import numpy as np
import subprocess
import os
import gzip
from scipy.optimize import curve_fit
from utils.readsim import ReadSim
from numba import njit



# python class for setting up and simulating 2D curvamers
class Curvamer2D:
    
    def __init__(self,directory="."):
    
        # path to lammps executable (mpi version)
        self.lmplocal = "/Users/kyle/Documents/Code/lammps-23Jun2022/src/lmp_mpi" 
        self.lmpunity = "/home/kyltsullivan_umass_edu/lammps-23Jun2022/src/lmp_mpi"
        
        self.cwd = os.getcwd()    # current working directory where curvsim code is being executed
        
        if directory != ".":    # directory =  where simulation files will be stored and run (relative to self.cwd)
            self.directory = directory
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)
        else:
            self.directory = "."
        
        self.nmoltypes = 0
        self.nmols = 0
        self.natoms = 0
        self.nbonds = 0
        self.moltypes = []
        self.atomtypes = []
        self.natomtypes = 0
        self.nbondtypes = 0
        self.data_atoms = []
        self.data_bonds = []
        self.data_bondcoeffs = []
        self.data_masses = []
        self.data_molecules = []
        
    def make_curvamer(self,moltype,rx,ry,theta,w,N,fraction,t0,k_0,k_i,kh,kckh,kvkh):
        
        
        natoms = 2*N
        
        ### Atoms Section
        atomids = np.arange(natoms).reshape(2,N)
        # reshape to be consistant with x, y arrays
        # xpos1[j,k]:
        # j = 0 -> bottom layer, j = 1 -> top layer
        # k = bead number in layer j
        
        # flat curvamer
        xflat = []
        yflat = []
        xbot = np.linspace(-w/2,w/2,N)
        ybot = (-t0/2)*np.ones(N)
        xtop = xbot
        ytop = -ybot
        xflat.append(xbot.tolist())
        xflat.append(xtop.tolist())
        yflat.append(ybot.tolist())
        yflat.append(ytop.tolist())
        xflat = np.array(xflat)
        yflat = np.array(yflat)
        atomtypes = np.ones(2*N,dtype=int).reshape(2,N)
        topmask = (yflat>0)
        bottommask = (yflat<0)
        leftmask = (xflat<(-w/2 + w*(1-fraction)/2))
        rightmask = (xflat>(w/2 - w*(1-fraction)/2))
        middlemask = (~leftmask) & (~rightmask)
        atomtypes[bottommask*leftmask] = int(1 + 7*(moltype-1))
        atomtypes[bottommask*rightmask] = int(5 + 7*(moltype-1))
        atomtypes[bottommask*middlemask] = int(3 + 7*(moltype-1))
        atomtypes[topmask*leftmask] = int(2 + 7*(moltype-1))
        atomtypes[topmask*rightmask] = int(6 + 7*(moltype-1))
        atomtypes[topmask*middlemask] = int(4 + 7*(moltype-1))
        atomtypes[0][0] = int(7 + 7*(moltype-1))
        atomtypes[1][0] = int(7 + 7*(moltype-1))
        atomtypes[0][N - 1] = int(7 + 7*(moltype-1))
        atomtypes[1][N - 1] = int(7 + 7*(moltype-1))
        print(atomtypes)
        
        # define atom positions - preferred shape
        xpreferred = []
        ypreferred = []
        if k_0 != 0:
            r_0 = 1/k_0
            angle = w*k_0    # angular spread of curvamer
            thetaleft = np.pi/2 - angle/2    # left angle
            thetaright = np.pi/2 + angle/2   # right angle
            atomangles = np.linspace(thetaleft,thetaright,N)
            rb = r_0 - t0/2    # radius of curvature of bottom layer
            rt = r_0 + t0/2    # radius of curvature of top layer
            xbot1 = rb*np.cos(atomangles)          # x pos of bottom beads
            ybot1 = rb*np.sin(atomangles) - r_0    # y pos of bottom beads
            xtop1 = rt*np.cos(atomangles)          # x pos of top beads
            ytop1 = rt*np.sin(atomangles) - r_0    # y pos of top beads
            xpreferred.append(xbot1.tolist())
            xpreferred.append(xtop1.tolist())
            ypreferred.append(ybot1.tolist())
            ypreferred.append(ytop1.tolist())
            
            # bond rest lengths
#             dtheta = (w*k_0)/(N-1)
#             thetaC = np.pi/2 - dtheta/2
#             lv = t0                              # vertical
#             lh_bottom = 2*rb*np.sin(dtheta/2)    # horizontal bottom
#             lc = np.sqrt(lv**2 + lh_bottom**2 -2*lv*lh_bottom*np.cos(thetaC))    # cross
#             lh_top = 2*rt*np.sin(dtheta/2)       # horizontal top

            dtheta = (w*k_0)/(N-1)
            lh_mid = w/(N-1)  # fictitious horizontal middle bond
            x = t0*np.tan(dtheta/2)    # extra bit on top of trapezoid
            lh_bottom = lh_mid - x     # horizontal bottom bond
            lh_top = lh_mid + x        # horizontal top bond
            lv = np.sqrt(t0**2 + x**2)                  # vertical bond
            lc = np.sqrt(t0**2 + (lh_bottom + x)**2)    # cross bond

            
        else:
            xpreferred = xflat
            ypreferred = yflat
            
            # bond rest lengths
            lv = t0
            lh_bottom = w/(N-1)
            lc = np.sqrt(lv**2 + lh_bottom**2)
            lh_top = lh_bottom
        
        xpreferred = np.array(xpreferred)
        ypreferred = np.array(ypreferred)
        
        # define atom positions - initial shape
        xinitial = []
        yinitial = []
        if k_i != 0:
            r_i = 1/k_i
            angle = w*k_i    # angular spread of curvamer
            thetaleft = np.pi/2 - angle/2    # left angle
            thetaright = np.pi/2 + angle/2   # right angle
            atomangles = np.linspace(thetaleft,thetaright,N)
            rb = r_i - t0/2    # radius of curvature of bottom layer
            rt = r_i + t0/2    # radius of curvature of top layer
            xbot1 = rb*np.cos(atomangles)          # x pos of bottom beads
            ybot1 = rb*np.sin(atomangles) - r_i    # y pos of bottom beads
            xtop1 = rt*np.cos(atomangles)          # x pos of top beads
            ytop1 = rt*np.sin(atomangles) - r_i    # y pos of top beads
            xinitial.append(xbot1.tolist())
            xinitial.append(xtop1.tolist())
            yinitial.append(ybot1.tolist())
            yinitial.append(ytop1.tolist())
        else:
            xinitial = xflat
            yinitial = yflat
            
        xinitial = np.array(xinitial)
        yinitial = np.array(yinitial)
            
        xrot = np.cos(theta) * xinitial - np.sin(theta) * yinitial
        yrot = np.sin(theta) * xinitial + np.cos(theta) * yinitial
        
        xinitial = xrot + rx
        yinitial = yrot + ry
        
        # add atoms section to data file list
        zinitial = 0    # zero for 2d system
        for i in range(natoms):
            self.data_atoms.append('{} {} {} {} {} {}'.format(atomids.flatten()[i]+1 + self.natoms , self.nmols+1, atomtypes.flatten()[i], xinitial.flatten()[i], yinitial.flatten()[i], zinitial))
        
        ### Bonds section

        springids = []    # list of spring ids
        springtypes = []  # list of spring types
        atom1 = []        # atom1[i] gives 1st atom coord for spring i
        atom2 = []        # atom2[i] gives 2nd atom coord for spring i
#         springconsts = [] # springconsts[i] gives spring const for spring i
#         restlengths = []  # restlengths[i] gives rest lenth of spring i
        
#         kc = kckh*kh    # cross spring constant
#         kv = kvkh*kh    # vertical spring constant
        
        springid_counter = 0
        
        for k in range(N-1):
            ## horizontal springs
            # bottom
            springids.append(springid_counter)
            springtypes.append(2)
#             springconsts.append(kh)
#             restlengths.append(lh_bottom)
            atom1.append(atomids[0,k])
            atom2.append(atomids[0,k+1])
            springid_counter += 1
            # top
            springids.append(springid_counter)
            springtypes.append(4)
#             springconsts.append(kh)
#             restlengths.append(lh_top)
            atom1.append(atomids[1,k])
            atom2.append(atomids[1,k+1])
            springid_counter += 1
            ## cross springs
            # lower left to upper right
            springids.append(springid_counter)
            springtypes.append(3)
#             springconsts.append(kc)
#             restlengths.append(lc)
            atom1.append(atomids[0,k])
            atom2.append(atomids[1,k+1])
            springid_counter += 1
            # lower right to upper left
            springids.append(springid_counter)
            springtypes.append(3)
#             springconsts.append(kc)
#             restlengths.append(lc)
            atom1.append(atomids[0,k+1])
            atom2.append(atomids[1,k])
            springid_counter += 1
        
        ## vertical springs
        for k in range(N):
            springids.append(springid_counter)
            springtypes.append(1)
#             springconsts.append(kv)
#             restlengths.append(lv)
            atom1.append(atomids[0,k])
            atom2.append(atomids[1,k])
            springid_counter += 1

        springids = np.array(springids)
        springtypes = np.array(springtypes)
        atom1 = np.array(atom1)
        atom2 = np.array(atom2)
#         springconsts = np.array(springconsts)
#         restlengths = np.array(restlengths)
        nbonds = springid_counter
        nbondtypes = 4
        
        for s in range(nbonds):
            self.data_bonds.append("{} {} {} {}".format(\
                springids[s]+1 + self.nbonds,\
                springtypes[s] + self.nbondtypes,\
                atom1[s]+1 + self.natoms,\
                atom2[s]+1 + self.natoms))
    
            
        ### Bond Coeffs Section 
        kc = kckh*kh    # cross spring constant
        kv = kvkh*kh    # vertical spring constant
        
        self.data_bondcoeffs.append("{} {} {}".format(1 + self.nbondtypes, 0.5*kv, lv))    # vertical bond - type 1
        self.data_bondcoeffs.append("{} {} {}".format(2 + self.nbondtypes, 0.5*kh, lh_bottom))    # horizontal bottom bond - type 2
        self.data_bondcoeffs.append("{} {} {}".format(3 + self.nbondtypes, 0.5*kc, lc))    # cross bond - type 3
        self.data_bondcoeffs.append("{} {} {}".format(4 + self.nbondtypes, 0.5*kh, lh_top))    # horizontal top bond - type 4
        
        ### Molecules section (to keep track of molecule type and preferred curvatures in trajectory dump file)
        for i in range(natoms):
            self.data_molecules.append('{} {} {} {:0.5f}'.format(atomids.flatten()[i]+1 + self.natoms, self.nmols+1, moltype, k_0))
        
        ### Update attributes
        if moltype in self.moltypes:
            pass
        else:
            self.moltypes.append(moltype)
        self.nmoltypes = np.size(self.moltypes)
        self.nmols += 1
        self.natoms += natoms
        for atype in np.unique(atomtypes):
            if atype in self.atomtypes:
                pass
            else:
                self.atomtypes.append(atype)
        self.natomtypes = self.nmoltypes * 7 #int(np.size(np.unique(self.atomtypes)))
        self.nbonds += nbonds
        self.nbondtypes += nbondtypes
        
        ### Masses section
        mass = 1
        masslist = []
        for i in range(self.natomtypes):
            masslist.append('{} {}'.format(i+1, mass))
        self.data_masses = masslist
        
        
        
    def make_datafile(self,xlo,xhi,ylo,yhi,zlo=-0.5,zhi=0.5,datadir="simdir",gz=True):
        
        """
        datadir is location where data file will be saved.
        If datadir = "simdir", then data file will be saved in the simulation directory.
        If not, datadir gives the path where to save.
        
        gz:
            Whether to write the data file as a compressed gzip file.  Default is True.
            Requires lammps to have been built with the compress package, e.g make yes-compress.
        
        """
        
        if datadir == "simdir":
            loc = self.directory
        else:
            loc = datadir
        
        if not os.path.exists(loc):
            os.makedirs(loc)
      
            
        if gz==True:
            with gzip.open("{}/data.lammps.gz".format(loc),"wt") as fdata:
                fdata.write('# LAMMPS data file - 2D Curvamers\n\n')
                if self.natoms > 0:
                    fdata.write('{} atoms\n'.format(self.natoms))
                if self.nbonds > 0:
                    fdata.write('{} bonds\n'.format(self.nbonds))
                if self.natomtypes > 0:
                    fdata.write('{} atom types\n'.format(self.natomtypes))
                if self.nbondtypes > 0:
                    fdata.write('{} bond types\n'.format(self.nbondtypes))
                fdata.write('{} {} xlo xhi\n'.format(xlo,xhi))
                fdata.write('{} {} ylo yhi\n'.format(ylo,yhi))
                fdata.write('{} {} zlo zhi\n'.format(zlo,zhi))

                section_length = np.size(self.data_atoms)
                if section_length > 0:
                    fdata.write('\nAtoms\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_atoms[i]))

                section_length = np.size(self.data_bonds)
                if section_length > 0:
                    fdata.write('\nBonds\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_bonds[i]))

                section_length = np.size(self.data_bondcoeffs)
                if section_length > 0:
                    fdata.write('\nBond Coeffs\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_bondcoeffs[i]))

                section_length = np.size(self.data_masses)
                if section_length > 0:
                    fdata.write('\nMasses\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_masses[i]))
                        
                section_length = np.size(self.data_molecules)
                if section_length > 0:
                    fdata.write('\nMolecules\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_molecules[i]))
                        
        else:
            
            with open("{}/data.lammps".format(loc),'w') as fdata:
                fdata.write('# LAMMPS data file - 2D Curvamers\n\n')
                if self.natoms > 0:
                    fdata.write('{} atoms\n'.format(self.natoms))
                if self.nbonds > 0:
                    fdata.write('{} bonds\n'.format(self.nbonds))
                if self.natomtypes > 0:
                    fdata.write('{} atom types\n'.format(self.natomtypes))
                if self.nbondtypes > 0:
                    fdata.write('{} bond types\n'.format(self.nbondtypes))
                fdata.write('{} {} xlo xhi\n'.format(xlo,xhi))
                fdata.write('{} {} ylo yhi\n'.format(ylo,yhi))
                fdata.write('{} {} zlo zhi\n'.format(zlo,zhi))

                section_length = np.size(self.data_atoms)
                if section_length > 0:
                    fdata.write('\nAtoms\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_atoms[i]))

                section_length = np.size(self.data_bonds)
                if section_length > 0:
                    fdata.write('\nBonds\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_bonds[i]))

                section_length = np.size(self.data_bondcoeffs)
                if section_length > 0:
                    fdata.write('\nBond Coeffs\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_bondcoeffs[i]))

                section_length = np.size(self.data_masses)
                if section_length > 0:
                    fdata.write('\nMasses\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_masses[i]))
                        
                section_length = np.size(self.data_molecules)
                if section_length > 0:
                    fdata.write('\nMolecules\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_molecules[i]))
                   
    
    def run_minimization(self,simpath,etol,maxiter,dumpfreq,epsilon,sigma,shift,ljcut,wcacut):
        """ Run energy minimization of curvamer setup """
        with open("{}/in.lammps".format(simpath),'w') as fdata:
            fdata.write('# 2D Curvamer Input File\n')
            fdata.write("\nunits lj")
            fdata.write("\ndimension 2")
            fdata.write("\nboundary s s p")
            fdata.write("\natom_style molecular")
            fdata.write("\nbond_style harmonic")
            fdata.write("\nangle_style none")
            fdata.write("\ndihedral_style none")
            fdata.write("\nimproper_style none\n")

            fdata.write("\nread_data {}/data.lammps\n".format(simpath))

        #     fdata.write("\npair_style none")
            fdata.write("\npair_style hybrid lj/expand {}".format(ljcut))
            fdata.write("\npair_coeff 1 1 none")
            fdata.write("\npair_coeff 2 2 none")
            fdata.write("\npair_coeff 3 3 none")
            fdata.write("\npair_coeff 4 4 none")
            fdata.write("\npair_coeff 1 3 none")
            fdata.write("\npair_coeff 2 4 none")
            fdata.write("\npair_coeff 1 4 lj/expand {} {} {} {}".format(epsilon, sigma, shift, wcacut-shift))
            fdata.write("\npair_coeff 2 3 lj/expand {} {} {} {}".format(epsilon, sigma, shift, wcacut-shift))
            fdata.write("\npair_coeff 1 2 lj/expand {} {} {} {}".format(epsilon, sigma, shift, wcacut-shift))
            fdata.write("\npair_coeff 3 4 lj/expand {} {} {} {}".format(epsilon, sigma, shift, ljcut-shift))
            fdata.write("\npair_modify shift yes\n")

            fdata.write("\nfix 1 all enforce2d\n")
        #     fdata.write("\nfix 2 all setforce NULL 0.0 NULL\n")    # fix in place
            fdata.write("\nneigh_modify exclude molecule/intra all every 1 delay 0 check yes")
        #     fdata.write("\nspecial_bonds lj 0.0 0.0 0.0\n")

            fdata.write("\ncompute 1 all property/local btype batom1 batom2")
            fdata.write("\ncompute 2 all bond/local engpot\n")

            fdata.write("\nthermo_style custom step pe ke ebond etotal")
            fdata.write("\nthermo 500")
            fdata.write("\nthermo_modify norm no\n")

            fdata.write("\ndump 1 all custom {} {}/trajectory.dump mol id type x y z".format(dumpfreq,simpath))
            fdata.write("\ndump 2 all local {} {}/bondinfo.dump index c_1[*] c_2\n".format(dumpfreq,simpath))
            fdata.write("\ndump_modify 1 append no sort id\n")

            fdata.write("\nminimize {} 0.0 {} {}".format(etol,maxiter,10*maxiter))

        runlammps = subprocess.run(["lmp","-in","{}/in.lammps".format(simpath),"-l","{}/log.lammps".format(simpath),"-screen","none"])
    
    def curvfocus(k_i,t):
        r_i = 1/k_i
        rfocus = r_i + t
        kfocus = 1/rfocus
        return kfocus
    
    def gapdelta(r,t,w):
        radical = np.sqrt(1 - ( (2*r-t)*np.sin(w/(2*r)) )**2 / (2*r + t)**2 )
        return -t - r*( np.cos(w/(2*r)) - 1 ) + t/2 * np.cos(w/(2*r)) + t/2*radical + r*(radical - 1)
    

# misc funtions
@njit
def ulj(epsilon,sigma,shift,r,rcut):
    """ shifted at cutoff lj potential like lj/expand in lammps"""
    if r > rcut:
        energy = 0
    else:
        sr6 = (sigma/(r-shift))**6
        sr6cut = (sigma/(rcut-shift))**6
        ucut = 4*epsilon*(sr6cut*sr6cut - sr6cut)
        u = 4*epsilon*(sr6*sr6 - sr6)
        energy = u - ucut
    return energy

@njit
def ubead(type1,type2,r12,pair_flag,soft_ints,sigma,epsilon,shift,ljcut,wcacut,softsigma,softepsilon,softshift,softcut):

    """
    Calculate bead-bead interaction according to bead types and pair style.
    
    Note: Using integer flags to indicate pair style.
        "1patch" = 1
        "patchy" = 3
    """
    
    types = (type1,type2)

    # soft repulsion
    if (type1==1 and type2==1) \
    or (type1==2 and type2==2) \
    or (type1==3 and type2==3) \
    or (type1==4 and type2==4) \
    or (type1==1 and type2==3) \
    or (type1==3 and type2==1) \
    or (type1==2 and type2==4) \
    or (type1==4 and type2==2):
        if soft_ints == True:
            u0 = ulj(softepsilon,softsigma,softshift,r12,softcut)
        else:
            u0 = ulj(epsilon,sigma,shift,r12,wcacut)

    elif pair_flag == 1: # aka "1patch"            
        # attractive
        if (type1==3 and type2==4) \
        or (type1==4 and type2==3):
            u0 = ulj(epsilon,sigma,shift,r12,ljcut) 
        # repulsive
        elif (type1==1 and type2==2) \
        or (type1==2 and type2==1) \
        or (type1==1 and type2==4) \
        or (type1==4 and type2==1) \
        or (type1==2 and type2==3) \
        or (type1==3 and type2==2):
            u0 = ulj(epsilon,sigma,shift,r12,wcacut)

    elif pair_flag == 3: # aka "patchy"            
        # attractive
        if (type1==3 and type2==4) \
        or (type1==4 and type2==3) \
        or (type1==1 and type2==2) \
        or (type1==2 and type2==1):
            u0 = ulj(epsilon,sigma,shift,r12,ljcut) 
        # repulsive
        elif (type1==1 and type2==4) \
        or (type1==4 and type2==1) \
        or (type1==2 and type2==3) \
        or (type1==3 and type2==2):
            u0 = ulj(epsilon,sigma,shift,r12,wcacut)

    else: 
        u0 = 0

    return u0                

@njit
def ushell(list1len,list2len,type1list,type2list,x1list,x2list,y1list,y2list,rshift,
          pair_flag,soft_ints,sigma,epsilon,shift,ljcut,wcacut,softsigma,softepsilon,softshift,softcut):
    # perform double sum over atom interactions to calculate adhesive energy
    eadh = 0
    for a1 in range(list1len): # mol1 beads
        for a2 in range(list2len): # mol2 beads
            type1 = type1list[a1]
            type2 = type2list[a2]
            r1 = np.array([x1list[a1],y1list[a1]])
            r2 = np.array([x2list[a2],y2list[a2]])
            r12 = r2 - r1 - 2*rshift * np.floor( (r2-r1)/(2*rshift) + 0.5)

            eadh += ubead(type1,type2,np.sqrt(np.sum(r12**2)),pair_flag,soft_ints,sigma,epsilon,shift,ljcut,wcacut,softsigma,softepsilon,softshift,softcut)
            
    return eadh
                
   
        
        