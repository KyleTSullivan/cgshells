import numpy as np
import os
import subprocess
import time
from scipy.optimize import curve_fit
import gzip
from utils.readsim import ReadSim

class Curvamer3D:

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

        self.nmols = 0
        self.natoms = 0
        self.nbonds = 0
        self.natomtypes = 0
        self.nbondtypes = 0
        self.nangletypes = 0
        self.data_atoms = []
        self.data_bonds = []
        self.data_bondcoeffs = []
        self.data_masses = []

    def make_mesh(self, mesh_dir, a, wx, wy):
        """
        Makes curvamer (triangular) lattice mesh.
        """

#         mesh_directory = "{}/MeshDesigns/{}".format(self.directory,mesh_name)
        t1 = time.time()
        
        if not os.path.exists(mesh_dir):
            os.makedirs(mesh_dir)

        ##### Mesh Nodes #####
        # finding nearest indeces to target dimensions
        ## m = index along x axis
        ## n = index along diagonal
        ## l = index along z axis
        nmax = 2*int( np.round( wy / (np.sqrt(3) * a) ) ) + 1
        dm = int( np.round(0.5*nmax) )
        mmax = int(np.round(wx/a)) +  dm + 1
        lmax = 2   # only ever two layers (top/bottom)
        wy = np.sqrt(3)/2 * (nmax-1) * a   # actual length
        wx = int(np.round(wx/a))*a   # actual width

        # rectangular cutout variables
        xmin = dm*a
        xmax = xmin + wx
        ymin = 0
        ymax = wy
        dx = a
        dy = a
        t0 = 1    # thickness: always 1 for mesh.  Can adjust later by multiplying by desired t0 when making curvamer.
        dz = t0

        ### Create rectangular lattice (to be transformed)
        idlist = []
        xrect = []
        yrect = []
        zrect = []
        i = 0

        for m in np.arange(mmax):
            for n in np.arange(nmax):
                for l in np.arange(lmax):
                    idlist.append(i)
                    xrect.append(m*dx)
                    yrect.append(n*dy)
                    zrect.append(l*dz)
                    i = i + 1

        idlist = np.array(idlist).reshape(mmax,nmax,lmax)
        xrect = np.array(xrect)
        yrect = np.array(yrect)
        zrect = np.array(zrect) - 0.5


        # find (m,n) values of edges (to be used later with for special edge bonds)

        edge_mn_left = []
        mn_left = np.array([dm,0])
        edge_mn_left.append(mn_left)

        edge_mn_right = []
        mn_right = np.array([mmax-1,0])
        edge_mn_right.append(mn_right)

        while mn_left[1]<=(nmax-2):
            mn_left = mn_left + np.array([-1,2])
            edge_mn_left.append(mn_left)

        while mn_right[1]<=(nmax-2):
            mn_right = mn_right + np.array([-1,2])
            edge_mn_right.append(mn_right)

        edge_mn_left = np.array(edge_mn_left)
        edge_mn_right = np.array(edge_mn_right)

        ### Transform to triangular lattice
        xtri = xrect + 0.5 * yrect
        ytri = np.sqrt(3)/2 * yrect
        ztri = zrect

        ### Trim to shape
        mask = (xtri>=0.99*xmin)&(xtri<= 1.001*xmax)
        xpos = xtri[mask]
        ypos = ytri[mask]
        zpos = ztri[mask]

        ##### Mesh Bond Network #####

        ### Creating neighbor lists

        # neighbor[i] returns list of id's that are bonded neighbors to i

        neighbors = []
        bond_idlist = []
        bond_typelist = []
        # bond types: (see 3DUnitCells.nb mathematica notebook for visuals)
        # here, horizontal refers to bonds in the same z layer
        # 1 = vertical
        # 2 = top horizontal type 1 (along diagonals - aka change in n index)
        # 3 = top horizontal type 2 (along x axis - aka no change in n index, only change m index)
        # 4 = bot horizontal type 1 (along diagonals)
        # 5 = bot horizontal type 2 (along x axis)
        # 6 = cross type 1 (along diagonals)
        # 7 = cross type 2 (along diagonals)
        # 8 = cross type 3 (along x axis)
        # 9 = special edge case horizontal
        # 10 = special edge case cross

        #  make empty neighbor lists for each atom
        for i in range(np.size(idlist)):
            neighbors.append([])
            bond_idlist.append([])
            bond_typelist.append([])

        b = 0    # bond id counter
        for m in range(mmax):
            for n in range(nmax):
                for l in range(lmax):
                    current_id = idlist[m,n,l]
                    neigh = []
                    bid = []
                    btype = []
                    lp = (l+1)%2

                    neigh.append(idlist[m,n,lp])    # vertical bond
                    btype.append(1)
                    bid.append(b)
                    b += 1

                    if m < mmax-1:
                        neigh.append(idlist[m+1,n,l])    # horizontal bond along x-axis
                        if l==0:
                            btype.append(5)
                        else:
                            btype.append(3)
                        bid.append(b)
                        b += 1

                        neigh.append(idlist[m+1,n,lp])    # cross bond along x-axis
                        btype.append(8)
                        bid.append(b)
                        b += 1

                        if n != 0:
                            neigh.append(idlist[m+1,n-1,l])   # horizontal bond along diagonal
                            if l==0:
                                btype.append(4)
                            else:
                                btype.append(2)
                            bid.append(b)
                            b += 1

                            neigh.append(idlist[m+1,n-1,lp])   # cross bond along diagonal
                            if l==0:
                                btype.append(6)
                            else:
                                btype.append(7)
                            bid.append(b)
                            b += 1

                    if m != 0:
                        neigh.append(idlist[m-1,n,l])    # horizontal bond along x-axis
                        if l==0:
                            btype.append(5)
                        else:
                            btype.append(3)
                        bid.append(b)
                        b += 1

                        neigh.append(idlist[m-1,n,lp])    # cross bond along x-axis
                        btype.append(8)
                        bid.append(b)
                        b += 1

                        if n != nmax-1:
                            neigh.append(idlist[m-1,n+1,l])    # horizontal bond along diagonal
                            if l==0:
                                btype.append(4)
                            else:
                                btype.append(2)
                            bid.append(b)
                            b += 1

                            neigh.append(idlist[m-1,n+1,lp])    # cross bond along diagonal
                            if l==0:
                                btype.append(6)
                            else:
                                btype.append(7)
                            bid.append(b)
                            b += 1

                    if n != nmax-1:
                        neigh.append(idlist[m,n+1,l])    # horizontal bond along diagonal
                        if l==0:
                            btype.append(4)
                        else:
                            btype.append(2)
                        bid.append(b)
                        b += 1

                        neigh.append(idlist[m,n+1,lp])    # cross bond along diagonal
                        if l==0:
                            btype.append(6)
                        else:
                            btype.append(7)
                        bid.append(b)
                        b += 1

                    if n != 0:
                        neigh.append(idlist[m,n-1,l])    # horizontal bond along diagonal
                        if l==0:
                            btype.append(4)
                        else:
                            btype.append(2)
                        bid.append(b)
                        b += 1

                        neigh.append(idlist[m,n-1,lp])    # cross bond along diagonal
                        if l==0:
                            btype.append(6)
                        else:
                            btype.append(7)
                        bid.append(b)
                        b += 1

                    # special edge cases
                    if ([m,n] in edge_mn_left.tolist()) or ([m,n] in edge_mn_right.tolist()):
                        if n != 0:
                            neigh.append(idlist[m+1,n-2,l])    # edge horizontal bond
                            btype.append(9)
                            bid.append(b)
                            b += 1

                            neigh.append(idlist[m+1,n-2,lp])    # edge cross bond
                            btype.append(10)
                            bid.append(b)
                            b += 1

                        if n != nmax-1:
                            neigh.append(idlist[m-1,n+2,l])    # edge horizontal bond
                            btype.append(9)
                            bid.append(b)
                            b += 1

                            neigh.append(idlist[m-1,n+2,lp])    # edge cross bond
                            btype.append(10)
                            bid.append(b)
                            b += 1


                    neighbors[current_id] = neigh    # replace empty list with neighbor list for atom current_id
                    bond_idlist[current_id] = bid
                    bond_typelist[current_id] = btype

        elim = idlist.flatten()[~mask]   # ids of sites outside desired area (to be deleted)
        keep = idlist.flatten()[mask]    # ids of sites inside target shape (to be kept)
        natoms = np.size(keep)

        ### get rid of neighbor lists for eliminated ids
        new_neighbors = []
        new_bond_idlist = []
        new_bond_typelist = []
        for i in keep:
            new_neighbors.append(neighbors[i])
            new_bond_idlist.append(bond_idlist[i])
            new_bond_typelist.append(bond_typelist[i])

        ### get rid of eliminated ids from neighbor lists
        for i in range(natoms):
            badidmask = np.ones(len(new_neighbors[i]))
            for j in range(len(new_neighbors[i])):
                atom2 = new_neighbors[i][j]
                if atom2 in elim:
        #             new_neighbors[i].remove(j)
                    badidmask[j] = 0

            badidmask = np.array(badidmask, dtype=bool)
            new_neighbors[i] = np.array(new_neighbors[i])[badidmask].tolist()
            new_bond_idlist[i] = np.array(new_bond_idlist[i])[badidmask].tolist()
            new_bond_typelist[i] = np.array(new_bond_typelist[i])[badidmask].tolist()

        ### update atomids so they start from 0 and have no gaps

        atomids = []
        for i in range(natoms):
            oldid = keep[i]
            newid = i
            atomids.append(newid)
            for k in range(natoms):
                for j in range(len(new_neighbors[k])):
                    atom2 = new_neighbors[k][j]
                    if atom2 == oldid:
                        new_neighbors[k][j] = newid
        atomids = np.array(atomids)
        ndup = []    # finished neighbor lists (with duplicates e.g. 1 is bonded to 2 and 2 is bonded to 1)
        for i in range(len(new_neighbors)):
            nn = []
            for j in new_neighbors[i]:
                nn.append(j)
            ndup.append(nn)

        ### eliminate duplicate bonds (now 1 is bonded to 2 but 2 doesn't show bond to 1)
        bonded = []
        for i in range(natoms):
            # atom1 = keep[i]
            atom1 = i
            badidmask = np.ones(len(new_neighbors[i]))
            for j in range(len(new_neighbors[i])):
                atom2 = new_neighbors[i][j]
                if {atom1,atom2} in bonded:
                    badidmask[j] = 0
                else:
                    bonded.append({atom1,atom2})

            badidmask = np.array(badidmask, dtype=bool)
            new_neighbors[i] = np.array(new_neighbors[i])[badidmask].tolist()
            new_bond_idlist[i] = np.array(new_bond_idlist[i])[badidmask].tolist()
            new_bond_typelist[i] = np.array(new_bond_typelist[i])[badidmask].tolist()

        ### update bondids so they start from 0 and have no gaps

        bcount = 0
        for i in range(len(new_bond_idlist)):
            for j in range(len(new_bond_idlist[i])):
                oldid = new_bond_idlist[i][j]
                newid = bcount
                for u in range(len(new_bond_idlist)):
                    for v in range(len(new_bond_idlist[u])):
                        atom2 = new_bond_idlist[u][v]
                        if atom2 == oldid:
                            new_bond_idlist[u][v] = newid
                bcount += 1

        nbonds = bcount

        ### convert nested lists to flattened lists
        # nlist is list of neighbor ids
        # indexlist[i] gives index of nlist where particle i's neighbor list ends
        nlist = []
        bidlist = []
        btypelist = []
        indexlist = []
        index1 = 0
        for i in range(natoms):
            atom1 = atomids[i]
            index2 = 0
            for j in range(len(new_neighbors[i])):
                atom2 = new_neighbors[i][j]
                nlist.append(atom2)
                bidlist.append(new_bond_idlist[i][j])
                btypelist.append(new_bond_typelist[i][j])
                index2 += 1

            index1 = index1 + index2
            indexlist.append(index1)

        nlist = np.array(nlist)
        bidlist = np.array(bidlist)
        btypelist = np.array(btypelist)
        
        ### convert nested ndup list to flattened list
        npairs = []
        ipairs = []
        index1 = 0
        for i in range(natoms):
            atom1 = atomids[i]
            index2 = 0
            for j in range(len(ndup[i])):
                atom2 = ndup[i][j]
                npairs.append(atom2)
                index2 += 1
            index1 = index1 + index2
            ipairs.append(index1)
        npairs = np.array(npairs)
        ipairs = np.array(ipairs)

        xpos = xpos - np.min(xpos) - wx/2
        ypos = ypos - np.min(ypos) - wy/2
        zpos = zpos - np.min(zpos) - t0/2

        natoms = np.size(atomids)    # per molecule
        nbonds = np.size(nlist)   # per molecule

        # ### Random displacements
        # if delta_r != 0:
        #     ## Choose random direction and random distance to displace each lattice site
        #     randdists = np.random.normal(loc=0,scale = delta_r,size=natoms)
        #     randangles = np.pi * np.random.random(natoms)
        #     randx = randdists*np.cos(randangles)
        #     randy = randdists*np.sin(randangles)
        #     xposr = xpos + randx
        #     yposr = ypos + randy
        #     zposr = zpos
        # else:
        #     xposr = xpos
        #     yposr = ypos
        #     zposr = zpos

        ### Atom Type Patches: Divide particle surface into nine sectors
        topmask = zpos>0
        botmask = zpos<0

        s1 = (xpos <= (-wx/2 + wx/3)) & (ypos >= (wy/2 - wy/3 ))
        s2 = (xpos <= (-wx/2 + 2*wx/3)) & (xpos > (-wx/2 + wx/3)) & (ypos >= (wy/2 - wy/3) )
        s3 = (xpos > (-wx/2 + 2*wx/3)) & (ypos >= (wy/2 - wy/3) )
        s4 = (xpos <= (-wx/2 + wx/3)) & (ypos < (wy/2 - wy/3) ) & (ypos >= (wy/2 - 2*wy/3))
        s5 = (xpos > (-wx/2 + wx/3)) & (xpos <= (-wx/2 + 2*wx/3)) & (ypos < (wy/2 - wy/3) ) & (ypos >= (wy/2 - 2*wy/3))
        s6 = (xpos > (-wx/2 + 2*wx/3)) & (ypos < (wy/2 - wy/3) ) & (ypos >= (wy/2 - 2*wy/3))
        s7 = (xpos <= (-wx/2 + wx/3)) & (ypos < (wy/2 - 2*wy/3 ))
        s8  = (xpos <= (-wx/2 + 2*wx/3)) & (xpos > (-wx/2 + wx/3)) & (ypos < (wy/2 - 2*wy/3) )
        s9 = (xpos > (-wx/2 + 2*wx/3)) & (ypos < (wy/2 - 2*wy/3) )

        atomtypes = np.ones(len(atomids))
        
        # corners: type 3 = top (C); type 6 = bottom (C')
        atomtypes[topmask*s1] = 3
        atomtypes[topmask*s3] = 3
        atomtypes[topmask*s7] = 3
        atomtypes[topmask*s9] = 3
        atomtypes[botmask*s1] = 6
        atomtypes[botmask*s3] = 6
        atomtypes[botmask*s7] = 6
        atomtypes[botmask*s9] = 6
        
        # "checkered cross": type 2 = top (B); type 5 = bottom (B')
        atomtypes[topmask*s2] = 2
        atomtypes[topmask*s4] = 2
        atomtypes[topmask*s6] = 2
        atomtypes[topmask*s8] = 2
        atomtypes[botmask*s2] = 5
        atomtypes[botmask*s4] = 5
        atomtypes[botmask*s6] = 5
        atomtypes[botmask*s8] = 5
        
        # center: type 1 = top (A); type 4 = bottom (A')
        atomtypes[topmask*s5] = 1
        atomtypes[botmask*s5] = 4

        atomtypes = np.array(atomtypes,dtype=int)

        ### save mesh data lists to files
        with open("{}/xpos".format(mesh_dir),'w') as fdata:
                for i in range(np.size(xpos)):
                    fdata.write("{}\n".format(xpos[i]))

        with open("{}/ypos".format(mesh_dir),'w') as fdata:
                for i in range(np.size(ypos)):
                    fdata.write("{}\n".format(ypos[i]))

        with open("{}/zpos".format(mesh_dir),'w') as fdata:
                for i in range(np.size(zpos)):
                    fdata.write("{}\n".format(zpos[i]))

        with open("{}/nlist".format(mesh_dir),'w') as fdata:
                for i in range(np.size(nlist)):
                    fdata.write("{}\n".format(nlist[i]))
                   
        with open("{}/indexlist".format(mesh_dir),'w') as fdata:
                for i in range(np.size(indexlist)):
                    fdata.write("{}\n".format(indexlist[i]))
                    
        with open("{}/npairs".format(mesh_dir),'w') as fdata:
                for i in range(np.size(npairs)):
                    fdata.write("{}\n".format(npairs[i]))
                   
        with open("{}/ipairs".format(mesh_dir),'w') as fdata:
                for i in range(np.size(ipairs)):
                    fdata.write("{}\n".format(ipairs[i]))

        with open("{}/bidlist".format(mesh_dir),'w') as fdata:
                for i in range(np.size(bidlist)):
                    fdata.write("{}\n".format(bidlist[i]))

        with open("{}/btypelist".format(mesh_dir),'w') as fdata:
                for i in range(np.size(btypelist)):
                    fdata.write("{}\n".format(btypelist[i]))

        with open("{}/atomids".format(mesh_dir),'w') as fdata:
                for i in range(np.size(atomids)):
                    fdata.write("{}\n".format(atomids[i]))

        with open("{}/atomtypes".format(mesh_dir),'w') as fdata:
                for i in range(np.size(atomtypes)):
                    fdata.write("{}\n".format(atomtypes[i]))

        ### review of lattice parameters
        t2 = time.time()
        telapsed = t2 - t1
        hrs = telapsed//3600
        mins = (telapsed%3600)//60
        secs = (telapsed%3600)%60
        print("Mesh Calculation Runtime: {}hrs {}min {:.1f}sec".format(hrs,mins,secs))
        print("Review of lattice parameters: (absolute units)\n")
        print("lattice constant = {}".format(a))
        # print("std dev of random displacement = {}".format(delta_r))
        #         print("t0 = {}".format(t0))
        print("curvamer width (x) = {}".format(wx))
        print("curvamer length (y) = {}".format(wy))

        print("# of sites per curvamer: {}".format(natoms))
        print("# of bonds per curvamer: {}".format(nbonds))


    def make_curvamer(self,mesh_dir,rx,ry,rz,theta_z,t0,kx_0,ky_0,kxy_0,kx_i,ky_i,kxy_i,kh,kckh,kvkh,dcore=1.0):

        ### load mesh data
        xpos = np.loadtxt("{}/xpos".format(mesh_dir))
        ypos = np.loadtxt("{}/ypos".format(mesh_dir))
        zpos = t0 * np.loadtxt("{}/zpos".format(mesh_dir))
        nlist = np.loadtxt("{}/nlist".format(mesh_dir),dtype=int)
        indexlist = np.loadtxt("{}/indexlist".format(mesh_dir),dtype=int)
        atomids = np.loadtxt("{}/atomids".format(mesh_dir),dtype=int)
        atomtypes = np.loadtxt("{}/atomtypes".format(mesh_dir),dtype=int)
        bidlist = np.loadtxt("{}/bidlist".format(mesh_dir),dtype=int)
        btypelist = np.loadtxt("{}/btypelist".format(mesh_dir),dtype=int)

        wx = np.max(xpos)-np.min(xpos)
        wy = np.max(ypos)-np.min(ypos)
        natoms = np.size(atomids)
        nbonds = np.size(nlist)
        natomtypes = np.size(np.unique(atomtypes))
        nbondtypes = np.size(bidlist)
        kv = kvkh * kh
        kc = kckh * kh

        ### apply curvature to triangular lattice sites for a single curvamer
        xpreferred = []
        ypreferred = []
        zpreferred = []
        xinitial = []
        yinitial = []
        zinitial = []
        ellipsoid_angle = []
        for i in range(np.size(xpos)):
            x = xpos[i]
            y = ypos[i]
            if zpos[i]>0:
                z = t0/2
            elif zpos[i]<0:
                z = -t0/2
                
            prefx = x + z * (kx_0 * x + kxy_0 * y) / np.sqrt(1 + (kx_0*x + kxy_0*y)**2 + (ky_0*y + kxy_0*x)**2)
            prefy = y + z * (ky_0 * y + kxy_0 * x) / np.sqrt(1 + (kx_0*x + kxy_0*y)**2 + (ky_0*y + kxy_0*x)**2)
            prefz = -0.5*kx_0 * x**2 -0.5*ky_0 * y**2 - kxy_0*x*y + z/np.sqrt(1 + (kx_0*x + kxy_0*y)**2 + (ky_0*y + kxy_0*x)**2)
                          
            xi = x + z * (kx_i * x + kxy_i * y) / np.sqrt(1 + (kx_i*x + kxy_i*y)**2 + (ky_i*y + kxy_i*x)**2)
            yi = y + z * (ky_i * y + kxy_i * x) / np.sqrt(1 + (kx_i*x + kxy_i*y)**2 + (ky_i*y + kxy_i*x)**2)
            zi = -0.5*kx_i * x**2 -0.5*ky_i * y**2 - kxy_i*x*y + z/np.sqrt(1 + (kx_i*x + kxy_i*y)**2 + (ky_i*y + kxy_i*x)**2)
            
            xpreferred.append(prefx)
            ypreferred.append(prefy)
            zpreferred.append(prefz)
            xinitial.append(xi)
            yinitial.append(yi)
            zinitial.append(zi)

        xpreferred = np.array(xpreferred)
        ypreferred = np.array(ypreferred)
        zpreferred = np.array(zpreferred)

        xinitial = np.array(xinitial)
        yinitial = np.array(yinitial)
        zinitial = np.array(zinitial)


        # rotate initial positions by angle theta_z around z axis

        xinitial_rot = np.cos(theta_z) * xinitial - np.sin(theta_z) * yinitial
        yinitial_rot = np.sin(theta_z) * xinitial + np.cos(theta_z) * yinitial
        zinitial_rot = zinitial

        # displace initial positions by (rx,ry,rz)

        xinitial = xinitial_rot + rx
        yinitial = yinitial_rot + ry
        zinitial = zinitial_rot + rz

        self.nmols += 1
        self.natoms += natoms
        self.nbonds += nbonds
        self.natomtypes = natomtypes
        self.nbondtypes = nbondtypes

        ### add to atoms section of data file list
        for i in range(natoms):
            self.data_atoms.append('{} {} {} {} {} {}'.format(atomids[i]+1 + (self.nmols - 1)*natoms , self.nmols, atomtypes[i], xinitial[i], yinitial[i], zinitial[i]))

        ### add to bonds section of data file list
        for atom1 in atomids:
            if atom1 == 0:
                istart = 0
                istop = indexlist[atom1]
            else:
                istart = indexlist[atom1-1]
                istop = indexlist[atom1]

            for i in np.arange(istart,istop):
                atom2 = nlist[i]
                self.data_bonds.append('{} {} {} {}'.format(bidlist[i]+1 + (self.nmols - 1)*nbonds, bidlist[i]+1, atom1+1 + (self.nmols - 1)*natoms, atom2+1 + (self.nmols - 1)*natoms))

        ### add to bond coeffs section of data file list
        if self.nmols == 1:    # current version only works with one mesh design (no dispersity in shape or design yet)
            for atom1 in atomids:
                if atom1 == 0:
                    istart = 0
                    istop = indexlist[atom1]
                else:
                    istart = indexlist[atom1-1]
                    istop = indexlist[atom1]

                for i in np.arange(istart,istop):

                    if btypelist[i] == 1:    # vertical bond
                        bstrength = kv
#                         blength = t0

                    if btypelist[i] == 2:    # horizontal top bond type 1 - diagonal
                        bstrength = kh

                    if btypelist[i] == 3:    # horizontal top bond type 2 - x axis
                        bstrength = kh

                    if btypelist[i] == 4:    # horizontal bottom bond type 1 - diagonal
                        bstrength = kh

                    if btypelist[i] == 5:    # horizontal bottom bond type 2 - x axis
                        bstrength = kh
                        
                    if btypelist[i] == 6:    # cross bond type 1 - diagonal
                        bstrength = kc
                        
                    if btypelist[i] == 7:    # cross bond type 2 - diagonal
                        bstrength = kc
                        
                    if btypelist[i] == 8:    # cross bond type 3 - x axis
                        bstrength = kc
                        
                    if btypelist[i] == 9:    # special edges case horizontal
                        bstrength = kh
                        
                    if btypelist[i] == 10:    # special edges case horizontal
                        bstrength = kc
                        
                    atom2 = nlist[i]
                    dx = xpreferred[atom1]-xpreferred[atom2]
                    dy = ypreferred[atom1]-ypreferred[atom2]
                    dz = zpreferred[atom1]-zpreferred[atom2]
                    blength = np.sqrt(dx*dx + dy*dy + dz*dz)

                    self.data_bondcoeffs.append('{} {} {}'.format(bidlist[i]+1, bstrength, blength))
#                     self.data_bondcoeffs.append('{} {} {:.5f}'.format(btypelist[i], bstrength, blength))


        ### masses section
        mass = 1.0
        masslist = []
        for i in range(self.natomtypes):
            masslist.append('{} {}'.format(i+1, mass))
        self.data_masses = masslist

    def make_datafile(self,xlo=-1,xhi=1,ylo=-1,yhi=1,zlo=-1,zhi=1,name="data.lammps",
                      datadir="simdir",atoms=True,bonds=True,bondcoeffs=True,masses=True,gz=True):

        """
        datadir is location where data file will be saved.
        
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
        
        s = np.array(["Atoms","Bonds","Bond Coeffs","Masses"],dtype=str)
        mask = np.array([atoms, bonds, bondcoeffs, masses],dtype=bool)
        contents = ", ".join(s[mask])
        
        if gz==True:
            with gzip.open("{}/{}.gz".format(loc,name),"wt") as fdata:
                fdata.write('# LAMMPS data file\n')
                fdata.write('# Contains: {}\n\n'.format(contents))

                ### Write header

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

                ### Write sections

                section_length = np.size(self.data_atoms)
                if (atoms==True) & (section_length > 0):
                    fdata.write('\nAtoms\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_atoms[i]))

                section_length = np.size(self.data_bonds)
                if (bonds==True) & (section_length > 0):
                    fdata.write('\nBonds\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_bonds[i]))

                section_length = np.size(self.data_bondcoeffs)
                if (bondcoeffs==True) & (section_length > 0):
                    fdata.write('\nBond Coeffs\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_bondcoeffs[i]))

                section_length = np.size(self.data_masses)
                if (masses==True) & (section_length > 0):
                    fdata.write('\nMasses\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_masses[i]))
        else:
            with open("{}/{}".format(loc,name),'w') as fdata:
                fdata.write('# LAMMPS data file\n')
                fdata.write('# Contains: {}\n\n'.format(contents))

                ### Write header

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

                ### Write sections

                section_length = np.size(self.data_atoms)
                if (atoms==True) & (section_length > 0):
                    fdata.write('\nAtoms\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_atoms[i]))

                section_length = np.size(self.data_bonds)
                if (bonds==True) & (section_length > 0):
                    fdata.write('\nBonds\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_bonds[i]))

                section_length = np.size(self.data_bondcoeffs)
                if (bondcoeffs==True) & (section_length > 0):
                    fdata.write('\nBond Coeffs\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_bondcoeffs[i]))

                section_length = np.size(self.data_masses)
                if (masses==True) & (section_length > 0):
                    fdata.write('\nMasses\n\n')
                    for i in range(section_length):
                        fdata.write('{}\n'.format(self.data_masses[i]))
                    
    def run_minimization(self,nproc,sigma,epsilon,shift,ljcut,wcacut,dumpfreq,etol,emax,
                         minstyle="cg",lmpmpi="local",pairints=True,setforce=False,thermofreq=500,
                         datafile="data.lammps",gz=True):
        
        """
        setforce:
            setforce not equal to False tells lammps to fix certain forces components on all atoms.
            The forces used will be setforce[0], setforce[1], setforce[2].
            E.g setforce = ["NULL","NULL",0.0] allows atoms to move in x and y but z positions are fixed.

        
        pairints:
            pairints=True uses LJ and WCA potentials for pairwise interactions.
            pairints=False turns off pairwise interactions.  
            If False, sigma, epsilon, shift, ljcut and wcacut must still be specified for command to work,
            however, the values are arbitrary and are ignored.
            
        minstyle:
            Minimization method.  Either "cg", "hftn", "sd", "quickmin", or "fire".
            Default is "cg".
            
        thermofreq:
            How frequently (in iteration steps) to output thermo values to screen and log file.
            Default is 500 steps.
            
        datafile:
            The location and name of the data file to read in.  Path is relative to sim directory.
            Default is in the sim directory as "data.lammps".
            Note that LAMMPS can read gzipped data files.  Simply set datafile="data.lammps.gz".
            
        gz:
            Whether to zip the dump file to save memory.  
            Default is True.
            Requires lammps to have been built with the compress package, e.g make yes-compress.
            
        """
        
        px = nproc[0]    # number of processors in x
        py = nproc[1]
        pz = nproc[2]
        t1 = time.time()
        print("***\n***")
        print("N = {}".format(self.nmols))
        print("***\n***")
        with open("{}/in.lammps".format(self.directory),'w') as fdata:
            fdata.write('# 3D Curvamer Input File - Energy Minimization\n\n')
            fdata.write("units lj\n")
            fdata.write("dimension 3\n")
            fdata.write("boundary s s s\n")
            fdata.write("atom_style molecular\n")
            fdata.write("bond_style harmonic\n")
            fdata.write("angle_style none\n")
            fdata.write("dihedral_style none\n")
            fdata.write("improper_style none\n\n")

            fdata.write("processors {} {} {} grid onelevel\n\n".format(px,py,pz))
            
            fdata.write("read_data {}\n\n".format(datafile))
            
            if pairints==True:
                fdata.write("pair_style hybrid lj/expand {}\n".format(ljcut))
                
#                 # Interactions for old 1 patch model
#                 fdata.write("pair_coeff 1 1 none\n")
#                 fdata.write("pair_coeff 2 2 none\n")
#                 fdata.write("pair_coeff 3 3 none\n")
#                 fdata.write("pair_coeff 4 4 none\n")
#                 fdata.write("pair_coeff 1 3 none\n")
#                 fdata.write("pair_coeff 2 4 none\n")
#                 fdata.write("pair_coeff 1 4 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
#                 fdata.write("pair_coeff 2 3 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
#                 fdata.write("pair_coeff 1 2 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
#                 fdata.write("pair_coeff 3 4 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, ljcut-shift))
#                 fdata.write("pair_modify shift yes\n\n")
                
                # Attractive interactions
                fdata.write("pair_coeff 1 4 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, ljcut-shift))
                fdata.write("pair_coeff 2 5 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, ljcut-shift))
                fdata.write("pair_coeff 3 6 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, ljcut-shift))
                # Repulsive interactions
                fdata.write("pair_coeff 1 5 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
                fdata.write("pair_coeff 1 6 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
                fdata.write("pair_coeff 2 4 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
                fdata.write("pair_coeff 2 6 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
                fdata.write("pair_coeff 3 4 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
                fdata.write("pair_coeff 3 5 lj/expand {} {} {} {}\n".format(epsilon, sigma, shift, wcacut-shift))
                # No interactions
                fdata.write("pair_coeff 1 1 none\n")
                fdata.write("pair_coeff 2 2 none\n")
                fdata.write("pair_coeff 3 3 none\n")
                fdata.write("pair_coeff 4 4 none\n")
                fdata.write("pair_coeff 5 5 none\n")
                fdata.write("pair_coeff 6 6 none\n")
                fdata.write("pair_coeff 1 2 none\n")
                fdata.write("pair_coeff 1 3 none\n")
                fdata.write("pair_coeff 2 3 none\n")
                fdata.write("pair_coeff 4 5 none\n")
                fdata.write("pair_coeff 4 6 none\n")
                fdata.write("pair_coeff 5 6 none\n")
                # Shift interaction energy at cutoff
                fdata.write("pair_modify shift yes\n\n")
                fdata.write("neigh_modify exclude molecule/intra all every 1 delay 0 check yes one 20000 page 200000\n")

            else:
                fdata.write("pair_style none\n")
                
            
            if setforce!=False:
                fdata.write("fix 1 all setforce {} {} {}\n".format(setforce[0],setforce[1],setforce[2]))    
#             fdata.write("fix 1 all setforce NULL NULL 0.0\n")    

        #     fdata.write("special_bonds lj 0.0 0.0 0.0\n\n")

        #     fdata.write("compute 1 all property/local btype batom1 batom2\n")
        #     fdata.write("compute 2 all bond/local engpot\n\n")

            fdata.write("thermo_style custom step etotal pe ke ebond\n")
            fdata.write("thermo {}\n".format(thermofreq))
            fdata.write("thermo_modify norm no\n\n")
            
            if gz==True:
                dump1 = "dump 1 all custom/gz {} dump.lammps.gz mol id type x y z\n".format(dumpfreq)
            else:
                dump1 = "dump 1 all custom {} dump.lammps mol id type x y z\n".format(dumpfreq)


        #dump2  = "dump 2 all local 100 {}/DumpFiles/{}_bondinfo.dump index c_1[*] c_2".format(simpath,ncurvamers)

            fdata.write(dump1)
            fdata.write("\ndump_modify 1 append no sort id\n\n")

            fdata.write("min_style {}\n".format(minstyle))
            fdata.write("minimize {} 0.0 {} {}".format(etol,emax,1000*emax))

        os.chdir(self.directory)    # change dir to where sim files are located so lammps can be called

        # Run simulation on local machine
        if lmpmpi=="local":
            run = subprocess.run(["mpirun","-np","{}".format(px*py*pz),self.lmplocal,"-in","in.lammps","-l","log.lammps"])
        elif lmpmpi=="cluster":
            run = subprocess.run(["mpirun","-np","{}".format(px*py*pz),self.lmpcluster,"-in","in.lammps","-l","log.lammps"])
        else:
            print("Error: could not find lmp_mpi executable")
            
        
        os.chdir(self.cwd)    # change dir back to where code was originally executed

        t2 = time.time()
        telapsed = t2 - t1
        hrs = telapsed//3600
        mins = (telapsed%3600)//60
        secs = (telapsed%3600)%60
        print("***\n***")
        print("N = {} Simulation Runtime: {}hrs {}min {:.1f}sec".format(self.nmols,hrs,mins,secs))
        print("***\n***")
        

    def run_flat_adhesion(self,ysep,dcore,a,wx,wy,mesh_dir,t0,sigma,epsilon,px,py,pz,minstyle="hftn",fix="all",lmpmpi="local"):
        """
        Calculate adhesive energy between two flat plates separated by ysep between midsurfaces
        setforce 0.0 0.0 0.0
        """

        ### Curvamer Variables
        nmols = 2
        kh = 100 # doesn't matter
        kx_0 = 0
        ky_0 = 0 
        kxy_0 = 0
        kx_i = 0
        ky_i = 0
        kxy_i = 0
        kckh = 1 # doesn't matter
        kvkh = 1 # doesn't matter
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
        nproc = [px,py,pz]
        thermofreq = 1
        etol = 1e-14
        emax = 100000
        dumpfreq = emax
        shift = dcore - 2**(1/6)*sigma
        ljcut = shift + 5*sigma #t0 + 2*dcore
        wcacut = dcore
        if fix=="all":
            setforce=[0.0,0.0,0.0] # False or [0.0,0.0,0.0] or ["NULL","NULL",0.0]
#         elif fix=="xy":   # need to specify etol emax etc if I want to do this.  Need extra arguments.
#             setforce=["NULL","NULL",0.0]

        ### Make Curvamers
        sim = Curvamer3D(directory = self.directory)

        for n in range(nmols):
            rx_n = rx
            ry_n = ry
#             rz_n = rz + n*(t0+dcore)
            rz_n = rz + n*ysep

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
        sim.run_minimization(nproc,sigma,epsilon,shift,ljcut,wcacut,dumpfreq,etol,emax,
                                 minstyle=minstyle,lmpmpi=lmpmpi,pairints=True,setforce=setforce,thermofreq=thermofreq,
                                datafile="data.lammps.gz",gz=True)
        
        readsim = ReadSim(sim.directory)
        readsim.read_log_emin()
        return readsim.energyf
    

    def find_gammaA(self,dcore,a,wx,wy,mesh_dir,t0,sigma,epsilon,px,py,pz,minstyle="hftn",fix="all",lmpmpi="local"):
        params = [dcore,a,wx,wy,mesh_dir,t0,sigma,epsilon,px,py,pz]
        ### Golden ratio search for energy minimum
        accuracy = 1e-3         # required accuracy to stop search
        z = (1+np.sqrt(5))/2    # golden ratio

        # initial positions of search (x = ysep)
        x1 = 0.8*(t0+dcore)
        x4 = 1.1*(t0+dcore)
        x2 = x4 - (x4-x1)/z
        x3 = x1 + (x4-x1)/z

        # initial values of the function at the initial points
        sim = Curvamer3D(directory = self.directory)
        f1 = sim.run_flat_adhesion(x1,*params,minstyle=minstyle,fix=fix,lmpmpi=lmpmpi)
        f2 = sim.run_flat_adhesion(x2,*params,minstyle=minstyle,fix=fix,lmpmpi=lmpmpi)
        f3 = sim.run_flat_adhesion(x3,*params,minstyle=minstyle,fix=fix,lmpmpi=lmpmpi)
        f4 = sim.run_flat_adhesion(x4,*params,minstyle=minstyle,fix=fix,lmpmpi=lmpmpi)

        # main loop of search
        loops = 0
        while (np.abs(f4-f1) > accuracy) & (loops < 100):
            if f2<f3:
                x4,f4 = x3,f3
                x3,f3 = x2,f2
                x2 = x4 - (x4-x1)/z
                f2 = sim.run_flat_adhesion(x2,*params,minstyle=minstyle,fix=fix,lmpmpi=lmpmpi)
            else:
                x1,f1 = x2,f2
                x2,f2 = x3,f3
                x3 = x1 + (x4-x1)/z
                f3 = sim.run_flat_adhesion(x3,*params,minstyle=minstyle,fix=fix,lmpmpi=lmpmpi)
            loops += 1

        # result
        xmin = 0.5*(x1+x4)
        fmin = sim.run_flat_adhesion(xmin,*params,minstyle=minstyle,fix=fix,lmpmpi=lmpmpi)
        if (loops == 100):
            print("exceeded max iterations (100) in search for minimum")
        return xmin, fmin, loops
    
    
    def fit_gppA(self,t,dcore,a,wx,wy,mesh_dir,t0,sigma,epsilon,px,py,pz,minstyle="hftn",fix="all",lmpmpi="local"):
        sim = Curvamer3D(directory = self.directory)
        yseps = np.linspace(t-0.02*sigma,t+0.02*sigma,20)
        energies = []
        i = 0
        for ysep in yseps:
            i += 1
            print("Sim {}".format(i))
            energies.append(sim.run_flat_adhesion(ysep,dcore,a,wx,wy,mesh_dir,t0,sigma,epsilon,px,py,pz,minstyle=minstyle,fix=fix,lmpmpi=lmpmpi))
        energies = np.array(energies)

        def harmonic(x,e0,epp):
            return -e0 + 0.5*epp*(x-1)**2

        params, cov = curve_fit(harmonic, yseps/t,energies)
        gammaA = params[0]
        gppA = params[1] / t**2
        
        return (gammaA, gppA, yseps, energies)
    
    def find_BA(self,kh,dcore,a,wx,wy,mesh_dir,t0,kx_0,ky_0,kxy_0,nu,kckh,kvkh,etol,emax,minstyle,px,py,pz,dumpfreq,thermofreq=1,lmpmpi="local"):
        """Calculates flattening energy of a curvamer with horizontal spring constant kh"""

        ### Curvamer Variables
        nmols = 1
        kx_i = 0 
        ky_i = 0 
        kxy_i = 0
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
        # px = 2
        # py = 2
        # pz = 1
        nproc = [px,py,pz]
        sigma = 1    # doesn't matter
        epsilon = 1
        shift = 1    
        ljcut = 1
        wcacut = 1
        setforce=["NULL","NULL",0.0]

        ### Make Curvamers
        sim = Curvamer3D(directory = self.directory)

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
        sim.run_minimization(nproc,sigma,epsilon,shift,ljcut,wcacut,dumpfreq,etol,emax,
                                 minstyle=minstyle,lmpmpi=lmpmpi,pairints=False,setforce=setforce,thermofreq=thermofreq,
                                datafile="data.lammps.gz",gz=True)
        
        readsim = ReadSim(self.directory)
        readsim.read_log_emin()
        Eflat_i = readsim.energyi
        Eflat_f = readsim.energyf
        BA_i = 2*Eflat_i/(kx_0**2)
        BA_f = 2*Eflat_f/(kx_0**2)
        return Eflat_i, BA_i, Eflat_f, BA_f
