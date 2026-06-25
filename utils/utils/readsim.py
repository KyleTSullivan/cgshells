# python class for reading results of lammps simulations

import numpy as np
import gzip
import subprocess
import os

class ReadSim:
    
    def __init__(self,simdir):
        self.simdir = simdir
        self.files = os.listdir(simdir)
        
    def latest_file(self, file="trajectory", return_matches=False):
        # checks simdir for sequentially numbered files starting with [file] and returns latest one
            # e.g. if simdir contains trajectory1.dump, trajectory2.dump, trajectory3.dump
                # and file = "trajectory", then latest_file() will return "trajectory3.dump"
                
            # format of filename must be [file][integer].[extension]
                # works by parsing filenames from end of [file] to the "." character just before the extension
                
            # default is "trajectory" but can also be used with sequential log files
            
            # if there is only one match, returns it
                # this will work for files that are not numbered
                
            # return_matches: bool, if True returns all files that start with [file] in addition to last instance
            
            # returns:
                # filename, files (if return_matches=True)
                
        # collect files in simdir starting with [file]
        charlen = len(file)
        files = []  # list of all files that match starting characters
        for f in self.files:
            if f[:charlen] == file:
                files.append(f)
                
        # if only one match
        if len(files)==1:
            filename = files[0]

        # if multiple matches, find latest one
        elif len(files)>1:
            tlast = 0
            restart = 0
            for i in np.arange(len(files)):
                f = files[i]
                p = f.index(".")
                n = int(f[charlen:p])
                if n > restart:
                    tlast = i
                    restart = n
            filename = files[tlast]
            
        if return_matches:
            return (filename, files)
        else:
            return filename
    
    # read log file (works for both energy minimization and MD)
    def read_log(self,logname="log.lammps",latest=True):
                
        # look for matches and return last if latest=True
        if latest:
            # if logname has an extension, e.g. logname="log.lammps"
            try:
                p = logname.index(".")
                fname = logname[:p]
            # if logname has no extension, e.g. logname="log"
            except:
                fname = logname
            filename = self.latest_file(file=fname,return_matches=False)
        else:
            filename = logname
        
        opened = open("{}/{}".format(self.simdir,filename))
        in_text = opened.read()
        opened.close()
        
        # thermo output
        start_index = in_text.index("Step")
        end_index = in_text.index("\n",start_index)
        self.thermo_labels = in_text[start_index : end_index].split() 
        nlabels = len(self.thermo_labels)
        loop_index = in_text.index("Loop")

        thermostring = in_text[end_index:loop_index]
        thermostring = thermostring.splitlines()
        mask = np.ones(len(thermostring))
        for i in np.arange(len(thermostring)):
            if thermostring[i][:7]=="WARNING":
                mask[i] = 0
            if thermostring[i]=='':
                mask[i] = 0
        mask = np.array(mask,dtype = bool)
        thermo = []
        for i in np.arange(len(np.array(thermostring)[mask])):
            thermo.append(np.fromstring(np.array(thermostring)[mask][i],sep=" ").tolist())
        self.thermo = np.array(thermo)
        
        try:
            for i in range(nlabels):
                setattr(self,"thermo_"+self.thermo_labels[i],self.thermo[:,i])
        except:
            print("Could not assign attributes according to column labels.")
            # can't define attributes when names like c[1] are column labels (underscores OK; brackets NOT)
            # best to define computes as scalars to avoid brackets in labels
              
#         try:
#             warning_index = in_text.index("WARNING:",end_index)
#             warning_stop = in_text.index("\n",warning_index)+len("\n")
#             thermo_stop = warning_index
#             extra = in_text[warning_stop:loop_index]
#             thermostring = in_text[end_index:thermo_stop]+extra
#         except:
#             thermo_stop = loop_index
#             thermostring = in_text[end_index:thermo_stop]

#         lines = 0 
#         for i in thermostring:#in_text[start_index:thermo_stop]:
#             if i=="\n":
#                 lines += 1

#         self.thermo = np.fromstring(thermostring,sep="\n").reshape(lines-1,nlabels)
#         for i in range(nlabels):
#             setattr(self,"thermo_"+self.thermo_labels[i],self.thermo[:,i])
            
        # wall time
        time_index = in_text.index("Total wall time:") + len("Total wall time:")
        end_index = in_text.index("\n", time_index)
        self.walltime_hms = in_text[time_index:end_index]    # simulation wall time (string formatted as hrs:min:sec)
        self.walltime = int(np.sum(np.array(self.walltime_hms.split(":"),dtype=float)*np.array([3600,60,1])))   

            
        # energy minimization only info
        try:
            # get initial, 2nd to last, final energies
            start_index = in_text.index("Energy initial, next-to-last, final = \n") + len("Energy initial, next-to-last, final = \n")
            end_index = in_text.index("\n", start_index)
            energies = (in_text[start_index : end_index]).strip().split()
            self.energyi = float(energies[0])    # initial energy
            self.energyf = float(energies[2])    # final energy
            self.energy2 = float(energies[1])    # second to last energy

            # get stopping criterion
            start_index = in_text.index("Stopping criterion = ") + len("Stopping criterion = ")
            end_index = in_text.index("\n", start_index)
            reason = in_text[start_index : end_index]
            self.why_stop = reason
            self.sim_type = "Energy Minimization"  
        except:
            self.sim_type = "Molecular Dynamics"
#             pass

        # read dump file
    def read_dump(self,dumpname="dump.lammps",readall=False):
        
        """

        readall: Whether to read and collect data from all the dumps.  
        Default is False in which case only the first and last completed dumps are kept.
      
        
        """
        
        gz = (dumpname[-3:] == ".gz")    # test to see if dump file has been compressed
        
        if gz==True:
            try:
                opened = gzip.open("{}/{}".format(self.simdir,dumpname),"rt")
                in_text = opened.read()
                opened.close()
            except:
                convert = subprocess.run(['gzip','-dc','{}/{}'.format(self.simdir,dumpname)],stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                in_text = convert.stdout
        else:
            opened = open("{}/{}".format(self.simdir,dumpname))
            in_text = opened.read()
            opened.close()


        # count total number of lines in file and find their indeces    
        linecount = 0
        newline_indeces = []
        j = 0
        for i in in_text:
            if i == "\n":
                linecount += 1
                newline_indeces.append(j)
            j += 1

        # count number of lines to skip to get to first frame data
        end_index = in_text.index("ITEM: ATOMS ")
        toplines = 1
        for i in in_text[0 : end_index]:
            if i == "\n":
                toplines += 1
              
        # get sim box info
        start_index = in_text.index("ITEM: BOX BOUNDS") + len("ITEM: BOX BOUNDS")
        end_index = in_text.index("\n",start_index)
        self.boundary_types = in_text[start_index : end_index].split()
        start_index = end_index
        end_index = in_text.index("ITEM",start_index)
        self.boxdims = np.array(in_text[start_index : end_index].split(),dtype=float)
        self.xlo, self.xhi, self.ylo, self.yhi, self.zlo, self.zhi = np.array(in_text[start_index : end_index].split(),dtype=float)
        
            
        # find number of atoms in simulation
        start_index = in_text.index("ITEM: NUMBER OF ATOMS") + len("ITEM: NUMBER OF ATOMS")
        end_index = in_text.index("ITEM:", start_index)
        self.natoms = int(in_text[start_index : end_index])

        # fetch data column labels
        start_index = in_text.index("ITEM: ATOMS ") + len("ITEM: ATOMS ")
        end_index = in_text.index("\n",start_index)
        self.dump_labels = in_text[start_index : end_index].split()

        
        # gather data dumps
        self.nframes = linecount//(toplines+self.natoms)    # number of completed dumps in file
        tsteps = []
        dumps = []
        
        # initial dump
        ts = in_text.index("TIMESTEP\n") + len("TIMESTEP\n")
        tf = in_text.index("\n",ts)
        self.tstep_i = int(in_text[ts:tf])
        tsteps.append(self.tstep_i)
        d0 = in_text.index("ITEM: ATOMS ") + len("ITEM: ATOMS ")
        ds = in_text.index("\n",d0)+1
        if self.nframes == 1:
            self.dump_i = np.fromstring(in_text[ds:],sep='\n').reshape(self.natoms,len(self.dump_labels))
            self.dump_f = self.dump_i
        else:
            df = in_text.index("ITEM: TIMESTEP\n",tf)-1
            self.dump_i = np.fromstring(in_text[ds:df],sep='\n').reshape(self.natoms,len(self.dump_labels))
        dumps.append(self.dump_i)

        for fn in (np.arange(1,self.nframes)):

            ts = in_text.index("TIMESTEP\n",tf) + len("TIMESTEP\n")
            tf = in_text.index("\n",ts)
            tstep_n = int(in_text[ts:tf])
            d0 = in_text.index("ITEM: ATOMS ",ts) + len("ITEM: ATOMS ")
            ds = in_text.index("\n",d0)+1

            # final completed dump
            if (fn+1) == self.nframes:    
                self.tstep_f = tstep_n
                tsteps.append(self.tstep_f)
                try:   # for incomplete dump file
                    df = in_text.index("ITEM: TIMESTEP\n",tf)-1
                    dump_f = np.fromstring(in_text[ds:df],sep='\n').reshape(self.natoms,len(self.dump_labels))
                except:    # for completed dump file
                    dump_f = np.fromstring(in_text[ds:],sep='\n').reshape(self.natoms,len(self.dump_labels))
                self.dump_f = dump_f
                dumps.append(self.dump_f)

            # gather all dumps if readall is True (otherwise just collect first and last)
            elif readall == True:
                df = in_text.index("ITEM: TIMESTEP\n",tf)-1
                dump_n = np.fromstring(in_text[ds:df],sep='\n').reshape(self.natoms,len(self.dump_labels))
                dumps.append(dump_n)
                tsteps.append(tstep_n)

        self.tsteps = np.array(tsteps)    # list of timesteps associated with each gathered dump
        self.dumps = dumps

        # set attributes for each column label
        for i in range(len(self.dump_labels)):
            data = []
            for t in range(len(self.tsteps)):
                framedata = self.dumps[t][:,i].tolist()
                data.append(framedata)
            data = np.array(data)
            setattr(self,"dump_"+self.dump_labels[i],data)



    # read log file - energy minimization
    def read_log_emin(self,logname="log.lammps"):
        opened = open("{}/{}".format(self.simdir,logname))
        in_text = opened.read()
        opened.close()
        
        # get initial, 2nd to last, final energies
        start_index = in_text.index("Energy initial, next-to-last, final = \n") + len("Energy initial, next-to-last, final = \n")
        end_index = in_text.index("\n", start_index)
        energies = (in_text[start_index : end_index]).strip().split()
        self.energyi = float(energies[0])    # initial energy
        self.energyf = float(energies[2])    # final energy
        self.energy2 = float(energies[1])    # second to last energy
        
        # get stopping criterion
        start_index = in_text.index("Stopping criterion = ") + len("Stopping criterion = ")
        end_index = in_text.index("\n", start_index)
        reason = in_text[start_index : end_index]
        self.why_stop = reason
        
        # thermo output
        try:
            start_index = in_text.index("TotEng") + len("TotEng")
            end_index = in_text.index("Loop time ", start_index)
            thermos = (in_text[start_index : end_index]).strip().split()
            self.ebondi = float(thermos[0:6][3])
            self.ebondf = float(thermos[-5:][3])
            self.epoti = float(thermos[0:6][1])
            self.epotf = float(thermos[-5:][1])
        except:
            pass
        
    def read_log_md(self,logname="log.lammps"):
        opened = open("{}/{}".format(self.simdir,logname))
        in_text = opened.read()
        opened.close()
        
        start_index = in_text.index("Step E_bond E_angle PotEng KinEng TotEng \n") + len("Step E_bond E_angle PotEng KinEng TotEng \n")
        end_index = in_text.index("Loop time ", start_index)
        energies = (in_text[start_index : end_index]).strip().split()
        self.ebondi = float(energies[0:6][1])
        self.ebondf = float(energies[-6:][1])
        self.eanglei = float(energies[0:6][2])
        self.eanglef = float(energies[-6:][2])
        self.pei = float(energies[0:6][3])
        self.pef = float(energies[-6:][3])
        self.kei = float(energies[0:6][4])
        self.kef = float(energies[-6:][4])
        self.energyi = float(energies[0:6][5])
        self.energyf = float(energies[-6:][5])
    
#     def read_dump(self,dumpname="dump.lammps",auto_labels="yes"):
#         opened = open("{}/{}".format(self.simdir,dumpname))
#         in_text = opened.read()
#         opened.close()
        
#         linecount = 0
#         for i in in_text:
#             if i == "\n":
#                 linecount += 1
        
#         end_index = in_text.index("ITEM: ATOMS ")
#         toplines = 1
#         for i in in_text[0 : end_index]:
#             if i == "\n":
#                 toplines += 1
        
#         start_index = in_text.index("ITEM: NUMBER OF ATOMS") + len("ITEM: NUMBER OF ATOMS")
#         end_index = in_text.index("ITEM:", start_index)
#         self.natoms = int(in_text[start_index : end_index])

#         start_index = in_text.index("ITEM: ATOMS ") + len("ITEM: ATOMS ")
#         end_index = in_text.index("\n",start_index)
#         self.column_labels = in_text[start_index : end_index].split()
        
#         self.initial = np.loadtxt("{}/{}".format(self.simdir,dumpname), skiprows=toplines, max_rows=self.natoms)
#         self.final = np.loadtxt("{}/{}".format(self.simdir,dumpname),skiprows=linecount-self.natoms)
        
#         if auto_labels=="yes":
#             # initial config
#             for i in range(len(self.column_labels)):
#                 setattr(self,self.column_labels[i]+"_i",self.initial[:,i])
#             # final config
#             for i in range(len(self.column_labels)):
#                 setattr(self,self.column_labels[i]+"_f",self.final[:,i])

    def read_variables(self,varname="Variables.py"):
        opened = open("{}/../{}".format(self.simdir,varname))
        in_text = opened.read()
        opened.close()

        newlines = []
        for i in np.arange(len(in_text)):
            if in_text[i] == "\n":
                newlines.append(i+len("\n"))

        for i in np.arange(len(newlines)-1):
            substring = in_text[newlines[i]:newlines[i+1]]
            if ("=" in substring):
                variable_name = substring.split()[0]
                if (variable_name == "nbeads") or (variable_name == "maxiter"):
                    variable_value = int(substring.split()[2])
                elif (variable_name == "loaddirectory"):
                    variable_value = substring.split()[2]
                else:
                    variable_value = float(substring.split()[2])
                setattr(self,variable_name,variable_value)
        
                
    def fitcircle(x1,y1,x2,y2,x3,y3):
        if ((y1!=y2) and (y1!=y3) and (y2!=y3)): 
            try:
                x0 = (x2**2 * y1 - x3**2 * y1 - x1**2 * y2 + x3**2 * y2 - y1**2 * y2 + y1 * y2**2 + x1**2 * y3 - x2**2 * y3 + y1**2 * y3 - y2**2 * y3 - y1 * y3**2 + y2 * y3**2)/(2 * (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3))
                y0 = (x1**2 * x2 - x1 * x2**2 - x1**2 * x3 + x2**2 * x3 + x1 * x3**2 - x2 * x3**2 + x2 * y1**2 - x3 * y1**2 - x1 * y2**2 + x3 * y2**2 + x1 * y3**2 - x2 * y3**2)/(2 * (x2 * y1 - x3 * y1 - x1 * y2 + x3 * y2 + x1 * y3 - x2 * y3))
                radius = np.sqrt((x1-x0)**2 + (y1-y0)**2)
            except ZeroDivisionError:
                radius = 1e10
        else:
            radius = 1e10
        return radius
    
    def gapdelta(self):
        r = self.r0
        t = self.t
        w = self.wx
        radical = np.sqrt(1 - ( (2*r-t)*np.sin(w/(2*r)) )**2 / (2*r + t)**2 )
        return -t - r*( np.cos(w/(2*r)) - 1 ) + t/2 * np.cos(w/(2*r)) + t/2*radical + r*(radical - 1)
        
    def find_curvature(self,mol,step="final"):
        if step == "final":
            molmask = (self.mol_f == mol)
            botmask = (self.type_f == 1) | (self.type_f == 3)
            topmask = (self.type_f == 2) | (self.type_f == 4) 
            xbot = self.x_f[botmask*molmask]
            ybot = self.y_f[botmask*molmask]
            xtop = self.x_f[topmask*molmask]
            ytop = self.y_f[topmask*molmask]
        else:
            molmask = (self.mol_i == mol)
            botmask = (self.type_i == 1) | (self.type_i == 3)
            topmask = (self.type_i == 2) | (self.type_i == 4) 
            xbot = self.x_i[botmask*molmask]
            ybot = self.y_i[botmask*molmask]
            xtop = self.x_i[topmask*molmask]
            ytop = self.y_i[topmask*molmask]
        xmid = (xtop - xbot)/2 + xbot
        ymid = (ytop - ybot)/2 + ybot
        radii_n = []
        dx_n = []
        for i in (np.arange(len(xmid)-2)+1):
            radii_n.append(ReadSim.fitcircle(xmid[i-1],ymid[i-1],xmid[i],ymid[i],xmid[i+1],ymid[i+1]))
            dx_n.append(xmid[i])
        radii_n = np.array(radii_n)
        dx_n = np.array(dx_n)
        
        return radii_n, dx_n
    
    def find_curvature_mid(self,mol,step="final"):
        if step == "final":
            molmask = (self.mol_f == mol)
            botmask = (self.type_f == 3)
            topmask = (self.type_f == 4) 
            xbot = self.x_f[botmask*molmask]
            ybot = self.y_f[botmask*molmask]
            xtop = self.x_f[topmask*molmask]
            ytop = self.y_f[topmask*molmask]
        else:
            molmask = (self.mol_i == mol)
            botmask = (self.type_i == 3)
            topmask = (self.type_i == 4) 
            xbot = self.x_i[botmask*molmask]
            ybot = self.y_i[botmask*molmask]
            xtop = self.x_i[topmask*molmask]
            ytop = self.y_i[topmask*molmask]
        xmid = (xtop - xbot)/2 + xbot
        ymid = (ytop - ybot)/2 + ybot
        radii_n = []
        dx_n = []
        for i in (np.arange(len(xmid)-2)+1):
            radii_n.append(ReadSim.fitcircle(xmid[i-1],ymid[i-1],xmid[i],ymid[i],xmid[i+1],ymid[i+1]))
            dx_n.append(xmid[i])
        radii_n = np.array(radii_n)
        dx_n = np.array(dx_n)
        
        return radii_n, dx_n
    
    def find_curvature_midlayer(self,mol,layer,step="final"):
        if step == "final":
            molmask = (self.mol_f == mol)
            botmask = (self.type_f == 3)
            topmask = (self.type_f == 4) 
            xbot = self.x_f[botmask*molmask]
            ybot = self.y_f[botmask*molmask]
            xtop = self.x_f[topmask*molmask]
            ytop = self.y_f[topmask*molmask]
        else:
            molmask = (self.mol_i == mol)
            botmask = (self.type_i == 3)
            topmask = (self.type_i == 4) 
            xbot = self.x_i[botmask*molmask]
            ybot = self.y_i[botmask*molmask]
            xtop = self.x_i[topmask*molmask]
            ytop = self.y_i[topmask*molmask]
            
        
        xmid = (xtop - xbot)/2 + xbot
        ymid = (ytop - ybot)/2 + ybot
        radii_n = []
        dx_n = []
        for i in (np.arange(len(xmid)-2)+1):
            if layer == "mid":
                radii_n.append(ReadSim.fitcircle(xmid[i-1],ymid[i-1],xmid[i],ymid[i],xmid[i+1],ymid[i+1]))
                dx_n.append(xmid[i])
            elif layer == "top":
                radii_n.append(ReadSim.fitcircle(xtop[i-1],ytop[i-1],xtop[i],ytop[i],xtop[i+1],ytop[i+1]))
                dx_n.append(xtop[i])
            elif layer == "bot":
                radii_n.append(ReadSim.fitcircle(xbot[i-1],ybot[i-1],xbot[i],ybot[i],xbot[i+1],ybot[i+1]))
                dx_n.append(xbot[i])
            
        radii_n = np.array(radii_n)
        dx_n = np.array(dx_n)
        
        return radii_n, dx_n
    
    def find_strain(self,mol,step="final"):
        if step == "final":
            molmask = (self.mol_f == mol)
            botmask = (self.type_f == 1) | (self.type_f == 3)
            topmask = (self.type_f == 2) | (self.type_f == 4) 
            xbot = self.x_f[botmask*molmask]
            ybot = self.y_f[botmask*molmask]
            xtop = self.x_f[topmask*molmask]
            ytop = self.y_f[topmask*molmask]
        else:
            molmask = (self.mol_i == mol)
            botmask = (self.type_i == 1) | (self.type_i == 3)
            topmask = (self.type_i == 2) | (self.type_i == 4) 
            xbot = self.x_i[botmask*molmask]
            ybot = self.y_i[botmask*molmask]
            xtop = self.x_i[topmask*molmask]
            ytop = self.y_i[topmask*molmask]
        xmid = (xtop - xbot)/2 + xbot
        ymid = (ytop - ybot)/2 + ybot
        
        dx = np.diff(xmid)
        dy = np.diff(ymid)
        dr = np.sqrt(dx*dx + dy*dy)
        a = self.wx/(self.nbeads-1)
        strains = (dr-a)/a
        return strains
        
    def midgaps_instack(self,step="final"):
        ncurvamers = np.size(np.unique(self.mol_f))
        npart = int(self.nbeads/2)
        gaps = []
        if step == "final":
            for mol in np.arange(ncurvamers-1)+1:
                botmolmask = (self.mol_f == mol)
                topmolmask = (self.mol_f == mol+1)
                botmask = (self.type_f == 1) | (self.type_f == 3)
                topmask = (self.type_f == 2) | (self.type_f == 4) 
                xbot = self.x_f[botmask*topmolmask]
                ybot = self.y_f[botmask*topmolmask]
                xtop = self.x_f[topmask*botmolmask]
                ytop = self.y_f[topmask*botmolmask]
                midgap = np.mean(ybot[npart-5:npart+5] - ytop[npart-5:npart+5])
                gaps.append(midgap)
        else:
            for mol in np.arange(ncurvamers-1)+1:
                botmolmask = (self.mol_f == mol)
                topmolmask = (self.mol_f == mol+1)
                botmask = (self.type_f == 1) | (self.type_f == 3)
                topmask = (self.type_f == 2) | (self.type_f == 4) 
                xbot = self.x_f[botmask*topmolmask]
                ybot = self.y_f[botmask*topmolmask]
                xtop = self.x_f[topmask*botmolmask]
                ytop = self.y_f[topmask*botmolmask]
                midgap = np.mean(ybot[npart-5:npart+5] - ytop[npart-5:npart+5])
                gaps.append(midgap)

        gaps = np.array(gaps) - (self.t - self.t0)
        return gaps
        
    def centermidgap(gaps,step="final"):
#         gaps = ReadSim.midgaps_instack(step)
        ncurvamers = np.size(gaps)+1
        if (ncurvamers%2==0):
            # if even number of curvamers, calculate midgap at center of stack 
            # e.g if 4 curvamers, calculate midgap between particles 2 and 3
            centergap = gaps[int((ncurvamers)/2 - 1)]
        elif (ncurvamers%2==1):
            # if odd number of curvamers, avergage midgap between center particles
            # e.g if 5 curvamers, average midgap between particles 2 and 3, and 3 and 4
            centergap = (gaps[int((ncurvamers-1)/2 - 1)] + gaps[int((ncurvamers-1)/2)])/2
        return centergap
           
        
