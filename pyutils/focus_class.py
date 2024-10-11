import os
import focus_parser
import numpy as np

class Common:
    def __init__(self):
        self.parentdir = None
        self.workdir = None
        self.dtype = 0
        self.sorb = None
        self.nelec = None
        self.twom = None
        self.integral_file = None
        self.scratch = "scratch"

    def gen_input(self,f):
        f.write('dtype '+str(self.dtype)+'\n')
        f.write('sorb '+str(self.sorb)+'\n')
        f.write('nelec '+str(self.nelec)+'\n')
        f.write('twom '+str(self.twom)+'\n')
        f.write('integral_file '+str(self.integral_file)+'\n')
        f.write('scratch '+str(self.scratch)+'\n')
        return 0

    def build(self):
        print('\nCommon.build:')
        print('parentdir =',self.parentdir)
        self.workdir = self.parentdir+'/'+self.workdir
        print('workdir =',self.workdir)
        os.system('mkdir -p '+self.workdir)
        return 0


class SCI:
    def __init__(self,common):
        self.common = common
        self.dets = None
        self.nroots = 1
        self.eps0 = 1.e-2
        self.maxiter = 3
        self.schedule = [[0,1.e-2]]

    def gen_input(self,fname,iprt):
        f = open(fname,'w')
        self.common.gen_input(f)
        f.write('\n$ci\n')
        # dets
        f.write('dets\n')
        for i in range(len(self.dets)):
            line = ' '.join(str(k) for k in self.dets[i])+'\n'
            f.write(line)
        f.write('end\n')
        # schedule
        if self.schedule != None:
            f.write('schedule\n')
            for item in self.schedule:
                f.write(str(item[0])+' '+str(item[1])+'\n')
            f.write('end\n')
        f.write('nroots '+str(self.nroots)+'\n')
        f.write('eps0 '+str(self.eps0)+'\n')
        f.write('maxiter '+str(self.maxiter)+'\n')
        f.write('analysis\n')
        f.write('$end\n')
        f.close()
        if iprt > 0:
            print('\nInput file:')
            os.system('cat '+fname)
        return 0

    def kernel(self,fname='sci.dat',output='sci.out',iprt=0):
        os.chdir(self.common.workdir)
        cmd = "sci.x "+fname+" > "+output
        print('\nSCI calculations: '+cmd)
        self.gen_input(fname,iprt)
        info = os.system(cmd)
        assert info == 0
        if iprt > 0:
            print('\nOutput file:')
            os.system('cat '+output)
        elst = focus_parser.parse_sci(output)
        os.chdir(self.common.parentdir)
        return elst


class CTNS:
    def __init__(self,common):
        self.common = common
        self.qkind = "rNSz"
        self.nroots = 1
        self.maxdets = 10000
        self.thresh_proj = 1.e-10
        self.topology_file = "topo"
        self.topo = None
        self.schedule = None
        self.maxsweep = 4
        self.tasks = None
        self.oo_maxiter = None
        self.oo_macroiter = None
        self.oo_alpha = None
        self.alg_hvec = 4
        self.alg_renorm = 4
        self.rcanon_file = None

    def gen_input(self,fname,iprt):
        # TOPO
        f = open(self.topology_file,'w')
        for i in self.topo:
            f.write(str(i)+'\n')
        f.close()
        # CTNS
        f = open(fname,'w')
        self.common.gen_input(f)
        f.write('\n$ctns\n')
        f.write('qkind '+self.qkind+'\n')
        f.write('nroots '+str(self.nroots)+'\n')
        f.write('maxdets '+str(self.maxdets)+'\n')
        f.write('thresh_proj '+str(self.thresh_proj)+'\n')
        f.write('topology_file '+self.topology_file+'\n')
        # schedule
        if self.schedule != None:
            f.write('schedule\n')
            for item in self.schedule:
                f.write(str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n')
            f.write('end\n')
        f.write('maxsweep '+str(self.maxsweep)+'\n')
        for task in self.tasks:
            f.write(task+'\n')
        f.write('alg_hvec '+str(self.alg_hvec)+'\n')
        f.write('alg_renorm '+str(self.alg_renorm)+'\n')
        if self.oo_maxiter != None: f.write('oo_maxiter '+str(self.oo_maxiter)+'\n')
        if self.oo_macroiter != None: f.write('oo_macroiter '+str(self.oo_macroiter)+'\n')
        if self.oo_alpha != None: f.write('oo_alpha '+str(self.oo_alpha)+'\n')
        if self.rcanon_file != None: f.write('rcanon_file '+str(self.rcanon_file)+'\n')
        f.write('$end\n')
        f.close()
        if iprt > 0:
            print('\nInput file:')
            os.system('cat '+fname)
        return 0

    def kernel(self,fname='ctns.dat',output='ctns.out',iprt=0):
        os.chdir(self.common.workdir)
        cmd = "ctns.x "+fname+" > "+output
        print('\nCTNS calculations: '+cmd)
        self.gen_input(fname,iprt)
        info = os.system(cmd)
        assert info == 0
        if iprt > 0:
            print('\nOutput file:')
            os.system('cat '+output)
        os.chdir(self.common.parentdir)
        return 0

    def parse_dmrg(self,output='ctns.out',iprt=0):
        elst = focus_parser.parse_ctns(output)
        if iprt > 0:
            print('parse_dmrg: elst=',elst)
        return elst

    def parse_oodmrg(self,output='ctns.out',iprt=0):
        result = focus_parser.parse_oodmrg(output,iprt=iprt)
        if iprt > 0:
            print('parse_oodmrg: result=',result)
        return result

    def gen_savebin(self,fname,iprt,isweep,task_ham):
        f = open(fname,'w')
        self.common.gen_input(f)
        f.write('\n$ctns\n')
        f.write('qkind '+self.qkind+'\n')
        f.write('topology_file '+self.topology_file+'\n')
        f.write('rcanon_file rcanon_isweep'+str(isweep)+'\n')
        f.write('savebin\n')
        if task_ham: f.write('task_ham\n')
        f.write('$end\n')
        f.close()
        if iprt > 0:
            print('\nInput file:')
            os.system('cat '+fname)
        return 0

    def savebin(self,isweep=0,iprt=0,fname='savebin.dat',output='savebin.out'):
        os.chdir(self.common.workdir)
        cmd = "rdm.x "+fname+" > "+output
        print('\nSAVEBIN: '+cmd)
        self.gen_savebin(fname,iprt,isweep,False)
        info = os.system(cmd)
        assert info == 0
        if iprt > 0:
            print('\nOutput file:')
            os.system('cat '+output)
        return 0


class RDM:
    def __init__(self,common):
        self.common = common
        self.qkind = "rNSz"
        self.topology_file = "topo"
        self.alg_renorm = 4
        self.alg_rdm = 1
        self.rcanon_file = None
        
    def gen_input(self,fname,iprt):
        f = open(fname,'w')
        self.common.gen_input(f)
        f.write('\n$ctns\n')
        f.write('qkind '+self.qkind+'\n')
        f.write('topology_file '+self.topology_file+'\n')
        f.write('alg_renorm '+str(self.alg_renorm)+'\n')
        f.write('alg_rdm '+str(self.alg_rdm)+'\n')
        f.write('task_prop 2\n')
        if self.rcanon_file != None: f.write('rcanon_file '+self.rcanon_file+'\n')
        f.write('$end\n')
        f.close()
        if iprt > 0:
            print('\nInput file:')
            os.system('cat '+fname)
        return 0

    def kernel(self,fname='rdm.dat',output='rdm.out',iprt=0):
        os.chdir(self.common.workdir)
        cmd = "rdm.x "+fname+" > "+output
        print('\nRDM calculations: '+cmd)
        self.gen_input(fname,iprt)
        info = os.system(cmd)
        assert info == 0
        if iprt > 0:
            print('\nOutput file:')
            os.system('cat '+output)
        os.chdir(self.common.parentdir)
        return 0

    def parse_rdm(self,order=2,iprt=0):
        os.chdir(self.common.workdir)
        f = open('rdm'+str(order)+'mps.0.0.txt')
        lines = f.readlines()
        k2 = len(lines)
        rdm = np.zeros((k2,k2))
        idx = 0
        for line in lines:
            rdm[:,idx] = [eval(x) for x in line.split()]
            idx += 1
        if iprt > 0:
            print('trace(RDM'+str(order)+')=',np.trace(rdm),
                  ' |RDM'+str(order)+'|_F=',np.linalg.norm(rdm))
        os.chdir(self.common.parentdir)
        return rdm

    def get_cumulant(self,iprt=0):
        rdm1 = self.parse_rdm(order=1,iprt=iprt)
        rdm2 = self.parse_rdm(order=2,iprt=iprt)
        k2 = rdm2.shape[0]
        k = int(1 + np.sqrt(1+k2*8))//2
        cumulant = np.zeros((k2,k2))
        for p in range(k):
            for q in range(p):
                pq = p*(p-1)//2 + q
                for r in range(k):
                    for s in range(r):
                        rs = r*(r-1)//2 + s
                        cumulant[pq,rs] = rdm2[pq,rs] - rdm1[p,r]*rdm1[q,s] + rdm1[p,s]*rdm1[q,r]
        if iprt > 0:
            print('tr(C2)=',np.trace(cumulant),
                  ' |C2|_F=',np.linalg.norm(cumulant))
        return rdm1,rdm2,cumulant

