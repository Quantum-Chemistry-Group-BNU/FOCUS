import os
import focus_parser

class Common:
    def __init__(self):
        self.parrentdir = None
        self.workdir = None
        self.dtype = None
        self.nelec = None
        self.twoms = None
        self.integral_file = None
        self.scratch = None

    def gen_input(self,f):
        f.write('dtype '+str(self.dtype)+'\n')
        f.write('nelec '+str(self.nelec)+'\n')
        f.write('twoms '+str(self.twoms)+'\n')
        f.write('integral_file '+str(self.integral_file)+'\n')
        f.write('scratch '+str(self.scratch)+'\n')
        return 0

    def build(self):
        print('parrentdir =',self.parrentdir)
        self.workdir = self.parrentdir+'/'+self.workdir
        print('workdir =',self.workdir)
        os.system('mkdir -p '+self.workdir)
        return 0
            
class SCI:
    def __init__(self,common):
        self.common = common
        self.dets = None
        self.nroots = None
        self.eps0 = None
        self.maxiter = None
        self.schedule = None

    def gen_input(self,fname,iprt):
        f = open(fname,'w')
        self.common.gen_input(f)
        f.write('\n$sci\n')
        # dets
        f.write('dets\n')
        for i in range(len(self.dets)):
            line = ' '.join(str(k) for k in self.dets[i])+'\n'
            f.write(line)
        f.write('end\n')
        # schedule
        f.write('schedule\n')
        for item in self.schedule:
            f.write(str(item[0])+' '+str(item[1])+'\n')
        f.write('end\n')
        f.write('nroots '+str(self.nroots)+'\n')
        f.write('eps0 '+str(self.eps0)+'\n')
        f.write('maxiter '+str(self.maxiter)+'\n')
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
        os.chdir(self.common.parrentdir)
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
        self.tasks = ["task_oodmrg"]
        self.alg_hvec = 4
        self.alg_renorm = 4

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
        f.write('schedule\n')
        for item in self.schedule:
            f.write(str(item[0])+' '+str(item[1])+' '+str(item[2])+' '+str(item[3])+' '+str(item[4])+'\n')
        f.write('end\n')
        f.write('maxsweep '+str(self.maxsweep)+'\n')
        for task in self.tasks:
            f.write(task+'\n')
        f.write('alg_hvec '+str(self.alg_hvec)+'\n')
        f.write('alg_renorm '+str(self.alg_renorm)+'\n')
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
        elst = focus_parser.parse_ctns(output)
        os.chdir(self.common.parrentdir)
        return elst

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
        os.chdir(self.common.parrentdir)
        return 0

