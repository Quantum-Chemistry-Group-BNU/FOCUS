import os
import test_utils

mpiprefix = "mpirun -np 2 "
os.environ['OMP_NUM_THREADS'] = "4"
print('OMP_NUM_THREADS=',os.environ.get('OMP_NUM_THREADS'))

HOME = os.path.dirname(os.getcwd())
print('HOME=',HOME)
SCI  = HOME+"/bin/sci.x"
CTNS = mpiprefix + HOME+"/bin/ctns.x"
os.environ['SCI'] = SCI
os.environ['CTNS'] = CTNS

#import os
#print(os.environ['DYLD_LIBRARY_PATH'])

#cdir = os.getcwd()
#dirs = [tdir for tdir in os.listdir(cdir) if os.path.isdir(tdir)]
dirs = ['tests/0_h6_tns',
        'tests/1_lih3_dcg', 
        'tests/2_lih3+_dcg', 
        'tests/3_h6+_kr',
        'tests/4_h5_cNK',
        'tests/5_h5_rNSz_hvec4',
        'tests/5_h5_rNSz_hvec5',
        'tests/5_h5_rNSz_hvec6',
        'tests/5_h5_rNSz_renorm2',
        'tests/7_h6_cisolver',
        ]
test_utils.test_run(dirs)
