import os
import test_utils

mpiprefix = "" #mpirun -np 2 "
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
dirs = ['tests_gpu/h4',
        'tests_gpu/h5_rNSz',
        ]
test_utils.test_run(dirs)
