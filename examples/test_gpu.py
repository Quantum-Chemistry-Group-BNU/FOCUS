import os
import test_utils

mpiprefix = "mpirun -np 2 "
os.environ['OMP_NUM_THREADS'] = "10"
print('OMP_NUM_THREADS=',os.environ.get('OMP_NUM_THREADS'))
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#print('CUDA_VISIBLE_DEVICES=',os.environ.get('CUDA_VISIBLE_DEVICES'))

HOME = os.path.dirname(os.getcwd())
print('HOME=',HOME)
SCI  = HOME+"/bin/sci.x"
CTNS = mpiprefix + HOME+"/bin/ctns.x"
SADMRG = mpiprefix + HOME+"/bin/sadmrg.x"
os.environ['SCI'] = SCI
os.environ['CTNS'] = CTNS
os.environ['SADMRG'] = SADMRG

#import os
#print(os.environ['DYLD_LIBRARY_PATH'])

#cdir = os.getcwd()
#dirs = [tdir for tdir in os.listdir(cdir) if os.path.isdir(tdir)]
dirs = ['tests_gpu/h4',
        'tests_gpu/h5_rNSz_alg16',
        'tests_gpu/h5_rNSz_alg17',
        'tests_gpu/h5_rNSz_alg18',
        'tests_gpu/h5_rNSz_alg19',
        'tests_gpu/h5_rNSz_alg19b', # ifdist1
        'tests_gpu/h5_rNSz_cisolver',
        'tests_gpu/h6_sadmrg_cisolver',
        'tests_gpu/h6_sadmrg_onedot'
        ]
test_utils.test_run(dirs)
