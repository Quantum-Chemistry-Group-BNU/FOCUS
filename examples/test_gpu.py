import os
import test_utils

mpiprefix = "mpirun -np 2 --allow-run-as-root "
os.environ['OMP_NUM_THREADS'] = "10"
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
#print('CUDA_VISIBLE_DEVICES=',os.environ.get('CUDA_VISIBLE_DEVICES'))

HOME = os.path.dirname(os.getcwd())
SCI  = HOME+"/bin/sci.x"
CTNS = mpiprefix + HOME+"/bin/ctns.x"
SADMRG = mpiprefix + HOME+"/bin/sadmrg.x"
os.environ['SCI'] = SCI
os.environ['CTNS'] = CTNS
os.environ['SADMRG'] = SADMRG

print('mpiprefix=',mpiprefix)
print('OMP_NUM_THREADS=',os.environ.get('OMP_NUM_THREADS'))
print('HOME=',HOME)

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
        'tests_gpu/h6_sadmrg_onedot',
        'tests_gpu/h6_sadmrg_nccl',
        'tests_gpu/h6_ab2pq',
        'tests_gpu/h6_sadmrg_ab2pq',
        'tests_gpu/fe2s2_sadmrg',
        'tests_gpu/fe2s2_sadmrg_ab2pq'
        ]
test_utils.test_run(dirs)
