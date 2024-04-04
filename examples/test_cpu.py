import os
import test_utils

mpiprefix = "mpirun -np 1 "
os.environ['OMP_NUM_THREADS'] = "4"
print('OMP_NUM_THREADS=',os.environ.get('OMP_NUM_THREADS'))

HOME = os.path.dirname(os.getcwd())
print('HOME=',HOME)
SCI  = HOME+"/bin/sci.x"
CTNS = mpiprefix + HOME+"/bin/ctns.x"
SADMRG = mpiprefix + HOME+"/bin/sadmrg.x"
os.environ['SCI'] = SCI
os.environ['CTNS'] = CTNS
os.environ['SADMRG'] = SADMRG

dirs = ['tests_cpu/0_h6_tns', 
        'tests_cpu/1_lih3_dcg', 
        'tests_cpu/2_lih3+_dcg', 
        'tests_cpu/3_h6+_kr',
        'tests_cpu/4_h5_cNK',
        'tests_cpu/5_h5_rNSz_hvec4',
        'tests_cpu/5_h5_rNSz_hvec5',
        'tests_cpu/5_h5_rNSz_hvec6',
        'tests_cpu/5_h5_rNSz_renorm2',
        'tests_cpu/5_h5_cNSz_hvec5',
        'tests_cpu/7_h6_cisolver',
        'tests_cpu/8_h6+_sadmrg',
        'tests_cpu/9_h6+_sadmrg_cisolver',
        ]
test_utils.test_run(dirs)
