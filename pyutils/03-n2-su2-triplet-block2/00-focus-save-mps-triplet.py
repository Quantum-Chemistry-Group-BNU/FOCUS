
import focus_class
import ipyscf_real
import os

iface = ipyscf_real.iface()
fd_name = "N2.STO3G.FCIDUMP"
info = iface.get_integral_FCIDUMP(fname=fd_name)
iface.dump(info, fname='fmole.info')
header = open(fd_name, 'r').readlines()[0]
n_sites, n_elec, spin = [int(header.split('%s=' % x)[1].split(',')[0]) for x in ['NORB', 'NELEC', 'MS2']]
spin = 2
print(n_sites, n_elec, spin)

common = focus_class.Common()
common.parentdir = os.getcwd()
common.workdir = 'tmp'
common.dtype = 0
common.nelec = n_elec
common.sorb = n_sites * 2
common.twos = spin
common.twom = spin
common.integral_file = os.getcwd() + '/fmole.info'
common.scratch = 'scratch'
common.build()

sci = focus_class.SCI(common)
sci.additional = ['init_aufbau','checkms']
sci.nroots = 2
sci.eps0 = 1.e-4
sci.maxiter = 1
sci.schedule = [(0,1.e-3)]
sci.kernel(fname='sci.dat', output='sci.out', iprt=0)

ctns = focus_class.CTNS(common)
ctns.qkind = 'rNS'
ctns.additional = ['tosu2','singlet']
ctns.nroots = 1
ctns.maxdets = 100
ctns.thresh_proj = 1.e-14
ctns.topology_file = 'topo'
ctns.topo = range(n_sites)
ctns.schedule = [(0,2,64,1.e-4,1.e-8)]
ctns.maxsweep = 4
ctns.tasks = ['task_dmrg']
ctns.alg_hvec = 4
ctns.alg_renorm = 4
ctns.kernel(fname='ctns.dat', output='ctns.out', iprt=0)

os_system_orig = os.system
os.system = lambda cmd: os_system_orig(cmd.replace('rdm.x ', 'prop.x '))
ctns.savebin(isweep=1, iprt=0)

# will generate tmp/scratch/rcanon_isweep1_su2.bin
# copy it to ./rcanon_isweep1_su2.bin
