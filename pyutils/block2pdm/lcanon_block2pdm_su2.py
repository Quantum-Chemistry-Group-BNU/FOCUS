import time
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2.algebra.io import MPSTools

# Note: to align the fermion / su2 convention and orbital ordering between focus and block2:
#   U1:  Use SymmetryTypes.SAnySZ symmetry mode and hints=['ALT'] in ``driver.set_symmetry_groups``.
#        Use reorder=np.arange(ncas)[::-1] in ``driver.get_qc_mpo``.
#   SU2 (singlet only):
#        Use SymmetryTypes.SU2 symmetry mode
#        Use reorder=np.arange(ncas)[::-1] in ``driver.get_qc_mpo``.
#   SU2 (non-singlet or singlet):
#        Use SymmetryTypes.SU2 symmetry mode
#        Use reorder=np.arange(ncas)[::-1] in ``driver.get_qc_mpo``.
#        Use singlet_embedding=False in ``driver.initialize_system``.
#        When import to block2, use center=driver.n_sites - 1 in ``MPSTools.to_block2``.
#        When export from block2, use ref=driver.n_sites - 1 in ``driver.align_mps_center``.
#   For starting two-site dmrg in block2 using the imported mps:
#        Use ``mps = driver.adjust_mps(mps, dot=2)[0]``.

mpsfile = './scratch/rcanon_isweep11_su2.lcanon.singlet.bin'
fcidumpfile = 'FCIDUMP'
topofile = None #'./tmp/topo'
spin = 0
dmax = 1000 #100
ifSE = True #False

topo = np.arange(36) # no need to reverse ordering in lcanon case
#topo = []
#f = open(topofile)
#for line in f.readlines():
#   topo.append(eval(line))
#f.close()
#topo = np.array(topo)
#topo = np.array(topo[::-1])
print('topo=',topo)

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, \
      n_threads=18,
      stack_mem=1024*1024*1024*10)

driver.read_fcidump(filename=fcidumpfile, pg='nopg')

ncas = driver.n_sites
n_elec = driver.n_elec
orb_sym = driver.orb_sym
h1e = driver.h1e
g2e = driver.g2e
ecore = driver.ecore

driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=ifSE)

#=== import MPS from file ===
pymps = MPSTools.from_focus_mps_file(fname=mpsfile, is_su2=True)
left_vacuum = None if not ifSE else driver.bw.SX(spin, spin, 0)
mps = MPSTools.to_block2(pymps, driver.basis, center=driver.n_sites - 1, left_vacuum=left_vacuum)

ifEnergyExpect = False
if ifEnergyExpect:
   mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1, reorder=topo)
   impo = driver.get_identity_mpo()
   expt = driver.expectation(mps, mpo, mps) / driver.expectation(mps, impo, mps)
   print('Energy from expectation = %20.15f' % expt)

# pdms & expectations
for fac in [1]:
   dcut = fac*dmax
   print('\nGenerate n-pdm with dcut=',dcut)

   t0 = time.time()

   pdm1 = driver.get_1pdm(mps, max_bond_dim=dcut) #, iprint=2)
   np.save('pdm1',pdm1)
   print('|dm1-dm1.T|=',np.linalg.norm(pdm1-pdm1.T))
   pdm2 = driver.get_2pdm(mps, max_bond_dim=dcut) #, iprint=2)
   np.save('pdm2',pdm2)

   # spin-free rdms
   print(pdm1.shape)
   print(pdm2.shape)
   pdm1tmp = np.einsum('ijjk->ik',pdm2)/(n_elec-1)
   print(np.linalg.norm(pdm1tmp-pdm1))

   #
   # <ij|kl> <is+*jt+*lt*ks> = [ik|jl] G2[i,k,j,l]
   # G2[i,k,j,l] = <is+*jt+*lt*ks>
   #
   # block2 rdm is defined [https://block2.readthedocs.io/en/latest/api/pyblock2.html] as
   # g2[i,j,l,k] = <is+*jt+*lt*ks>
   #
   # G2[i,k,j,l] = g2[i,j,l,k] = g2.tranpose(0,3,1,2) [i,k,j,l]
   #
   pdm2b = pdm2.transpose(0, 3, 1, 2)
   print('Energy from pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1e)
       + 0.5 * np.einsum('ijkl,ijkl->', pdm2b, driver.unpack_g2e(g2e)) + ecore))

   t1 = time.time()
   print('\ndt=',t1-t0)

