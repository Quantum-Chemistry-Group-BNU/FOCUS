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

mpsfile = './scratch/rcanon_isweep1_su2.lcanon.singlet.bin'
#mpsfile = './tmp/scratch/rcanon_isweep0_su2.lcanon.bin'
#mpsfile = './scratch/rcanon_isweep39_su2.bin'
fcidumpfile = 'mole.info.FCIDUMP'
spin = 0
dmax = 100
ifSE = True #False

norb = 10
topo = np.array(range(norb)[::-1])
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
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1, reorder=topo)
impo = driver.get_identity_mpo()

#=== import MPS from file ===
pymps = MPSTools.from_focus_mps_file(fname=mpsfile, is_su2=True)
left_vacuum = None if not ifSE else driver.bw.SX(spin, spin, 0)
mps = MPSTools.to_block2(pymps, driver.basis, center=driver.n_sites - 1, left_vacuum=left_vacuum)

expt = driver.expectation(mps, mpo, mps) / driver.expectation(mps, impo, mps)
print('Energy from expectation = %20.15f' % expt)

# pdms & expectations
for fac in [1,2]:
   dcut = fac*dmax
   print('\nGenerate n-pdm with dcut=',dcut)

   t0 = time.time()

   pdm1 = driver.get_1pdm(mps, max_bond_dim=dcut) #, iprint=2)
   print('|dm1-dm1.T|=',np.linalg.norm(pdm1-pdm1.T))
   pdm2 = driver.get_2pdm(mps, max_bond_dim=dcut) #, iprint=2)

   # spin-free rdms
   print(pdm1.shape)
   print(pdm2.shape)
   pdm1tmp = np.einsum('ijjk->ik',pdm2)/(n_elec-1)
   print(np.linalg.norm(pdm1tmp-pdm1))

   print('topo=',topo)
   pdm1 = pdm1[np.ix_(topo,topo)]
   pdm2 = pdm2[np.ix_(topo,topo,topo,topo)]
   np.save('pdm1',pdm1)
   np.save('pdm2',pdm2)
   
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

