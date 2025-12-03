
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from pyblock2.algebra.io import MPSTools

bond_dims = [250] * 4 + [500] * 4
noises = [1e-4] * 4 + [1e-5] * 4 + [0]
thrds = [1e-10] * 8

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

driver = DMRGDriver(scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=4)

driver.read_fcidump(filename='N2.STO3G.FCIDUMP', pg='nopg')
exit(1)

ncas = driver.n_sites
n_elec = driver.n_elec
spin = driver.spin
orb_sym = driver.orb_sym
h1e = driver.h1e
g2e = driver.g2e
ecore = driver.ecore

spin = 2

driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin, orb_sym=orb_sym, singlet_embedding=False)
mpo = driver.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=1, reorder=np.arange(ncas)[::-1])
impo = driver.get_identity_mpo()

# import MPS from file
pymps = MPSTools.from_focus_mps_file(fname='rcanon_isweep1_su2.bin', is_su2=True)
mps = MPSTools.to_block2(pymps, driver.basis, center=driver.n_sites - 1)

# pdms & expectations
pdm1 = driver.get_1pdm(mps)
pdm2 = driver.get_2pdm(mps).transpose(0, 3, 1, 2)
print('Energy from pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1e)
    + 0.5 * np.einsum('ijkl,ijkl->', pdm2, driver.unpack_g2e(g2e)) + ecore))
expt = driver.expectation(mps, mpo, mps) / driver.expectation(mps, impo, mps)
print('Energy from expectation = %20.15f' % expt)
exit(1)

# two-site dmrg from imported mps
mps = driver.adjust_mps(mps, dot=2)[0]
energy = driver.dmrg(mpo, mps, n_sweeps=2, bond_dims=[500], noises=[0], thrds=[1E-10], iprint=0)
print('DMRG energy = %20.15f' % energy)

# dmrg start from random
ket = driver.get_random_mps(tag="KET", bond_dim=10, nroots=1)
energy = driver.dmrg(mpo, ket, n_sweeps=20, bond_dims=bond_dims, noises=noises, thrds=thrds, iprint=0)
print('DMRG energy = %20.15f' % energy)

# export MPS to file
ket = driver.adjust_mps(ket, dot=1)[0]
driver.align_mps_center(ket, ref=driver.n_sites - 1)
pyket = MPSTools.from_block2(ket)
MPSTools.to_focus_mps_file(fname='block2_mps_su2.bin', mps=pyket)

# import MPS from file
pymps = MPSTools.from_focus_mps_file(fname='block2_mps_su2.bin', is_su2=True)
mps = MPSTools.to_block2(pymps, driver.basis, center=driver.n_sites - 1)

# pdms & expectations
pdm1 = driver.get_1pdm(mps)
pdm2 = driver.get_2pdm(mps).transpose(0, 3, 1, 2)
print('Energy from pdms = %20.15f' % (np.einsum('ij,ij->', pdm1, h1e)
    + 0.5 * np.einsum('ijkl,ijkl->', pdm2, driver.unpack_g2e(g2e)) + ecore))
expt = driver.expectation(mps, mpo, mps) / driver.expectation(mps, impo, mps)
print('Energy from expectation = %20.15f' % expt)
