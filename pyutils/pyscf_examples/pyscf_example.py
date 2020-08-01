#
# For more details, see
# https://sunqm.github.io/pyscf/tutorial.html
#

#==========
# MOLECULE
#==========
from pyscf import gto
mol = gto.Mole()
mol.verbose = 5 #6
#==========
# Geometry
#==========
#R = 1.0
#mol.atom = [['N',(0,0,0)],
#	     ['N',(0,0,R)]]		
R = 1.0
natoms = 2
mol.atom = [['H',(0.0,0.0,i*R)] for i in range(natoms)]
#==========
mol.basis = '6-31g' #sto-3g' #cc-pvdz' #6-31g'#sto-3g'#cc-pvdz' #'dzp' #6-31g'#sto-3g' #6-31g'
mol.symmetry = True
mol.charge = 0
mol.spin = 0
mol.build()

#=====
# SCF
#=====
from pyscf import scf
mf = scf.RHF(mol)
mf.init_guess = 'atom'
mf.level_shift = 0.001
mf.max_cycle = 100
mf.conv_tol=1.e-14
escf=mf.scf()

print mf.mo_coeff.shape

#=====
# MP2
#=====
from pyscf import mp
mymp = mp.MP2(mf)
mymp.kernel()

#====
# CC
#====
from pyscf import cc
mycc = cc.CCSD(mf)
mycc.kernel()

#exit()

#==============
# FCIDUMP file
#==============
from pyscf import tools
tools.fcidump.from_mo(mol,'FCIDUMP',mf.mo_coeff)

#==============
# MO integrals
#==============
from pyscf import ao2mo
mo_coeff = mf.mo_coeff
h1 = mo_coeff.T.dot(mf.get_hcore()).dot(mo_coeff)
eri = ao2mo.kernel(mol, mo_coeff)
print h1.shape
print eri.shape

#=======================
# FCI with MO integrals
#=======================
from pyscf import fci
cisolver = fci.direct_spin1.FCI(mol)
cisolver.nroots = 2
norb = h1.shape[1]
nelec = mol.nelec
e, ci = cisolver.kernel(h1, eri, norb, nelec, ecore=mol.energy_nuc())
print 'e=',e
print 'ci.shape=',ci[0].shape
print ci[0]

