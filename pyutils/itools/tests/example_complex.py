import numpy
from pyscf import gto,scf

#==================================================================
# MOLECULE
#==================================================================
mol = gto.Mole()
mol.verbose = 5 #6

#==================================================================
# Coordinates and basis
#==================================================================
molname = 'random'

if molname == 'ne':
   R = 1.e0
   natoms = 1 #50 #20 #,14,20,50
   mol.atom = [['Ne',(0.0,0.0,i*R)] for i in range(natoms)]
   mol.basis = 'dzp'

elif molname == 'h':
   R = 1.0
   natoms = 6 #40 #,14,20,50
   mol.atom = [['H',(0.0,0.0,i*R)] for i in range(natoms)]
   mol.basis = 'sto-3g' #cc-pvdz' #sto-3g' #6-31g' 

elif molname == 'random':
   mol.atom = [['Li',(0,0,0)],
   	       ['H' ,(0,0,1)],
               ['H' ,(2,0.8,0)],
               ['H' ,(2,0,0)]]
   mol.basis = 'sto-3g'

elif molname == 'hf':
   R=1.0
   mol.atom = [['H',(0,0,0)],
   	       ['F',(0,0,R)]]		
   mol.basis = 'sto-3g'
   # RHF -98.57048973900092
   # DHF -98.6404374236861                DIRAC: -98.64043742396342
   # DHF -98.62934960957465  with Gaunt - DIRAC: -98.62934960987491

elif molname == 'ch3s':
   mol.atom = [['C', (1.22840691,   -1.18042226,    0.00000000)], 
               ['H', (0.63626191,   -0.24101826,    0.00000000)],
               ['H', (0.63626191,   -2.11982626,    0.00003900)],
               ['S', (2.79480691,   -1.18042226,    0.00000000)],
               ['H', (3.23209390,   -1.18281673,   -1.23485803)]]
   mol.basis = 'sto-3g' #sto-3g' #cc-pvdz'#sto-3G'

elif molname == 'o2':
   R=1.20752
   mol.atom = [['O',(0,0,0)],
   	       ['O',(0,0,R)]]	
   mol.basis = {'O': gto.uncontract_basis(gto.basis.load('sto-3g', 'O'))}

elif molname == 'f':
   mol.atom = [['F',(0,0,0)]]
   mol.basis = {'F': gto.uncontract_basis(gto.basis.load('sto-3g', 'F'))}

elif molname == 'ne':
   mol.atom = [['Ne',(0,0,0)]]
   mol.basis = {'Ne': gto.uncontract_basis(gto.basis.load('321g', 'Ne'))}

#==================================================================
mol.symmetry = False 
mol.charge = 0 #1 #-1 
mol.spin = 0
# Artificial world
#mol.light_speed = 1.0 
mol.build()
#==================================================================

#==================================================================
# DHF
#==================================================================
mf = scf.DHF(mol)
mf.init_guess = 'atom'
mf.level_shift = 0.0
mf.max_cycle = 100
mf.conv_tol=1.e-12
mf.with_gaunt = False
print(mf.scf())
print(mol.nelectron)

#==================================================================
# Dump integrals
#==================================================================
from itools import ipyscf_complex

iface = ipyscf_complex.iface()
iface.mol = mol
iface.mf = mf
iface.nfrozen = 4 # spinor
iface.ifgaunt = mf.with_gaunt
sbas = mol.nao_2c()
info = iface.get_integral4C(mf.mo_coeff[:,sbas:])
iface.dump(info,fname='./data/cmole.info')
exit(1)

#=================
# Results for H6 
#=================

# Without gaunt: -3.7119336|6035
# DIRAC		 -3.7119336|467079256	KRCI: -3.89505033
# With gaunt:    -3.2478523|7181
# DIRAC		 -3.2478523|674439819	KRCI: -3.92429101 (strange?)

#-------
#0g
#-------
#Core energy        =           4.60384171
#Active energies    =          -8.49889204         -8.37596564         -8.35617587
#Final CI energies  =          -3.89505033         -3.77212392         -3.75233416
#-------
#0u
#-------
#Core energy        =           4.60384171
#Active energies    =          -8.44057229         -8.32910670         -8.30965921
#Final CI energies  =          -3.83673058         -3.72526499         -3.70581750
#

#------------------------------------------------------
# D=80 [DC]
# >>> ic: 5
#  ieig =  0      -3.895050344024       0.000000000000
#  ieig =  1      -3.836730596251       0.000000000000
#  average =      -3.865890470138       0.000000000000 
#------------------------------------------------------
# >>> ic: 1
#  ieig =  0      -3.791624385106       0.000000000000
#  ieig =  1      -3.727988115851       0.000000000000
#  average =      -3.759806250479       0.000000000000 
#------------------------------------------------------

#------------------------------------------------------
# The convergence is not perfect !
#------------------------------------------------------
# ntype     D            E	 	nsweeps
#   2      10 	  -3.68059296657	   15
#   2      20	  -3.759251503328          30	
#   2      30     -3.791046295205          16
#------------------------------------------------------
# NEW CODES:
#   1      10 	  -3.655825671230	   6
#   1      20     -3.759251510361	   30
#------------------------------------------------------
