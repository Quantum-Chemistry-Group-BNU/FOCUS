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
molname = 'h' 

if molname == 'ne':
   R = 1.e0
   natoms = 1 #50 #20 #,14,20,50
   mol.atom = [['Ne',(0.0,0.0,i*R)] for i in range(natoms)]
   mol.basis = 'dzp'

elif molname == 'h':
   R = 1.0
   natoms = 6 #40 #,14,20,50
   mol.atom = [['H',(0.0,0.0,i*R)] for i in range(natoms)]
   #mol.atom = [['H',(0.0,0.0,0.0)],\
 	#       ['H',(1.0,0.0,0.0)],\
 	#       ['H',(1.1,1.2,1.3)],\
 	#       ['H',(0.0,0.0,1.1)],\
 	#       ['H',(1.0,1.0,0.0)],\
 	#       ['H',(0.0,1.0,1.2)]]
   mol.basis = 'sto-3g' #cc-pvdz' #sto-3g' #6-31g' 

elif molname == 'hf':
   R=3.4
   mol.atom = [['H',(0,0,0)],
   	       ['F',(0,0,R)]]		
   mol.basis = '6-31g' #sto-3g' #cc-pvdz'#sto-3G'

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
mol.charge = 0 #-1 
mol.spin = 0
# Artificial world
mol.light_speed = 1 
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
mf.with_gaunt = True
print(mf.scf())

#==================================================================
# Dump integrals
#==================================================================
from zmpo_dmrg.source.itools import ipyscf_complex 
iface = ipyscf_complex.iface(mol,mf)

iface.iflocal = False
iface.iflowdin = False
iface.ifreorder = False
iface.ifgaunt = mf.with_gaunt

iface.nelec = 5
iface.ifHFtest = False
# For Ne:
#iface.core = [0,1,2,3]
#iface.act  = [4,5,6,7,8,9]+[10,11,12,13,14,15]
iface.dump4C()

#=================
# Results for H6+ 
#=================

#################################################################
# Remark:
#################################################################
# In closed-shell case, we can precisely control Sz value by 
# controling the number of alpha and beta electrons in forming H 
# in the direct product space. But in the relativistic case, one 
# need to do some additional effect to ensure that the Kramers 
# pairs are both included. Otherwise, the energies will not be 
# degenerate for small D values!!!
#################################################################

#------------------------------------------------------
# D=10
#------------------------------------------------------
#  ieig =  0      -3.182826719200       0.000000000000
#  ieig =  1      -3.140862110786       0.000000000000
#  average =      -3.161844414993       0.000000000000 
#------------------------------------------------------
# D=20
#------------------------------------------------------
#  ieig =  0      -3.221399445068       0.000000000000
#  ieig =  1      -3.197123257506       0.000000000000
#  average =      -3.209261351287       0.000000000000 
#------------------------------------------------------
# D=30
#------------------------------------------------------
#  ieig =  0      -3.234050221976       0.000000000000
#  ieig =  1      -3.207861921777       0.000000000000
#  average =      -3.220956071877       0.000000000000 
#------------------------------------------------------
# D=40
#------------------------------------------------------
#  ieig =  0      -3.234069991699       0.000000000000
#  ieig =  1      -3.234069991668       0.000000000000
#  average =      -3.234069991683       0.000000000000 
#------------------------------------------------------
# D=50
#------------------------------------------------------
#  ieig =  0      -3.234069991678       0.000000000000
#  ieig =  1      -3.234069991618       0.000000000000
#  average =      -3.234069991648       0.000000000000 
#------------------------------------------------------
