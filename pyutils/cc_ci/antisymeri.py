import numpy
from pyscf import gto

# (ab|cd) - spherical
def fill4(mol,intor='cint2e_sph'):
   nao = mol.nao_nr()
   eri = numpy.empty([nao]*4)
   atm = numpy.array(mol._atm, dtype=numpy.int32)
   bas = numpy.array(mol._bas, dtype=numpy.int32)
   env = numpy.array(mol._env)
   pi = 0
   for i in range(mol.nbas):
       pj = 0
       for j in range(mol.nbas):
           pk = 0
           for k in range(mol.nbas):
               pl = 0
               for l in range(mol.nbas):
                   shls = (i,j,k,l)
                   buf = gto.moleintor.getints_by_shell(intor, shls, atm, bas, env)
                   di, dj, dk, dl = buf.shape
                   eri[pi:pi+di,pj:pj+dj,pk:pk+dk,pl:pl+dl] = buf
                   pl += dl
               pk += dk
           pj += dj
       pi += di
   return eri


def rhf_spinorb(mf,storage=True,h1e=None):
   print "\n*** Interface for RHF-Spin-orbital Integrals ***"
   mol = mf.mol
   mo_energy = mf.mo_energy
   mo_occ = mf.mo_occ
   # RHF interface
   nelec= mol.nelectron
   ecor = 0.0
   enuc=mol.energy_nuc()
   mo_coeff = mf.mo_coeff
   nbas = mo_coeff.shape[0]
   nb   = 2*nbas
   
   escf=mf.energy_elec(mf.make_rdm1())[0]
   etot=mf.energy_tot(mf.make_rdm1())
   print "E_nuc = ",mol.energy_nuc()
   print "E_ele = ",mf.energy_elec(mf.make_rdm1())[0]
   print "E_scf = ",mf.energy_tot(mf.make_rdm1())
   print "Nelec = ",mol.nelectron
   print "Nsorb = ",nb
   
   # print mo_coeff
   mo_coeffT= mo_coeff.T.copy()
   s_coeffT = numpy.zeros((2*nbas,2*nbas))
   for i in range(2*nbas):
      # alpha
      if i%2 == 0:
         s_coeffT[i][:nbas]=mo_coeffT[i//2]
      else:
         s_coeffT[i][nbas:]=mo_coeffT[i//2]
   # each col is an eigenvector
   s_coeff=s_coeffT.T
   
   # SPECIAL FORM of MO for TRANS
   b = numpy.zeros((nbas,2*nbas))
   b[:, ::2] = mo_coeff.copy()
   b[:,1::2] = mo_coeff.copy()
  
   # INT1e
   if h1e == None:
      h = mf.get_hcore()
   else:
      #model 	   
      h=h1e.copy() 
   hmo=reduce(numpy.dot,(b.T,h,b))
   hmo[::2,1::2]=hmo[1::2,::2]=0.

   # INT2e:
   from pyscf import ao2mo
   eri = ao2mo.outcore.general_iofree(mol,(b,b,b,b),compact=0).reshape(nb,nb,nb,nb)
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   #
   # <ij||kl>=[ik|jl]-[il|jk]
   #          [12|34]-[14|32] => <13|24> => <12||34>
   #          (01|23)-(03|21) => (02|13)
   #
   eriA=eri-eri.transpose(0,3,2,1)
   eriA=eriA.transpose(0,2,1,3).copy()
 
   if storage:
      # HDF5 store
      import h5py
      f = h5py.File("mole.h5", "w")
      cal = f.create_dataset("cal",(1,),dtype='i')
      cal.attrs["nelec"]=nelec
      cal.attrs["nspin"]=mol.spin
      cal.attrs["nb"]   =nb
      cal.attrs["enuc"]=enuc
      cal.attrs["escf"]=escf
      cal.attrs["ecor"]=ecor
      coeff = f.create_dataset("coeff", data=s_coeff)
      int1e = f.create_dataset("int1e", data=hmo)
      int2e = f.create_dataset("int2e", data=eriA)
      f.close()

      # Test
      f2 = h5py.File("mole.h5", "r")
      print "\nCHECK STORAGE:"
      print "nelec=",f2['cal'].attrs['nelec']
      print "nspin=",f2['cal'].attrs['nspin']
      print "nb   =",f2['cal'].attrs['nb']
      print "enuc =",f2['cal'].attrs['enuc']
      print "escf =",f2['cal'].attrs['escf']
      print "ecor =",f2['cal'].attrs["ecor"]
      print f2['coeff']
      print f2['int1e']
      print f2['int2e']
      f2.close()
 
   else: 
      return nb,nelec,hmo,eriA


def uhf_spinorb(mf,storage=True,h1e=None):
   print "\n*** Interface for UHF-Spin-orbital Integrals ***"
   mol = mf.mol
   mo_energy = mf.mo_energy
   mo_occ = mf.mo_occ
   nelec= mol.nelectron
   ecor = 0.0
   enuc=mol.energy_nuc()
   # UHF case => tuple
   mo_coeff = mf.mo_coeff
   nbas = mo_coeff[0].shape[0]
   nb   = 2*nbas
   
   escf=mf.energy_elec(mf.make_rdm1())[0]
   etot=mf.energy_tot(mf.make_rdm1())
   print "E_nuc = ",mol.energy_nuc()
   print "E_ele = ",mf.energy_elec(mf.make_rdm1())[0]
   print "E_scf = ",mf.energy_tot(mf.make_rdm1())
   print "Nelec = ",mol.nelectron
   print "Nsorb = ",nb
   
   # print mo_coeff
   mo_coeffTa = mo_coeff[0].T.copy()
   mo_coeffTb = mo_coeff[1].T.copy()
   s_coeffT = numpy.zeros((2*nbas,2*nbas))
   for i in range(2*nbas):
      # alpha
      if i%2 == 0:
         s_coeffT[i][:nbas]=mo_coeffTa[i//2]
      else:
         s_coeffT[i][nbas:]=mo_coeffTb[i//2]
   # each col is an eigenvector
   s_coeff=s_coeffT.T
   
   # SPECIAL FORM of MO for TRANS
   b = numpy.zeros((nbas,2*nbas))
   b[:, ::2] = mo_coeff[0].copy()
   b[:,1::2] = mo_coeff[1].copy()
  
   # INT1e
   if h1e == None:
      h = mf.get_hcore()
   else:
      #model 	   
      h=h1e.copy() 
   hmo=reduce(numpy.dot,(b.T,h,b))
   hmo[::2,1::2]=hmo[1::2,::2]=0.

   # INT2e:
   from pyscf import ao2mo
   eri = ao2mo.outcore.general_iofree(mol,(b,b,b,b),compact=0).reshape(nb,nb,nb,nb)
   eri[::2,1::2]=eri[1::2,::2]=eri[:,:,::2,1::2]=eri[:,:,1::2,::2]=0.
   #
   # <ij||kl>=[ik|jl]-[il|jk]
   #          [12|34]-[14|32] => <13|24> => <12||34>
   #          (01|23)-(03|21) => (02|13)
   #
   eriA=eri-eri.transpose(0,3,2,1)
   eriA=eriA.transpose(0,2,1,3).copy()
 
   if storage:
      # HDF5 store
      import h5py
      f = h5py.File("mole.h5", "w")
      cal = f.create_dataset("cal",(1,),dtype='i')
      cal.attrs["nelec"]=nelec
      cal.attrs["nspin"]=mol.spin
      cal.attrs["nb"]   =nb
      cal.attrs["enuc"]=enuc
      cal.attrs["escf"]=escf
      cal.attrs["ecor"]=ecor
      coeff = f.create_dataset("coeff", data=s_coeff)
      int1e = f.create_dataset("int1e", data=hmo)
      int2e = f.create_dataset("int2e", data=eriA)
      f.close()

      # Test
      f2 = h5py.File("mole.h5", "r")
      print "\nCHECK STORAGE:"
      print "nelec=",f2['cal'].attrs['nelec']
      print "nspin=",f2['cal'].attrs['nspin']
      print "nb   =",f2['cal'].attrs['nb']
      print "enuc =",f2['cal'].attrs['enuc']
      print "escf =",f2['cal'].attrs['escf']
      print "ecor =",f2['cal'].attrs["ecor"]
      print f2['coeff']
      print f2['int1e']
      print f2['int2e']
      f2.close()
 
   else: 
      return nb,nelec,hmo,eriA


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf import cc

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = [['H',(0.,0.,0.)],
		['F',(1.,0.,0.)]]
    mol.basis={'H':'6-31g',
	       'F':'6-31g'}
    mol.build()

    rhf = scf.RHF(mol)
    rhf.conv_tol=1.e-14
    rhf.scf() # -76.0267656731
    nb,ne,h1e,int2e=rhf_spinorb(rhf,storage=False)

    uhf = scf.UHF(mol)
    uhf.conv_tol=1.e-14
    uhf.scf() # -76.0267656731
    nb,ne,h1e,int2e=uhf_spinorb(uhf,storage=False)

