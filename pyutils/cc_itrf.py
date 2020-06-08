import numpy
from cc_ci import ccsd

class CC:
   def __init__(self,nb,ne,h1e,h2e):
      self.nb = nb
      self.ne = ne
      self.h1e = h1e.copy()
      # <ij||kl> = [ik|jl]-[il|jk]
      self.h2e = numpy.einsum('ikjl->ijkl',h2e)\
               - numpy.einsum('iljk->ijkl',h2e)
      # use cc 
      self.mcc = ccsd.CC(self.nb,self.ne,self.h1e,self.h2e)
      self.mcc.max_cycle = 100

   def cisd(self):
      print "\nCC.cisd"
      self.mcc.cisd()
      self.emp2 = self.mcc.emp2
      return self.mcc.ecisd

   def ccsd(self):
      print "\nCC.ccsd"
      self.mcc.ccsd()
      self.emp2 = self.mcc.emp2
      return self.mcc.eccsd

   def pt(self):
      print "\nCC.pt"
      self.mcc.pt()
      return self.mcc.ept


if __name__ == '__main__':
  
   import tools_io

   mol = "fe4s4"

   if mol == "fe2s2":
      ehf = -116.51269169894704
      nelec = 30
   elif mol == "fe4s4":
      ehf = -327.0619022451
      nelec = 54
   # load new integrals   
   e,h1e,h2e = tools_io.load_FCIDUMP("FCIDUMP_"+mol)
   k = h1e.shape[0]

   mycc = CC(k,nelec,h1e,h2e)
   ecisd = mycc.cisd()
   eccsd = mycc.ccsd()
   ept = mycc.pt()
   print "\nEhf=",ehf
   print "Ecisd=",ehf+ecisd
   print "Eccsd=",ehf+eccsd
   print "Eccsd(t)=",ehf+eccsd+ept

