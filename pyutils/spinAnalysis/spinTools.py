import numpy

#########################################################
# <Sz[local]> 
# G[i,j,k,l] =<a_is1^+*a_js2^+*a_ks2*a_ls1>
#########################################################
def szHS(onepdm,twopdm,n,s):
   spdm = onepdm*(2.0-0.5*n)-numpy.einsum('ikjk',twopdm)
   spdm = 0.5*spdm/(s+1.) # 0.5 for Sz
   return spdm

#########################################################
# <Si*Sj> = 0.5*(<(Si+Sj)^2>-<Si^2>-<Sj^2>)
#########################################################
def sisj(onepdm,twopdm,na,nb):
   bas1 = range(na)
   bas2 = range(na,na+nb)
   rdm1i = onepdm[numpy.ix_(bas1,bas1)]
   rdm2i = twopdm[numpy.ix_(bas1,bas1,bas1,bas1)]
   rdm1j = onepdm[numpy.ix_(bas2,bas2)]
   rdm2j = twopdm[numpy.ix_(bas2,bas2,bas2,bas2)]
   expi = local_spinsquare(rdm1i,rdm2i)
   expj = local_spinsquare(rdm1j,rdm2j)
   expij = local_spinsquare(onepdm,twopdm)
   sij = 0.5*(expij-expi-expj)
   return sij

#########################################################
# Local expectation value <S^2>
#########################################################
def local_spin(onepdm, twopdm):
    return from_spinsquare_to_spin(local_spinsquare(onepdm,twopdm))

def local_spinsquare(onepdm, twopdm):
    sum = 0.0
    sum += -0.5* numpy.einsum('ijij',twopdm)
    sum += -0.25* numpy.einsum('ijji',twopdm)
    sum += 0.75* numpy.einsum('ii',onepdm)
    return sum

def from_spinsquare_to_spin(spinsquare):
    #from S(S+1) to S
    from math import sqrt
    return sqrt(spinsquare+0.25) - 0.5
