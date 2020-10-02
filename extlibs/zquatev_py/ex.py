#!/usr/bin/env python

import numpy
import libzquatev
import zquatev

#a = libzquatev.gen_array(2)
#b = numpy.ones(4)
#err = libzquatev.test(a,b)
#exit(1)

n = 2 #1000
print("n=",n)
fock = libzquatev.gen_array(n) # return a Hermitian matrix
print(numpy.linalg.norm(fock - fock.conj().T))
print("F=\n",fock)

print("\nzquatev")
e,v = zquatev.eigh(fock,iop=1)
print("e=",e)
print("v=",v)
print("Fc0-c0*E0 Should be clouse to zero", e[0])
print(numpy.dot(fock, v[:,0]) - e[0]*v[:,0])
print("v0=",v[0,:])
print(numpy.isfortran(fock))
print(fock.flags)

print("\nnumpy")
e,v = numpy.linalg.eigh(fock)
print("e=",e.shape,e)
print("v0=",v[:,0])
print("v=\n",v)
print("Should be clouse to zero", e[0])
print(numpy.dot(fock, v[:,0]) - e[0]*v[:,0])
