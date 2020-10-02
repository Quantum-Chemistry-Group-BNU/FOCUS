#!/usr/bin/env python

import numpy
import libzquatev

#a = libzquatev.gen_array(2)
#b = numpy.ones(4)
#err = libzquatev.test(a,b)

n = 1000
print("n=",n)
eigvec = libzquatev.gen_array(n) # return a Hermitian matrix
print(numpy.linalg.norm(eigvec - eigvec.conj().T))

fock = eigvec.copy()
eigval = numpy.ones(2*n)

err = libzquatev.eigh(eigvec,eigval)
print("eigval=",eigval)
print("eigvec0=",eigvec[:,0])
#print numpy.dot(fock.T, eigvec[0,:]) - eigval[0]*eigvec[0,:]

#eigvec = eigvec.conj().T

v0 = eigvec[:,2]
print(v0)
print(v0.conj().dot(v0))
exit()

print("Fc0-c0*E0 Should be clouse to zero")
print(numpy.dot(fock, eigvec[:,0]) - eigval[0]*eigvec[:,0])

print("\nNumpy")
e,v = numpy.linalg.eigh(fock)
print("e=",e)
print("v0=",v[:,0])
print("Should be clouse to zero")
print(numpy.dot(fock, v[:,0]) - e[0]*v[:,0])
