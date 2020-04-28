import numpy

cA 	= numpy.array(
[[ 0.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 1.,  0.,  0.,  0.],
 [ 0.,  1.,  0.,  0.]] )

cB 	= numpy.array(
[[ 0.,  0.,  0.,  0.],
 [ 1.,  0.,  0.,  0.],
 [ 0.,  0.,  0.,  0.],
 [ 0.,  0., -1.,  0.]] )

c = [cA,cB]
a = [cA.T,cB.T]

print "c[0]                              \n",c[0]
print "c[1]                              \n",c[1]
print
print "a[0]                              \n",a[0]
print "a[1]                              \n",a[1]
print
print "c[0].dot(c[1])                    \n",c[0].dot(c[1])
print
print "c[0].dot(a[0])                    \n",c[0].dot(a[0])
print "c[0].dot(a[1])                    \n",c[0].dot(a[1])
print "c[1].dot(a[0])                    \n",c[1].dot(a[0])
print "c[1].dot(a[1])                    \n",c[1].dot(a[1])
print
print "c[0].dot(a[1].dot(a[0]))          \n",c[0].dot(a[1].dot(a[0]))
print "c[1].dot(a[1].dot(a[0]))          \n",c[1].dot(a[1].dot(a[0]))
print
print "c[0].dot(c[1].dot(a[1].dot(a[0])))\n",c[0].dot(c[1].dot(a[1].dot(a[0])))
