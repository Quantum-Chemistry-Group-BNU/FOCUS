
CSF
###

Configuration State Functions (CSFs)

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Storage
=======

CSF are stored in the following way:

Example: ud0uud (**little endian**)

Ordering: 543210 corresponds to MPS=A[0]A[1]A[2]A[3] **A[4]A[5]**

To be consistent with sweep optimization in DMRG, A[5] is set as the identity. 
Then, even with Dopt=1, after optimization the wavefunction can be multi-configurational, 
as the last two dot can accommodate several combinations:

.. code-block::

   i=0 idx=137 state=ud0uud pop=0.655 coeff=-0.81
   i=1 idx=58  state=020uud pop=0.219 coeff=-0.468
   i=2 idx=173 state=200uud pop=0.126 coeff=-0.354

From CSF to MPS
===============

CSF can be represented by a D=1 SA-MPS. The physical dimension is 3 corresponding to 
:math:`\{|N,S\rangle\}=\{|0,0\rangle,|2,0\rangle,|1,1/2\rangle\}`. 
The CSF is fully specified by the quantum numbers of all the bonds :math:`\{(N[k],S[k])\}`, 
where the difference between two adjacent bonds gives :math:`(\Delta N[k],\Delta S[k])
=(N[k]-N[k-1],S[k]-S[k-1])`.

For CSF/Det, two auxiliary arrays (narray & tsarray) can be generated, which record
the intermediate particle numbers and spins:

.. code-block::
   
   csf=000222 | 543210 => A[0]A[1]A[2]A[3]A[4]A[5]=222000
   narray= 6 4 2 0 0 0 0
   tsarray= 0 0 0 0 0 0 0
   
   
