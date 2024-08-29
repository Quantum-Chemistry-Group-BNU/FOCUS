
CI
##

Configuration Interaction (CI) algorithm

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2


Representation of dets
======================

In FOCUS, we adopted the following convention for occupation number vectors (ONVs).

.. math::
   |n_0n_1n_2n_3n_4n_5\rangle \triangleq a_0^{n_0} a_1^{n_1} a_2^{n_2} a_3^{n_3}a_4^{n_4} a_5^{n_5} |vac\rangle

It is internally stored and output as "n5n4n3n2n1n0" (little endian). This also means,
when one want to input the above state, the string "n5n4n3n2n1n0" needs to be used
for the construction of ``onstate``. For example,

.. code-block::

   fock::onstate state("2010ab",1);

The second argument 1 for specifying spatial orbital configuration.

.. note::
   Determinant basis used in string CI algorithm (used in most of packages for FCI) is usually defined as follows:

   .. math::
      |I_\alpha I_\beta\rangle \triangleq (a_0^{n_0} a_2^{n_2} a_4^{n_4}) (a_1^{n_1} a_3^{n_3} a_5^{n_5}) |vac\rangle

   The sign factor is :math:`(-1)^{n_1(n_2+n_4)+n_3n_4}=(-1)^{n_2n_1+n_4(n_1+n_3)}` to convert
   the ONV defined in our case to this convention.
   The former form corresponds to moving :math:`a_1` and :math:`a_3` into :math:`I_\alpha`,
   while the latter corresponds to moving :math:`a_4` and :math:`a_2` into
   :math:`I_\beta`. In general, it is given by

   .. math::
      (-1)^{\sum_{i=0}^{K-2} n_{2i+1}\sum_{j=i+1}^{K-1} n_{2j}} =(-1)^{\sum_{i=1}^{K-1}n_{2i}\sum_{j=0}^{i-1}n_{2j+1}}

   Thus, to compare the coefficients with string CI algorithm implemented in other programs. One need to include this additional factors.

