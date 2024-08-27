
CI
##

Configuration Interaction (CI) algorithm

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2


Representation of dets
======================

In FOCUS, we adopted the following convention for occupation number vectors (little endian):

.. math::
   |n_5n_4n_3n_2n_1n_0\rangle \triangleq a_0^{n_0} a_1^{n_1} a_2^{n_2} a_3^{n_3}a_4^{n_4} a_5^{n_5} |vac\rangle

.. note::
   However, determinant basis used in string CI algorithm (used in most of packages for FCI) is usually defined as follows:

   .. math::
      |I_\alpha I_\beta\rangle \triangleq (a_0^{n_0} a_2^{n_2} a_4^{n_4}) (a_1^{n_1} a_3^{n_3} a_5^{n_5}) |vac\rangle

   The sign factor is :math:`(-1)^{n_1(n_2+n_4)+n_3n_4}=(-1)^{n_2n_1+n_4(n_1+n_3)}`.
   The former form corresponds to moving :math:`a_1` and :math:`a_3` into :math:`I_\alpha`,
   while the latter corresponds to moving :math:`a_4` and :math:`a_2` into
   :math:`I_\beta`. In general, it is given by

   .. math::
      (-1)^{\sum_{i=0}^{K-2} n_{2i+1}\sum_{j=i+1}^{K-1} n_{2j}} =(-1)^{\sum_{i=1}^{K-1}n_{2i}\sum_{j=0}^{i-1}n_{2j+1}}

   Thus, to compare the coefficients with string CI algorithm implemented in other programs. One need to include this additional factors.

