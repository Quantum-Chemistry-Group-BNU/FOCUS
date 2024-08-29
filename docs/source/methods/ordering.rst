
Topology and Ordering
#####################

For CTNS, we can specify the topology in a file as follows:

.. code-block::

   0
   1,2,6
   4,3
   5

This gives a simple comb structure. To write down the
tensor network states, we need to be careful about the
ordering of occupation number vectors. In our settings,
the wavefunction in the right canonical form (RCF) is
written as

.. math::

   \langle n|\Psi\rangle = \sum_{\{n_i\}} \mathrm{tr}(\prod_i \mathbf{A}^{n_i})|n_0 (n_1 n_2 n_6) (n_4 n_3) n_5\rangle

The output of loading the above topology is

.. code-block::

   ctns::topology::print ifmps=0 ntotal=9 nphysical=7 nbackbone=4
   topo:
    i=0 : 0
    i=1 : -1 1 2 6
    i=2 : -1 4 3
    i=3 : 5
   rcoord:
    idx=0 coord=(3,0) rindex=0 node: lindex=6 porb=5 type=0 center=(-1,-1) left=(2,0) right=(-2,-2)
    idx=1 coord=(2,2) rindex=1 node: lindex=5 porb=3 type=0 center=(-1,-1) left=(2,1) right=(-2,-2)
    idx=2 coord=(2,1) rindex=2 node: lindex=4 porb=4 type=2 center=(-1,-1) left=(2,0) right=(2,2)
    idx=3 coord=(2,0) rindex=3 node: lindex=-1 porb=-1 type=3 center=(2,1) left=(1,0) right=(3,0)
    idx=4 coord=(1,3) rindex=4 node: lindex=3 porb=6 type=0 center=(-1,-1) left=(1,2) right=(-2,-2)
    idx=5 coord=(1,2) rindex=5 node: lindex=2 porb=2 type=2 center=(-1,-1) left=(1,1) right=(1,3)
    idx=6 coord=(1,1) rindex=6 node: lindex=1 porb=1 type=2 center=(-1,-1) left=(1,0) right=(1,2)
    idx=7 coord=(1,0) rindex=7 node: lindex=-1 porb=-1 type=3 center=(1,1) left=(0,0) right=(2,0)
    idx=8 coord=(0,0) rindex=8 node: lindex=0 porb=0 type=0 center=(-1,-1) left=(-2,-2) right=(1,0)
   rsupport/lsupport:
    idx=0 coord=(3,0) rsupport: 5 ; lsupport: 0 1 2 3 4 6
    idx=1 coord=(2,2) rsupport: 3 ; lsupport: 0 1 2 4 5 6
    idx=2 coord=(2,1) rsupport: 4 3 ; lsupport: 0 1 2 5 6
    idx=3 coord=(2,0) rsupport: 4 3 5 ; lsupport: 0 1 2 6
    idx=4 coord=(1,3) rsupport: 6 ; lsupport: 0 1 2 3 4 5
    idx=5 coord=(1,2) rsupport: 2 6 ; lsupport: 0 1 3 4 5
    idx=6 coord=(1,1) rsupport: 1 2 6 ; lsupport: 0 3 4 5
    idx=7 coord=(1,0) rsupport: 1 2 6 4 3 5 ; lsupport: 0
    idx=8 coord=(0,0) rsupport: 0 1 2 6 4 3 5 ; lsupport:
   corbs/rorbs/lorbs:
    idx=0 coord=(3,0) corbs: 5 ; rorbs: ; lorbs: 0 1 2 3 4 6
    idx=1 coord=(2,2) corbs: 3 ; rorbs: ; lorbs: 0 1 2 4 5 6
    idx=2 coord=(2,1) corbs: 4 ; rorbs: 3 ; lorbs: 0 1 2 5 6
    idx=3 coord=(2,0) corbs: 4 3 ; rorbs: 5 ; lorbs: 0 1 2 6
    idx=4 coord=(1,3) corbs: 6 ; rorbs: ; lorbs: 0 1 2 3 4 5
    idx=5 coord=(1,2) corbs: 2 ; rorbs: 6 ; lorbs: 0 1 3 4 5
    idx=6 coord=(1,1) corbs: 1 ; rorbs: 2 6 ; lorbs: 0 3 4 5
    idx=7 coord=(1,0) corbs: 1 2 6 ; rorbs: 4 3 5 ; lorbs: 0
    idx=8 coord=(0,0) corbs: 0 ; rorbs: 1 2 6 4 3 5 ; lorbs:
   image2:
    0 1 2 3 4 5 12 13 8 9 6 7 10 11

Note that ``image2`` corresponds to the spin-orbital form of :math:`|n_0 (n_1 n_2 n_6) (n_4 n_3) n_5\rangle`,
which is consistent with ``porb``. The position of the **physical node** (``porb``!=-1) in the occupation number vector is recorded
in ``lindex``.

.. note::
   We should distingush the **mathematical form** :math:`|n_0 (n_1 n_2 n_6) (n_4 n_3) n_5\rangle` from its **output form**
   obtained in sampling, which will be "n5n3n4n6n2n1n0" (little endian).

.. note::

   This ordering :math:`|n_0 (n_1 n_2 n_6) (n_4 n_3) n_5\rangle` is changing during the sweep optimization of CTNS.
   For instatnce, it can change from :math:`|lcr\rangle=|[n_0][(n_1n_2n_6)][(n_4n_3)n_5]\rangle` to :math:`|[n_0(n_4n_3)n_5][n_1][n_2n_6]\rangle` to visit the first branch of the above comb structure. 
   This is different from the case in MPS, where the underlying ordering is not changed
   during the sweep.
