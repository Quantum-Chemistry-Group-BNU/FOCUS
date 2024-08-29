
Kramers symmetry
#################

Basics
======
In Kramers symmetry-adapted relativistic DMRG, the matrix representation :math:`[a_{\bar{p}}^\dagger]` can be computed from :math:`[a_p^\dagger]`.K(1):

.. math::

   \langle l|a_{\bar{p}}^\dagger|l'\rangle=(\hat{K}\langle l|a_{\bar{p}}^\dagger|l'\rangle)^*=(\langle \bar{l}|-a_{p}^\dagger|\bar{l}'\rangle)^*=-\langle \bar{l}|a_{p}^\dagger|\bar{l}'\rangle^*

Likewise, the matrix representation of the annihilation operator 
:math:`[a_p]` is computed from :math:`[a_p^\dagger]`.H(), while
:math:`[a_{\bar{p}}]` is computed from :math:`[a_p^\dagger]`.H().K(1).
