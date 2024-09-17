
Reduced density matrices
########################

Reduced density matrices (RDMs) and transition density matrices (TDMs) in DMRG are formed by assembling normal operators in onedot algorithm. The necessary type patterns for up to two-particle density matrices are as follows:

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Definitions
===========

The 1,2-RDM are defined as follows:

.. math::

   \gamma_{pq}^{IJ} = \langle\Psi_I|p^\dagger q|\Psi_J\rangle
   
   \Gamma_{pq,rs}^{IJ} = \langle\Psi_I|p^\dagger q^\dagger s r |\Psi_J\rangle

The 2-RDM is also storaged as a matrix by restricting :math:`p>q` and :math:`r>s`.
Together, they can be used to compute measures of correlation such as the
trace or the norm of the cumulants

.. math::

   \Delta_{pq,rs} = \Gamma_{pq,rs} - \gamma_{pr}\gamma_{qs} + \gamma_{ps}\gamma_{qr}
   
For determinants, :math:`\Delta_{pq,rs}=0` such that :math:`\mathrm{tr}(\Delta)=0` and :math:`\|\Delta\|_F=0`.
Note that :math:`\mathrm{tr}(\Delta)=0` is determined by 1-RDM:

.. math::

   \mathrm{tr}(\Delta)=\sum_{pq}\Delta_{pq,pq}=(N-1)N-N^2+\mathrm{tr}(\gamma^2) = \mathrm{tr}(\gamma^2) - N
   = - \mathrm{tr}[\gamma(1-\gamma)] \le 0

Patterns
========

1-RDM
-----

.. code-block::

   ctns::display_patterns name=tpatterns size=2
    i=0 020:|+-| hermi=1
    i=1 110:+|-| hermi=0
   ctns::display_patterns name=fpatterns size=1
    i=0 200:+-|| hermi=1
   ctns::display_patterns name=lpatterns size=3
    i=0 101:+||- hermi=0
    i=1 011:|+|- hermi=0
    i=2 002:||+- hermi=1

1-TDM
-----

.. code-block::

   ctns::rdm_sweep order=1 ifab=1 alg_rdm=0 alg_renorm=0 mpsize=1 maxthreads=1
   ctns::display_patterns name=tpatterns size=3
    i=0 020:|+-| hermi=1
    i=1 110:-|+| hermi=0
    i=2 110:+|-| hermi=0
   ctns::display_patterns name=fpatterns size=1
    i=0 200:+-|| hermi=1
   ctns::display_patterns name=lpatterns size=5
    i=0 101:-||+ hermi=0
    i=1 101:+||- hermi=0
    i=2 011:|-|+ hermi=0
    i=3 011:|+|- hermi=0
    i=4 002:||+- hermi=1

2-RDM
-----

.. code-block::

   ctns::display_patterns name=tpatterns size=10
    i=0 040:|++--| hermi=1
    i=1 031:|++-|- hermi=0
    i=2 130:+|+--| hermi=0
    i=3 121:+|+-|- hermi=0
    i=4 121:+|--|+ hermi=0
    i=5 220:++|--| hermi=0
    i=6 220:+-|+-| hermi=1
    i=7 211:++|-|- hermi=0
    i=8 211:-+|+|- hermi=0
    i=9 211:+-|+|- hermi=0
   ctns::display_patterns name=fpatterns size=3
    i=0 301:++-||- hermi=0
    i=1 310:++-|-| hermi=0
    i=2 400:++--|| hermi=1
   ctns::display_patterns name=lpatterns size=9
    i=0 202:++||-- hermi=0
    i=1 202:+-||+- hermi=1
    i=2 112:+|+|-- hermi=0
    i=3 112:+|-|+- hermi=0
    i=4 022:|++|-- hermi=0
    i=5 022:|+-|+- hermi=1
    i=6 103:+||+-- hermi=0
    i=7 013:|+|+-- hermi=0
    i=8 004:||++-- hermi=1

2-TDM
-----

.. code-block::

   ctns::display_patterns name=tpatterns size=16
    i=0 040:|++--| hermi=1
    i=1 031:|+--|+ hermi=0
    i=2 031:|++-|- hermi=0
    i=3 130:-|++-| hermi=0
    i=4 130:+|+--| hermi=0
    i=5 121:-|+-|+ hermi=0
    i=6 121:-|++|- hermi=0
    i=7 121:+|--|+ hermi=0
    i=8 121:+|+-|- hermi=0
    i=9 220:--|++| hermi=0
    i=10 220:+-|+-| hermi=1
    i=11 220:++|--| hermi=0
    i=12 211:--|+|+ hermi=0
    i=13 211:+-|-|+ hermi=0
    i=14 211:+-|+|- hermi=0
    i=15 211:++|-|- hermi=0
   ctns::display_patterns name=fpatterns size=5
    i=0 301:+--||+ hermi=0
    i=1 301:++-||- hermi=0
    i=2 310:+--|+| hermi=0
    i=3 310:++-|-| hermi=0
    i=4 400:++--|| hermi=1
   ctns::display_patterns name=lpatterns size=15
    i=0 202:--||++ hermi=0
    i=1 202:+-||+- hermi=1
    i=2 202:++||-- hermi=0
    i=3 112:-|-|++ hermi=0
    i=4 112:-|+|+- hermi=0
    i=5 112:+|-|+- hermi=0
    i=6 112:+|+|-- hermi=0
    i=7 022:|--|++ hermi=0
    i=8 022:|+-|+- hermi=1
    i=9 022:|++|-- hermi=0
    i=10 103:-||++- hermi=0
    i=11 103:+||+-- hermi=0
    i=12 013:|-|++- hermi=0
    i=13 013:|+|+-- hermi=0
    i=14 004:||++-- hermi=1
  
Computational cost
==================

The computational cost for 2-RDMs comes from three parts:

1. Prepare right environment via ``rdm_renom("cr",...)``: :math:`O(K^2D^3)`

2. Assemble RDMs by patterns: :math:`O(K^2D^3+K^4D^2)` dominated by patterns 2|1|1

.. code-block::

    i=0 pattern=040:|++--| opkey=I0:F0:I0 sizes=1:1:1 TIMING=7.9e-05 S
    i=1 pattern=031:|++-|- opkey=I0:T1:C1 sizes=1:2:76 TIMING=0.00614 S
    i=2 pattern=130:+|+--| opkey=C0:T0:I0 sizes=74:2:1 TIMING=0.000187 S
    i=3 pattern=121:+|+-|- opkey=C0:B0:C1 sizes=74:4:76 TIMING=0.0199 S
    i=4 pattern=121:+|--|+ opkey=C0:A1:C0 sizes=74:1:76 TIMING=0.00609 S
    i=5 pattern=220:++|--| opkey=A0:A1:I0 sizes=2701:1:1 TIMING=0.00153 S
    i=6 pattern=220:+-|+-| opkey=B0:B0:I0 sizes=2775:4:1 TIMING=0.0057 S
    i=7 pattern=211:++|-|- opkey=A0:C1:C1 sizes=2701:2:76 TIMING=0.208 S
    i=8 pattern=211:-+|+|- opkey=B1:C0:C1 sizes=2775:2:76 TIMING=0.646 S
    i=9 pattern=211:+-|+|- opkey=B0:C0:C1 sizes=2775:2:76 TIMING=0.218 S

3. Left renormalization via ``rdm_renorm("lc",...)``: :math:`O(K^3D^3)` 

