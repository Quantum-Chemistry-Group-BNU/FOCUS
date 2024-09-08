
Reduced density matrix
######################

.. contents:: Table of contents
   :local:
   :backlinks: entry
   :depth: 2

Basics
======
Reduced density matrices (RDMs) and transition density matrices (TDMs) in DMRG are formed by assembling normal operators in onedot algorithm. The necessary type patterns for up to two-particle density matrices are as follows:

1-RDM
-----

.. code-block::

   ctns::display_patterns name=tpatterns size=2
    i=0 020:|+-|
    i=1 110:+|-|
   ctns::display_patterns name=fpatterns size=1
    i=0 200:+-||
   ctns::display_patterns name=lpatterns size=3
    i=0 101:+||-
    i=1 011:|+|-
    i=2 002:||+-

1-TDM
-----

.. code-block::

   ctns::display_patterns name=tpatterns size=3
    i=0 020:|+-|
    i=1 110:-|+|
    i=2 110:+|-|
   ctns::display_patterns name=fpatterns size=1
    i=0 200:+-||
   ctns::display_patterns name=lpatterns size=5
    i=0 101:-||+
    i=1 101:+||-
    i=2 011:|-|+
    i=3 011:|+|-
    i=4 002:||+-

2-RDM
-----

.. code-block::

   ctns::display_patterns name=tpatterns size=11
    i=0 040:|++--|
    i=1 031:|++-|-
    i=2 130:+|+--|
    i=3 121:+|+-|-
    i=4 121:+|--|+
    i=5 220:++|--|
    i=6 220:-+|+-|
    i=7 220:+-|+-|
    i=8 211:++|-|-
    i=9 211:-+|+|-
    i=10 211:+-|+|-
   ctns::display_patterns name=fpatterns size=3
    i=0 301:++-||-
    i=1 310:++-|-|
    i=2 400:++--||
   ctns::display_patterns name=lpatterns size=10
    i=0 202:++||--
    i=1 202:-+||+-
    i=2 202:+-||+-
    i=3 112:+|+|--
    i=4 112:+|-|+-
    i=5 022:|++|--
    i=6 022:|+-|+-
    i=7 103:+||+--
    i=8 013:|+|+--
    i=9 004:||++--

2-TDM
-----

.. code-block::

   ctns::display_patterns name=tpatterns size=16
    i=0 040:|++--|
    i=1 031:|+--|+
    i=2 031:|++-|-
    i=3 130:-|++-|
    i=4 130:+|+--|
    i=5 121:-|+-|+
    i=6 121:-|++|-
    i=7 121:+|--|+
    i=8 121:+|+-|-
    i=9 220:--|++|
    i=10 220:+-|+-|
    i=11 220:++|--|
    i=12 211:--|+|+
    i=13 211:+-|-|+
    i=14 211:+-|+|-
    i=15 211:++|-|-
   ctns::display_patterns name=fpatterns size=5
    i=0 301:+--||+
    i=1 301:++-||-
    i=2 310:+--|+|
    i=3 310:++-|-|
    i=4 400:++--||
   ctns::display_patterns name=lpatterns size=15
    i=0 202:--||++
    i=1 202:+-||+-
    i=2 202:++||--
    i=3 112:-|-|++
    i=4 112:-|+|+-
    i=5 112:+|-|+-
    i=6 112:+|+|--
    i=7 022:|--|++
    i=8 022:|+-|+-
    i=9 022:|++|--
    i=10 103:-||++-
    i=11 103:+||+--
    i=12 013:|-|++-
    i=13 013:|+|+--
    i=14 004:||++--
   
