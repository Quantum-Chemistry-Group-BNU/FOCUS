import numpy
import scipy.linalg
import tools_io
import ghf

mol = "fe4s4"

if mol == "fe2s2":
   integrals = "../../database/benchmark/fes/fe2s2/FCIDUMP"
   det = "0 2 4 6 8 10 12 14 16 18 20 22 24 36 38   1 3 15 17 19 21 23 25 27 29 31 33 35 37 39"
   nelec = 30
elif mol == "fe4s4":
   integrals = "../../database/benchmark/fes/fe4s4/FCIDUMP"
   det = "0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 68 70   1 3 5 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 65 67 69 71"
   nelec = 54
elif mol == "fe8s7":
   integrals = "../../database/benchmark/fes/fe8s7/FCIDUMP" 
   det = "0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 40 42 44 46 56 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98 100 102 104 106 108 110 112 114 116 118 120 122 124 126 128 138 140 142 144   1 3 5 7 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 65 67 69 71 73 75 77 79 81 91 101 103 105 107 109 111 113 115 117 119 129 131 133 135 137 139 141 143 145"
   nelec = 114

e,h1,h2 = tools_io.load_FCIDUMP(integrals)
h1e,h2e = tools_io.to_SpinOrbitalERI(h1,h2)

mf = ghf.ghf()
mf.nelec = nelec
mf.ints = (e,h1e,h2e)
mf.solve(det)
mf.trans(mol)

e,h1,h2 = tools_io.load_FCIDUMP("FCIDUMP_"+mol)
mf.ints = (e,h1,h2)
mf.energy_det(mf.det)

import cc_itrf
k = h1.shape[0]
mycc = cc_itrf.CC(k,nelec,h1,h2)
ecisd = mycc.cisd()
eccsd = mycc.ccsd()
ept = mycc.pt()
print "\nEhf=",ehf
print "Ecisd=",ehf+ecisd
print "Eccsd=",ehf+eccsd
print "Eccsd(t)=",ehf+eccsd+ept
