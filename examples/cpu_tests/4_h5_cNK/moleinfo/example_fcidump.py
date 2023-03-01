import sys
sys.path.append("..")
from itools import ipyscf_real

iface = ipyscf_real.iface()
info = iface.get_integral_FCIDUMP(fname="FCIDUMP")
iface.dump(info,fname='fmole.info')
