import sys
sys.path.append("..")
import ipyscf_real

iface = ipyscf_real.iface()
info = iface.get_integral_FCIDUMP(fname="FCIDUMP_c2")
iface.dump(info,fname='fmole.info')
