from itools import ipyscf_real

iface = ipyscf_real.iface()
info = iface.get_integral_FCIDUMP(fname="./data/FCIDUMP_h6")
iface.dump(info,fname='./data/fmole.info')
