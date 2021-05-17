import os

HOME = os.path.dirname(os.getcwd())
SCI = HOME+"/bin/sci.x"
CTNS = HOME+"/bin/ctns.x"

print('HOME=',HOME)
os.environ.setdefault('HOME',HOME)

os.chdir("0_h6_tns")
os.system("pwd")
os.system(SCI+" results/input.dat")
#
#os.chdir("..")
#os.system("pwd")
