import os

fname = "test.out2"
os.system("grep 'GPUmem' "+fname+" > gpumem.stat")

f = open("gpumem.stat")
lines = f.readlines()
used = -1
for line in lines:
    dat = line.split()[2]
    if "used" in dat: 
        used = max(used,eval(dat.split('=')[1]))

print('max usage=',used,'GB')
