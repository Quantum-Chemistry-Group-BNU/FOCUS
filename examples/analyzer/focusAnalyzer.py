
f = open("ctns.out","r")

maxsweep = 45

send = "summary of sweep optimization up to isweep="

lines = []
iread = 0
for line in f.readlines():
   if send+str(maxsweep-1) in line:
      iread = 1
   if iread >= 1 and iread <= 2*maxsweep+3:
      if iread >= 3: lines.append(line)
      iread += 1

data = {}
for isweep in range(maxsweep):
   line = lines[isweep].split()
   msweep = eval(line[2])
   nmvp = eval(line[6])
   time = eval(line[8])
   data[isweep] = [msweep,nmvp,time]

for isweep in range(maxsweep):
   line = lines[isweep+maxsweep+1].split()
   dwt = eval(line[2])
   ener = eval(line[3].split('=')[-1])
   data[isweep].append(dwt)
   data[isweep].append(ener)

#print
print("isweep,msweep,nmvp,time,ener,dwt")
for isweep in range(maxsweep):
   msweep,nmvp,time,dwt,ener = data[isweep]
   print(isweep,msweep,nmvp,time,ener,dwt)

