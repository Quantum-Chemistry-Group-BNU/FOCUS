import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import os
import h5py

ijdx = [7,5,4,6,2,3,1,0]
ijdx = np.ix_(ijdx,ijdx)

#=== Plot each figure ===
a = -9
b = +9
clmap = cm.bwr #seismic #bwr #coolwarm
ncase = 5 # 2*10 
ischeme = 'nearest'

fig = plt.figure(figsize=(2.5,3.5))

l = 0.25
r = 0.95
x = -14 #3.5
y = 4
fsize = 14
def plotData(gs1,irow,icol,i,ioff=0,fname='new1'):
   f = h5py.File(fname+'/ss'+str(i)+'.h5','r')
   ss0 = f['sisj'].value
   ss0 = ss0[ijdx]
   assert np.linalg.norm(ss0-ss0.T)<1.e-10
   ax = plt.subplot(gs1[ncase*irow+ioff,icol])
   ax.set_xticks([])
   ax.set_yticks([])
   im = ax.imshow(ss0,vmin=a,vmax=b,interpolation=ischeme,cmap=clmap)
   f.close()
   if icol == 0 and ioff == 0:
      ax.text(x,y,r"$D_{\mathrm{SP}}=1   $",fontsize=fsize)
   elif icol == 0 and ioff == 1:
      ax.text(x,y,r"$D_{\mathrm{SP}}=50  $",fontsize=fsize)
   elif icol == 0 and ioff == 2:
      ax.text(x,y,r"$D_{\mathrm{SA}}=1000$",fontsize=fsize)
   elif icol == 0 and ioff == 3:
      ax.text(x,y,r"$D_{\mathrm{SA}}=2000$",fontsize=fsize)
   elif icol == 0 and ioff == 4:
      ax.text(x,y,r"$D_{\mathrm{SA}}=3000$",fontsize=fsize)
   fig.add_subplot(ax)
   return ax,im
	
gs1 = gridspec.GridSpec(ncase,1,wspace=0.05,hspace=0.0)
gs1.update(left=l,right=r,top=0.9,bottom=0.02)
for i in [26]:
   irow = 0
   icol = 0
   print 'i,irow,icol=',(i,irow,icol)
   ax,im = plotData(gs1,irow,icol,i,ioff=0,fname='new1')
   ax.set_title(str(i+1),fontsize=fsize) # Labels
   ax,im = plotData(gs1,irow,icol,i,ioff=1,fname='new50')
   ax,im = plotData(gs1,irow,icol,i,ioff=2,fname='new1000')
   ax,im = plotData(gs1,irow,icol,i,ioff=3,fname='new2000')
   ax,im = plotData(gs1,irow,icol,i,ioff=4,fname='new3000')
 
   cbar_ax = fig.add_axes([0.8, 0.1, 0.05, 0.75]) # left,bottom,width,top
   fig.colorbar(im, cax=cbar_ax)
   
   plt.savefig('ss_pn'+str(i)+'.pdf')
   plt.show()
