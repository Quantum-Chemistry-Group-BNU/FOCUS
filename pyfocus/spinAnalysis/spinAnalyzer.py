import numpy as np
from spinAnalysis import spinTools
import matplotlib.pyplot as plt
from matplotlib import cm

def spinAnalyzer(twordm_spatial_lmo,groups,k,n,s):
    print('\n[spinAnalyzer]: k,n,s=',k,n,s)
    twordm = twordm_spatial_lmo
    onerdm = np.einsum('ijjl->il',twordm)/(n-1)
    print('<n>=',np.trace(onerdm))
    print('<s2>=',spinTools.local_spinsquare(onerdm,twordm))
    
    ng = len(groups)

    # <Si^2>
    si2 = np.zeros(ng)
    for i,ig in enumerate(groups):
        rdm1 = onerdm[np.ix_(ig,ig)]
        rdm2 = twordm[np.ix_(ig,ig,ig,ig)]
        sij = spinTools.local_spinsquare(rdm1,rdm2)
        si2[i] = sij
    for i in range(len(groups)):
        print('igroup=',i,'s2exp=',si2[i],'seff=',spinTools.from_spinsquare_to_spin(si2[i]))

    # <Si*Sj>
    sisj = np.zeros((ng,ng))
    for i,ig in enumerate(groups):
        for j,jg in enumerate(groups):
            if i>=j: continue 
            bas1 = ig+jg
            rdm1 = onerdm[np.ix_(bas1,bas1)].copy()
            rdm2 = twordm[np.ix_(bas1,bas1,bas1,bas1)].copy()
            sij = spinTools.sisj(rdm1,rdm2,len(ig),len(jg))
            sisj[i,j] = sij
            sisj[j,i] = sij
        sisj[i,i] = si2[i] 
    print('s2sum=',np.sum(sisj))
    print(sisj)

    # <Sz>
    srdm = spinTools.szHS(onerdm,twordm,n,s)
    print('tr(Sz)=',np.trace(srdm))
    sz = np.zeros(ng)
    for i,ig in enumerate(groups):
        rdm1 = srdm[np.ix_(ig,ig)]
        sz[i] = np.trace(rdm1)
    print('sz=',sz,np.sum(sz))

    # <Ne>
    ne = np.zeros(ng)
    for i,ig in enumerate(groups):
        rdm1 = onerdm[np.ix_(ig,ig)]
        ne[i] = np.trace(rdm1)
    print('ne=',ne,np.sum(ne))
    return sisj

def genSpinSpinPlot(sisj,fname='fe4s4.pdf'):
    # Plot the weighted
    ischeme = 'nearest'
    fig, axes = plt.subplots(1, 1, figsize=(12, 6),
    			 subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.3, wspace=0.05)
    clmap = cm.coolwarm
    im = axes.imshow(sisj,interpolation=ischeme,cmap=clmap)
    #axes[1].imshow(sisjList[1],interpolation=ischeme)
    #axes[2].imshow(sisjList[2],interpolation=ischeme)
    #axes[3].imshow(sisjList[3],interpolation=ischeme)
    #>fig.subplots_adjust(right=0.85)
    #>cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    #>fig.colorbar(im, cax=cbar_ax)
    plt.savefig(fname)
    plt.show()
    print('saved to fname=',fname)
    return 0
