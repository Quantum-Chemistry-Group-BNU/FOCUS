import numpy as np
import scipy

def get_dims(qs1):
    return [len(qs1[key]) for key in qs1]

def get_dimtot(qs1):
    return np.sum(get_dims(qs1))

def combine_index12(dim1,dim2,indx1i,indx2j):
    indx12ij = []
    for ki in indx1i:
        for kj in indx2j:
            kij = ki*dim2 + kj
            indx12ij.append(kij)
    return indx12ij

def direct_product(qs1,qs2):
    dim1 = get_dimtot(qs1)
    dim2 = get_dimtot(qs2)
    dic = {}
    for key1 in qs1:
        for key2 in qs2:
            n12 = key1[0]+key2[0]
            tm12 = key1[1]+key2[1]
            key = (n12,tm12)
            if key not in dic:
                dic[key] = combine_index12(dim1,dim2,qs1[key1],qs2[key2])
            else:
                dic[key] += combine_index12(dim1,dim2,qs1[key1],qs2[key2])
    return dic

def sweep_projection(sites,thresh=1.e-14,debug=False):
    nsite = len(sites)
    nroot = sites[0].shape[0]
    if debug: print('[sweep_projection] nsite=',nsite,'nroot=',nroot)
    # build enviroment
    env = [None]*(nsite+1)
    env[0] = np.identity(nroot)/nroot
    for i in range(nsite):
        env[i+1] = np.einsum('lnr,la,anb->rb',sites[i].conj(),env[i],sites[i])
    assert env[nsite].shape == (1,1)
    assert np.abs(env[nsite][0,0]-1.0) < 1.e-10
    # sweep projection from the rightmost site
    assert sites[nsite-1].shape[2] == 1
    wmat = np.einsum('lnr->ln',sites[nsite-1]) # expansion from nosym to sym basis
    qnum_phys = {(0,0):[0],(2,0):[1],(1,1):[2],(1,-1):[3]}
    qnum_right = qnum_phys
    rmps_sites = [np.identity(4).reshape(4,4,1)]
    rmps_qnums = [qnum_phys]
    for i in range(nsite-2,-1,-1):
        if debug: print('\nisite=',i,' shape=',sites[i].shape)
        csite = np.einsum('lnr,ra->lna',sites[i],wmat)
        # build a reduced density matrix in the subspace by
        # rho_R = tr_l(psi[lna]|lna><l'n'a'|psi[l'n'a'])
        shape = csite.shape
        cmat = csite.reshape(shape[0],shape[1]*shape[2]) # lL
        dm = np.einsum('lL,lr,rR->LR',cmat.conj(),env[i],cmat).conj()
        qnum_super = direct_product(qnum_phys,qnum_right)
        if debug: print('qnum_super=',qnum_super)
        dim_phys  = get_dimtot(qnum_phys)
        dim_right = get_dimtot(qnum_right)
        dim_super = get_dimtot(qnum_super)
        # diagonalize dm
        dim_reduced = 0
        qnum_reduced = {}
        vbas = {}
        for key in qnum_super:
            indices = qnum_super[key]
            dim = len(indices)
            dm0 = dm[np.ix_(indices,indices)]
            e,v = scipy.linalg.eigh(-dm0)
            e = -e
            kept = np.argwhere(e>thresh)
            nkept = min(dim,len(kept))
            if nkept > 0:
                vbas[key] = v[:,:nkept]
                qnum_reduced[key] = list(range(dim_reduced,dim_reduced+nkept))
                dim_reduced += nkept
        # assemble into a full matrix
        rmat = np.zeros((dim_super,dim_reduced))
        for key in qnum_reduced:
            rindices = qnum_super[key]
            cindices = qnum_reduced[key]
            rmat[np.ix_(rindices,cindices)] = vbas[key]
        # Update wmat & qnum_right
        wmat = np.einsum('lL,Lr->lr',cmat,rmat.conj())
        qnum_right = qnum_reduced
        # Update site tensor
        tensor = rmat.reshape(dim_phys,dim_right,dim_reduced)
        tensor = np.einsum('nrl->lnr',tensor)
        rmps_sites.append(tensor)
        rmps_qnums.append(qnum_reduced)
        if debug:
            print('tensor=',tensor.shape)
            print('qnum_reduced=',qnum_reduced)
    # rwfuns
    rwfuns = wmat.copy()
    # form qbonds
    qbonds = [np.array([[0,0,1]],dtype=np.int32)]
    for i in range(nsite):
        qnum = rmps_qnums[i]
        nq = len(qnum)
        qdata = np.zeros((nq,3),dtype=np.int32)
        idx = 0
        for key in qnum:
            qdata[idx][0] = key[0]
            qdata[idx][1] = key[1]
            qdata[idx][2] = len(qnum[key])
            idx += 1
        qbonds.append(qdata)
    return qbonds,rmps_sites,rwfuns

def dumpMPSforFOCUS(qbonds,rmps_sites,rwfuns,prefix='rmps',debug=False):
    nsite = len(rmps_sites)
    print('[dumpMPSforFOCUS] nsite=',nsite)
    # dump qbonds
    nsectors = np.zeros(nsite+1,dtype=np.int32)
    for i in range(nsite+1):
        qbonds[i].tofile(prefix+'.qbond'+str(i))
        nsectors[i] = qbonds[i].shape[0]
    nsectors.tofile(prefix+'.nsectors')
    # dump rsites
    qphys = qbonds[1]
    for i in range(nsite):
        rsite = rmps_sites[i]
        qrow = qbonds[i+1]
        qcol = qbonds[i]
        qmid = qphys
        # to be consistent with qinfo3 in FOCUS
        nnz = 0
        ir = 0
        blks = []
        for qr in qrow:
            ic = 0
            for qc in qcol:
                im = 0
                for qm in qmid:
                    if qr[0]==qc[0]+qm[0] and qr[1]==qc[1]+qm[1]:
                        #print('qr=',qr,'qc=',qc,'qm=',qm)
                        dr = qr[2]
                        dc = qc[2]
                        dm = qm[2]
                        blk = rsite[ir:ir+dr,im:im+dm,ic:ic+dc] # lnr
                        blk = blk.transpose(1,2,0).reshape(-1) # nrl
                        blks.append(blk)
                        nnz += blk.shape[0]
                    im += qm[2]
                ic += qc[2]
            ir += qr[2]
        data = np.hstack(blks)
        data.tofile(prefix+'.site'+str(i))
        if debug: 
            dimL = np.sum(qrow[:,2])
            dimC = np.sum(qmid[:,2])
            dimR = np.sum(qcol[:,2])
            print(' i=',i,'(dl,dn,dr)=',(dimL,dimC,dimR),'nnz=',nnz)
    # dump rwfuns
    assert rwfuns.shape[0] == rwfuns.shape[1]
    data = rwfuns.reshape(-1) # ir [C order]
    data.tofile(prefix+'.rwfuns')
    if debug: print(' rwfuns.shape=',rwfuns.shape)
    return 0
