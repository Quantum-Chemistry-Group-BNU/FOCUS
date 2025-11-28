import numpy as np

def extractKij(fname='fmole.info'):
    f = open(fname,'r')
    lines = f.readlines()
    dic = {}
    for line in lines:
        lst = line.split()
        if len(lst) > 1:
            i = eval(lst[0])
            j = eval(lst[1])
            k = eval(lst[2])
            l = eval(lst[3])
            eri = eval(lst[4])
            dic[(i,j,k,l)] = eri
        elif len(lst) == 1:
            sorb = eval(lst[0])
    
    norb = sorb//2
    kij = np.zeros((norb,norb))
    for i in range(norb):
        for j in range(norb):
            # we want to get <iAjB||jAiB> = <iAjB|jAiB> - <iAjB|iBjA> = [ij|ji] from dic (compressed storage)
            iA = 2*i
            iB = 2*i+1
            jA = 2*j
            jB = 2*j+1
            # canonical order (iA,jB,jA,iB)
            p = iA+1 # integral file count from 1
            q = jB+1
            sgn1 = 1
            if p<q: 
                sgn1 = -1
                p,q = q,p
            pq = (p-1)*(p-2)/2+q-1
            r = jA+1
            s = iB+1
            sgn2 = 1
            if r<s:
                sgn2 = -1
                r,s = s,r
            rs = (r-1)*(r-1)/2+s-1
            quad = (p,q,r,s)
            if pq<rs:
                quad = (r,s,p,q)
            sgn = sgn1*sgn2
            kij[i,j] = dic[quad]*sgn # for real orbitals
    print(kij.shape)
    print(np.linalg.norm(kij-kij.T))
    return kij

