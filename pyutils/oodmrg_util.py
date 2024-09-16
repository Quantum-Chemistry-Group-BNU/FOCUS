import os
import numpy as np
from pyscf import lo
import ipyscf_real
from focus_class import *
import focus_parser

def vonNeumannEntropy(pop,thresh=1.e-100):
    s = 0.0
    for p in pop:
        if p < thresh: continue
        s += -p*np.log(p)
    return s

def renyiEntropy(pop,alpha):
    if alpha == 1:
        return vonNeumannEntropy(pop)
    else:
        return 1/(1-alpha)*np.sum(pop**alpha)

def loadUrot(fname,norb):
    f = open(fname)
    urot = np.zeros((norb,norb))
    lines = f.readlines()
    idx = 0
    for line in lines:
        urot[idx,:] = np.array([eval(x) for x in line.split()])
        idx += 1
    assert(idx == norb)
    f.close()
    assert(np.allclose(urot.T.dot(urot), np.identity(norb)))
    return urot

def loadSchmidtValues(fname,alpha=1):
    f = open(fname)
    line = f.readline()
    data = []
    lines = f.readlines()
    for line in lines:
        sigs2 = [eval(x) for x in line.split()]
        data.append(np.sort(sigs2)[-1::-1])
    # entropy
    svnlst = []
    for i in range(len(data)):
        svn = renyiEntropy(data[i],alpha)
        svnlst.append(svn)
    return data,svnlst

def OO_DMRG(mol,mf,mo,dcut,oo_maxiter,oo_alpha,parentdir,workdir='oodmrg_tmp',iprt=0):
    print('### OO_DMRG ###')
    print('mo=',mo)
    print('parentdir=',parentdir)
    print('workdir=',workdir)

    #--- Define starting MO ---
    cmo = mf.mo_coeff
    norb = cmo.shape[1]
    nfrozen = 0
    s = mf.get_ovlp()
    if mo == 'cmo':
        urot = np.identity(norb)

    elif mo == 'lmo':
        # localized orbitals
        loc_occ = lo.PM(mol, cmo[:,:mol.nelectron//2]).kernel()
        loc_vir = lo.PM(mol, cmo[:,mol.nelectron//2:]).kernel()
        lmo = np.hstack([loc_occ,loc_vir])
        urot = cmo.T.dot(s.dot(lmo))

    elif mo == 'oao':
        mo_coeff = lo.orth_ao(mol, method="meta-lowdin")
        urot = cmo.T.dot(s.dot(mo_coeff))

    mo_coeff = cmo.dot(urot)

    #--- Setup Common & Save initial integrals ---
    os.chdir(parentdir)
    common = Common()
    common.parentdir = parentdir
    common.workdir = workdir
    common.build()

    os.chdir(common.workdir)
    os.system("pwd")
    iface = ipyscf_real.iface()
    iface.mol = mol
    iface.mf = mf
    iface.nfrozen = nfrozen
    info = iface.get_integral(mo_coeff)
    integral_file = 'rmole_lo.info'
    iface.dump(info,fname=integral_file)

    common.nelec = mol.nelectron
    common.twom = mol.spin
    common.integral_file = integral_file

    #--- SCI ---
    sci = SCI(common)
    sci.dets = [list(range(mol.nelectron))]
    sci.gen_input("sci.dat",iprt)
    esci = sci.kernel(iprt=0)
    print("esci=",esci)

    #--- OO-DMRG ---
    ctns = CTNS(common)
    ctns.topo = range(mol.nao)
    ctns.schedule = [[0,2,dcut,1.e-4,1.e-8]]
    ctns.tasks = ['task_oodmrg']
    ctns.oo_maxiter = oo_maxiter
    ctns.oo_alpha = oo_alpha
    ctns.gen_input("ctns.dat",iprt=iprt)
    ctns.kernel(iprt=iprt)

    print('finished! currentdir=',os.getcwd())
    return common,sci,ctns,mo_coeff

