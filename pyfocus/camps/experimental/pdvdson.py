import time
# import numpy
# import scipy.linalg
import torch
from functools import reduce
from camps.utils.config import dtype_config


#
# Mask for Matrix vector product subroutine
#
class mask:
    def __init__(self, info, mvp):
        self.info = info  # 存储额外的参数信息
        self.mvp = mvp    # 存储矩阵向量乘积函数

    def matvec(self, vec):
        # 调用矩阵向量乘积函数，传递向量和额外参数
        return self.mvp(vec, *self.info)


#
# LOBPCG solver with Davidson precondition
#
class eigenSolver:
    def __init__(self):
        self.maxcycle = 20       # 最大迭代次数
        self.crit_e = 1.e-8      # 特征值收敛阈值
        self.crit_vec = 1.e-8    # 残差收敛阈值  
        self.crit_indp = 1.e-12  # 线性无关性阈值
        self.crit_demo = 1.e-10  # 数值稳定性阈值
        # Basic setting
        self.rank = 0
        self.iprt = 1            # 打印级别
        self.ndim = 0            # 矩阵维度
        self.neig = 0            # 需要求解的特征值个数
        self.diag = None         # 矩阵对角元
        self.HVec = None         # 矩阵向量乘积函数
        self.noise = True        # 是否添加噪声
        self.const = 0.0         # 能量常数偏移
        # self.comm = None         # MPI通信器
        self.dtype = dtype_config.default_dtype # 数据类型
        # self.mtype = MPI.DOUBLE  # MPI数据类型
        self.nmvp = 0            # 矩阵向量乘积计数
        self.lshift = 1.e-4      # Level-shift参数
        self.nz = 1.e-5          # 噪声大小
        # Projeciton
        self.ifex = False        # 是否使用外部投影
        self.qbas = None         # 投影基
        # Full space of vectors
        self.ifall = True        # 是否保留全部基向量
        self.nfac = 10           # 子空间最大倍数

    # Q*V=(1-C*C^d)*V, vbas.shape=(neig,ndim)
    def projection(self, vbas):
        vbas2 = vbas - reduce(torch.matmul, (vbas, self.qbas.T.conj(), self.qbas))
        return vbas2

    # Debug: full diagonalization
    def fullDiag(self):
        print('diag=', self.diag)
        print('sumdiag=', torch.sum(self.diag))
        v0 = torch.identity(self.ndim, dtype=dtype_config.default_dtype).to(dtype_config.device)
        hmat = self.HVecs(v0)
        print('hmat\n', hmat)
        eig, vr = torch.linalg.eigh(hmat)
        print('eig=', eig)
        exit()

    # Matrix vector product H[iproc]*V
    def HVecs(self, vbas):
        # print(f"{self.rank}:{vbas.shape}")
        n = vbas.shape[0]
        self.nmvp += n
        wbas = torch.zeros((n, self.ndim), dtype=self.dtype).to(dtype_config.device)
        for i in range(n):
            wbas_iproc = self.HVec(vbas[i])      # 这个矩阵向量乘需要具体定义
            wbas[i] = wbas_iproc
        return wbas

    # Generate V0 from real diagonal values
    def genV0(self, neig):
        # Break the degeneracy artifically in generating v0
        diag = self.diag + 1.e-12 * torch.arange(1, 1 + self.ndim).to(dtype_config.device) / float(self.ndim)
        index = torch.argsort(diag.real)[:neig]
        v0 = torch.zeros((neig, self.ndim), dtype=dtype_config.default_dtype).to(dtype_config.device)
        v0[range(neig), index] = 1.0
        return v0

    # Main solver: LOBPCG + Davidson preconditioning
    def solve_iter(self, v0=None, iop=4, ifplot=False):
        if self.neig > self.ndim:
            print(' error in dvdson: neig>ndim, neig/ndim=', self.neig, self.ndim)
            exit(1)
        # Clear counter
        self.nmvp = 0
        t0 = time.time()
        #
        # ONLY rank-0 generates the basis, input v0 is an np.array (neig,ndim)
        #
        if self.rank == 0:
            if v0 is None:
                vbas = self.genV0(self.neig)
            elif v0.shape[0]<self.neig:
                v0_num = v0.shape[0]
                vbas = self.genV0(self.neig)
                vbas[:v0_num] = v0
            else:
                vbas = v0.clone()
            # Add random noise to interact with the whole space
            if self.noise: 
                vbas = vbas + self.nz * torch.rand(self.neig, self.ndim).to(dtype_config.device) * 2 - 1
            if self.ifex: 
                vbas = self.projection(vbas)
            genOrthoBas(vbas, self.crit_indp)
        else:
            vbas = None
        wbas = self.HVecs(vbas)
        #
        # Begin to solve
        #
        ifconv = False
        neig = self.neig
        eigs = torch.zeros(neig, dtype=dtype_config._default_dtype).to(dtype_config.device) + 1.e3
        # Record history
        rnorm = torch.zeros((neig, self.maxcycle), dtype=dtype_config.default_dtype).to(dtype_config.device)
        eigval = torch.zeros((neig, self.maxcycle + 1), dtype=dtype_config.default_dtype).to(dtype_config.device) + 1.e3
        # initial dimension
        ndim = neig
        for niter in range(1, self.maxcycle):

            # =======================================
            # ONLY rank-0 compute the small problem
            # =======================================
            if self.rank == 0:
                # Check othonormality of basis
                iden = torch.matmul(vbas, vbas.T.conj())
                diff = torch.linalg.norm(iden - torch.eye(ndim, dtype=dtype_config.default_dtype).to(dtype_config.device))
                if diff > 1.e-10:
                    print(' diff_VBAS=', diff)
                    print(iden)
                    exit(1)
                # An important note: Vh*H*V \= Vt*Hc*Vc
                # WRONG  : tmpH = numpy.dot(vbas,wbas.T.conj())
                # CORRECT:
                # print(f'vbas_device: {vbas.device}, wbas_device: {wbas.device}')
                tmpH = torch.matmul(vbas.conj(), wbas.T)
                diff = torch.linalg.norm(tmpH - tmpH.T.conj())
                # print(print(' diff_skewH=', diff))
                # if diff > 1.e-10:
                #     print(' diff_skewH=', diff)
                #     print(' tmpH =\n', tmpH)
                #     print('warning: non-hermitian!')
                # Explicit symmetrizaiton
                tmpH = 0.5 * (tmpH + tmpH.T.conj())
                # EigenSolve
                eig, vr = torch.linalg.eigh(tmpH)
                vr = vr[:, :neig].T.clone()
                # CHECK ORTHOGONALITY for vr[neig,ndim]
                over = torch.matmul(vr, vr.T.conj())
                diff = torch.linalg.norm(over - torch.eye(neig, dtype=dtype_config.default_dtype).to(dtype_config.device))
                if diff > 1.e-10:
                    print(' diff_VEC=', diff)
                    print(over)
                    exit(1)
                # Save
                teig = eig[:neig]

                # Eigenvalue convergence
                nconv1 = 0
                econv = [False] * neig
                for i in range(neig):
                    tmp = teig[i] - eigs[i]
                    if abs(tmp) <= self.crit_e:
                        econv[i] = True
                        nconv1 += 1
                eigs = teig.clone()
                eigval[:, niter] = teig.clone()

                # Full Residuals: Res[i]=Res'[i]-w[i]*X[i]
                rbas = torch.matmul(vr, vbas)   # 对精确本征矢在子空间中的近似 (neig, ndim)
                rbas = torch.matmul(vr, wbas) - torch.matmul(torch.diag(eigs.to(dtype_config.default_dtype)), rbas)
                if self.ifex: 
                    rbas = self.projection(rbas)
                nconv2 = 0
                rindx = []
                rconv = [0] * neig
                for i in range(neig):
                    tmp = torch.linalg.norm(rbas[i, :])
                    rnorm[i, niter - 1] = tmp
                    # Criteria for convergence
                    if tmp <= self.crit_vec:
                        nconv2 += 1
                        rconv[i] = (True, tmp)
                    else:
                        rconv[i] = (False, tmp)
                        if not econv[i]: 
                            rindx.append(i)

                # Printing
                t1 = time.time()
                if self.iprt >= 0:
                    if niter == 1:
                        print('[pdvdson]: Hc=ce with (noise,size,ndim,crit_e,crit_v) = (%i,%d,%d,%.1e,%.1e)' % \
                              (self.noise, 1, self.ndim, self.crit_e, self.crit_vec))
                        print(' iter  dim  nmvp   ieig         eigenvalue        ediff      rnorm     time/s  ')
                        print(' ------------------------------------------------------------------------------')
                    
                    symbol = '+' if niter % 2 == 1 else '-'
                    for i in range(neig):
                        status_char = str(rconv[i][0] or econv[i])[0]
                        print('%4d %4d %5d %4d %s %s %20.12f %11.3e %10.3e %9.2e' % 
                              (niter, ndim, self.nmvp, i, symbol, status_char,
                               self.const + eigval[i, niter],
                               eigval[i, niter] - eigval[i, niter - 1],
                               rconv[i][1], t1 - t0))
                t0 = time.time()

            # =======================================
            # Check convergence on rank-0 and then
            # broadcast the result to each proc.
            # =======================================
            if self.rank == 0:
                # Convergence by either criteria (NO - just residual)
                # ifconv = (nconv1 == neig) or (nconv2 == neig)
                ifconv = len(rindx) == 0
                ifconv = torch.tensor(ifconv).to(dtype_config.device)
            else:
                ifconv = None
            # If converged, exit in all processors
            if ifconv or niter == self.maxcycle - 1:
                # =======================================
                # Only return the eigens from rank-0
                # =======================================
                if self.rank == 0:
                    eigs = eigs + self.const
                    rbas = torch.matmul(vr, vbas)
                else:
                    eigs = None
                    rbas = None
                break

            # =======================================
            # If not converged, use processor-0 to
            # generate the new basis and bcast them.
            # =======================================
            if self.rank == 0:

                # Reduce the basis to span{x[k],x[k]-x[k-1]}
                if (not self.ifall) or (self.ifall and ndim > self.nfac * self.neig):
                    # Rotated basis to minimal subspace that
                    # can give the exact [neig] eigenvalues
                    # Also, the difference vector = xold - xnew as correction
                    pr = (torch.eye(ndim, dtype=dtype_config.default_dtype).to(dtype_config.device))[:neig, :] - vr
                    nindp, vr2 = dvdson_ortho(vr, pr[rindx, :], self.crit_indp)
                    if nindp != 0: 
                        vr = torch.vstack((vr, vr2))
                    vbas = torch.matmul(vr, vbas)
                    wbas = torch.matmul(vr, wbas)

                # New directions from residuals
                for i in range(neig):
                    if rconv[i][0] == True: 
                        continue
                    # Various PRECONDITIONER:
                    if iop == 0:
                        # gradient
                        pass
                    elif iop == 1:
                        # Davidson
                        tmp = self.diag - eigs[i]
                        tmp[abs(tmp) < self.crit_demo] = self.crit_demo
                        rbas[i, :] = rbas[i, :] / tmp
                    elif iop == 2:
                        # Olsen's algorithm works for close diag ~ H : 0.00067468 [3]
                        tmp = self.diag - eigs[i]
                        tmp[abs(tmp) < self.crit_demo] = self.crit_demo
                        e1 = torch.matmul(vbas[i, :], rbas[i, :] / tmp) / torch.matmul(vbas[i, :], vbas[i, :] / tmp)
                        rbas[i, :] = -(rbas[i, :] - e1 * vbas[i, :]) / tmp
                    elif iop == 3:
                        # ABS
                        tmp = abs(self.diag - eigs[i])
                        tmp[abs(tmp) < self.crit_demo] = self.crit_demo
                        rbas[i, :] = rbas[i, :] / tmp
                    elif iop == 4:
                        # ABS+LEVEL-SHIFT ~ Davidson+Gradient
                        tmp = abs(self.diag - eigs[i]) + self.lshift
                        rbas[i, :] = rbas[i, :] / tmp

                # Projection
                if self.ifex: 
                    rbas = self.projection(rbas)
                # Re-orthogonalization and get Nindp
                nindp, vbas2 = dvdson_ortho(vbas, rbas[rindx, :], self.crit_indp)
                if self.iprt > 0: 
                    print(' final nindp = ', nindp)
                nindp = torch.tensor(nindp).to(dtype_config.device)
            else:
                nindp = None
                
            if nindp != 0:
                wbas2 = self.HVecs(vbas2)
                if self.rank == 0:
                    vbas = torch.vstack((vbas, vbas2))
                    wbas = torch.vstack((wbas, wbas2))
                    ndim = vbas.shape[0]
            else:
                print('Convergence failure: unable to generate new direction: Nindp=0 !')
                exit(1)

        if not ifconv:
            print('Convergence failure: out of maxcycle ! maxcycle =', self.maxcycle)
            # exit(1)

        #
        # Plot iteration history if necessary
        #
        if ifplot:
            import matplotlib.pyplot as plt
            plt.plot(range(self.ndim), self.diag)
            plt.show()
            plt.savefig("diag.png")

            for i in range(self.neig):
                plt.plot(range(1, niter + 1), torch.log10(rnorm[i, :niter]), label=str(i + 1))
            plt.legend()
            plt.savefig("res_conv.png")
            plt.show()

            for i in range(self.neig):
                plt.plot(range(1, niter + 1), eigval[i, 1:niter + 1], label=str(i + 1))
            plt.legend()
            plt.savefig("eig_conv.png")
            plt.show()

        # Only processor-0 holds the correct [eigs,rbas]
        return eigs, rbas, self.nmvp


#
# From vbas to generate orthonormal basis
#
def genOrthoBas(vbas, crit_indp):
    vbas[0] = vbas[0] / torch.linalg.norm(vbas[0])
    nbas = vbas.shape[0]
    if nbas != 1:
        nindp, vbas2 = dvdson_ortho(vbas[0:1], vbas[1:], crit_indp)
        if nindp + 1 == nbas:
            vbas[1:] = vbas2
        else:
            print('error: insufficient orthonormal basis: nbas/nindp', nbas, nindp + 1)
            exit()
    return 0


#
# Orthonormalization basis from rbas against previous vbas
#
def dvdson_ortho(vbas, rbas, crit_indp):
    debug = False
    if debug: 
        print('[dvdson_ortho]')
    
    ndim = vbas.shape[0]
    nres = rbas.shape[0]
    nindp = 0
    vbas2 = torch.zeros(rbas.shape, dtype=rbas.dtype).to(dtype_config.device)
    
    # 投影: (I - V·V†)·R
    maxtimes = 5
    for k in range(maxtimes):
        # 展开 reduce 操作以便理解
        # rbas = rbas - reduce(numpy.dot, (rbas, vbas.T.conj(), vbas))
        proj_coeff = torch.matmul(rbas, vbas.T.conj())
        proj_component = torch.matmul(proj_coeff, vbas)
        rbas = rbas - proj_component
    
    # 从残差中提取新的正交基
    for i in range(nres):
        rvec = rbas[i, :].clone()
        rii = torch.linalg.norm(rvec)
        
        if rii <= crit_indp: 
            continue
        
        if debug: 
            print(f'  i,rii= {i}, {rii}')
        
        # 归一化
        rvec = rvec / rii
        rii = torch.linalg.norm(rvec)  # 再次归一化确保精度
        rvec = rvec / rii
        
        vbas2[nindp] = rvec
        nindp += 1
        
        # 从剩余残差中减去新基的分量
        for k in range(maxtimes):
            # 减去相对于vbas的投影
            proj_coeff1 = torch.matmul(rbas[i:, :], vbas.T.conj())
            proj_component1 = torch.matmul(proj_coeff1, vbas)
            rbas[i:, :] -= proj_component1
            
            # 减去相对于新基vbas2的投影
            proj_coeff2 = torch.matmul(rbas[i:, :], vbas2[:nindp, :].T.conj())
            proj_component2 = torch.matmul(proj_coeff2, vbas2[:nindp, :])
            rbas[i:, :] -= proj_component2
    
    # 返回线性无关的新基
    vbas2 = vbas2[:nindp].clone()
    
    # 调试：检查正交性
    if debug and nindp != 0:
        tmp = torch.vstack((vbas, vbas2))
        iden = torch.matmul(tmp, tmp.T.conj())
        diff = torch.linalg.norm(iden - torch.identity(iden.shape[0], dtype=rbas.dtype))
        if diff > 1.e-10:
            print(f' error in mgs_ortho: diff= {diff}')
            print(iden)
            exit(1)
        else:
            print(f' final nindp from mgs_ortho = {nindp}, diffIden= {diff}')
    
    return nindp, vbas2
