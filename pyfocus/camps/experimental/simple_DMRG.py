import numpy as np

import copy
# import numpy as np
import torch
# from torch.types import Tensor
# import scipy
from typing import Tuple, List, Union
from torch import Tensor
from functools import partial
from opt_einsum import contract, contract_expression

from pyfocus.camps.mps.disentangled import minimize_entropy_multiSweep, update_ham_multiSweep
from pyfocus.camps.mps.mpo import construct_mpo_pauli
from pyfocus.camps.mps.mps import random_state
from pyfocus.camps.experimental.pdvdson_sg import mask, eigenSolver
from pyfocus.camps.mps.mps_simple import leftCanonicalization, rightCanonicalization
from pyfocus.camps.utils.config import dtype_config
from pyfocus.camps.utils.typing import Clifford, SaveInfo, Sites, Hamiltonian
from pyfocus.camps.utils.environ import Environ, env_L_i
from loguru import logger
import time

# def env_L_i(mps_i, mpo_i, env_Li):
#     """
#         S-a-S-f
#             d  
#         O-b-O-g
#             e  
#         S-c-S-h
#     """
#     env_L_new = contract('abc, adf, bdeg, ceh -> fgh', 
#                         env_Li, mps_i.conj(), mpo_i, mps_i)
#     return env_L_new

# def env_R_i(mps_i, mpo_i, env_Ri):
#     """
#         -f-S-a-S
#            d    
#         -g-O-b-O
#            e    
#         -h-S-c-S
#     """
#     # print(mps_i.dtype, env_Ri.dtype, mpo_i.dtype)
#     env_R_new = contract('fda, abc, gdeb, hec -> fgh', 
#                         mps_i.conj(), env_Ri, mpo_i, mps_i)
#     return env_R_new

# def get_envs_L(mps: List[Tensor], 
#                mpo: Union[List[Tensor], List[List[Tensor]]], 
#                method: str, 
#                mpo_type: str = 'S'):
#     if method == '1site':
#         remain = 1
#     elif method == '2site':
#         remain = 2
#     else:
#         assert False
#     nsites = len(mps)
#     envs_L_list = []
#     # env_L_tmp = np.ones([1,]*3)
#     if mpo_type == 'S':
#         env_L_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
#         envs_L_list.append(env_L_tmp)
#         for i in range(nsites-remain):
#             env_L_tmp = env_L_i(mps[i], mpo[i], env_L_tmp)
#             envs_L_list.append(env_L_tmp)
#     elif mpo_type == 'M':
#         M_MPO_len = len(mpo)
#         for L in range(M_MPO_len):
#             single_env_list = []
#             env_L_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
#             single_env_list.append(env_L_tmp)
#             for i in range(nsites-remain):
#                 env_L_tmp = env_L_i(mps[i], mpo[L][i], env_L_tmp)
#                 single_env_list.append(env_L_tmp)
#             envs_L_list.append(single_env_list)
#     return envs_L_list

# def get_envs_R(mps: List[Tensor], 
#                mpo: Union[List[Tensor], List[List[Tensor]]], 
#                method: str, 
#                mpo_type: str = 'S'):
#     if method == '1site':
#         remain = 1
#     elif method == '2site':
#         remain = 2
#     else:
#         assert False
#     nsites = len(mps)
#     envs_R_list = []
#     if mpo_type == 'S':
#         env_R_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
#         envs_R_list.append(env_R_tmp)
#         for i in range(nsites-1, remain - 1, -1):
#             env_R_tmp = env_R_i(mps[i], mpo[i], env_R_tmp)
#             envs_R_list.append(env_R_tmp)
#     elif mpo_type == 'M':
#         M_MPO_len = len(mpo)
#         for L in range(M_MPO_len):
#             single_env_list = []
#             env_R_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
#             single_env_list.append(env_R_tmp)
#             for i in range(nsites-1, remain - 1, -1):
#                 env_R_tmp = env_R_i(mps[i], mpo[L][i], env_R_tmp)
#                 single_env_list.append(env_R_tmp)
#             envs_R_list.append(single_env_list)
#     return envs_R_list

def get_ham_direct(
    ltensor,
    rtensor,
    cmo,
):
    mo_length = len(cmo)
    if mo_length == 1:
        ham = contract('abc, bdef, lfk -> adlcek', ltensor, cmo[0], rtensor)
        sp = ham.shape
        size = sp[:3]
    else:
        ham = contract('abc,bdef,fghj,ljk->adglcehk', ltensor, cmo[0], cmo[1], rtensor)
        sp = ham.shape
        size = sp[:4]
    size_num = size.numel()
    ham_2d = ham.reshape(size_num, size_num)
    return ham_2d, size

def get_Hx(ltensor, cmos, rtensor, method, cshape):
    if method == '1site':
        assert ltensor.shape[-1] == cshape[0] and cmos[0].shape[-2] == cshape[1] and \
            rtensor.shape[-1] == cshape[2]
        expr = contract_expression(
                "abc, bdef, lfk, cek -> adl",
                ltensor, cmos[0], rtensor, cshape,
                constants=[0, 1, 2],
            )
    elif method == '2site':
        assert ltensor.shape[-1] == cshape[0] and cmos[0].shape[-2] == cshape[1] and \
            cmos[1].shape[-2] == cshape[2] and rtensor.shape[-1] == cshape[3]
        expr = contract_expression(
                "abc, bdef, fghj, ljk, cehk -> adgl",
                ltensor, cmos[0], cmos[1], rtensor, cshape,
                constants=[0, 1, 2, 3],
            )
    return expr

def get_diags(ltensor, cmos, rtensor, method):   # 需要返回主程序
    # if mpo_type == 'S':
    ltensor_diag = contract('aba->ba', ltensor)
    cmos_0_diag = contract('abbc->abc', cmos[0])
    rtensor_diag = contract('aba->ba', rtensor)
    if method == '1site':
        hdiag = contract('ba,bcd,de->ace', ltensor_diag, cmos_0_diag, rtensor_diag).reshape(-1)   #要转成1维
    elif method == '2site':
        cmos_1_diag = contract('abbc->abc', cmos[1])
        hdiag = contract('ba,bcd,def,fg->aceg', ltensor_diag, cmos_0_diag, cmos_1_diag, rtensor_diag).reshape(-1)
    # elif mpo_type == 'M':
    #     M_MPO_len = len(ltensor)
    #     ltensor_diag_stack = [contract('aba->ba', ltensor[i]) for i in range(M_MPO_len)]
    #     cmos_0_diag_stack = [contract('abbc->abc', cmos[i][0]) for i in range(M_MPO_len)]
    #     rtensor_diag_stack = [contract('aba->ba', rtensor[i]) for i in range(M_MPO_len)]
    #     if method == '1site':
    #         hdiag = torch.sum(torch.stack(
    #             [contract('ba,bcd,de->ace', ltensor_diag_stack[i], cmos_0_diag_stack[i], rtensor_diag_stack[i]) 
    #              for i in range(M_MPO_len)]
    #             ), axis = 0).reshape(-1)
    #     elif method == '2site':
    #         cmos_1_diag_stack = [contract('abbc->abc', cmos[i][1]) for i in range(M_MPO_len)]
    #         hdiag = torch.sum(torch.stack(
    #             [contract('ba,bcd,def,fg->aceg', ltensor_diag_stack[i], cmos_0_diag_stack[i], cmos_1_diag_stack[i], rtensor_diag_stack[i])
    #              for i in range(M_MPO_len)]
    #             ), axis = 0).reshape(-1)
    return hdiag

def turn22block(mat, max_bond, domain, size, cutoff=10**-18):
    # cutoff可能是不必要的
    DL, di, dj, DR = size
    # (DL*di) x (dj*DR)
    M = mat.reshape(DL * di, dj * DR)
    # econ SVD
    U, s, Vh = torch.linalg.svd(M, full_matrices=False, driver = dtype_config.driver)
    dcutoff = len(s[s>cutoff])
    d_keep = min(max_bond, dcutoff)
    error = sum(s[d_keep:]**2)
    # Turncate
    U_keep = U[:, :d_keep]
    Vh_keep = Vh[:d_keep]
    s_keep = s[:d_keep]
    s_keep_n = s_keep/torch.linalg.norm(s_keep)
    if domain == 'to_right':
        A_L = U_keep.reshape(DL, di, d_keep)
        A_R = contract('i,ij->ij', s_keep_n, Vh_keep).reshape(d_keep, dj, DR)
    else:
        A_L = contract('ij,j->ij', U_keep, s_keep_n).reshape(DL, di, d_keep)
        A_R = Vh_keep.reshape(d_keep, dj, DR)
    return A_L, A_R, error

def func_reshape(x, func, size):
    return func(x.reshape(size)).reshape(-1)

def mpo_exp(mps, mpo, mpo_type='S'):
    if mpo_type == 'S':
        env_L_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
        for seq_i, mps_s in enumerate(mps):
            env_L_tmp = env_L_i(mps_s, mpo[seq_i], env_L_tmp)
        exp = env_L_tmp.reshape(-1)
    elif mpo_type == 'M':
        M_MPO_len = len(mpo)
        exp = 0.0
        for L in range(M_MPO_len):
            env_L_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
            for seq_i, mps_s in enumerate(mps):
                env_L_tmp = env_L_i(mps_s, mpo[L][seq_i], env_L_tmp)
            exp += env_L_tmp.reshape(-1)
    return exp

def single_sweep(
    mps: List[Tensor],
    mpo: Union[List[Tensor],List[List[Tensor]]],
    environ: Union[Environ, List[Environ]],
    domain: str = 'to_right',
    method: str = '2site',
    algo: str = 'direct',
    mpo_type: str = 'S'
    #max_bond: int = 20,
    #cut_off: float = 10**-5,
):
    """_summary_

    Args:
        mps (List[Tensor]): _description_
        mpo (Union[List[Tensor],List[List[Tensor]]]): _description_
        environ (Union[List[Tensor], List[List[Tensor]]]): _description_
        domain (str, optional): 'to_right' means . Defaults to 'to_right'.
        method (str, optional): _description_. Defaults to '2site'.
        algo (str, optional): _description_. Defaults to 'direct'.
        mpo_type (str, optional): Single (S) or Multiple (M). Defaults to 'S'.

    Returns:
        _type_: _description_
    """
    if mpo_type != 'S':
        raise ValueError("Camps_single only supports a single full MPO.")
    mps_c = copy.deepcopy(mps)
    nsites = len(mps_c)
    # energies after optimizing each site
    micro_iteration_result = []
    if domain == 'to_right':
        idx_list = list(range(*[0, nsites, 1]))
    elif domain == 'to_left':
        idx_list = list(range(*[nsites-1, -1, -1]))
    else:
        assert False
    
    #itensor = np.ones([1,]*3)   # 这个是最左/右边的环境
    
    for imps in idx_list:
        if method == '2site' and (
            (domain == 'to_right' and imps == nsites-1)
            or ((not domain == 'to_right') and imps == 0)
        ):
            break
        
        if domain == 'to_right':
            lmethod, rmethod = "System", "Enviro"
        else:
            lmethod, rmethod = "Enviro", "System"
        
        if method == "1site":
            lidx = imps - 1
            cidx = [imps]
            ridx = imps + 1
        elif method == "2site":
            if domain == 'to_right':
                lidx = imps - 1
                cidx = [imps, imps + 1]
                ridx = imps + 2
            else:
                lidx = imps - 2
                cidx = [imps - 1, imps]  # center site
                ridx = imps + 1
        else:
            assert False
        logger.debug(f"optimize site: {cidx}, process: {domain}")
        
        if mpo_type == 'S':
            # if domain == 'to_right':
                # if lidx == -1:
                #     ltensor = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
                # else:
                #     ltensor = env_L_i(mps_c[lidx], mpo[lidx], ltensor) # 环境部分得存储下来，不能重复构建
                # rtensor = environ[nsites-ridx]
            # else:
            #     if ridx == nsites:
            #         rtensor = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
            #     else:
            #         rtensor = env_R_i(mps_c[ridx], mpo[ridx], rtensor)
            #     ltensor = environ[lidx+1]
            ltensor = environ.GetLR("L", lidx, mps_c, mpo, itensor=None, method=lmethod)
            rtensor = environ.GetLR("R", ridx, mps_c, mpo, itensor=None, method=rmethod)
            cmo = [mpo[idx] for idx in cidx]
            
        elif mpo_type == 'M':
            M_MPO_len = len(mpo)              # split the MPO into several MPOs
            # if domain == 'to_right':
            #     if lidx == -1:
            #         ltensor = [torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device) 
            #                    for _ in range(M_MPO_len)]   # Todo: 该部分之后可能会混合并行进行处理
            #     else:
            #         ltensor = [env_L_i(mps_c[lidx], mpo[i][lidx], ltensor[i]) for i in range(M_MPO_len)]
            #     rtensor = [environ[i][nsites-ridx] for i in range(M_MPO_len)]
            # else:
            #     if ridx == nsites:
            #         rtensor = [torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device) 
            #                    for _ in range(M_MPO_len)]
            #     else:
            #         rtensor = [env_R_i(mps_c[ridx], mpo[i][ridx], rtensor[i]) for i in range(M_MPO_len)]
            #     ltensor = [environ[i][lidx+1] for i in range(M_MPO_len)]
            ltensor = [environ_item.GetLR("L", lidx, mps_c, operator_item, itensor=None, method=lmethod) for environ_item, operator_item in zip(environ, mpo)]
            rtensor = [environ_item.GetLR("R", ridx, mps_c, operator_item, itensor=None, method=rmethod) for environ_item, operator_item in zip(environ, mpo)]
            cmo = [[mpo[i][idx] for idx in cidx] for i in range(M_MPO_len)]
        
        
        # 求解方法
        if algo.lower() == 'direct':
            if mpo_type == 'M':
                raise ValueError(
                    f"The direct diagonalization method does not support the case of the multiple MPO."
                )
            ham, size = get_ham_direct(ltensor, rtensor, cmo)
            w, v = torch.linalg.eigh(ham)
            e = w[0]
            c = v[:,0]
            cstruct = c.reshape(size)

        elif algo.lower() == 'davidson':
            # 构建试探矢量 v_init
            if method == '1site':
                v_init_mp = mps_c[cidx[0]]
                size = v_init_mp.shape
                v_init = v_init_mp.reshape(-1)
            else:
                v_init_mp_L, v_init_mp_R = mps_c[cidx[0]], mps_c[cidx[1]]
                v_init_mp = contract('abc,cde->abde', v_init_mp_L, v_init_mp_R)
                size = v_init_mp.shape
                v_init = v_init_mp.reshape(-1)
            # Davidson方法的参数设置
            if mpo_type == 'S':
                Hdiag = get_diags(ltensor, cmo, rtensor, method)   # 
                expr = get_Hx(ltensor, cmo, rtensor, method, size)
                Hx_func = partial(func_reshape, func = expr, size = size)
                masker = mask([], Hx_func)
                solver = eigenSolver()
                solver.iprt = -1
                solver.crit_vec = 1.e-4
                solver.crit_e = 1.e-8
                solver.nz = 1.e-8
                solver.maxcycle = 200
                solver.ndim = size.numel()
                solver.diag = Hdiag
                solver.neig = 1 # 这里直接设为常数进行尝试
                solver.HVec = masker.matvec
                solver.noise = True
                #solver.ifall = False
                solver.dtype = dtype_config.default_dtype
                # 求解
                eigs, civec, nmvp = solver.solve_iter(v0=v_init.unsqueeze(0), iop=4)
                e = eigs[0]
                c = civec[0]
                cstruct = c.reshape(size)
            elif mpo_type == 'M':
                Hdiag = torch.sum(torch.stack([get_diags(ltensor[i], cmo[i], rtensor[i], method) for i in range(M_MPO_len)]), axis=0)
                exprs = [get_Hx(ltensor[i], cmo[i], rtensor[i], method, size) for i in range(M_MPO_len)]
                Hx_funcs = [partial(func_reshape, func = exprs[i], size = size) for i in range(M_MPO_len)]
                maskers = [mask([], Hx_func) for Hx_func in Hx_funcs]
                solver = eigenSolver_M()
                solver.iprt = -1
                solver.crit_vec = 1.e-4
                solver.crit_e = 1.e-8
                solver.nz = 1.e-8
                solver.maxcycle = 200
                solver.ndim = size.numel()
                solver.diag = Hdiag
                solver.neig = 1 # 这里直接设为常数进行尝试
                solver.HVec = [masker.matvec for masker in maskers]
                solver.noise = True
                solver.M_MPO_len = M_MPO_len
                solver.dtype = dtype_config.default_dtype
                eigs, civec, nmvp = solver.solve_iter(v0=v_init.unsqueeze(0), iop=4)
                e = eigs[0]
                c = civec[0]
                cstruct = c.reshape(size)
                
        logger.debug(f"energy: {e}")
        micro_iteration_result.append((e, cidx))
        if method == "1site":
            mps_c[cidx[0]] = cstruct
        else:
            max_bond = mps_c[cidx[0]].shape[-1]
            A_L, A_R, error = turn22block(cstruct, max_bond, domain, size)
            mps_c[cidx[0]:cidx[1]+1] = [A_L, A_R]
            logger.debug(f"Turncation error: {error}.")
        mps = None
    return micro_iteration_result, mps_c

def DMRG(mps: List[Tensor],
         mpo: Union[List[Tensor],List[List[Tensor]]],
         method: str = '2site',
         sweep_time: int = 5,
         algo: str = 'direct',
         mpo_type: str = 'S'):  # 环境张量的构建需要更有效的写法
    if mpo_type != 'S':
        raise ValueError("Camps_single only supports a single full MPO.")
    iteration_result = []
    mps = rightCanonicalization(mps)
    environ = Environ(mps, mpo, 'R')
    for time in range(1, 1+sweep_time):
        logger.debug(f'Sweep {time}')
        logger.debug(f'Process: L->R')
        # env_R = get_envs_R(mps, mpo, method, mpo_type)
        micro_iteration_result, mps = single_sweep(mps, mpo, environ, 'to_right', method, algo, mpo_type)
        iteration_result.append(micro_iteration_result)
        logger.debug(f'Process: R->L')
        # env_L = get_envs_L(mps, mpo, method, mpo_type)
        micro_iteration_result, mps = single_sweep(mps, mpo, environ, 'to_left', method, algo, mpo_type)
        iteration_result.append(micro_iteration_result)
    return mps, iteration_result

def optimize_mps_disentangle(
    hams_lst: list[Hamiltonian] | Hamiltonian,
    clifford: Clifford,   # 考虑清楚这个的数据类型
    basis: list,
    Sweep0: int,
    Sweep1: int,
    dmax: int,
    n_sites: int,
    method: str = '2site',
    algo: str = 'davidson',
    n_dim: int = 2,
    use_orb: bool = False,
    mps_list: list[Tensor] = None,
    mpo_list: list[Tensor]= None,
    *,
    mps_is_optimized: bool = False,
    use_random_gates: bool = False,
    random_nums: int = 200,
    given_clifford: Clifford = None,
    alpha: int | float = 1,
    mpo_type: str = 'S',
    n_clusters: int = 10,
):
    if mpo_type != 'S':
        raise ValueError("Camps_single only supports a single full MPO.")
    if not isinstance(hams_lst, (tuple, list)):
        hams_lst = [hams_lst]
    assert len(hams_lst) == 1 or 2
    assert isinstance(alpha, (int, float)) and alpha >= 1
    hams = hams_lst[0]
    ham_array = hams["array"]
    Ham_coeff = hams["coeff"]
    nterms = len(ham_array)
    assert n_dim in (2, 4)
    assert (n_dim == 4 and use_orb) or (n_dim == 2 and not use_orb)
    
    if mpo_list is None:
        logger.info(f"Start Construct MPO")
        # get mpo in CPU
        mpo, model = construct_mpo_pauli(hams, basis, use_orb)
        # Transmit mpo to GPU
        mpo_list = [torch.from_numpy(mpo_s.array).to(dtype_config.default_dtype).to(dtype_config.device) 
                    for mpo_s in mpo]
        del mpo
        logger.info(f"End Construct MPO")
    
    if mps_list is None:
        logger.info(f"random MPS")
        # create a random mps in cpu
        mps = random_state(model, 0, dmax, percent=1.0)
        # to GPU
        mps_list_ini = [torch.from_numpy(mps_s.array).to(dtype_config.default_dtype).to(dtype_config.device) 
                        for mps_s in mps]
    else:
        logger.info(f"Copy MPS")
        # a GPU mps
        mps_list_ini = copy.deepcopy(mps_list)
    
    nqubits = len(mps_list_ini)
    # opt_ene length
    ene_length = nqubits if method == '1site' else nqubits-1
    
    logger.info(f"Start procedure-one optimize mps")
    if not mps_is_optimized:
        # DMRG in GPU
        mps_list_res, iteration_result_res = DMRG(mps_list_ini, mpo_list, method, Sweep0, algo, mpo_type)
        # Transmit energy to the CPU
        ene_process = [[iteration_result_res[i][k][0].to('cpu').numpy() for k in range(ene_length)] 
                       for i in range(2*Sweep0)]
        ene_init_arr = np.array(ene_process)
        ene_init_min = np.min(ene_init_arr, axis=1)
    else:
        mps_list_res = mps_list.copy()
        ene_init_min = np.array([0.0])
    logger.info(f"End procedure-one optimize mps")
    # calculate the expectation of energy in GPU
    e_old = mpo_exp(mps_list_res, mpo_list, mpo_type)
    
    if len(hams_lst) > 1:
        # 是否需要拆分
        # get mpo for op in CPU
        mpo_op, model_op = construct_mpo_pauli(hams_lst[1], basis, use_orb)
        # Transmit mpo to GPU
        mpo_op_list = [torch.from_numpy(mpo_s.array).to(dtype_config.default_dtype).to(dtype_config.device) 
                       for mpo_s in mpo_op]
        # calculate the expectation of energy in GPU
        ops_e_old = mpo_exp(mps_list_res, mpo_op_list)
    
    logger.info(f"dcut: {dmax}, alpha: {alpha}, dmax: {int(dmax * alpha)}")
    # the process for the minimization of the entropy in GPU
    sites_new, gates_idx, ee_diff, random_clifford = minimize_entropy_multiSweep(
        sites=mps_list_res,
        dmax= int(dmax * alpha),
        clifford=clifford,
        microiter=10,
        n_dim=n_dim,
        use_random_gates=use_random_gates,
        random_endian=clifford.endian,
        random_nums=random_nums,
        given_clifford=given_clifford,
    )
        
    if use_random_gates:
        res_clif = random_clifford
    else:
        res_clif = clifford
    logger.info(f"gate: {gates_idx}")
    
    # the process for the updating hamiltonian in the CPU
    gates_idx = gates_idx.to('cpu').numpy()
    ham_array_new, sign = update_ham_multiSweep(
        Ham_array=ham_array,
        idx=gates_idx,
        clifford=res_clif,
        n_sites=n_sites,
        n_dim=n_dim,
    )
    # get mpo for op in CPU
    ham_coeff_new = Ham_coeff * sign
    logger.info(f"Start Construct MPO")
    hams_new = Hamiltonian(array=ham_array_new, coeff=ham_coeff_new)
    mpo_new, model_new = construct_mpo_pauli(hams_new, basis, use_orb)
    for i, p in enumerate(mpo_new):
        logger.info(f"MPO: {i} shape: {p.array.shape}")
    # Transmit the full MPO to GPU.
    mpo_new_list = [torch.from_numpy(mpo_s.array).to(dtype_config.default_dtype).to(dtype_config.device) 
                    for mpo_s in mpo_new]
    del mpo_new
        
    logger.info(f"End Construct MPO")
        
    #--------Testing Na/Nb ops----
    if len(hams_lst) > 1:
        ops_array = hams_lst[1]["array"]
        ops_coeff = hams_lst[1]["coeff"]
        ops_array_new, sign = update_ham_multiSweep(
            Ham_array=ops_array,
            idx=gates_idx,
            clifford=res_clif,
            n_sites=n_sites,
            n_dim=n_dim,
        )
        ops_coeff_new = ops_coeff * sign
        logger.info(f"Start Construct MPO")
        ops_hams_new = Hamiltonian(array=ops_array_new, coeff=ops_coeff_new)
        
    # A must be greater than or equal to 1
    #if alpha>1:
    # A new DMRG process in GPU
    sites_tmp = leftCanonicalization(sites_new, dmax)
    mps_new_ini = rightCanonicalization(sites_tmp, dmax)
    
    e_new = mpo_exp(mps_new_ini, mpo_new_list, mpo_type)
    logger.info(f"mps-old expectation: {e_old}")
    logger.info(f"mps-new expectation: {e_new}")
    logger.info(f"Start procedure-two optimize mps")
    mps_new_res, iteration_new_res  = DMRG(mps_new_ini, mpo_new_list, method, Sweep1, algo, mpo_type)
    # energy output in CPU
    ene_process = [[iteration_new_res[i][k][0].to('cpu').numpy() for k in range(ene_length)] 
                   for i in range(2*Sweep1)]
    ene_new_arr = np.array(ene_process)
    ene_new_min = np.min(ene_new_arr, axis=1)
    logger.info(f"End procedure-two optimize mps")
    e_opt = mpo_exp(mps_new_res, mpo_new_list, mpo_type)
    logger.info(f"mps-opt expectation: {e_opt}")
    
    e = (ene_init_min, ene_new_min)
    
    if len(hams_lst) > 1:
        mpo_op, model_op = construct_mpo_pauli(ops_hams_new, basis, use_orb)
        mpo_op_list = [torch.from_numpy(mpo_s.array).to(dtype_config.default_dtype).to(dtype_config.device) 
                       for mpo_s in mpo_op]
        ops_e_new = mpo_exp(mps_new_ini, mpo_op_list)
        ops_e_opt = mpo_exp(mps_new_res, mpo_op_list)
        logger.info(f"Ops-expectations")
        logger.info(f"ops-e-old: {ops_e_old}")
        logger.info(f"ops-e-new: {ops_e_new}")
        logger.info(f"ops-e-opt: {ops_e_opt}")
    if use_random_gates:
        save_clifford = [res.to_dict() for res in random_clifford]
    else:
        save_clifford = None
    sites_dict = {"init": mps_list_ini, 
                  "disentangle": sites_new, 
                  "last": mps_new_res}
    save_info = SaveInfo(
            hams=[hams, hams_new],      # CPU
            clifford=save_clifford,     # CPU and GPU (gates in GPU)
            sites_dict=sites_dict,      # GPU
            energy=e,                   # CPU
            clifford_idx=gates_idx,     # CPU
            random_clifford=use_random_gates,
        )
    
    hams = Hamiltonian(array=ham_array_new, coeff=ham_coeff_new)
    res = [hams]
    if len(hams_lst) > 1:
        res.append(ops_hams_new)
    return e, mps_new_res, mpo_new_list, res, ee_diff, save_info
