import numpy as np
from numpy.typing import NDArray
import copy
import torch
# from torch.types import Tensor
# import scipy
from typing import Tuple, List, Union
from torch import Tensor
from functools import partial
from opt_einsum import contract #, contract_expression, contract_path
from pyfocus.camps.experimental.batched_einsum import TorchCachedBatchedEinsum
from pyfocus.camps.mps.disentangled import minimize_entropy_multiSweep, update_ham_multiSweep
from pyfocus.camps.mps.mpo import construct_mpo_pauli
from pyfocus.camps.mps.mps import random_state
from pyfocus.camps.mps.storage import MPSStorage, MPOStorage
from pyfocus.camps.experimental.pdvdson import eigenSolver, mask
from pyfocus.camps.mps.mps_simple import leftCanonicalization, rightCanonicalization
from pyfocus.camps.utils.config import dtype_config
from pyfocus.camps.utils.typing import Clifford, SaveInfo, Sites, Hamiltonian
from pyfocus.camps.utils.environ import Environ, env_L_i
from pyfocus.camps.utils.memorytrack import MemoryTrack, calculate_tensor_memory
from loguru import logger
import time

def create_mp_mpo(hams, basis, use_orb, nterms=None, rank=0, world_size=1, save_mode='save'):
    mpo, model = construct_mpo_pauli(hams, basis, use_orb)
    logger.info(f"MPO shapes: {[p.array.shape[-1] for i, p in enumerate(mpo)]}")
    logger.info(f"Start Construct MPO")
    mpo_new = [mpo_s.array for mpo_s in mpo]
    if save_mode == 'save':
        mpo_new = MPOStorage(mpo_new)
    del mpo
    logger.info(f"End Construct MPO")
    return mpo_new, model


def update_operator(ops_array, ops_coeff, gates_idx, cliff, n_sites, n_dim, rank=None):
    ops_array_new, sign = update_ham_multiSweep(
        Ham_array=ops_array,
        idx=gates_idx, 
        clifford=cliff,
        n_sites=n_sites,
        n_dim=n_dim,
    )
    ops_coeff_new = ops_coeff * sign
    return ops_array_new, ops_coeff_new

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

# def _peak_tensor_from_pathinfo(info, itemsize):
#     # info.size_list: 每一步 contraction 的输出张量元素数
#     # info.contraction_list[i][2]: 该步的 einsum_str, 形如 "kl,jk->jl"
#     imax = int(np.argmax(info.size_list))
#     einsum_str = info.contraction_list[imax][2]
#     out_inds = einsum_str.split("->", 1)[1]          # 例: "jl"
#     peak_shape = tuple(int(info.size_dict[c]) for c in out_inds)
#     peak_elems = int(info.size_list[imax])
#     peak_gb = (peak_elems * itemsize) / (1024**3)

#     return {
#         "step": imax,
#         "einsum_step": einsum_str,
#         "inds": out_inds,
#         "shape": peak_shape,
#         "numel": peak_elems,
#         "GB": peak_gb,
#     }

def get_Hx_batched(
    ltensor,
    cmos,
    rtensor,
    method,
    *,
    batch_char: str = "a",
    max_memory_gib: float = 0.2,
    safety: float = 1.05,
):
    """
    returns:
      expr(ctensor) -> output  (callable, compatible with your old code)
      engine: TorchCachedBatchedEinsum (so you can inspect cache_info / clear cache)
    """
    if method == "1site":
        eq = "abc, bdef, lfk, cek -> adl"
        # operands order: 0=ltensor, 1=cmos[0], 2=rtensor, 3=ctensor(variable)
        engine = TorchCachedBatchedEinsum(
            eq,
            batch_char=batch_char,
            max_memory_gib=max_memory_gib,
            safety=safety,
            constants_in_expr=[0, 1, 2],  # 注意：如果常量里含 batch_char，会被自动排除，不会 embed
        )
        engine.bind_operand(0, ltensor)
        engine.bind_operand(1, cmos[0])
        engine.bind_operand(2, rtensor)

        def expr(ctensor):
            # ctensor shape must match cshape (= tuple(size) in your code)
            return engine([None, None, None, ctensor])

    elif method == "2site":
        eq = "abc, bdef, fghj, ljk, cehk -> adgl"
        # operands order: 0=ltensor, 1=cmos[0], 2=cmos[1], 3=rtensor, 4=ctensor(variable)
        engine = TorchCachedBatchedEinsum(
            eq,
            batch_char=batch_char,
            max_memory_gib=max_memory_gib,
            safety=safety,
            constants_in_expr=[0, 1, 2, 3],
        )
        engine.bind_operand(0, ltensor)
        engine.bind_operand(1, cmos[0])
        engine.bind_operand(2, cmos[1])
        engine.bind_operand(3, rtensor)

        def expr(ctensor):
            return engine([None, None, None, None, ctensor])

    else:
        raise ValueError(f"Unknown method: {method}")

    return expr, engine

# def get_Hx(ltensor, cmos, rtensor, method, cshape,
#            optimize="auto", itemsize=16, memory_limit=None):
#     """
#     returns:
#       expr(ctensor, backend=...) -> output
#       stats: dict (peak intermediate shape + GB, path info)
#     """
#     # if dtype is None:
#     #     dtype = getattr(ltensor, "dtype", np.complex128)

#     if method == "1site":
#         eq = "abc, bdef, lfk, cek -> adl"
#         shapes = (ltensor.shape, cmos[0].shape, rtensor.shape, cshape)
#         constants = [0, 1, 2]   # ltensor, cmos[0], rtensor 常量；只留下 cek 作为输入
#         const_arrays = (ltensor, cmos[0], rtensor)

#     elif method == "2site":
#         eq = "abc, bdef, fghj, ljk, cehk -> adgl"
#         shapes = (ltensor.shape, cmos[0].shape, cmos[1].shape, rtensor.shape, cshape)
#         constants = [0, 1, 2, 3]
#         const_arrays = (ltensor, cmos[0], cmos[1], rtensor)

#     else:
#         raise ValueError(f"Unknown method: {method}")

#     # 1) 只用 shapes 拿收缩路径与 PathInfo（不会真的做收缩）
#     path, info = contract_path(eq, *shapes, shapes=True,
#                                  optimize=optimize, memory_limit=memory_limit)  # :contentReference[oaicite:1]{index=1}

#     # 2) 用 PathInfo 报告峰值中间张量 shape 与显存
#     peak = _peak_tensor_from_pathinfo(info, itemsize)

#     # 3) 生成 ContractExpression：只需要传入 cshape 对应的张量
#     # 注意：constant 的位置传“真实数组”，非 constant 位置传“shape tuple”
#     expr = contract_expression(
#         eq,
#         *const_arrays,
#         cshape,
#         constants=constants,
#         optimize=path,  # 用刚刚算出的路径（list-of-tuples 也是合法 optimize 参数）:contentReference[oaicite:2]{index=2}
#     )
#     logger.info(f"max shape: {peak['shape']}, max memory: {peak['GB']} GB")
#     # stats = {
#     #     "eq": eq,
#     #     "path": path,
#     #     "peak_intermediate": peak,     # {"shape": ..., "GB": ..., "einsum_step": ...}
#     #     "largest_intermediate_numel": int(info.largest_intermediate),  # :contentReference[oaicite:3]{index=3}
#     # }
#     return expr

# def get_Hx(ltensor, cmos, rtensor, method, cshape):
#     if method == '1site':
#         assert ltensor.shape[-1] == cshape[0] and cmos[0].shape[-2] == cshape[1] and \
#             rtensor.shape[-1] == cshape[2]
#         expr = contract_expression(
#                 "abc, bdef, lfk, cek -> adl",
#                 ltensor, cmos[0], rtensor, cshape,
#                 constants=[0, 1, 2],
#             )
#     elif method == '2site':
#         assert ltensor.shape[-1] == cshape[0] and cmos[0].shape[-2] == cshape[1] and \
#             cmos[1].shape[-2] == cshape[2] and rtensor.shape[-1] == cshape[3]
#         expr = contract_expression(
#                 "abc, bdef, fghj, ljk, cehk -> adgl",
#                 ltensor, cmos[0], cmos[1], rtensor, cshape,
#                 constants=[0, 1, 2, 3],
#             )
#     # _print_path_info(expr)
#     return expr

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
    return hdiag

def turn22block(mat, max_bond, domain, size):
    # cutoff可能是不必要的
    DL, di, dj, DR = size
    # (DL*di) x (dj*DR)
    M = mat.reshape(DL * di, dj * DR)
    # econ SVD
    if dtype_config.device == 'cpu':
        U, s, Vh = torch.linalg.svd(M, full_matrices=False,)
    else:
        U, s, Vh = torch.linalg.svd(M, full_matrices=False, driver=dtype_config.driver)
    #dcutoff = len(s[s>cutoff])
    #d_keep = min(max_bond, dcutoff)
    error = sum(s[max_bond:]**2)
    # Turncate
    U_keep = U[:, :max_bond]
    Vh_keep = Vh[:max_bond]
    s_keep = s[:max_bond]
    s_keep_n = s_keep/torch.linalg.norm(s_keep)
    if domain == 'to_right':
        A_L = U_keep.reshape(DL, di, max_bond).contiguous()
        A_R = contract('i,ij->ij', s_keep_n, Vh_keep).reshape(max_bond, dj, DR).contiguous()
        A_L_c = A_L.clone()
        A_R_c = A_R.clone()
    else:
        A_L = contract('ij,j->ij', U_keep, s_keep_n).reshape(DL, di, max_bond).contiguous()
        A_R = Vh_keep.reshape(max_bond, dj, DR).contiguous()
        A_L_c = A_L.clone()
        A_R_c = A_R.clone()
    del A_L, A_R
    return A_L_c, A_R_c, error

def func_reshape(x, func, size):
    return func(x.reshape(size)).reshape(-1)

def mpo_exp(mps, mpo, save_mode):
    env_L_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
    if save_mode == 'save':
        length = mps.get_system_size()
    elif save_mode == 'normal':
        length = len(mps)
    for seq in range(length):
        if save_mode == 'save':
            mps_i = torch.from_numpy(mps.read(seq)).to(dtype_config.device)
            mpo_i = torch.from_numpy(mpo.read(seq)).to(dtype_config.device)
        elif save_mode == 'normal':
            mps_i = torch.from_numpy(mps[seq]).to(dtype_config.device)
            mpo_i = torch.from_numpy(mpo[seq]).to(dtype_config.device)
        env_L_tmp = env_L_i(mps_i, mpo_i, env_L_tmp)
    exp = env_L_tmp.reshape(-1)
    return exp.to('cpu').numpy()

def mpo_exp_s(mps, mpo, save_mode):  # 这个一般用于很小的mpo的情况
    # mpo不进行存储
    env_L_tmp = torch.ones([1,]*3, dtype=dtype_config.default_dtype).to(dtype_config.device)
    if save_mode == 'save':
        length = mps.get_system_size()
    elif save_mode == 'normal':
        length = len(mps)
    for seq in range(length):
        mpo_i = torch.from_numpy(mpo[seq]).to(dtype_config.device)
        if save_mode == 'save':
            mps_i = torch.from_numpy(mps.read(seq)).to(dtype_config.device)
        elif save_mode == 'normal':
            mps_i = torch.from_numpy(mps[seq]).to(dtype_config.device)            
        env_L_tmp = env_L_i(mps_i, mpo_i, env_L_tmp)
    exp = env_L_tmp.reshape(-1)
    return exp.to('cpu').numpy()

# 仅将收缩操作放置GPU进行，其他操作均使用CPU操作，最大限度减少
def single_sweep(
    mps: Union[list[Tensor], MPSStorage] ,
    mpo: Union[list[Tensor], MPSStorage] ,
    environ: Union[Environ, List[Environ]],
    domain: str = 'to_right',
    method: str = '2site',
    algo: str = 'direct',
    save_mode: str = 'save'
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
    dtype = dtype_config.default_dtype
    device = dtype_config.device
    # mps_c = copy.deepcopy(mps)
    # np.savez(f'test/mps_init.npz', *mps_c)
    if save_mode == 'save':
        nsites = mps.get_system_size()  # mps需要是read
    elif save_mode == 'normal':
        nsites = len(mps)
    else:
        assert False
    # energies after optimizing each site
    micro_iteration_result = []
    if domain == 'to_right':
        idx_list = list(range(*[0, nsites, 1]))
    elif domain == 'to_left':
        idx_list = list(range(*[nsites-1, -1, -1]))
    else:
        assert False
    
    #itensor = np.ones([1,]*3)   # 这个是最左/右边的环境
    ltensor = None
    rtensor = None
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
        logger.info(f"optimize site: {cidx}, process: {domain}")
        with MemoryTrack(torch.tensor(0, dtype=dtype, device=device).device) as track:
            ltensor = environ.GetLR("L", lidx, mps, mpo, itensor=ltensor, method=lmethod)  # may lead to a bug?
            rtensor = environ.GetLR("R", ridx, mps, mpo, itensor=rtensor, method=rmethod)
            if save_mode == 'save':
                cmo = [torch.from_numpy(mpo.read(idx)).to(dtype=dtype, device=device) for idx in cidx]
            elif save_mode == 'normal':
                cmo = [torch.from_numpy(mpo[idx]).to(dtype=dtype, device=device) for idx in cidx]
            if method == "2site":
                if save_mode == 'save':
                    max_bond = mps.read(cidx[0]).shape[-1]
                elif save_mode == 'normal':
                    max_bond = mps[cidx[0]].shape[-1]
            lmem = calculate_tensor_memory(ltensor)
            rmem = calculate_tensor_memory(rtensor)
            cmomem = sum([calculate_tensor_memory(cmo_s) for cmo_s in cmo])
            logger.info(f'ltensor using {lmem} GB, rtensor using {rmem} GB, mpos using {cmomem} GB.')
            logger.info(f'The total of these use {lmem+rmem+cmomem} GB.')
                # 求解方法
            if algo.lower() == 'direct':
                ham, size = get_ham_direct(ltensor, rtensor, cmo)
                w, v = torch.linalg.eigh(ham)
                e = w[0]
                c = v[:,0]
                cstruct = c.reshape(size)
            elif algo.lower() == 'davidson':
                # 构建试探矢量 v_init
                if method == '1site':
                    if save_mode == 'save':
                        v_init_mp = torch.from_numpy(mps.read(cidx[0])).to(dtype=dtype, device=device)
                    elif save_mode == 'normal':
                        v_init_mp = torch.from_numpy(mps[cidx[0]]).to(dtype=dtype, device=device)
                    size = torch.tensor(v_init_mp.shape).to(device=device)
                    v_init = v_init_mp.reshape(-1)
                else:
                    if save_mode == 'save':
                        v_init_mp_L = torch.from_numpy(mps.read(cidx[0])).to(dtype=dtype, device=device)
                        v_init_mp_R = torch.from_numpy(mps.read(cidx[1])).to(dtype=dtype, device=device)
                    elif save_mode == 'normal':
                        v_init_mp_L = torch.from_numpy(mps[cidx[0]]).to(dtype=dtype, device=device)
                        v_init_mp_R = torch.from_numpy(mps[cidx[1]]).to(dtype=dtype, device=device)
                    max_bond = v_init_mp_L.shape[-1]
                    v_init_mp = contract('abc,cde->abde', v_init_mp_L, v_init_mp_R)
                    size = torch.tensor(v_init_mp.shape).to(device=device)
                    del v_init_mp_L, v_init_mp_R
                    v_init = v_init_mp.reshape(-1)
                # Davidson方法的参数设置
                Hdiag = get_diags(ltensor, cmo, rtensor, method)
                expr, engine = get_Hx_batched(ltensor, cmo, rtensor, method,
                                                batch_char="a", max_memory_gib=3.0,          # 例：切 c 这个 contracted bond
                                                )
                # expr = get_Hx(ltensor, cmo, rtensor, method, tuple(size))
                Hx_func = partial(func_reshape, func = expr, size = tuple(size))
                masker = mask([], Hx_func)
                solver = eigenSolver()
                solver.iprt = -1
                solver.crit_vec = 1.e-4
                solver.crit_e = 1.e-8
                solver.nz = 1.e-8
                solver.maxcycle = 50
                solver.ndim = torch.prod(size)
                solver.diag = Hdiag
                solver.neig = 1 # 这里直接设为常数进行尝试
                solver.HVec = masker.matvec
                solver.noise = False
                solver.dtype = dtype
                # 求解
                # if torch.cuda.is_available():
                #     torch.cuda.synchronize()
                #     start_event = torch.cuda.Event(enable_timing=True)
                #     end_event = torch.cuda.Event(enable_timing=True)
                #     start_event.record()
                eigs, civec, nmvp = solver.solve_iter(v0=v_init.unsqueeze(0), iop=4)
                # if torch.cuda.is_available():
                #     end_event.record()
                #     torch.cuda.synchronize()  # 等待CUDA操作完成
                #     elapsed_time = start_event.elapsed_time(end_event) / 1000.0
                #     logger.info(f'[time needed for Davidson algorithm]: {elapsed_time}.')
                e = eigs[0]
                c = civec[0]
                cstruct = c.reshape(tuple(size))
                engine.clear_expr_cache()
                del v_init, solver, masker, Hx_func, expr, Hdiag, engine
            track.manually_clean_cache((ltensor, rtensor, *cmo))
            
            logger.info(f"energy: {e}")
            micro_iteration_result.append((e, cidx))
            
            if method == "1site":
                if save_mode == 'save':
                    mps.close()
                    mps.reopen('a')
                    mps.write(cidx[0], cstruct.to('cpu').numpy())
                    mps.close()
                    mps.reopen('r')
                elif save_mode == 'normal':
                    mps[cidx[0]] = cstruct.to('cpu').numpy()
            else:
                A_L, A_R, error = turn22block(cstruct, max_bond, domain, size)
                logger.info(f"Turncation error: {error}.")
                if save_mode == 'save':
                    mps.close()
                    mps.reopen('a')
                    mps.write(cidx[0], A_L.to('cpu').numpy())
                    mps.write(cidx[1], A_R.to('cpu').numpy())
                    mps.close()
                    mps.reopen('r')
                elif save_mode == 'normal':
                    mps[cidx[0]] = A_L.to('cpu').numpy()
                    mps[cidx[1]] = A_R.to('cpu').numpy()
                track.manually_clean_cache((cstruct, A_L, A_R))

    return micro_iteration_result, mps

def DMRG(mps: Union[list[NDArray], MPSStorage],
         mpo: Union[list[NDArray], MPSStorage],
         method: str = '2site',
         sweep_time: int = 5,
         algo: str = 'direct',
         save_mode: str = 'save'):  # 环境张量的构建需要更有效的写法
    # save_mode: 'save' and 'normal'.
    iteration_result = []
        
    # 注意临时文件的命名
    environ = Environ(mps, mpo, 'R', save_mode)
    for time in range(1, 1+sweep_time):
        logger.info(f'Sweep {time}')
        logger.info(f'Process: L->R')
        # env_R = get_envs_R(mps, mpo, method, mpo_type)
        micro_iteration_result, mps = single_sweep(mps, mpo, environ, 'to_right', method, algo, save_mode)
        iteration_result.append(micro_iteration_result)
        # exit()
        logger.info(f'Process: R->L')
        # env_L = get_envs_L(mps, mpo, method, mpo_type)
        micro_iteration_result, mps = single_sweep(mps, mpo, environ, 'to_left', method, algo, save_mode)
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
    mps_c: Union[list[Tensor], MPSStorage] = None,
    mpo_c: Union[list[Tensor], MPOStorage]= None,
    *,
    mps_is_optimized: bool = False,
    use_random_gates: bool = False,
    random_nums: int = 200,
    given_clifford: Clifford = None,
    alpha: int | float = 1,
    save_mode: str = 'save',
):
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
    
    dtype = dtype_config.default_dtype
    device = dtype_config.device
    
    if mpo_c is None:
        mpo_c, model = create_mp_mpo(hams, basis, use_orb, save_mode=save_mode)
    elif save_mode == 'save' and type(mpo_c) is list:
        mpo_c = MPOStorage(mpo_c)
        model = None
    else:
        model = None
    
    if mps_c is None:
        if model is None:
            raise ValueError("mps_c must be provided when mpo_c is provided.")
        logger.info(f"random MPS")
        # create a random mps in cpu
        mps = random_state(model, 0, dmax, percent=1.0)
        # to GPU
        # mps_list_ini = [torch.from_numpy(mps_s.array).to(dtype_config.default_dtype).to(dtype_config.device) 
        #                 for mps_s in mps]
        mps_c = [mps_s.array for mps_s in mps]  # 加入左右正则化
        if save_mode == 'normal':   # 所有的东西都先放内存
            mps_c = [torch.from_numpy(mps_c[i]).to(dtype=dtype, device=device) for i in range(n_sites)]
            mps_c = leftCanonicalization(mps_c)
            mps_c = rightCanonicalization(mps_c)
            mps_c = [mps_c[i].to('cpu').numpy() for i in range(n_sites)]
        elif save_mode == 'save':   # 所有的东西都先放硬盘
            mps_c = MPSStorage(mps_c)
            mps_c.leftCanonicalization()
            mps_c.rightCanonicalization()
            mps_c.close()
            mps_c = MPSStorage()  # 所有进程都读取h5文件
    else:
        logger.info(f"Copy MPS")
        # a CPU mps
        if save_mode == 'normal':
            mps_c = [torch.from_numpy(mps_c[i]).to(dtype=dtype, device=device).to('cpu').numpy() for i in range(n_sites)]
        elif save_mode == 'save' and type(mps_c) is list:
            mps_tmp = MPSStorage(mps_c)
            #mps_tmp.leftCanonicalization()
            #mps_tmp.rightCanonicalization()
            mps_tmp.close()
            mps_c = MPSStorage()
        elif save_mode == 'save' and not isinstance(mps_c, MPSStorage):
            mps_c = MPSStorage() # 所有进程都读取h5文件

        # mps_ini = copy.deepcopy(mps_c)
    
    ene_length = n_sites if method == '1site' else n_sites-1
    if not mps_is_optimized:
        # DMRG in GPU
        logger.info(f"Start procedure-one optimize mps")
        mps_c, iteration_result_res = DMRG(mps_c, mpo_c, method, Sweep0, algo, save_mode)
        # Transmit energy to the CPU
        ene_process = [[iteration_result_res[i][k][0].to('cpu').numpy() for k in range(ene_length)] 
                        for i in range(2*Sweep0)]
        ene_init_arr = np.array(ene_process)
        ene_init_min = np.min(ene_init_arr, axis=1)
        logger.info(f"End procedure-one optimize mps")
    else:
        # mps_list_res = copy.deepcopy(mps_c)
        ene_init_min = np.array([0.0])
    # exit()
    
    # calculate the expectation of energy in GPU
    e_old = mpo_exp(mps_c, mpo_c, save_mode)
    
    if save_mode == 'save':
        mpo_c.delete_file()
    elif save_mode == 'normal':
        del mpo_c
    
    if len(hams_lst) > 1:
        # 是否需要拆分
        # get mpo for op in CPU
        mpo_op, model_op = construct_mpo_pauli(hams_lst[1], basis, use_orb)
        mpo_op_list = [mpo_s.array for mpo_s in mpo_op]
        # calculate the expectation of energy in GPU
        ops_e_old = mpo_exp_s(mps_c, mpo_op_list, save_mode)
    logger.info(f"dcut: {dmax}, alpha: {alpha}, dmax: {int(dmax * alpha)}")
    # the process for the minimization of the entropy in GPU
    sites_new, gates_idx, ee_diff, random_clifford = minimize_entropy_multiSweep(
        sites=mps_c,
        dmax= int(dmax * alpha),
        clifford=clifford,
        microiter=10,
        n_dim=n_dim,
        use_random_gates=use_random_gates,
        random_endian=clifford.endian,
        random_nums=random_nums,
        given_clifford=given_clifford,
        save_mode = save_mode,
    )   # 此时sites_new是做完Clifford变换后的mps
    if use_random_gates:
        res_clif = random_clifford
    else:
        res_clif = clifford
    logger.info(f"gate: {gates_idx}")

    # the process for the updating hamiltonian in the CPU
    gates_idx = gates_idx.to('cpu').numpy()
    ham_array_new, ham_coeff_new = update_operator(ham_array, Ham_coeff, gates_idx, res_clif, n_sites, n_dim)
    hams_new = Hamiltonian(array=ham_array_new, coeff=ham_coeff_new)
    mpo_new, model_new = create_mp_mpo(hams_new, basis, use_orb, save_mode=save_mode)
    logger.info(f"End Construct MPO")
        
    #--------Testing Na/Nb ops----
    if len(hams_lst) > 1:
        ops_array = hams_lst[1]["array"]
        ops_coeff = hams_lst[1]["coeff"]
        ops_array_new, ops_coeff_new = update_operator(ops_array, ops_coeff, gates_idx, res_clif, n_sites, n_dim)
        ops_hams_new = Hamiltonian(array=ops_array_new, coeff=ops_coeff_new)
    
    # 将仅在主进程中进行的解纠缠mps得到传递给其他进程
    if save_mode == 'save':
        mps_res = MPSStorage(sites_new.read_all(), file_name = 'mps_res.h5')
        sites_new.reopen('r')
        mps_res.close()
        mps_res = MPSStorage(file_name='mps_res.h5')  # 以read模式进行读取
    elif save_mode == 'normal':
        mps_res = copy.deepcopy(sites_new)
        
    e_new = mpo_exp(mps_res, mpo_new, save_mode)
    logger.info(f"mps-old expectation: {e_old}")
    logger.info(f"mps-new expectation: {e_new}")
    logger.info(f"Start procedure-two optimize mps")
        
    mps_res, iteration_new_res = DMRG(mps_res, mpo_new, method, Sweep1, algo, save_mode)
    # energy output in CPU
    e_opt = mpo_exp(mps_res, mpo_new, save_mode)
    ene_process = [[iteration_new_res[i][k][0].to('cpu').numpy() for k in range(ene_length)] 
                   for i in range(2*Sweep1)]
    ene_new_arr = np.array(ene_process)
    ene_new_min = np.min(ene_new_arr, axis=1)
    logger.info(f"End procedure-two optimize mps")
    logger.info(f"mps-opt expectation: {e_opt}")
    
    e = (ene_init_min, ene_new_min)
    
    if len(hams_lst) > 1:
        mpo_op, model_op = construct_mpo_pauli(ops_hams_new, basis, use_orb)
        mpo_op_list = [mpo_s.array for mpo_s in mpo_op]
        ops_e_new = mpo_exp_s(sites_new, mpo_op_list, save_mode)
        ops_e_opt = mpo_exp_s(mps_res, mpo_op_list, save_mode)
        del mpo_op_list
        logger.info(f"Ops-expectations")
        logger.info(f"ops-e-old: {ops_e_old}")
        logger.info(f"ops-e-new: {ops_e_new}")
        logger.info(f"ops-e-opt: {ops_e_opt}")
    if use_random_gates:
        save_clifford = [res.to_dict() for res in random_clifford]
    else:
        save_clifford = None
    if save_mode == 'save':
        sites_dict = {"init": mps_c.read_all(), 
                      "disentangle": sites_new.read_all(), 
                      "last": mps_res.read_all()}
    elif save_mode == 'normal':
        sites_dict = {"init": mps_c, 
                      "disentangle": sites_new, 
                      "last": mps_res}
    # 之后再考虑这个问题怎么写
    save_info = SaveInfo(
            hams=[hams, hams_new],      # CPU
            clifford=save_clifford,     # CPU and GPU (gates in GPU)
            sites_dict=sites_dict,      # CPU
            energy=e,                   # CPU
            clifford_idx=gates_idx,     # CPU
            random_clifford=use_random_gates,
        )
    if save_mode == 'save':
        mps_c.close()
        sites_new.close()
        mps_c.delete_file()
        sites_new.delete_file()
        mps_c = MPSStorage(mps_res.read_all(), 'mps.h5')
        mps_c.close()
        mps_res.close()
        mps_c = MPSStorage()
        mps_res.delete_file()
    else:
        del mps_c
        mps_c = copy.deepcopy(mps_res)
        del sites_new, mps_res
    res = [hams_new]
    if len(hams_lst) > 1:
        res.append(ops_hams_new)
    return e, mps_c, mpo_new, res, ee_diff, save_info
