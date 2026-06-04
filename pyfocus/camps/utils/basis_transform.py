import quimb
import quimb.tensor as qtn
import torch
import numpy as np
from pyfocus.camps.utils.config import dtype_config

def to_backend(x):
    import torch
    return torch.tensor(x, dtype=torch.complex128, device=dtype_config.device)

# input and output in quimb
def get_MPS_psi(mps_list):   # numpy.array和torch.tensor均可
    L = len(mps_list)
    mps_ini = []
    for i in range(L):
        if i == 0:
            data = mps_list[0][0].T
            mps_ini.append(torch.from_numpy(data).to(dtype_config.device))
        elif i == L - 1:
            data = mps_list[i][:, :, 0]
            mps_ini.append(torch.from_numpy(data).to(dtype_config.device))
        else:
            data = mps_list[i]
            data = data.transpose(0, 2, 1)
            mps_ini.append(torch.from_numpy(data).to(dtype_config.device))
    Psi0_MPS = qtn.MatrixProductState(mps_ini)
    return Psi0_MPS

def mps_psi2list(mps):
    mps_list_new = []
    length = mps.L
    for i in range(length):
        if i == 0:
            mps_list_new.append((mps[i].data.T)[None, :, :].cpu().numpy())
        elif i == length-1:
            mps_list_new.append((mps[i].data)[:, :, None].cpu().numpy())
        else:
            data = mps[i].data
            data = data.permute(0, 2, 1)
            mps_list_new.append(data.cpu().numpy())
    return mps_list_new

def parity_transform(mps, max_bond):
    length = mps.L
    cir = qtn.CircuitMPS(length, psi0=mps, max_bond=2*max_bond, cutoff=0.0, to_backend=to_backend)
    for i in range(length-1):
        cir.apply_gate('CX', qubits=[i, i+1])
    mps_new = cir.psi
    mps_new.compress(form = 'left', max_bond = max_bond)
    mps_new.compress(form = 'right', max_bond = max_bond)
    return mps_new

def jw_to_bk_cnots(n: int):
    # returns list of (control, target) in 0-indexed qubits
    ops = []
    for i in range(1, n + 1):
        lsb = i & -i
        j = i + lsb
        if j <= n:
            ops.append((i - 1, j - 1))
    return ops

def BK_transform(mps, max_bond):
    length = mps.L
    ops = jw_to_bk_cnots(length)
    cir = qtn.CircuitMPS(length, psi0=mps, max_bond=2*max_bond, cutoff=0.0, to_backend=to_backend)
    for op in ops:
        cir.apply_gate('CX', qubits=list(op))
    mps_new = cir.psi
    mps_new.compress(form = 'left', max_bond = max_bond)
    mps_new.compress(form = 'right', max_bond = max_bond)
    return mps_new

def turn2parityMPS(mps_list, max_bond):
    mps = get_MPS_psi(mps_list)
    mps_parity = parity_transform(mps, max_bond)
    mps_list_new = mps_psi2list(mps_parity)
    return mps_list_new

def turn2BKMPS(mps_list, max_bond):
    mps = get_MPS_psi(mps_list)
    mps_BK = BK_transform(mps, max_bond)
    mps_list_new = mps_psi2list(mps_BK)
    return mps_list_new

################################### to aabb ##############################
def min_adjacent_swaps_with_parallel(A, B):
    n = len(A)
    # 步骤 1: 计算目标位置映射 T
    T = {}
    for idx, x in enumerate(B, start=1):
        T[x] = idx  # 1-based 位置

    # 步骤 2: 构建序列 Q
    Q = [T[x] for x in A]  # Q[i] 是初始位置 i+1 的目标位置（1-based）

    steps = []  # 存储输出步骤
    t = 1  # 阶段计数器
    sorted_flag = False

    while not sorted_flag and t <= n:  # 最多 n 步
        swaps_step = []  # 当前并行步骤的交换对

        if t % 2 == 1:  # 奇数阶段
            for i in range(0, n - 1, 2):  # i 从 0 到 n-2 (0-based 索引), 对应 1-based 的奇数索引
                if Q[i] > Q[i + 1]:
                    # 执行交换
                    Q[i], Q[i + 1] = Q[i + 1], Q[i]
                    swaps_step.append((i + 1, i + 2))  # 转换为 1-based 索引输出
        else:  # 偶数阶段
            for i in range(1, n - 1, 2):  # i 从 1 到 n-2 (0-based 索引), 对应 1-based 的偶数索引
                if Q[i] > Q[i + 1]:
                    Q[i], Q[i + 1] = Q[i + 1], Q[i]
                    swaps_step.append((i + 1, i + 2))  # 转换为 1-based 索引输出

        # 如果当前步骤有交换，添加到输出
        if swaps_step:
            steps.append(swaps_step)

        # 检查序列是否有序 (Q 应为 [1,2,...,n])
        if all(Q[i] == i + 1 for i in range(n)):
            sorted_flag = True
        # 如果连续无交换且有序，可终止；这里简化：检查有序即终止
        elif not swaps_step and sorted_flag:  # 但 sorted_flag 在有序时已设
            pass

        t += 1

    return steps

def transJWababmps2JWaabbmps(mps, max_bond):
    length = mps.L
    no = length//2
    A = list(range(2*no))
    B = list(np.vstack((np.arange(no),np.arange(no,2*no))).T.reshape(-1))
    parallel_swaps = min_adjacent_swaps_with_parallel(A, B)[::-1]
    cir = qtn.CircuitMPS(length, psi0=mps, max_bond=16*max_bond, cutoff=0.0, to_backend=to_backend)
    for swaps in parallel_swaps:
        for (i, j) in swaps:
            cir.apply_gate('CZ', i - 1, j - 1)
            cir.apply_gate('SWAP', i - 1, j - 1)
    mps_new = cir.psi
    mps_new.compress(form = 'left', max_bond = 4*max_bond)
    mps_new.compress(form = 'right', max_bond = max_bond)
    return mps_new

def abab2aabb_mps(mps_list, max_bond):
    mps = get_MPS_psi(mps_list)
    mps_new = transJWababmps2JWaabbmps(mps, max_bond)
    mps_list_new = mps_psi2list(mps_new)
    return mps_list_new