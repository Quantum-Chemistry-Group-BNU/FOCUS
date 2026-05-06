import numpy as np
import opt_einsum
import torch
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any, Iterable, List, Optional, Sequence, Tuple
from loguru import logger


def _split_batch_idx(dim: int, min_batch: int) -> Tuple[int, ...]:
    n_batches = int(np.ceil(dim / min_batch))
    idx_arr = np.full(n_batches, min_batch, dtype=np.int64)
    idx_arr[-1] = dim - (n_batches - 1) * min_batch
    return tuple([0] + idx_arr.cumsum().tolist())


@dataclass(frozen=True)
class BatchPlan:
    input_subscripts: Tuple[str, ...]
    output_subscript: str
    size_dict: Dict[str, int]
    batch_char: str
    batch_dim: int
    # positions where batch_char appears: (operand_index, axis_index)
    pos: Tuple[Tuple[int, int], ...]
    # index list boundaries: (0, ..., dim)
    idx_lst: Tuple[int, ...]
    min_batch: int
    out_shape: Tuple[int, ...]
    # if batch_char appears in output, where is that output axis
    out_batch_axis: Optional[int]


class TorchCachedBatchedEinsum:
    """
    Cached opt_einsum.contract_expression + batching over any index char.

    - batch_char can be an output index (then we write into output slices), or a contracted index
      (then we accumulate into the full output; this is common for cutting internal bond dims).
    - constants_in_expr: operands embedded into expression via opt_einsum constants (true constants)
      NOTE: any operand that contains batch_char must NOT be embedded, otherwise you can't slice it.
    """

    _EXPR_CACHE: Dict[Any, Any] = {}

    def __init__(
        self,
        path: str,
        *,
        batch_char: str,
        max_memory_gib: float = 0.5,
        safety: float = 1.05,
        constants_in_expr: Optional[Iterable[int]] = None,
    ):
        self.path = path
        self.batch_char = batch_char
        self.max_memory_gib = float(max_memory_gib)
        self.safety = float(safety)
        self.constants_in_expr = tuple(sorted(set(constants_in_expr or ())))

        # you can bind fixed operands (like ltensor/rtensor/cmo) so __call__ only takes x
        self._bound: Dict[int, torch.Tensor] = {}

        # counters for debugging cache hits
        self._expr_builds = 0
        self._expr_hits = 0

    # ------------ cache control ------------
    @classmethod
    def clear_expr_cache(cls) -> None:
        cls._EXPR_CACHE.clear()
        cls._plan_cached.cache_clear()

    def cache_info(self) -> Dict[str, int]:
        return {
            "expr_cache_size": len(self._EXPR_CACHE),
            "expr_builds": self._expr_builds,
            "expr_hits": self._expr_hits,
            "plan_cache_info": self._plan_cached.cache_info()._asdict(),  # type: ignore
        }

    # ------------ binding fixed operands ------------
    def bind_operand(self, index: int, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("bind_operand expects a torch.Tensor")
        self._bound[int(index)] = tensor

    def unbind_operand(self, index: int) -> None:
        self._bound.pop(int(index), None)

    # ------------ planning ------------
    @staticmethod
    def _result_dtype_and_itemsize(arrays: Sequence[torch.Tensor]) -> Tuple[torch.dtype, int]:
        dt = arrays[0].dtype
        for a in arrays[1:]:
            dt = torch.promote_types(dt, a.dtype)   # ✅ 正确：dtype + dtype
        itemsize = torch.empty((), dtype=dt).element_size()
        return dt, int(itemsize)

    @classmethod
    @lru_cache(maxsize=256)
    def _plan_cached(
        cls,
        path: str,
        array_shapes: Tuple[Tuple[int, ...], ...],
        batch_char: str,
        max_memory_gib: float,
        itemsize: int,
        safety: float,
        inter_memory_override: Optional[float],
    ) -> BatchPlan:
        # parse using opt_einsum planner (no batching)
        info = opt_einsum.contract_path(path, *array_shapes, shapes=True)
        plan = info[1]

        input_subs = tuple(plan.input_subscripts.split(","))
        out_sub = plan.output_subscript
        size_dict = dict(plan.size_dict)

        if batch_char not in size_dict:
            raise ValueError(
                f"batch_char '{batch_char}' not found in size_dict; check your einsum subscripts."
            )

        batch_dim = int(size_dict[batch_char])
        out_shape = tuple(int(size_dict[c]) for c in out_sub)

        # find occurrences of batch_char in inputs
        pos: List[Tuple[int, int]] = []
        for i, subs in enumerate(input_subs):
            for axis, ch in enumerate(subs):
                if ch == batch_char:
                    pos.append((i, axis))

        if not pos:
            raise ValueError(f"batch_char '{batch_char}' does not appear in any input operand?!")

        # baseline largest intermediate memory (GiB)
        largest_elems_full = float(plan.largest_intermediate)
        largest_gib_full = (largest_elems_full * itemsize) / (2**30)

        # decide min_batch
        budget = float(max_memory_gib if inter_memory_override is None else inter_memory_override)
        if budget == -1:
            min_batch = batch_dim
        else:
            n = np.ceil((largest_gib_full / budget) * safety)
            n = max(1.0, float(n))
            min_batch = int(np.ceil(batch_dim / n))
            min_batch = max(1, min(min_batch, batch_dim))

        idx_lst = _split_batch_idx(batch_dim, min_batch)

        out_batch_axis = out_sub.find(batch_char)
        out_batch_axis = None if out_batch_axis < 0 else int(out_batch_axis)

        # recompute largest intermediate memory after batching (GiB)
        shapes_mod = [list(s) for s in array_shapes]
        for op_i, axis in pos:
            shapes_mod[op_i][axis] = int(min_batch)
        shapes_mod = tuple(tuple(s) for s in shapes_mod)

        info_b = opt_einsum.contract_path(path, *shapes_mod, shapes=True)
        plan_b = info_b[1]
        largest_elems_b = float(plan_b.largest_intermediate)
        largest_gib_b = (largest_elems_b * itemsize) / (2**30)

        # only keep these two logs
        logger.info(
            f"[compute Hx] batch_char='{batch_char}', dim {batch_dim} -> {min_batch} "
            f"(max_memory_gib={max_memory_gib}, itemsize={itemsize} bytes)."
        )
        logger.info(
            f"[compute Hx] largest intermediate memory: {largest_gib_full:.3f} GiB -> {largest_gib_b:.3f} GiB."
        )

        return BatchPlan(
            input_subscripts=input_subs,
            output_subscript=out_sub,
            size_dict=size_dict,
            batch_char=batch_char,
            batch_dim=batch_dim,
            pos=tuple(pos),
            idx_lst=idx_lst,
            min_batch=int(min_batch),
            out_shape=tuple(out_shape),
            out_batch_axis=out_batch_axis,
        )

    # ------------ expression building ------------
    def _expr_key(
        self,
        shapes_for_expr: Tuple[Tuple[int, ...], ...],
        constants_in_expr_effective: Tuple[int, ...],
        constant_ids: Tuple[int, ...],
    ) -> Any:
        # include constant_ids ONLY for those embedded constants, otherwise miss/cache confusion
        return ("torch_expr", self.path, self.batch_char, shapes_for_expr, constants_in_expr_effective, constant_ids)

    def _get_or_build_expr(
        self,
        shapes_for_expr: Tuple[Tuple[int, ...], ...],
        constants_in_expr_effective: Tuple[int, ...],
        constants_tensors: Dict[int, torch.Tensor],
    ):
        const_ids = tuple(id(constants_tensors[i]) for i in constants_in_expr_effective)
        key = self._expr_key(shapes_for_expr, constants_in_expr_effective, const_ids)

        expr = self._EXPR_CACHE.get(key)
        if expr is not None:
            self._expr_hits += 1
            return expr

        # build
        operands_spec = []
        constants_list = []
        for i in range(len(shapes_for_expr)):
            if i in constants_in_expr_effective:
                operands_spec.append(constants_tensors[i])  # embed real tensor
                constants_list.append(i)
            else:
                operands_spec.append(shapes_for_expr[i])    # just shape
        expr = opt_einsum.contract_expression(
            self.path,
            *operands_spec,
            constants=constants_list,
        )
        self._EXPR_CACHE[key] = expr   # 将expr存下来
        self._expr_builds += 1
        return expr

    # ------------ call ------------
    def __call__(
        self,
        operands: Sequence[Optional[torch.Tensor]],
        *,
        inter_memory_gib: Optional[float] = None,
    ) -> torch.Tensor:
        """
        operands: full operand list (same length as number of inputs in einsum).
                 You may pass None for those already bound via bind_operand().
        """
        ops: List[torch.Tensor] = []
        for i, t in enumerate(operands):
            if t is None:
                if i not in self._bound:
                    raise ValueError(f"Operand {i} is None and not bound.")
                ops.append(self._bound[i])
            else:
                ops.append(t)

        if not all(isinstance(t, torch.Tensor) for t in ops):
            raise TypeError("All operands must be torch.Tensor (or None for bound ones).")

        # device sanity
        dev = ops[0].device
        for t in ops[1:]:
            if t.device != dev:
                raise ValueError("All operands must be on the same device.")

        shapes = tuple(tuple(int(x) for x in t.shape) for t in ops)
        out_dtype, itemsize = self._result_dtype_and_itemsize(ops)

        plan = self._plan_cached(
            self.path, shapes, self.batch_char,
            self.max_memory_gib, itemsize, self.safety, inter_memory_gib
        )

        # operands that contain batch_char (need slicing)
        batched_operands = {i for (i, _) in plan.pos}

        # effective constants: only those (a) user requested, (b) not batched (cannot embed), (c) actually fixed in this call
        # In your case: rtensor, cmo 可以 embed；ltensor 如果被切片就不能 embed，但仍然可绑定为固定张量
        # 这个非常重要
        constants_tensors = {i: ops[i] for i in self.constants_in_expr}
        constants_in_expr_effective = tuple(
            i for i in self.constants_in_expr
            if (i not in batched_operands)
        )

        # build the template shapes for min_batch expr
        shapes_mod = [list(s) for s in shapes]
        for op_i, axis in plan.pos:
            shapes_mod[op_i][axis] = plan.min_batch
        shapes_mod = tuple(tuple(s) for s in shapes_mod)

        expr = self._get_or_build_expr(shapes_mod, constants_in_expr_effective, constants_tensors)

        # expression expects only non-embedded operands
        dyn_indices = [i for i in range(len(ops)) if i not in constants_in_expr_effective]

        # which axes to slice per operand
        affected_axes: Dict[int, List[int]] = {}
        for op_i, axis in plan.pos:
            affected_axes.setdefault(op_i, []).append(axis)

        out = torch.zeros(plan.out_shape, dtype=out_dtype, device=dev)

        for start, end in zip(plan.idx_lst[:-1], plan.idx_lst[1:]):
            bs = int(end - start)

            # assemble dynamic inputs for this batch
            dyn_inputs: List[torch.Tensor] = []
            for op_i in dyn_indices:
                t = ops[op_i]
                if op_i in affected_axes:
                    sl = [slice(None)] * t.ndim
                    for ax in affected_axes[op_i]:
                        sl[ax] = slice(start, end)
                    dyn_inputs.append(t[tuple(sl)])
                else:
                    dyn_inputs.append(t)

            if bs == plan.min_batch:
                val = expr(*dyn_inputs)
            else:
                # tail expr (cached by tail shapes)
                tail_shapes = [list(s) for s in shapes]
                for op_i, axis in plan.pos:
                    tail_shapes[op_i][axis] = bs
                tail_shapes = tuple(tuple(s) for s in tail_shapes)
                tail_expr = self._get_or_build_expr(tail_shapes, constants_in_expr_effective, constants_tensors)
                val = tail_expr(*dyn_inputs)

            if plan.out_batch_axis is None:
                # batch_char is contracted away -> accumulate
                out += val
            else:
                # write slice along the correct output axis
                slicer = [slice(None)] * out.ndim
                slicer[plan.out_batch_axis] = slice(start, end)
                out[tuple(slicer)] = val

        return out
