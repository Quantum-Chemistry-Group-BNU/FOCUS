# CAMPS

CAMPS implements Clifford-augmented matrix product state workflows for reducing entanglement in electronic-structure simulations. The method follows the idea of using local Clifford disentanglers to reshape the entanglement structure of a qubit Hamiltonian while preserving its Pauli-string form.

The implementation in FOCUS is intended for qubit Hamiltonians obtained from molecular integrals and fermion-to-qubit mappings such as Jordan-Wigner, parity, or Bravyi-Kitaev. The main workflow combines a low-bond-dimension DMRG calculation, Clifford disentangling of the resulting MPS, Hamiltonian transformation, MPO reconstruction, and a follow-up DMRG optimization.

## Method Summary

The CAMPS workflow is based on the following observations.

- MPS accuracy at a fixed bond dimension is limited by bipartite entanglement.
- Clifford operators map Pauli strings to Pauli strings under conjugation, so a Clifford-transformed qubit Hamiltonian can still be represented as a Pauli expansion.
- Local Clifford gates can be selected by minimizing the bipartite Renyi entropy across MPS bonds.
- The optimized Clifford transformation is then applied to the Hamiltonian, and the transformed Hamiltonian is converted back to an MPO for the next DMRG step.

For the electronic-structure tests described in the accompanying manuscript, Clifford disentanglers reduce the energy error at fixed MPS bond dimension for the studied systems, reduce sensitivity to orbital ordering, and provide an entropy-based diagnostic for whether a CAMPS optimization is likely to help. The method can also be used as Hamiltonian preprocessing for shallow-circuit VQE calculations.

## Package Layout

```text
pyfocus/camps/
  experimental/
    DMRG.py              # current CAMPS/DMRG driver
    simple_DMRG.py       # simpler reference driver
    pdvdson.py           # Davidson solver utilities
  examples/
    h_chain_sto3g_tets.py
  mps/
    mps.py               # MPS initialization helpers
    mpo.py               # Pauli Hamiltonian to MPO construction
    operators.py         # integral/OpenFermion to Pauli Hamiltonian helpers
    disentangled.py      # Clifford entropy-minimization and Hamiltonian updates
    storage.py           # HDF5-backed MPS/MPO storage
  linalg/
    batched_svd.py       # batched SVD backend
  utils/
    clifford.py          # Clifford gate helpers
    config.py            # dtype/device configuration
    typing.py            # typed containers for Hamiltonian and Clifford data
```

## Installation And Imports

Install the CAMPS requirements and FOCUS from the repository root:

```bash
python -m pip install -r pyfocus/camps/requirements.txt
python -m pip install -e .
```

If your Python environment already provides GPU-enabled PyTorch, install or verify that environment first, then install the remaining CAMPS requirements.

Use the package through the `pyfocus.camps` namespace:

```python
from pyfocus.camps.experimental.DMRG import optimize_mps_disentangle
from pyfocus.camps.utils.config import dtype_config, batch_svd_config
from pyfocus.camps.utils.typing import Clifford, Hamiltonian
```

The historical top-level `camps` import path is not the preferred public entry point.

## Examples

Runnable examples are collected in `pyfocus/camps/examples/`.

| File | Description |
| --- | --- |
| `h_chain_sto3g_tets.py` | Linear H-chain STO-3G CAMPS smoke test converted from the old notebook. It keeps parameters explicit at the top of the script and defaults to a small H4 calculation. |

Run the H-chain smoke test from the repository root:

```bash
python -m pyfocus.camps.examples.h_chain_sto3g_tets
```

Direct script execution is also supported:

```bash
python pyfocus/camps/examples/h_chain_sto3g_tets.py
```

The H-chain example uses `SAVE_MODE = "save"` by default so that MPS/MPO intermediates are stored through the HDF5-backed CAMPS storage classes. Switch it to `"normal"` in the script for small in-memory tests.

## Basic Calling Pattern

The driver expects Pauli-array Hamiltonian data, Clifford representative data, and an MPS/MPO basis compatible with the Renormalizer-backed MPO construction. Keep the parameters explicit in scripts so that the calculation setup is easy to audit.

```python
import numpy as np

from pyfocus.camps.experimental.DMRG import optimize_mps_disentangle
from pyfocus.camps.utils.config import dtype_config, batch_svd_config
from pyfocus.camps.utils.typing import Clifford, Hamiltonian

# Device and dtype settings.
dtype_config.apply(
    use_float64=True,
    use_complex=True,
    device="cuda",        # use "cpu" for CPU-only testing
    driver="gesvd",
    batch=-1,
    direct_limit=300,
    scratch_dir="./scratch",
)
batch_svd_config.apply(
    use_float64=True,
    use_complex=True,
    device="cuda",
    driver="gesvd",
    batch=-1,
    direct_limit=300,
    scratch_dir="./scratch",
)

# Pauli Hamiltonian: each row in ham_array encodes a Pauli string and ham_coeff
# stores the corresponding coefficient. Build these from PySCF/OpenFermion or
# another qubit-Hamiltonian generator.
ham_array = np.asarray([...], dtype=np.bytes_)
ham_coeff = np.asarray([...], dtype=np.complex128)
hamiltonian = Hamiltonian(array=ham_array, coeff=ham_coeff)

# Clifford data. For production calculations, use the preclassified 2Q or 4Q
# Clifford representatives distributed/generated for the calculation.
clifford = Clifford(
    gates=np.asarray(...),
    mapping=np.asarray(..., dtype=np.int64),
    phases=np.asarray(..., dtype=np.int64),
    endian="big",
)

# MPS/MPO settings.
N_SITES = 12
MAX_BOND = 64
SWEEP0 = 2
SWEEP1 = 2
METHOD = "2site"
ALGO = "davidson"
SAVE_MODE = "save"      # "normal" keeps arrays in memory; "save" uses HDF5 storage
USE_ORB = False         # False for 2Q spin-orbital CAMPS
N_DIM = 2               # 2 for 2Q Clifford gates; 4 for 4Q spatial-orbital gates

energy, mps, mpo, hams, entropy_change, save_info = optimize_mps_disentangle(
    hams_lst=hamiltonian,
    clifford=clifford,
    basis=basis,
    Sweep0=SWEEP0,
    Sweep1=SWEEP1,
    dmax=MAX_BOND,
    n_sites=N_SITES,
    method=METHOD,
    algo=ALGO,
    n_dim=N_DIM,
    use_orb=USE_ORB,
    mps_c=None,
    mpo_c=None,
    mps_is_optimized=False,
    use_random_gates=False,
    random_nums=200,
    given_clifford=None,
    alpha=1,
    save_mode=SAVE_MODE,
)
```

The returned tuple contains the energy history, final MPS object, transformed MPO object, transformed Hamiltonian data, entropy-change information, and a `SaveInfo` dictionary with intermediate MPS and Clifford data.

## 2Q And 4Q Modes

Use `n_dim=2` and `use_orb=False` for the spin-orbital 2Q Clifford workflow. This is the cheaper and usually preferred mode.

Use `n_dim=4` and `use_orb=True` for spatial-orbital 4Q Clifford workflows. The Clifford search space is much larger, so this mode is substantially more expensive and should be used only when the basis/model setup supports it.

## Storage Modes

`save_mode="normal"` keeps tensors in memory. This is convenient for small smoke tests.

`save_mode="save"` stores intermediate MPS/MPO tensors through the HDF5-backed storage classes in `pyfocus.camps.mps.storage`. This mode is more appropriate when the transformed MPO becomes large, which is common for larger electronic-structure systems after Clifford transformation.

## Practical Notes

- CAMPS generally does not preserve particle-number symmetry because generic Clifford disentanglers do not preserve U(1) symmetry. Dense tensor algebra is therefore used.
- The entropy reduction after the disentangling step is a useful diagnostic, but it is empirical rather than a strict guarantee of final energy improvement.
- 4Q Clifford disentanglers have a much larger search space than 2Q disentanglers; the manuscript reduces this through equivalence-class classification, but 4Q calculations remain significantly more expensive.
- For clean package usage, prefer `pyfocus.camps.*` imports in new code.
