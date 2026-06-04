#!/usr/bin/env python
"""Linear H-chain STO-3G CAMPS example converted from the old TETS notebook."""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from loguru import logger
from pyscf import ao2mo, fci, gto, lo, scf

import pyfocus.camps as camps_pkg
from openfermion.transforms import jordan_wigner

from pyfocus.camps.experimental.DMRG import optimize_mps_disentangle
from pyfocus.camps.mps.mpo import construct_mpo_pauli
from pyfocus.camps.mps.mps import set_basis
from pyfocus.camps.mps.operators import get_fermion_ham, get_ham_from_qop, integral2pauli_JW
from pyfocus.camps.utils.clifford import check_operator_endian
from pyfocus.camps.utils.config import batch_svd_config, dtype_config
from pyfocus.camps.utils.integral import spinorb_from_spatial
from pyfocus.camps.utils.tools import setup_seed
from pyfocus.camps.utils.typing import Clifford


# =========================
# User-facing parameters
# =========================

N_H = 4
H_SPACING = 1.0
BASIS = "sto-3g"
LOCALIZED_ORB = True
LOCALIZED_METHOD = "meta-lowdin"

DEVICE = "auto"  # "auto", "cpu", or "cuda"
SEED = 2025
USE_FLOAT64 = True
USE_COMPLEX = True
SVD_DRIVER = "gesvd"
BATCH_SVD_DRIVER = "gesvda"

DCUT = 4
SWEEP0 = 1
SWEEP1 = 1
N_MICRO = 1
ENTROPY_MICROITER = 3
ALGO = "direct"  # Set to "davidson" for the large-H-chain workflow in the old notebook.

SAVE_MODE = "save"  # Matches the old notebook path: MPS/MPO use HDF5 storage during DMRG.
PENALTY_NANB = 0.1
KEEP_INTEGRALS_PATH = None  # Example: Path("examples/camps/H4_sto3g_integrals.npz")


# =========================
# Runtime setup
# =========================

device = "cuda" if DEVICE == "auto" and torch.cuda.is_available() else DEVICE
if device == "auto":
    device = "cpu"

logger.remove()
logger.add(lambda msg: print(msg, end=""), format="{message}")
setup_seed(SEED)
torch.set_num_threads(1)

dtype_config.apply(
    use_float64=USE_FLOAT64,
    use_complex=USE_COMPLEX,
    device=device,
    driver=SVD_DRIVER,
    batch=2000,
    direct_limit=300,
)
batch_svd_config.apply(
    use_float64=USE_FLOAT64,
    use_complex=False,
    device=device,
    driver=BATCH_SVD_DRIVER,
    batch=2000,
    direct_limit=300,
)


# =========================
# PySCF integrals
# =========================

atoms = ";".join(f"H 0.0 0.0 {i * H_SPACING:.12g}" for i in range(N_H))
mol = gto.Mole(atom=atoms, verbose=0, basis=BASIS, symmetry=False)
mol.build()

mf = scf.RHF(mol)
mf.init_guess = "atom"
mf.max_cycle = 200
mf.conv_tol = 1.0e-12
e_hf = mf.kernel()

mo_coeff = lo.orth_ao(mf, LOCALIZED_METHOD) if LOCALIZED_ORB else mf.mo_coeff
ecore = float(mol.energy_nuc())
hcore = mf.get_hcore()
int1e = mo_coeff.T @ hcore @ mo_coeff
int2e = ao2mo.general(mol, (mo_coeff, mo_coeff, mo_coeff, mo_coeff), compact=0)
int2e = int2e.reshape(mo_coeff.shape[1], mo_coeff.shape[1], mo_coeff.shape[1], mo_coeff.shape[1])

sorb = mol.nao * 2
nele = mol.nelectron
e_fci = np.nan
if mol.nao <= 12:
    cisolver = fci.FCI(mf, mo_coeff)
    e_fci = float(cisolver.kernel()[0])

logger.info(f"H-chain: n_h={N_H}, spacing={H_SPACING}, sorb={sorb}, nele={nele}")
logger.info(f"Reference energies [FCI, HF]: {[e_fci, float(e_hf)]}")

if KEEP_INTEGRALS_PATH is not None:
    KEEP_INTEGRALS_PATH = Path(KEEP_INTEGRALS_PATH)
    KEEP_INTEGRALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        KEEP_INTEGRALS_PATH,
        ecore=ecore,
        int1e=int1e,
        int2e=int2e,
        energies=np.asarray([e_fci, float(e_hf)]),
    )


# =========================
# Hamiltonians and Clifford data
# =========================

na = nb = nele // 2
h1e, h2e = spinorb_from_spatial(int1e, int2e)
ham_op = get_fermion_ham(h1e, h2e, ecore)

na_h1e = np.zeros((sorb, sorb))
nb_h1e = np.zeros((sorb, sorb))
n_h2e = np.zeros((sorb, sorb, sorb, sorb))
na_h1e[0::2, 0::2] = np.eye(sorb // 2)
nb_h1e[1::2, 1::2] = np.eye(sorb // 2)
na_op = get_fermion_ham(na_h1e, n_h2e, 0.0)
nb_op = get_fermion_ham(nb_h1e, n_h2e, 0.0)
penalty_op = ((na_op - na) + (nb_op - nb)) ** 2
ham_penalty_op = ham_op + PENALTY_NANB * penalty_op
ham_jw = jordan_wigner(ham_penalty_op)
hams_mol = get_ham_from_qop(ham_jw, sorb)
hams_n_total = integral2pauli_JW(na_h1e + nb_h1e, n_h2e, 0.0, False)

use_orb = False
n_dim = 2
n_sites = sorb
basis = set_basis(sorb, use_orb)

clifford_file = Path(camps_pkg.__file__).resolve().parent / "file" / "clifford_2qubit_big.npz"
clifford_data = np.load(clifford_file)
operator = clifford_data["clifford_ops"]
tableau = clifford_data["tableau"]
phases = np.zeros((len(operator), tableau.shape[1]), dtype=np.int64)
check_operator_endian(operator[:20], tableau, n_dim, phases, endian="big")
clifford = Clifford(
    gates=operator,
    mapping=tableau,
    phases=phases,
    endian="big",
)


# =========================
# Initial MPO and MPS
# =========================

t0 = time.time_ns()
mpo_origin, model_origin = construct_mpo_pauli(hams_mol, basis, spatial_orb=use_orb)
logger.info(f"Construct MPO: {(time.time_ns() - t0) / 1.0e9:.3f} s")

for i, mpo_s in enumerate(mpo_origin):
    logger.info(f"site: {i} shape: {mpo_s.array.shape}")


# =========================
# CAMPS optimization
# =========================

hams_lst = [hams_mol, hams_n_total]
energy_micro = []
mps_c = None
mpo_c = None

for micro in range(N_MICRO):
    mps_is_optimized = micro > 0
    if not mps_is_optimized:
        mps_c = None
        mpo_c = None

    e, mps_c, mpo_c, hams_lst, ee_diff, _save_info = optimize_mps_disentangle(
        hams_lst,
        clifford,
        basis,
        SWEEP0,
        SWEEP1,
        DCUT,
        n_sites,
        algo=ALGO,
        n_dim=n_dim,
        use_orb=use_orb,
        mps_c=mps_c,
        mpo_c=mpo_c,
        mps_is_optimized=mps_is_optimized,
        save_mode=SAVE_MODE,
    )
    energy_micro.append(e)
    logger.info(f"micro={micro} energy blocks={e}")
    logger.info(f"micro={micro} entropy diff shape={np.shape(ee_diff)}")

final_energy = np.asarray(energy_micro[-1][1], dtype=float)[-1]
logger.info(f"Final optimized energy estimate: {final_energy:.12f}")

if SAVE_MODE == "save":
    for storage in (mps_c, mpo_c):
        if hasattr(storage, "delete_file"):
            storage.delete_file()
