import os
import sys
import tempfile
import h5py
from pathlib import Path

import torch
# from pyfocus.camps.mps.mps_simple import leftCanonicalization, rightCanonicalization
from pyfocus.camps.utils.config import dtype_config

from renormalizer import Model, Mps, Mpo
from renormalizer.model import Op, OpSum
from renormalizer.model.basis import BasisHalfSpin, BasisSet

try:
    from renormalizer.model.basis import BasisTwoHalfSpin
except ImportError:
    BasisTwoHalfSpin = None

######################################### basis ############################################
def set_basis(nsites, use_orb):
    if use_orb:
        if BasisTwoHalfSpin is None:
            raise ImportError(
                "The installed renormalizer package does not provide "
                "BasisTwoHalfSpin; use_orb=True is not supported in this "
                "environment."
            )
        return [BasisTwoHalfSpin(i) for i in range(nsites)]
    else:
        return [BasisHalfSpin(i) for i in range(nsites)]
    
    
def random_state(model, qntot, m_max, percent=1.0):
    return Mps.random(model, qntot, m_max, percent)
