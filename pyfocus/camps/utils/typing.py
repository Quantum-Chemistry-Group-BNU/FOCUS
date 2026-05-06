import numpy as np

from dataclasses import dataclass
from typing import Literal, NamedTuple, TypedDict
from numpy.typing import NDArray


Sites = list[NDArray[np.float64]] | list[NDArray[np.complex128]]
PauliArray = NDArray[np.bytes_]
GateArray = NDArray[np.float64] | NDArray[np.complex128]
Endianness = Literal["little", "big"]


class CliffordInfo(TypedDict):
    gates: GateArray
    mapping: NDArray[np.int64]
    phases: NDArray[np.int64]
    endian: Endianness


# TODO: using dataclass or TypedDict???
@dataclass(frozen=True)
class Clifford:
    """
    clifford gates, mapping, phases and endian
    """

    gates: GateArray
    mapping: NDArray[np.int64]
    phases: NDArray[np.int64]
    endian: Endianness

    # this is deprecated in the future
    def __getitem__(self, key: str):
        return getattr(self, key)

    def to_dict(self) -> CliffordInfo:
        """
        save to dict
        """
        return {
            "gates": self.gates,
            "mapping": self.mapping,
            "phases": self.phases,
            "endian": self.endian,
            **({"with_I": self.with_I} if isinstance(self, RandomClifford) else {}),
        }


@dataclass(frozen=True)
class RandomClifford(Clifford):
    with_I: bool = True


class Hamiltonian(TypedDict):
    array: PauliArray
    coeff: NDArray


class SaveInfo(TypedDict):
    """
    save info
    """

    hams: list[Hamiltonian]
    clifford: CliffordInfo | list[CliffordInfo]
    sites_dict: dict[str, Sites]
    energy: list[float]
    clifford_idx: NDArray[np.int64]
    random_clifford: bool = False
    
