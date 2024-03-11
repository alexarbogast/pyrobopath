# for Python >= 3.9

from typing import Tuple, Union, List, Any
from typing import Literal as L
from numpy import ndarray, dtype, floating
from numpy.typing import NDArray
from quaternion import quaternion

ArrayLike = Union[List[float], Tuple[float, ...], ndarray[Any, dtype[floating]]]
ArrayLike2 = Union[
    List[float],
    Tuple[float, float],
    ndarray[
        Tuple[L[2,]],
        dtype[floating],
    ],
]
ArrayLike3 = Union[
    List[float],
    Tuple[float, float, float],
    ndarray[
        Tuple[L[3,]],
        dtype[floating],
    ],
]
ArrayLike4 = Union[
    List[float],
    Tuple[float, float, float, float],
    ndarray[
        Tuple[L[4,]],
        dtype[floating],
    ],
]

# real vectors
R1 = ndarray[
    Tuple[L[1]],
    dtype[floating],
]  # R^1
R2 = ndarray[
    Tuple[L[2]],
    dtype[floating],
]  # R^2
R3 = ndarray[
    Tuple[L[3]],
    dtype[floating],
]  # R^3
R4 = ndarray[
    Tuple[L[4]],
    dtype[floating],
]  # R^4
R6 = ndarray[
    Tuple[L[6]],
    dtype[floating],
]  # R^6

# real matrices
R1x1 = ndarray[Tuple[L[1], L[1]], dtype[floating]]  # R^{1x1} matrix
R2x2 = ndarray[Tuple[L[2], L[2]], dtype[floating]]  # R^{2x2} matrix
R3x3 = ndarray[Tuple[L[3], L[3]], dtype[floating]]  # R^{3x3} matrix
R4x4 = ndarray[Tuple[L[4], L[4]], dtype[floating]]  # R^{4x4} matrix
R6x6 = ndarray[Tuple[L[6], L[6]], dtype[floating]]  # R^{6x6} matrix
R8x8 = ndarray[Tuple[L[8], L[8]], dtype[floating]]  # R^{8x8} matrix
R1x3 = ndarray[Tuple[L[1], L[3]], dtype[floating]]  # R^{1x3} row vector
R3x1 = ndarray[Tuple[L[3], L[1]], dtype[floating]]  # R^{3x1} column vector
R1x2 = ndarray[Tuple[L[1], L[2]], dtype[floating]]  # R^{1x2} row vector
R2x1 = ndarray[Tuple[L[2], L[1]], dtype[floating]]  # R^{2x1} column vector

Quat = Union[ArrayLike4, quaternion]
