from typing import Tuple, Union, List, Set

from numpy import floating 
from numpy.typing import NDArray

#NDArray = ndarray
PyArrayLike = Union[List[float], Tuple[float, ...], Set[float]]

ArrayLike = Union[NDArray, PyArrayLike]

# real vectors
R1 = NDArray[floating]  # R^1
R2 = NDArray[floating]  # R^2
R3 = NDArray[floating]  # R^3
R4 = NDArray[floating]  # R^4
R6 = NDArray[floating]  # R^6
R8 = NDArray[floating]  # R^8

# real matrices
R1x1 = NDArray  # R^{1x1} matrix
R2x2 = NDArray  # R^{3x3} matrix
R3x3 = NDArray  # R^{3x3} matrix
R4x4 = NDArray  # R^{4x4} matrix
R6x6 = NDArray  # R^{6x6} matrix
R8x8 = NDArray  # R^{8x8} matrix
R1x3 = NDArray  # R^{1x3} row vector
R3x1 = NDArray  # R^{3x1} column vector
R1x2 = NDArray  # R^{1x2} row vector
R2x1 = NDArray  # R^{2x1} column vector
