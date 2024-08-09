import numpy as np
from typing import Iterable


def sidx(*indexes: Iterable[int]):
    return np.array(indexes, dtype=np.int32)