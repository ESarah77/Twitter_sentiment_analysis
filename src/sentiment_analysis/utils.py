import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    This controls Python's random module, NumPy, and hash-based operations
    where possible. Some runtime values such as execution time can still vary.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
