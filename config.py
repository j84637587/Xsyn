from dataclasses import dataclass, field
from typing import List


@dataclass
class RunConfig:
    """Placeholder config used by BoxDiff (commented out in gligen_inference.py)."""
    run_standard_sd: bool = False
    scale_factor: int = 20
    thresholds: List[float] = field(default_factory=lambda: [0.05, 0.5, 1.0])
    max_iter: List[int] = field(default_factory=lambda: [10, 10, 10])
    max_index_step: int = 10
    token_indices: List[int] = field(default_factory=list)
    attention_res: int = 16
    smooth_attentions: bool = True
    sigma: float = 0.5
    kernel_size: int = 3
    normalize_eot: bool = False
