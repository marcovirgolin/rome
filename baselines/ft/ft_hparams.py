from dataclasses import dataclass
from typing import List

from util.hparams import HyperParams


@dataclass
class FTHyperParams(HyperParams):
    # Method
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # MV: added stuff for KL
    context_template_length_params: List[List[int]]
    loss_layer : int

    # Defaults
    batch_size: int = 128

    
