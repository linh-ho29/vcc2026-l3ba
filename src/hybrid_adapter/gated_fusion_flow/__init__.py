"""Gated latent multi-context fusion + flow pipeline (system architecture,
posted on the ``main`` branch's README).

Grouped together because all four modules implement pieces of that one
diagram: :mod:`gated_fusion` (the fusion step, also reused as an optional
extension of the original ``HybridAdapter``'s Module A - see
``hybrid_adapter.model``), :mod:`control_branch` (Phase 1's control-cell
side: frozen scGPT pooling, a PCA encoder fit on real control cells, and
the empirical NTC latent bank), :mod:`latent_flow` (the input conditioning
layer, the discrete latent flow block, and the decoder), and
:mod:`flow_model` (:class:`GatedFusionFlowAdapter`, wiring all of the above
into the full Phase 2 pipeline as a model separate from ``HybridAdapter``).

SYNTHETIC PHASE ONLY, same as the rest of this package - see the package
root docstring.
"""

from .gated_fusion import GatedMultiContextFusion
from .control_branch import ControlCellEncoder, NTCLatentBank, ScGPTGeneEmbeddingPooling
from .latent_flow import DiscreteLatentFlow, InputConditioningLayer, LatentDecoder
from .flow_model import GatedFusionFlowAdapter

__all__ = [
    "GatedMultiContextFusion",
    "ControlCellEncoder",
    "NTCLatentBank",
    "ScGPTGeneEmbeddingPooling",
    "DiscreteLatentFlow",
    "InputConditioningLayer",
    "LatentDecoder",
    "GatedFusionFlowAdapter",
]
