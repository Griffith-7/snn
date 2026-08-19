"""Exact-SNN: Exact-gradient training for Spiking Neural Networks.

Train SNNs with mathematically exact gradients instead of surrogate gradients.
Solves the non-differentiable spike problem using IFT, saltation matrices,
and escape-noise — so spike networks train as well as dense networks.

Quick start:
    from exact_snn import TTFSNet
    net = TTFSNet([784, 128, 10])

Fast event-driven engine:
    from exact_snn import EventTTFSNet
    net = EventTTFSNet([784, 128, 10])

Extended architectures:
    from exact_snn.extended import ConvTTFSLayer, SNNConvNet, MultiSpikeNet
"""

__version__ = "1.1.0"
__author__ = "Sumith Kumar"

from exact_snn.core import (
    TTFSNetTorch,
    forward_layer_torch,
    backward_layer_torch,
    backward_layer_saltation,
    peak_margin_torch,
    edge_peak_guard,
    device,
    forward_multispike_layer,
    forward_multispike_layer_torch,
    backward_multispike_layer,
    backward_multispike_layer_torch,
)

TTFSNet = TTFSNetTorch

from exact_snn.event import EventTTFSNet
from exact_snn.reset import ResetLIF
from exact_snn.losses import (
    latency_cross_entropy,
    spike_count_cross_entropy,
    rate_latency_loss,
)
from exact_snn.optim import AdamTorch
from exact_snn.extended import xavier_init, kaiming_init

__all__ = [
    "TTFSNet",
    "TTFSNetTorch",
    "EventTTFSNet",
    "forward_layer_torch",
    "backward_layer_torch",
    "backward_layer_saltation",
    "peak_margin_torch",
    "edge_peak_guard",
    "device",
    "ResetLIF",
    "forward_multispike_layer",
    "forward_multispike_layer_torch",
    "backward_multispike_layer",
    "backward_multispike_layer_torch",
    "latency_cross_entropy",
    "spike_count_cross_entropy",
    "rate_latency_loss",
    "AdamTorch",
    "xavier_init",
    "kaiming_init",
]
