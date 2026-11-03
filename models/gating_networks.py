import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from typing import Tuple
import warnings

# Display the k=1 warning only once.
warnings.filterwarnings(
    "once",
    category=RuntimeWarning,
    message=r"With softmax over the selected experts.*",
)


class BaseGatingNetwork(nn.Module):
    """Noisy Top-K Gating Network base class"""
    def __init__(self, *,
                 num_experts: int,
                 noisy_gating: bool,
                 noise_epsilon:float = 1e-2,
                 routing_type:str = "topk_softmax"
                 ):
        super().__init__()
        self.num_experts = num_experts  # Number of experts in the MoE layer
        self.noisy_gating = noisy_gating  # Whether to perform "noisy top-k sampling" (True) or simple "top-k sampling" (False)
        self.noise_epsilon = noise_epsilon  # Small constant added to the estimated noise standard deviation
        self.min_temperature = 1e-3  # Minimum temperature to avoid numerical problems
        # Fixed valid parameters; disable CDF sample validation because its
        # Python boolean checks would synchronize CUDA during load-loss computation.
        self.noise_distribution = Normal(0.0, 1.0, validate_args=False)

        if routing_type not in ("topk_softmax", "full_softmax"):
            raise ValueError(f'Unknown routing_type: {routing_type!r}. Expected "topk_softmax" or "full_softmax".')
        self.routing_type = routing_type

        # ===== Abstract layers ===== #
        # All child classes MUST implement all of these.
        # Use nn.Identity() when an operation is intentionally redundant.
        self.feature_mapping = None # Maps input tensor of arbitrary shape, e.g., (B,C,F,T), to features of shape (B,N)
        self.to_logits = None  # Projects features (B,N) to gating network logits (B,num_experts)
        self.noise_std = None # Projects features (B,N) to noise stddev (B,num_experts) - needed for noisy top-k gating.

    @staticmethod
    def cv_squared(x: torch.tensor) -> torch.tensor:
        """Coefficient of Variation (CV) squared := (std(x) / mean(x))**2
        Minimizing CV squared encourages the underlying distribution to be as uniform as possible.
        """
        eps = 1e-10
        x = x.float()

        # if num_experts is 1, then CV is ill-defined: return a single-element zero tensor
        if x.shape[0] <= 1:
            return x.new_zeros(())  # torch.tensor([0], device=x.device, dtype=x.dtype)

        # else, return the coefficient of variation squared (epsilon avoids division by zero)
        return x.var(correction=0) / (x.mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values, k):
        """Adapted from https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py#L188"""
        # Get the threshold values directly using 2D indexing
        threshold_if_in = noisy_top_values[:, -1].unsqueeze(1)   # (k+1)-th largest
        threshold_if_out = noisy_top_values[:, -2].unsqueeze(1)  # k-th largest

        # Boolean mask: whether a given logit is currently in the top-k under noisy values
        is_in = (noisy_values > threshold_if_in)

        # Probability of exceeding each threshold.
        prob_if_in = 1.0 - self.noise_distribution.cdf((threshold_if_in - clean_values) / noise_stddev)
        prob_if_out = 1.0 - self.noise_distribution.cdf((threshold_if_out - clean_values) / noise_stddev)

        # Select correct probability based on whether value is currently inside or outside top-k
        prob = torch.where(is_in, prob_if_in, prob_if_out)

        return prob  # shape: (B, num_experts)

    @staticmethod
    def topk_softmax_routing(topk_logits: torch.tensor, temperature: float) -> torch.tensor:
        """ topk-softmax routing

          The following snippet is more faithful to the paper "Outrageously Large Neural Networks: The Sparsely-Gated
          Mixture-of-Experts Layer" (https://arxiv.org/pdf/1701.06538)

          However, during training, when k=1, softmax of a singleton logit always assigns the chosen expert a weight of 1,
          regardless of the router's scores. The gradients of the main loss function (i.e., fake/bonafide classification)
          cannot teach the router to choose a better expert.

          This is known issue. The original paper hints at that at page 4:
              "Training the Gating Network: We train the gating network by simple back-propagation, along with the rest
               of the model. If we choose k > 1, the gate values for the top k experts have nonzero derivatives with
               respect to the weights of the gating network."
        """
        # Apply softmax to the sparse logits to obtain (top-K) expert gates
        topk_gates = F.softmax(topk_logits / temperature, dim=-1)
        return topk_gates

    @staticmethod
    def full_softmax_routing(noisy_logits: torch.tensor, topk_idx: torch.tensor, temperature: float) -> torch.tensor:
        """ full-softmax routing

         The following snippet computes softmax over all experts before selecting the top-k probabilities. This
         preserves a gradient path through the softmax denominator, including for k=1 when multiple experts exist.
         Expert selection remains unchanged for the same logits, but selected weights no longer necessarily sum to 1,
         changing the output scale and training dynamics.

         Note: With nearly uniform routing probabilities, each selected gate is approximately 1 / num_experts, and the
         selected gates sum to approximately k / num_experts. This can substantially attenuate the MoE output when k is
         small relative  to num_experts. The old snippet keeps selected weights 1-normalized, but requires training
         with k >= 2 for gradients to flow through the gate weights.

         The following snippet is similar to the routing in Google V-MoE: https://github.com/google-research/vmoe/
             See https://github.com/google-research/vmoe/blob/main/vmoe/nn/routing.py#L80
             and https://github.com/google-research/vmoe/blob/main/vmoe/moe.py#L567
         """
        # Apply softmax over all experts.
        router_probs = F.softmax(noisy_logits / temperature, dim=-1)

        # Keep only the probabilities of the selected experts.
        topk_gates = router_probs.gather(dim=1, index=topk_idx)

        return topk_gates

    def forward(self, x, k:int = 1, temperature:float = 1.0) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        assert 1 <= k <= self.num_experts, 'k in top-k must be less or equal than the number of experts.'

        if (self.training
            and k == 1
            and self.num_experts > 1
            and self.routing_type == "topk_softmax"
        ):
            warnings.warn(
                "With softmax over the selected experts and k=1, the selected gate is always 1. "
                "The main loss (fake/bonafide classification) and the auxiliary importance loss therefore "
                "cannot train the gating network through the gate weights. "
                "Note that, with k=1, the auxiliary load loss can still provide router gradients when noisy gating "
                "is enabled. This may potentially hide this issue when visually inspecting the gradients. "
                "Use k>=2 or explicitly choose full-softmax routing.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Clip minimum temperature
        temperature = max(temperature, self.min_temperature)

        # Map input tensor to features of shape (B,N), i.e., ready for the subsequent fully connected layers
        x = self.feature_mapping(x)

        # Compute gating network logits -> (B,num_experts)
        clean_logits = self.to_logits(x)

        # Add white Gaussian noise to the logits
        if self.noisy_gating and self.training:
            noise_std = self.noise_std(x) + self.noise_epsilon
            noise = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + noise_std * noise
        else:
            noise_std = 0.0
            noisy_logits = clean_logits

        # Compute top-(K+1) logit values. This is only needed for load_loss computation below.
        topkp1_logits, topkp1_idx = torch.topk(noisy_logits, k=min(k+1, self.num_experts), dim=-1)

        # Select the top-K values, i.e., discard the (K+1)-th largest logit, effectively performing top-K sampling.
        topk_logits = topkp1_logits[:, :k]  # shape: (B, K)
        topk_idx = topkp1_idx[:, :k]  # shape: (B, K)

        # ===== Routing ===== #
        if self.routing_type == "topk_softmax":
            topk_gates = self.topk_softmax_routing(topk_logits, temperature)
        else:
            topk_gates = self.full_softmax_routing(noisy_logits, topk_idx, temperature)

        # ===== Compute Importance ===== #
        # We want to go from sparse gates (B, K) to dense gates (B,num_experts)
        gates_full = torch.zeros_like(clean_logits) #, requires_grad=True)  # Zero tensor of shape (B,num_experts)
        gates_full = torch.scatter(gates_full, 1, topk_idx, topk_gates) # Write topk_gates at indices topk_idx, leaving the rest at zero.
        importance = gates_full.sum(0) # Now we sum across batch dimension to obtain the "Importance" of each expert.

        # ===== Compute Load ===== #
        if self.noisy_gating and k < self.num_experts and self.training:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_std, topkp1_logits, k).sum(dim=0)
        else:
            # True load: number of examples for which the corresponding gate is positive. This op is nondifferentiable;
            # it will be treated as a constant during back-propagation.
            load = (gates_full > 0).sum(0)

        # Compute auxiliary losses
        importance_loss = self.cv_squared(importance)
        load_loss = self.cv_squared(load)

        # Note that topk_gates shape is not (B,num_experts) but rather (B,K), with K <= num_experts. Therefore, you also
        # need topk_idx to correctly dispatch the features to the active experts.
        return topk_gates, topk_idx, importance_loss, load_loss


class FCNNGatingNetwork(BaseGatingNetwork):
    """Based on the paper https://arxiv.org/pdf/1701.06538

        Input: (batch_size, input_dim)
    """
    def __init__(self, *,
                 num_experts: int,
                 noisy_gating: bool,
                 input_dim:int,
                 routing_type: str = "topk_softmax",
                 ):
        super().__init__(num_experts=num_experts, noisy_gating=noisy_gating, routing_type=routing_type)
        self.feature_mapping = nn.Identity()  # no-op
        self.to_logits = nn.Linear(input_dim, num_experts, bias=False) # W_g
        self.noise_std = nn.Sequential(
            nn.Linear(input_dim, num_experts, bias=False),  # W_noise
            nn.Softplus()
        )

        # Initialize linear transformation matrices to zero
        nn.init.zeros_(self.to_logits.weight)
        nn.init.zeros_(self.noise_std[0].weight)


class DepthwiseSeparableResBlock(nn.Module):
    """Depthwise-separable residual convolutional block."""

    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
        )

        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.conv(x))


class TFGatingNetwork(BaseGatingNetwork):
    """Lightweight Time-Frequency (TF) Gating Network - it works with STFTs, Mel-Spec, etc.

        Input: (batch_size, in_channels, F, T)
    """
    def __init__(self, *,
                 num_experts: int,
                 noisy_gating: bool,
                 in_channels: int,
                 redux_channels: int,
                 hidden_dim: int,
                 routing_type: str = "topk_softmax",
                 ):
        super().__init__(num_experts=num_experts, noisy_gating=noisy_gating, routing_type=routing_type)

        # Feature extraction
        self.feature_mapping = nn.Sequential(
            nn.Conv2d(in_channels, redux_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, redux_channels),
            nn.SiLU(),
            DepthwiseSeparableResBlock(redux_channels),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
        )

        # Expert logits
        self.to_logits = nn.Sequential(
            nn.Linear(redux_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_experts)
        )

        self.noise_std = nn.Sequential(
            nn.Linear(redux_channels, num_experts, bias=False),  # W_noise
            nn.Softplus()
        )

        # Initialize the linear transformation matrix to zero
        nn.init.zeros_(self.noise_std[0].weight)
