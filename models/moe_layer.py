import torch
import torch.nn as nn
from typing import List


def _expand_for_broadcast(x: torch.tensor, target: torch.tensor) -> torch.tensor:
    """
    Expand a 1-D tensor of shape (N,) into shape (N, 1, 1, ..., 1) with (target.ndim - 1) singleton dims.
    This is safe for broadcasting against tensors with batch-dimension N and arbitrary rank >= 1.
    """
    if target.ndim <= 1:
        return x
    return x.reshape(-1, *([1] * (target.ndim - 1)))


class MoELayer(nn.Module):
    """MoE layer with efficient routing. N.B. it supports only experts with the same output shape."""
    def __init__(self,
                 gating_net: nn.Module,
                 experts: List[nn.Module],
                 ):
        super().__init__()
        self.gating_net = gating_net
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.tensor, k:int = 1, temperature: float = 1.0) -> torch.tensor:
        # ---- Graceful fallback to single expert ----
        if len(self.experts) == 1:
            output = self.experts[0](x)
            zero_loss = x.new_zeros(())  # scalar 0.0 on correct device/dtype
            return output, zero_loss, zero_loss

        if not 1 <= k <= len(self.experts):
            raise ValueError(f"k must be between 1 and {len(self.experts)}")

        # Store the current batch size
        batch_size = x.shape[0]

        # Compute sparse expert_gates, activated expert indices, auxiliary losses
        expert_gates, expert_idx, importance_loss, load_loss = self.gating_net(x, k=k, temperature=temperature)

        assert expert_gates.shape == (batch_size, k)
        assert expert_idx.shape == (batch_size, k)

        # Assign an integer index to all samples in the batch
        sample_ids = torch.arange(batch_size, device=x.device, dtype=torch.long)
        # Expand sample_ids to match the shape of expert_gates and expert_idx
        sample_ids = sample_ids.unsqueeze(1).expand(-1, k)  # (B,) -> (B,1) -> (B,K)

        # Flatten sample_ids, expert_gates, and expert indices to simplify masking
        flat_sample_ids = sample_ids.reshape(-1)
        flat_expert_gates = expert_gates.reshape(-1)
        flat_expert_idx = expert_idx.reshape(-1)

        # Initialize output buffer lazily
        output_buffer = None

        # Loop over all (activated) experts
        for i, expert in enumerate(self.experts):
            # Compute binary mask for the i-th expert
            mask = (flat_expert_idx == i)

            # Skip unused experts
            if not mask.any():
                continue

            masked_sample_ids = flat_sample_ids[mask]
            masked_gates = flat_expert_gates[mask]

            # Create sub-batch to be routed to the i-th expert
            x_sub = torch.index_select(x, 0, masked_sample_ids)
            # Forward pass through the i-th expert
            out_sub = expert(x_sub)
            # Multiply the expert output with the corresponding gate values
            out_sub = out_sub * _expand_for_broadcast(masked_gates, out_sub)

            # Delay the creation of the buffer until the experts' output shape becomes available
            if output_buffer is None:
                output_buffer = torch.zeros(batch_size, *out_sub.shape[1:], device=x.device, dtype=out_sub.dtype)

            # Sum the (weighted) i-th expert output to the buffer by writing only in the (batch) indices (dim=0)
            # corresponding to the (masked) sample ids.
            output_buffer.index_add_(0, masked_sample_ids, out_sub)

        return output_buffer, importance_loss, load_loss
