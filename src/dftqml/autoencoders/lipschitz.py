import torch
import torch.nn as nn
import torch.nn.functional as F


class LipschitzLinear(nn.Linear):
    """
    A linear layer with a learnable Lipschitz constant.

    This layer normalizes its weights to ensure the Lipschitz constant is bounded.
    The Lipschitz constant is defined as the maximum absolute row sum of the weight matrix.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        self.reset_c()

    def reset_parameters(self):
        super().reset_parameters()
        self.reset_c()

    def reset_c(self):
        # Initialize the Lipschitz constant as a learnable parameter
        with torch.no_grad():
            init_c = l_inf_norm(self.weight)
        self.c = nn.Parameter(init_c.unsqueeze(0))

    def forward(self, input):
        # Apply softplus to ensure positivity of weight normalization factor
        softplus_c = F.softplus(self.c)

        # Compute L-inf norm (max absolute row sum) for each row
        norm = l_inf_norm(self.weight)

        # Compute scaling factor (only scale down, never up)
        scale = torch.minimum(torch.ones_like(softplus_c), softplus_c / norm)

        # Apply scaling to weight matrix
        W_normalized = self.weight * scale

        # Standard linear transformation
        out = F.linear(input, W_normalized, self.bias)
        return out

    @property
    def lipschitz_bound(self):
        """Return softplus(c): the bound on the Lipschitz constant of this layer"""
        return F.softplus(self.c)


def l_inf_norm(tensor):
    """Compute the L-infinity norm (max absolute row sum) of a 2D tensor."""
    return torch.max(torch.sum(torch.abs(tensor), dim=1))
