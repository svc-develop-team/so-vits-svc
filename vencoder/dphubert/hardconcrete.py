"""Implementation of the hard Concrete distribution.

Originally from:
https://github.com/asappresearch/flop/blob/master/flop/hardconcrete.py

"""

import math

import torch
import torch.nn as nn


class HardConcrete(nn.Module):
    """A HarcConcrete module.
    Use this module to create a mask of size N, which you can
    then use to perform L0 regularization.

    To obtain a mask, simply run a forward pass through the module
    with no input data. The mask is sampled in training mode, and
    fixed during evaluation mode, e.g.:

    >>> module = HardConcrete(n_in=100)
    >>> mask = module()
    >>> norm = module.l0_norm()
    """

    def __init__(
        self,
        n_in: int,
        init_mean: float = 0.5,
        init_std: float = 0.01,
        temperature: float = 2/3,     # from CoFi
        stretch: float = 0.1,
        eps: float = 1e-6
    ) -> None:
        """Initialize the HardConcrete module.
        Parameters
        ----------
        n_in : int
            The number of hard concrete variables in this mask.
        init_mean : float, optional
            Initial drop rate for hard concrete parameter,
            by default 0.5.,
        init_std: float, optional
            Used to initialize the hard concrete parameters,
            by default 0.01.
        temperature : float, optional
            Temperature used to control the sharpness of the
            distribution, by default 1.0
        stretch : float, optional
            Stretch the sampled value from [0, 1] to the interval
            [-stretch, 1 + stretch], by default 0.1.
        """
        super().__init__()

        self.n_in = n_in
        self.limit_l = -stretch
        self.limit_r = 1.0 + stretch
        self.log_alpha = nn.Parameter(torch.zeros(n_in))
        self.beta = temperature
        self.init_mean = init_mean
        self.init_std = init_std
        self.bias = -self.beta * math.log(-self.limit_l / self.limit_r)

        self.eps = eps
        self.compiled_mask = None
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of this module."""
        self.compiled_mask = None
        mean = math.log(1 - self.init_mean) - math.log(self.init_mean)
        self.log_alpha.data.normal_(mean, self.init_std)

    def l0_norm(self) -> torch.Tensor:
        """Compute the expected L0 norm of this mask.
        Returns
        -------
        torch.Tensor
            The expected L0 norm.
        """
        return (self.log_alpha + self.bias).sigmoid().sum()

    def forward(self) -> torch.Tensor:
        """Sample a hard concrete mask.
        Returns
        -------
        torch.Tensor
            The sampled binary mask
        """
        if self.training:
            # Reset the compiled mask
            self.compiled_mask = None
            # Sample mask dynamically
            u = self.log_alpha.new(self.n_in).uniform_(self.eps, 1 - self.eps)
            s = torch.sigmoid((torch.log(u / (1 - u)) + self.log_alpha) / self.beta)
            s = s * (self.limit_r - self.limit_l) + self.limit_l
            mask = s.clamp(min=0., max=1.)

        else:
            # Compile new mask if not cached
            if self.compiled_mask is None:
                # Get expected sparsity
                expected_num_zeros = self.n_in - self.l0_norm().item()
                num_zeros = round(expected_num_zeros)
                # Approximate expected value of each mask variable z;
                # We use an empirically validated magic number 0.8
                soft_mask = torch.sigmoid(self.log_alpha / self.beta * 0.8)
                # Prune small values to set to 0
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
                self.compiled_mask = soft_mask
            mask = self.compiled_mask

        return mask

    def extra_repr(self) -> str:
        return str(self.n_in)

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.extra_repr())
