import torch
from .metropolis import metropolis_step


class Temperer(torch.nn.Module):
    """Class to perform parallel tempering optimization using Metropolis steps."""

    def __init__(self, N, K, Q, beta, dtype=torch.float32, device=torch.device("cpu")):
        """Initialize Temperer class.

        Args:
            N (int): Number of spins.
            K (int): Number of replicas.
            Q (torch.Tensor): Interaction matrix with shape (W, N, N).
            beta (torch.Tensor): Inverse temperatures with shape (L,).
            dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float16.
            device (torch.device, optional): Device for computations. Defaults to CPU.
        """
        super(Temperer, self).__init__()

        self.N = N
        self.K = K
        self.dtype = dtype
        self.device = device

        self.beta = beta.to(device, dtype=dtype).contiguous()
        self.L = beta.size(0)

        self.Q = Q.to(device, dtype=dtype).contiguous()  # Interaction matrices
        self.W = self.Q.size(0)

        # Initialize random binary states: shape (W, L, K, N)
        self.states = torch.randint(0, 2, (self.W, self.L, self.K, self.N),
                                    device=self.device, dtype=self.dtype)

    def step(self):
        """Perform one Metropolis step for state updates."""
        metropolis_step(self.states.contiguous(),
                        self.Q, self.beta, self.W, self.L, self.K, self.N,
                        torch.seed())

    def energy(self):
        """Compute energies of the current states.

        Returns:
            torch.Tensor: Energies with shape (W, L, K).
        """
        return torch.einsum("wlki,wij,wlkj->wlk", self.states, self.Q, self.states)

    def exchange(self):
        """Perform parallel tempering exchanges between neighboring replicas.

        Returns:
            torch.Tensor: Energies before the exchange step.
        """
        energies = self.energy()
        # Energy differences for exchanges between neighboring replicas
        delta = (energies[:, 1:] - energies[:, :-1]) * (self.beta[1:] - self.beta[:-1]).view(1, self.L - 1, 1)

        # Determine exchanges based on Metropolis criterion
        mask = (torch.rand_like(delta) < torch.exp(delta))[..., None].to(self.dtype)

        # Perform exchanges
        states_left = self.states[:, :-1].clone()
        states_right = self.states[:, 1:].clone()

        self.states[:, :-1] = mask * states_right + (1 - mask) * states_left
        self.states[:, 1:] = mask * states_left + (1 - mask) * states_right

        return energies