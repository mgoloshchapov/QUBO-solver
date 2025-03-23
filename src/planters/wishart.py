import torch


def generate_solutions(n, batch_size, device='cpu'):
    r"""
    Generates random arrays with inputs {-1, 1}.

    Args:
        n:              length of each array
        batch_size:     number of generated arrays
    """
    return torch.randint(low=0, high=2, size=(batch_size,n), device=device) * 2 - 1


def gauge_problems(s, J) -> None:
    r"""
    Given problems with {1}^n solution,
    turn them into problems with s solutions.

    Before gauge, each matrix from J has solution [1,1,...,1]. If solution
    has -1 at position i, i-th row and column of matrix is multiplied by -1. 
    This transforms problems with solution [1,1,...,1] to problems with given 
    solutions s. 

    Energy of configuration s with coupling matrix J is

    H(s) = -1/2 s^T J s

    Args:
        s:      solutions to Ising's problems
        J:      coupling matrices
    """
    masks = ((-s + 1)//2).bool()
    for i in range(masks.shape[0]):
        J[i, masks[i], ::] *= -1
        J[i, ::, masks[i]] *= -1


def generate_problems(s, m: int, device='cpu'):
    r"""
    Generation of Ising problem using Wishart planted 
    solutions. 

    Generate Ising's prolems with known solutions s. 
    H(s) = -1/2 s^T J s

    Args:
        s:              array of {-1, 1} of length n
        m:              hardness parameter
        batch_size:     number of generated problems
    """
    batch_size, n = s.shape
    W = torch.normal(mean=0, std=1, size=(batch_size, n,m), device=device)
    J = (-1/2) *  W @ torch.transpose(W, 1, 2) 
    J -= torch.diag_embed(torch.diagonal(J, dim1=-2, dim2=-1))
    gauge_problems(s, J)

    return J