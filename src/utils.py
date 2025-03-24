import torch


def J2Q(J):
    row_sums = torch.sum(J, axis=1)
    Q = 4 * J - 4 * torch.diag(row_sums)
    return Q

def Q2J(Q):
    row_sums = torch.sum(Q, axis=1)
    J = (Q + torch.diag(row_sums)) / 4
    return J
