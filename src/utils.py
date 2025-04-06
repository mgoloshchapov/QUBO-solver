import torch
import networkx as nx
from tqdm import tqdm

def J2Q(J):
    row_sums = torch.sum(J, dim=1)
    Q = 4 * J - 4 * torch.diag(row_sums)
    return Q

def Q2J(Q):
    row_sums = torch.sum(Q, dim=1)
    J = (Q + torch.diag(row_sums)) / 4
    return J

def coloring(AM):
    colored_idx = {}
    
    for w in tqdm(range(AM.size(0))):
        G = nx.from_edgelist(AM[w]._indices().numpy().T)
        # largest_first (fast), smallest_last (slow), random_sequential (fast), saturation_largest_first (very slow, best quality)
        coloring = nx.algorithms.coloring.greedy_color(G, strategy='largest_first')
        coloring = torch.tensor(list(coloring.items()))
        ncolors = coloring[:, 1].max() + 1
        
        for j in range(ncolors):
            idx = coloring[:, 0][coloring[:, 1] == j]
            if j not in colored_idx.keys(): colored_idx[j] = {'w':[], 'n':[]}
            colored_idx[j]['w'].append(torch.ones_like(idx) * w)
            colored_idx[j]['n'].append(idx)
            
    for j in range(max(colored_idx.keys()) + 1):
        colored_idx[j]['w'] = torch.concatenate(colored_idx[j]['w'])
        colored_idx[j]['n'] = torch.concatenate(colored_idx[j]['n'])
    return colored_idx