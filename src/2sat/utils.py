from collections import defaultdict
import random

"""
Example:
expression = [2*i, 2*j+1, 2*k+1, 2*l+1]

expression corresponds to (not xi or xj) and (not xk or not xl)
"""
def graph_from_2sat(expression):
    n = max(expression) | 1
    adjacency_list = {i : set() for i in range(n+1)}
    
    for k in range(0, len(expression), 2):
        xi, xj = expression[k], expression[k+1]
        if not (xi ^ xj == 1):
            adjacency_list[xi^1].add(xj)
            adjacency_list[xj^1].add(xi)

    return adjacency_list


def is_solution(expression, solution):
    n = len(expression)
    for i in range(0, n, 2):
        left = solution[expression[i]//2] ^ (expression[i]%2)
        right = solution[expression[i+1]//2] ^ (expression[i+1]%2)
        if left and right:
            return False
    return True
    

def check_satisfiability(low_link):
    for i in range(0, len(low_link), 2):
        if low_link[i] == low_link[i+1]:
            return False 
    return True

# n - number of variables, m - number of parentheses
def generate_expression(n, m):
    expression = [0] * (2*m)
    for i in range(2*m):
        expression[i] = random.randint(0, 2*n-1)
    return expression



