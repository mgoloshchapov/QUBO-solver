from collections import defaultdict

class SCC:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list
        self.node_idx = {key: -1 for key in adjacency_list}
        self.low_link = {key: -1 for key in adjacency_list}
        self.on_stack = {key: False for key in adjacency_list}
        self.stack = []
        self.cnt = -1

    def dfs(self, v):
        self.cnt += 1
        self.stack.append(v)
        self.on_stack[v] = True
        self.low_link[v] = self.cnt
        self.node_idx[v] = self.cnt

        for u in self.adjacency_list[v]:
            if self.node_idx[u] == -1:
                self.dfs(u)
                self.low_link[v] = min(self.low_link[v], self.low_link[u])
            elif self.on_stack[u]:
                self.low_link[v] = min(self.low_link[v], self.low_link[u])

        # Create strongly connected component/cycle
        if self.low_link[v] == self.node_idx[v]:
            while self.stack[-1] != v:
                u = self.stack.pop()
                self.on_stack[u] = False
            u = self.stack.pop()
            self.on_stack[u] = False


    def get_scc(self):
        if self.cnt == -1:
            for v in range(len(self.adjacency_list)):
                if self.node_idx[v] == -1:
                    self.dfs(v)

        return self.low_link
    


def check_satisfiability(low_link):
    for i in range(0, len(low_link), 2):
        if low_link[i] == low_link[i+1]:
            return False 
    return True


def toposort_step(v, adjacency_list, colors, stack):
    colors[v] = 1
    for u in adjacency_list[v]:
        if colors[u] == 0:
            toposort_step(u, adjacency_list, colors, stack)
        elif colors[u] == 1:
            raise ValueError("Found cycle in graph, topological sort doesn't exist")
    stack.append(v)
    colors[v] = 2


def toposort(adjacency_list):
    n = len(adjacency_list)
    colors = dict()
    order = dict()
    for key in adjacency_list:
        colors[key] = 0
        order[key] = -1

    stack = []
    for v in adjacency_list:
        if colors[v] == 0:
            toposort_step(v, adjacency_list, colors, stack)

    for i in range(n-1, -1, -1):
        v = stack.pop()
        order[v] = n-1-i

    return order
    


def get_solution(adjacency_list, low_link):
    scc_adjacency_list = defaultdict(set)

    # Build graph, where vertices are strongly connected components
    for v in adjacency_list:
        scc_adjacency_list[low_link[v]] = set()
        for u in adjacency_list[v]:
            if low_link[u] != low_link[v]:
                scc_adjacency_list[low_link[v]].add(low_link[u])
                if low_link[v] in scc_adjacency_list[low_link[u]]:
                    raise ValueError("2-SAT problem doesn't have a solution")
    
    
    # Topological sort of scc graph
    n = len(scc_adjacency_list)
    scc_toporder = toposort(scc_adjacency_list)

    # Retrieve answer
    n = len(adjacency_list)
    solution = [0] * (n//2)
    for i in range(n//2):
        solution[i] = int(scc_toporder[low_link[2*i]] < scc_toporder[low_link[2*i+1]])

    return solution
                

def find_answer(adjacency_list, low_link):
    scc_adjacency_list = defaultdict(set)

    # Build graph, where vertices are strongly connected components
    for v in adjacency_list:
        scc_adjacency_list[low_link[v]] = set()
        for u in adjacency_list[v]:
            if low_link[u] != low_link[v]:
                scc_adjacency_list[low_link[v]].add(low_link[u])
                if low_link[v] in scc_adjacency_list[low_link[u]]:
                    raise ValueError("2-SAT problem doesn't have a solution")
    
    
    # Topological sort of scc graph
    n = len(scc_adjacency_list)
    scc_toporder = toposort(scc_adjacency_list)

    # Retrieve answer
    n = len(adjacency_list)
    answer = [0] * (n//2)
    for i in range(n//2):
        answer[i] = scc_toporder[low_link[2*i]] < scc_toporder[low_link[2*i+1]]

    return answer
                
