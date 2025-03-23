import networkx as nx
import matplotlib.pyplot as plt 


def get_colors(low_link):
    scc = set(low_link.values())
    colors_cc = {cc : (i+1)/len(scc) for (i, cc) in enumerate(scc)}
    colors = {i : colors_cc[low_link[i]] for i in range(len(low_link))}
    return colors


def get_labels(adjacency_list):
    return {key: "$\overline{x}_{%s}$" % (key//2) if key % 2 == 0 else "$x_{%s}$" % (key//2)
            for key in adjacency_list}


def draw_2sat(adjacency_list, low_link, seed=3113794):
    # G = []
    # for i in range(len(adjacency_list)):
    #     for j in adjacency_list[i]:
    #         G.append([i, j])

    # G = nx.from_edgelist(G, create_using=nx.DiGraph)
    G = nx.from_dict_of_lists(adjacency_list, create_using=nx.DiGraph)

    pos = nx.spring_layout(G, k=1, seed=seed)

    temp = get_colors(low_link)
    colors = [temp.get(node) for node in G.nodes()]
    labels = get_labels(adjacency_list)
    
    nx.draw(G, 
            pos=pos, 
            with_labels=True, 
            labels=labels,
            width = 1.5,
            edge_color='black',
            arrows=True,
            arrowstyle='->',
            arrowsize=15,
            alpha=0.5,
            node_color='white',
            connectionstyle='arc3, rad = 0.3')
    
    nodes = nx.draw_networkx_nodes(G, 
                                   pos=pos, 
                                   node_size=500,
                                   node_color=colors,
                                   cmap=plt.get_cmap('tab20c'),
                                   alpha=0.75)
    nodes.set_edgecolor('black')
   