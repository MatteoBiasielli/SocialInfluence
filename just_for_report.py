import numpy as np
from graph.graph import Node


def generate_graph(n_nodes=100, n_init_nodes=3, n_conn_per_node=2, randomstate=np.random.RandomState(1234)):
    num_nodes = 0  # current number of nodes
    nodes = []  # list of nodes
    tot_degree = 0  # sum of all degrees in the network

    # INITIALIZATION - CREATING INITIAL NODES
    for i in range(n_init_nodes):
        newnode = Node()
        for node in nodes:
            node.attach(newnode)  # the initial nodes are connected as a "Clique"
            newnode.attach(node)
            tot_degree += 2
        nodes.append(newnode)
        num_nodes += 1

    # CREATING REST OF THE GRAPH
    while num_nodes < n_nodes:
        random_numbers = randomstate.randint(low=0, high=tot_degree, size=n_conn_per_node)  # select nodes to attach
        attached = []  # prevent attaching a node to another twice                          # the new node
        newnode = Node()

        for n in random_numbers:
            acc = 0
            for node in nodes:
                acc += node.degree
                if n < acc and node not in attached:
                    attached.append(node)
                    node.attach(newnode)
                    newnode.attach(node)
                    tot_degree += 2
                    break
        nodes.append(newnode)
        num_nodes += 1
    return nodes, tot_degree



