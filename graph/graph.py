import numpy as np
import seaborn
import matplotlib.pyplot as mplt
import os


class Node:

    count_id = 0

    def __init__(self):
        self.attrs = None
        self.seed = False
        self.active = False
        self.suceptible = False
        self.inactive = False
        self.adjacency_list = []
        self.adjacency_weights = []
        self.degree = 0
        self.id = Node.count_id
        Node.count_id += 1

    def attach(self, node, prob=1):
        self.adjacency_list.append(node)
        self.adjacency_weights.append(prob)
        self.degree += 1

    def sort_probabilities(self, seed, adj_matrix, common_args):
        for i in range(self.degree):
            self.adjacency_weights[i] = np.mean([sigmoid(self.degree/self.adjacency_list[i].degree - 1),
                                                 seed.rand(),
                                                 self.n_common_neighbors(self.adjacency_list[i], adj_matrix)/self.adjacency_list[i].degree,
                                                 common_args[self.id][self.adjacency_list[i].id]])

    def n_common_neighbors(self, node, adj_matr):
        return np.dot(adj_matr[self.id, :], adj_matr[node.id, :])


def sigmoid(x, scale=0.2):
    return 1 / (1 + np.exp(-x * scale))

class GraphScaleFree:

    def __init__(self, nodes=100, n_init_nodes=3, n_conn_per_node=2, randomstate=np.random.RandomState(1234)):
        self.adj_matr = np.zeros([nodes, nodes], dtype=np.uint8)
        self.common_args = np.zeros([nodes, nodes], dtype=np.float16)
        self.num_nodes = 0
        self.nodes = []
        self.tot_degree = 0
        for i in range(n_init_nodes):
            newnode = Node()
            for node in self.nodes:
                node.attach(newnode)
                newnode.attach(node)
                self.tot_degree += 2
            self.nodes.append(newnode)
            self.num_nodes += 1

        while self.num_nodes < nodes:
            random_numbers = randomstate.randint(low=0, high=self.tot_degree, size=n_conn_per_node)
            attached = []
            newnode = Node()

            for n in random_numbers:
                acc = 0
                for node in self.nodes:
                    acc += node.degree
                    if n < acc and node not in attached:
                        attached.append(node)
                        node.attach(newnode)
                        newnode.attach(node)
                        self.adj_matr[node.id][newnode.id] = 1
                        self.adj_matr[newnode.id][node.id] = 1
                        self.common_args[newnode.id][node.id] = randomstate.rand()
                        self.common_args[node.id][newnode.id] = self.common_args[newnode.id][node.id]
                        self.tot_degree += 2
                        break
            self.nodes.append(newnode)
            self.num_nodes += 1

        self.randstate = randomstate

    def n_common_neighbors(self, node1, node2):
        return node1.n_common_neighbors(node2, self.adj_matr)

    def sort_probabilities(self):
        for n in self.nodes:
            n.sort_probabilities(self.randstate, self.adj_matr, self.common_args)

    def plot_degrees(self):
        degrees = []
        for node in self.nodes:
            degrees.append(node.degree)

        seaborn.distplot(degrees)
        mplt.show()

    def to_csv(self, name="test", dir="../data/saved_graphs/"):
        file = open(os.path.join(dir, name + "_nodes.csv"), "w")
        file.write("Id,Label\n")
        for node in self.nodes:
            file.write(str(node.id) + "," + str(node.id) + "\n")
        file.close()

        file = open(os.path.join(dir, name + "_edges.csv"), "w")
        file.write("Source,Target,Type,Weight\n")
        for node in self.nodes:
            for i in range(node.degree):
                file.write(str(node.id) + "," +
                           str(node.adjacency_list[i].id) + "," +
                           "Directed" + "," +
                           str(node.adjacency_weights[i]) + "\n")
        file.close()




if __name__ == '__main__':
    gr = GraphScaleFree(nodes=10000, n_init_nodes=3, n_conn_per_node=2)
    gr.plot_degrees()
    gr.sort_probabilities()
    gr.to_csv(name="test2")





