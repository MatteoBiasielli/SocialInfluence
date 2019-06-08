import numpy as np
import seaborn
import matplotlib.pyplot as mplt
import os


class Node:

    count_id = 0

    def __init__(self):
        self.attrs = None
        self.seed = False
        self.cost = 0
        self.active = False
        self.suceptible = False
        self.inactive = False
        self.adjacency_list = []
        self.adjacency_weights = []
        self.adjacency_live = []
        self.degree = 0
        self.id = Node.count_id
        Node.count_id += 1

    def attach(self, node, prob=1):
        self.adjacency_list.append(node)
        self.adjacency_weights.append(prob)
        self.adjacency_live.append(0)
        self.degree += 1

    def isSeed(self):
        return self.seed

    def isActive(self):
        return self.active

    def isInactive(self):
        return self.inactive

    def isSuceptible(self):
        return self.suceptible

    def setActive(self):

        self.active = True
        self.suceptible = False
        self.inactive = False

    def setInactive(self):
        self.active = False
        self.suceptible = False
        self.inactive = True

    def setSuceptible(self):
        self.active = False
        self.suceptible = True
        self.inactive = False

    def setSeed(self):
        self.seed = True

    def resetNode(self):
        if not self.isSeed():
            self.suceptible = True
        self.inactive = False
        self.active = False

    def removeSeed(self):
        self.seed = False

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
        Node.count_id = 0
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

    def plot_degrees(self, name=""):
        degrees = []
        for node in self.nodes:
            degrees.append(node.degree)

        ax = seaborn.distplot(degrees, kde=None)
        ax.set_title("Degrees Distribution " + name)
        ax.set(xlabel="Degree")
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

    def assign_nodes_costs(self):
        maxdeg = np.max([n.degree for n in self.nodes])
        for n in self.nodes:
            n.cost = 10 + (n.degree ** (2 - 0.5*n.degree/maxdeg)) * np.mean(n.adjacency_weights)

    def print_costs_degrees(self):
        degrees = []
        costs = []
        for n in self.nodes:
            degrees.append(n.degree)
            costs.append(n.cost)
        print(degrees)
        print(costs, sum(costs), sum(list(reversed(sorted(costs)))[:int(0.1*len(costs))]))

    @staticmethod
    def create_graph100():
        gra = GraphScaleFree(nodes=100, n_init_nodes=3, n_conn_per_node=2, randomstate=np.random.RandomState(1234))
        gra.sort_probabilities()  # must do this or probabilities will all be 1
        gra.assign_nodes_costs()  # must do this or costs will all be 0
        return gra

    @staticmethod
    def create_graph1000():
        gra = GraphScaleFree(nodes=1000, n_init_nodes=3, n_conn_per_node=2, randomstate=np.random.RandomState(1234))
        gra.sort_probabilities()  # must do this or probabilities will all be 1
        gra.assign_nodes_costs()  # must do this or costs will all be 0
        return gra

    @staticmethod
    def create_graph10000():
        gra = GraphScaleFree(nodes=10000, n_init_nodes=5, n_conn_per_node=10, randomstate=np.random.RandomState(1234))
        gra.sort_probabilities()  # must do this or probabilities will all be 1
        gra.assign_nodes_costs()  # must do this or costs will all be 0
        return gra

    def activate_live_edges(self):
        for node in self.nodes:
            for i in range(len(node.adjacency_list)):
                if np.random.binomial(1, node.adjacency_weights[i]) == 1:
                    node.adjacency_live[i] = 1

    def deactivate_live_edges(self):
        for node in self.nodes:
            for i in range(len(node.adjacency_live)):
                if node.adjacency_live[i] == 1:
                    node.adjacency_live[i] = 0

    def prepare_for_cascade(self):
        for node in self.nodes:
            node.setSuceptible()

    def get_inactive_nodes_ids(self):
        result = []
        for node in self.nodes:
            if node.isInactive():
                result.append(node.id)
        return result

    def propagate_cascade(self):
        self.prepare_for_cascade()
        just_activated_nodes = []
        propagation = True
        # Starting from seeds, it activates all the node corresponding to live edges
        for node in self.nodes:
            if node.isSeed():
                for i in range(len(node.adjacency_list)):
                    if (node.adjacency_list[i].isSuceptible()) & (not node.adjacency_list[i].isSeed()) & (
                            node.adjacency_live[i] == 1):
                        node.adjacency_list[i].setActive()

        # For all the live edges, spreads the activation to all the neighbours
        while propagation:
            propagation = False

            for node in self.nodes:
                if (node.isActive()) & (not node.isSeed()) & (node not in just_activated_nodes):
                    propagation = True
                    for i in range(len(node.adjacency_list)):
                        if (node.adjacency_list[i].isSuceptible()) & (not node.adjacency_list[i].isSeed()) & (
                                node.adjacency_live[i] == 1):
                            node.adjacency_list[i].setActive()
                            just_activated_nodes.append(node.adjacency_list[i])
                    node.setInactive()

            just_activated_nodes.clear()

    def monte_carlo_sampling(self, number_of_iterations, seeds):
        result = np.zeros(len(self.nodes))
        for seed in seeds:
            self.nodes[seed].setSeed()

        for i in range(number_of_iterations):
            self.activate_live_edges()
            self.propagate_cascade()
            tmp = self.get_inactive_nodes_ids()
            for id in tmp:
                result[id] += 1
            self.deactivate_live_edges()
        result = (1 / number_of_iterations) * result

        return result


if __name__ == '__main__':
    # HOW TO CREATE THE GRAPH WITH 100 NODES WE WILL USE
    gr = GraphScaleFree(nodes=100, n_init_nodes=3, n_conn_per_node=2, randomstate=np.random.RandomState(1234))
    gr.plot_degrees(name="- Scale-Free 100 Nodes")  # in case you want to plot the distribution of the degrees
    gr.sort_probabilities()  # must do this or probabilities will all be 1
    gr.assign_nodes_costs()  # must do this or costs will all be 0
    gr.to_csv(name="graph100")  # in case you want to save it

    # Select one or more seeds for the m.c. sampling
    seeds = [35, 45]
    probabilities = gr.monte_carlo_sampling(1000, seeds)
    print(probabilities)

    # HOW TO CREATE THE GRAPH WITH 1000 NODES WE WILL USE
    gr = GraphScaleFree(nodes=1000, n_init_nodes=3, n_conn_per_node=2, randomstate=np.random.RandomState(1234))
    gr.plot_degrees(name="- Scale-Free 1000 Nodes")  # in case you want to plot the distribution of the degrees
    gr.sort_probabilities()  # must do this or probabilities will all be 1
    gr.assign_nodes_costs()  # must do this or costs will all be 0
    gr.to_csv(name="graph1000")  # in case you want to save it

    # HOW TO CREATE THE GRAPH WITH 10000 NODES WE WILL USE
    gr = GraphScaleFree(nodes=10000, n_init_nodes=5, n_conn_per_node=10, randomstate=np.random.RandomState(1234))
    gr.plot_degrees(name="- Scale-Free 10000 Nodes")  # in case you want to plot the distribution of the degrees
    gr.sort_probabilities()  # must do this or probabilities will all be 1
    gr.assign_nodes_costs()  # must do this or costs will all be 0
    gr.to_csv(name="graph10000")  # in case you want to save it





