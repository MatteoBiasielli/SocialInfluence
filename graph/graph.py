import numpy as np
import seaborn
import matplotlib.pyplot as mplt
import os
from sklearn.linear_model import LinearRegression
import tqdm


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
        self.adjacency_features = []
        self.ucb1_estimate_param = []  # each element is [empirical mean, bound, numOfSamples, hasBeenVisited]
        self.ts_estimate_param = []
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

    def sort_probabilities(self, seed, adj_matrix, common_args, lin_comb_params):
        for i in range(self.degree):
            feats = [sigmoid(0.2 * (self.degree / self.adjacency_list[i].degree - 1)),
                     seed.rand(),
                     self.n_common_neighbors(self.adjacency_list[i], adj_matrix) / self.adjacency_list[i].degree,
                     common_args[self.id][self.adjacency_list[i].id]]
            self.adjacency_weights[i] = lin_comb_params[0] * feats[0] + \
                                        lin_comb_params[1] * feats[1] + \
                                        lin_comb_params[2] * feats[2] + \
                                        lin_comb_params[3] * feats[3]
            self.adjacency_features.append(feats)

    def update_probabilities(self, lin_comb_params):
        for i in range(self.degree):
            self.adjacency_weights[i] = lin_comb_params[0] * self.adjacency_features[0] + \
                                        lin_comb_params[1] * self.adjacency_features[1] + \
                                        lin_comb_params[2] * self.adjacency_features[2] + \
                                        lin_comb_params[3] * self.adjacency_features[3]

    def n_common_neighbors(self, node, adj_matr):
        return np.dot(adj_matr[self.id, :], adj_matr[node.id, :])


def sigmoid(x, scale=0.2):
    return 1 / (1 + np.exp(-x * scale))


class GraphScaleFree:
    LIN_COMB_PARAMS = [0.3, 0.15, 0.4, 0.15]

    def __init__(self, nodes=100, n_init_nodes=3, n_conn_per_node=2, randomstate=np.random.RandomState(1234),
                 max_n_neighbors=None):
        Node.count_id = 0
        self.adj_matr = np.zeros([nodes, nodes], dtype=np.uint8)
        self.common_args = np.zeros([nodes, nodes], dtype=np.float16)
        self.num_nodes = 0
        self.nodes = []
        self.tot_degree = 0
        self.lin_comb_params = None
        for i in range(n_init_nodes):
            newnode = Node()
            for node in self.nodes:
                node.attach(newnode)
                newnode.attach(node)
                self.tot_degree += 2
            self.nodes.append(newnode)
            self.num_nodes += 1

        while self.num_nodes < nodes:

            attached = []
            newnode = Node()
            while not attached:
                random_numbers = randomstate.randint(low=0, high=self.tot_degree, size=n_conn_per_node)
                for n in random_numbers:
                    acc = 0
                    for node in self.nodes:
                        acc += node.degree
                        if n < acc and node not in attached and \
                                (max_n_neighbors is None or len(node.adjacency_list) < max_n_neighbors):
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

    def set_lin_comb_params(self, pars=None):
        self.lin_comb_params = pars if pars is not None else [p for p in GraphScaleFree.LIN_COMB_PARAMS]

    def n_common_neighbors(self, node1, node2):
        return node1.n_common_neighbors(node2, self.adj_matr)

    def sort_probabilities(self):
        for n in self.nodes:
            n.sort_probabilities(self.randstate, self.adj_matr, self.common_args, self.lin_comb_params)

    def update_probabilities(self):
        for n in self.nodes:
            n.update_probabilities(self.lin_comb_params)

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
            n.cost = 10 + (n.degree ** (2 - 0.5 * n.degree / maxdeg)) * np.mean(n.adjacency_weights)

    def print_costs_degrees(self):
        degrees = []
        costs = []
        for n in self.nodes:
            degrees.append(n.degree)
            costs.append(n.cost)
        print(degrees)
        print(costs, sum(costs), sum(list(reversed(sorted(costs)))[:int(0.1 * len(costs))]))

    @staticmethod
    def create_graph100(max_n_neighbors=None, lin_comb_parameters=None):
        gra = GraphScaleFree(nodes=100, n_init_nodes=3, n_conn_per_node=2,
                             randomstate=np.random.RandomState(1234), max_n_neighbors=max_n_neighbors)
        gra.set_lin_comb_params(pars=lin_comb_parameters)
        gra.sort_probabilities()  # must do this or probabilities will all be 1
        gra.assign_nodes_costs()  # must do this or costs will all be 0
        return gra

    @staticmethod
    def create_graph1000(max_n_neighbors=None, lin_comb_parameters=None):
        gra = GraphScaleFree(nodes=1000, n_init_nodes=3, n_conn_per_node=2,
                             randomstate=np.random.RandomState(1234), max_n_neighbors=max_n_neighbors)
        gra.set_lin_comb_params(pars=lin_comb_parameters)
        gra.sort_probabilities()  # must do this or probabilities will all be 1
        gra.assign_nodes_costs()  # must do this or costs will all be 0
        return gra

    @staticmethod
    def create_graph10000(max_n_neighbors=None, lin_comb_parameters=None):
        gra = GraphScaleFree(nodes=10000, n_init_nodes=3, n_conn_per_node=2,
                             randomstate=np.random.RandomState(1234), max_n_neighbors=max_n_neighbors)
        gra.set_lin_comb_params(pars=lin_comb_parameters)
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

    def compute_budget(self):
        costs = []
        for node in self.nodes:
            costs.append(node.cost)
        costs.sort(reverse=True)
        num_of_considered_nodes = int(0.1 * len(self.nodes))
        cg = 0
        for i in range(0, num_of_considered_nodes):
            cg += costs[i]
        return np.power(0.4 * cg, 1 + 0.5 / np.log2(cg))

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

        for node in self.nodes:
            if node.isSeed():
                result[node.id] += number_of_iterations
                node.removeSeed()
        result = (1 / number_of_iterations) * result

        return result

    def find_best_seeds(self, initial_seeds, budget=None, greedy_approach="standard", m_c_sampling_iterations=100,
                        file_name=""):
        self.assign_nodes_costs()
        feasible_nodes = []
        generated_increments = []
        result = []
        deletion_indexes = []

        for i in range(len(self.nodes)):
            feasible_nodes.append(self.nodes[i])
            generated_increments.append(0)

        budget = self.compute_budget() if budget is None else budget

        for seed in initial_seeds:
            result.append(self.nodes[seed].id)
            budget -= self.nodes[seed].cost

        if budget < 0:
            print("The initial seeds exceed the available budget (" + str(self.compute_budget()) +
                  ". Try again with a cheaper initial seed set.")
            return

        for i in range(len(feasible_nodes)):
            if (feasible_nodes[i].cost > budget or feasible_nodes[i].id in initial_seeds) and i not in deletion_indexes:
                deletion_indexes.append(i)

        deletion_indexes.sort(reverse=True)

        for index in deletion_indexes:
            feasible_nodes.__delitem__(index)
            generated_increments.__delitem__(index)

        deletion_indexes.clear()

        while len(feasible_nodes) != 0:
            print("queue nodes " + str(len(feasible_nodes)) + " - " + "remaining budget " + str(budget) + " ...")

            for i in range(len(feasible_nodes)):

                result.append(feasible_nodes[i].id)
                m_c_probabilities = self.monte_carlo_sampling(m_c_sampling_iterations, seeds=result)
                increment = sum(m_c_probabilities)
                result.__delitem__(-1)

                if greedy_approach == "standard":
                    generated_increments[i] = increment

                elif greedy_approach == "cost_based":
                    generated_increments[i] = increment / feasible_nodes[i].cost

            winner_index = np.argmax(generated_increments)
            winner_node = feasible_nodes[int(winner_index)]
            result.append(winner_node.id)
            budget -= winner_node.cost

            for i in range(len(feasible_nodes)):
                if feasible_nodes[i].id == winner_node.id:
                    deletion_indexes.append(i)
                if feasible_nodes[i].cost > budget and i not in deletion_indexes:
                    deletion_indexes.append(i)

            deletion_indexes.sort(reverse=True)
            for index in deletion_indexes:
                feasible_nodes.__delitem__(index)
                generated_increments.__delitem__(index)

            deletion_indexes.clear()

        result.sort()

        if file_name:
            file = open(file_name + ".csv", "w")
            file.write(str(result[0]))
            for i in range(1, len(result)):
                file.write("," + str(result[i]))
            file.close()

        print("Best nodes found: {}".format(result))
        return result

    def init_estimates(self, estimator="ucb1", approach="pessimistic"):
        """Initializes the estimated probabilities of all edges"""
        for i in self.nodes:
            for j in range(i.degree):
                if estimator == "ucb1":
                    if approach == "pessimistic":
                        i.ucb1_estimate_param = [[0, 0, 1, False] for _ in range(i.degree)]
                    elif approach == "optimistic":
                        i.ucb1_estimate_param = [[1, 0, 1, False] for _ in range(i.degree)]
                    elif approach == "neutral":
                        i.ucb1_estimate_param = [[0.5, 0, 1, False] for _ in range(i.degree)]

                elif estimator == "ts":
                    i.ts_estimate_param = [[1, 1] for _ in range(i.degree)]

    def update_estimate(self, id_from, realizations, time, estimator="ucb1"):
        """Updates the parameters of each edge of the specified node"""
        if estimator == "ucb1":
            estimate_param = self.nodes[id_from].ucb1_estimate_param

            for i in range(len(realizations)):
                # if the edge was "stimulated"
                if realizations[i] != -1:
                    # if first sample ever observed, overwrite mean
                    if not estimate_param[i][3]:
                        estimate_param[i][0] = realizations[i]
                        estimate_param[i][3] = True
                    else:
                        # update empirical mean
                        estimate_param[i][0] = (estimate_param[i][0] * estimate_param[i][2] + realizations[i]) / (
                                estimate_param[i][2] + 1)
                        # increase number of samples
                        estimate_param[i][2] += 1

                # update bound
                estimate_param[i][1] = np.sqrt((2 * np.log(time)) / estimate_param[i][2])

        elif estimator == "ts":
            estimate_param = self.nodes[id_from].ts_estimate_param

            for i in range(len(realizations)):
                if realizations[i] != -1:
                    estimate_param[i][0] += realizations[i]
                    estimate_param[i][1] += 1 - realizations[i]

    def update_weights(self, estimator="ucb1", use_features=False, exp_coeff=1):
        """Updates the estimated probabilities of each edge in the graph (weights)"""
        if estimator == "ucb1":
            all_weights = []
            for node in self.nodes:
                for i in range(node.degree):
                    # new weight = sum of empirical mean and exploration coeff. * ucb1 bound
                    node.adjacency_weights[i] = node.ucb1_estimate_param[i][0] + exp_coeff * \
                                                node.ucb1_estimate_param[i][1]
                    all_weights.append(node.adjacency_weights[i])

            # normalize all weights
            for node in self.nodes:
                for i in range(node.degree):
                    node.adjacency_weights[i] = (node.adjacency_weights[i] - min(all_weights)) / (
                            max(all_weights) - min(all_weights))

        elif estimator == "ts":
            for node in self.nodes:
                for i in range(node.degree):
                    node.adjacency_weights[i] = np.random.beta(a=node.ts_estimate_param[i][0],
                                                               b=node.ts_estimate_param[i][1])

        if use_features:
            self.set_lin_comb_params(self.estimate_features_parameters())
            self.sort_probabilities()

    def estimate_features_parameters(self):
        dataset_x = []
        dataset_y = []
        for node in self.nodes:
            for i in range(node.degree):
                dataset_x.append(node.adjacency_features[i])
                dataset_y.append(node.adjacency_weights[i])

        regression_model = LinearRegression(fit_intercept=False)
        regression_model.fit(X=dataset_x, y=dataset_y)

        return list(regression_model.coef_)

    def prog_cascade(self, seeds):
        explore_next_ids = [s for s in seeds]
        realizations_per_node = []

        for s in seeds:
            self.nodes[s].setActive()

        for i in explore_next_ids:
            realizations = []

            for j in range(self.nodes[i].degree):
                adjacent_node_id = self.nodes[i].adjacency_list[j].id

                if not self.nodes[adjacent_node_id].isActive():
                    realization = np.random.binomial(1, self.nodes[i].adjacency_weights[j])

                    if realization == 1:
                        explore_next_ids.append(adjacent_node_id)
                        self.nodes[adjacent_node_id].setActive()

                    realizations.append(realization)

                else:
                    realizations.append(-1)

            realizations_per_node.append([i, realizations])

        for id in explore_next_ids:
            self.nodes[id].setInactive()

        return realizations_per_node

    def seeds_at_time_zero(self, budget):
        seeds = []
        nodes_deg = [i.degree for i in self.nodes]

        while budget > 0 and len(nodes_deg) > 0:
            seed = int(np.argmax(nodes_deg))

            if budget - self.nodes[seed].cost > 0:
                budget -= self.nodes[seed].cost
                seeds.append(seed)

            nodes_deg.pop(seed)

        return seeds, budget

    def print_estimates(self, estimator="ucb1"):
        for i in self.nodes:
            if estimator == "ucb1":
                estimates = i.ucb1_estimate_param
            elif estimator == "ts":
                estimates = i.ts_estimate_param

            for j in range(len(estimates)):
                print(estimates[j])

    def print_edges(self):
        for i in self.nodes:
            for j in range(len(i.adjacency_weights)):
                print(i.adjacency_weights[j])

    def get_edges(self):
        edges = []
        for i in self.nodes:
            for j in range(len(i.adjacency_weights)):
                edges.append(i.adjacency_weights[j])
        return edges

    def get_empirical_means(self):
        means = []
        for i in self.nodes:
            for j in range(i.degree):
                means.append(i.ucb1_estimate_param[j][0])
        return means


if __name__ == '__main__':
    max_neigh = None
    # HOW TO CREATE THE GRAPH WITH 100 NODES WE WILL USE
    gr = GraphScaleFree(nodes=100, n_init_nodes=3, n_conn_per_node=2,
                        randomstate=np.random.RandomState(1234), max_n_neighbors=max_neigh)
    gr.plot_degrees(name="- Scale-Free 100 Nodes")  # in case you want to plot the distribution of the degrees
    gr.sort_probabilities()  # must do this or probabilities will all be 1
    gr.assign_nodes_costs()  # must do this or costs will all be 0
    gr.to_csv(
        name="graph100" + ("max" + str(max_neigh)) if max_neigh is not None else "")  # in case you want to save it

    seeds = []
    greedy_approach = "standard"
    m_c_sampling_iterations = 100
    file_name = ""
    # initial_seeds: the nodes, if any, that we have already bought as seeds, if the algorithm start from zero,
    # leave an empty list (default = empty list)
    # greedy_approach: "standard" as described in the slides, "cost_based" the increment is divided by the cost of the
    # node that generates the increment (default = "standard)
    # file_name: if not empty, creates and writes the result in a file called "file_name.csv" (default = empty)
    # returns: an ordered list of node ids corresponding to the best seed set found
    print(gr.find_best_seeds(initial_seeds=seeds, greedy_approach=greedy_approach,
                             m_c_sampling_iterations=m_c_sampling_iterations, file_name=file_name))

    # HOW TO CREATE THE GRAPH WITH 1000 NODES WE WILL USE
    gr = GraphScaleFree(nodes=1000, n_init_nodes=3, n_conn_per_node=2,
                        randomstate=np.random.RandomState(1234), max_n_neighbors=max_neigh)
    gr.plot_degrees(name="- Scale-Free 1000 Nodes")  # in case you want to plot the distribution of the degrees
    gr.sort_probabilities()  # must do this or probabilities will all be 1
    gr.assign_nodes_costs()  # must do this or costs will all be 0
    gr.to_csv(
        name="graph1000" + ("max" + str(max_neigh)) if max_neigh is not None else "")  # in case you want to save it

    # Select one or more seeds for the m.c. sampling

    # HOW TO CREATE THE GRAPH WITH 10000 NODES WE WILL USE
    gr = GraphScaleFree(nodes=10000, n_init_nodes=3, n_conn_per_node=2,
                        randomstate=np.random.RandomState(1234), max_n_neighbors=max_neigh)
    gr.plot_degrees(name="- Scale-Free 10000 Nodes")  # in case you want to plot the distribution of the degrees
    gr.sort_probabilities()  # must do this or probabilities will all be 1
    gr.assign_nodes_costs()  # must do this or costs will all be 0
    gr.to_csv(
        name="graph10000" + ("max" + str(max_neigh)) if max_neigh is not None else "")  # in case you want to save it
    """
    seeds = [i for i in range(8000, 10000)]
    probabilities = gr.monte_carlo_sampling(20, seeds)
    print(probabilities, sum(probabilities))
    """
