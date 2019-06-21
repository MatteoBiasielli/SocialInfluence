import graph.graph as g
import numpy as np
import matplotlib.pyplot as mpl

# PARAMETERS
approach = "pessimistic"
repetitions = 50
budget = 50
B = 0.5  # exploration coefficient
time = 0

true_graph = g.GraphScaleFree.create_graph100()
est_graph = g.GraphScaleFree.create_graph100()

# INITIALIZE REGRET
clairvoyant_best_seeds = true_graph.find_best_seeds(initial_seeds=[], budget=budget, m_c_sampling_iterations=1000,
                                                    verbose=False)
optimal_n_of_activations = true_graph.prog_cascade(clairvoyant_best_seeds)[1]
regret = [optimal_n_of_activations for _ in range(repetitions)]

# INITIALIZATION
est_graph.init_estimates(estimator="ucb1", approach=approach)
time += 1

# Buy seed based on model
seeds, remainder = true_graph.seeds_at_time_zero(budget)

for i in range(repetitions):
    print("\n# Repetition: " + str(i))
    # Witness cascade
    realizations_per_node, nodes_activated = true_graph.prog_cascade(seeds)
    regret[i] = regret[i] - nodes_activated
    print("n° of nodes activated: {}".format(nodes_activated))
    time += 1
    # Update representation (est_graph) based on observations
    for record in realizations_per_node:
        est_graph.update_estimate(record[0], record[1], time=time, estimator="ucb1")

    est_graph.update_weights(estimator="ucb1", use_features=False, exp_coeff=B)

    # Find the best seeds for next repetition
    seeds = est_graph.find_best_seeds(initial_seeds=[], budget=budget, m_c_sampling_iterations=10, verbose=False)
    print("best seeds found: {}".format(seeds))

final_n_of_nodes_activated = true_graph.prog_cascade(seeds)[1]
print("\nFinal num. of nodes activated: {}".format(final_n_of_nodes_activated))

# PLOT REGRET
mpl.plot([0 for _ in range(repetitions)], "--k")
mpl.plot(regret)
mpl.show()
"""
difference = abs(np.subtract(est_graph.get_edges(), true_graph.get_edges()))
print("Differences in edges estimations and true values:")
print(difference)
print("Cumulative error: {}".format(np.sum(difference)))
"""
