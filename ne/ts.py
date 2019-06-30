import graph.graph as g
import tqdm
import numpy as np

# PARAMETERS
approach = "pessimistic"
stimulations = 5
repetitions = 10
time = 0

true_graph = g.GraphScaleFree.create_graph100()
est_graph = g.GraphScaleFree.create_graph100()
est_graph.init_estimates(estimator="ts")

budget = true_graph.compute_budget()
performance_per_repetition = []

# Buy seed based on model
seeds, remainder = true_graph.seeds_at_time_zero(budget)
for i in range(repetitions):
    print("Repetition " + str(i))
    for j in range(stimulations):
        # Witness cascade
        realizations_per_node = true_graph.prog_cascade(seeds)[0]
        # Update representation (est_graph) based on observations
        for record in realizations_per_node:
            est_graph.update_estimate(record[0], record[1], estimator="ts")

    est_graph.update_weights(estimator="ts", use_features=False)

    # Find the best seeds for next repetition
    seeds = est_graph.find_best_seeds(initial_seeds=[], budget=budget)
    performance_per_repetition.append(sum(true_graph.monte_carlo_sampling(1000, seeds)))
    print("Performance: " + str(performance_per_repetition[len(performance_per_repetition)-1]))

difference = abs(np.subtract(est_graph.get_edges(), true_graph.get_edges()))
print("Differences in edges estimations and true values:")
print(difference)
print("Cumulative error: {}".format(np.sum(difference)))
print("Performance per repetition:")
print(performance_per_repetition)
