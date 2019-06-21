import graph.graph as g
import tqdm
import numpy as np

# PARAMETERS
approach = "pessimistic"
repetitions = 100
budget = 50
time = 0

true_graph = g.GraphScaleFree.create_graph100()
est_graph = g.GraphScaleFree.create_graph100()
est_graph.init_estimates(estimator="ts")
time += 1

# Buy seed based on model
seeds, remainder = true_graph.seeds_at_time_zero(budget)
true_graph.print_edges()
for i in range(repetitions):
    print("Repetition " + str(i))
    # Witness cascade
    realizations_per_node = true_graph.prog_cascade(seeds)[0]
    time += 1
    # Update representation (est_graph) based on observations
    for record in realizations_per_node:
        est_graph.update_estimate(record[0], record[1], time=time, estimator="ts")

    est_graph.update_weights(estimator="ts", use_features=False)

    # Find the best seeds for next repetition
    seeds = est_graph.find_best_seeds(initial_seeds=[], budget=budget, m_c_sampling_iterations=10)

difference = abs(np.subtract(est_graph.get_edges(), true_graph.get_edges()))
print("Differences in edges estimations and true values:")
print(difference)
print("Cumulative error: {}".format(np.sum(difference)))
