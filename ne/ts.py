import graph.graph as g
import tqdm

# PARAMETERS
approach = "pessimistic"
repetitions = 1
budget = 50

true_graph = g.GraphScaleFree.create_graph100()
est_graph = g.GraphScaleFree.create_graph100()
est_graph.init_estimates(estimator="ts")


# Buy seed based on model
seeds, remainder = true_graph.seeds_at_time_zero(budget)
true_graph.print_edges()
for i in range(repetitions):
    print("Repetition " + str(i))
    # Witness cascade
    realizations_per_node = true_graph.prog_cascade(seeds)

    # Update representation (est_graph) based on observations
    for record in realizations_per_node:
        est_graph.update_estimate(record[0], record[1], estimator="ts")

    est_graph.update_weights(estimator="ts", scenario="features")

    # Find the best seeds for next repetition
    seeds = est_graph.find_best_seeds(initial_seeds=[], budget=budget, m_c_sampling_iterations=10)

est_graph.print_estimates(estimator="ts")
est_graph.print_edges()




