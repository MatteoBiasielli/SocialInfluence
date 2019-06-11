import graph.graph as g

# PARAMETERS
approach = "pessimistic"
repetitions = 3
budget = 10

true_graph = g.GraphScaleFree.create_graph100()
est_graph = g.GraphScaleFree.create_graph100()
est_graph.init_estimates(estimator="ts")


# Buy seed based on model
seeds, remainder = true_graph.seeds_at_time_zero(budget)
for i in range(repetitions):
    # Witness cascade
    realizations_per_node = true_graph.prog_cascade(seeds)

    # Update representation (est_graph) based on observations
    for record in realizations_per_node:
        node = est_graph.nodes[record[0]]
        est_graph.update_estimate(node, record[1], estimator="ts")

    est_graph.update_weights(estimator="ts")

    # Find the best seeds for next repetition
    seeds = est_graph.find_best_seeds(initial_seeds=[], m_c_sampling_iterations=100)





