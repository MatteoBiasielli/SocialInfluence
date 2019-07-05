import graph.graph as g
import numpy as np
import csv


# PARAMETERS
graph_nodes = 1000
stimulations = 10
repetitions = 10
experiments = 20
from_exp = 0
delta = 0.95
budget = 800
use_features = False
save_subname = "feats" if use_features else "no_feats"

for k in range(from_exp, experiments):
    print("Experiment " + str(k))

    if graph_nodes == 100:
        true_graph = g.GraphScaleFree.create_graph100()
        est_graph = g.GraphScaleFree.create_graph100()
    elif graph_nodes == 1000:
        true_graph = g.GraphScaleFree.create_graph1000()
        est_graph = g.GraphScaleFree.create_graph1000()

    est_graph.init_estimates(estimator="ts")

    differences_per_repetition = []
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

        est_graph.update_weights(estimator="ts", use_features=use_features)

        # Find the best seeds for next repetition
        seeds = est_graph.find_best_seeds(initial_seeds=[], budget=budget, delta=delta)
        performance_per_repetition.append(sum(true_graph.monte_carlo_sampling(1000, seeds)))
        differences_per_repetition.append(abs(np.subtract(est_graph.get_edges(), true_graph.get_edges())))

    with open("results/ts_" + save_subname + "_" + str(graph_nodes) + "nodes_" + str(repetitions) + "repetitions" +
              str(stimulations) + "stimulations_delta" + str(delta) + "__exp" + str(k) + "_differences.csv",
              "w") as writeFile:
        writer = csv.writer(writeFile)
        for data in differences_per_repetition:
            writer.writerow(data)
    writeFile.close()

    with open("results/ts_" + save_subname + "_" + str(graph_nodes) + "nodes_" + str(repetitions) + "repetitions" +
              str(stimulations) + "stimulations_delta" + str(delta) + "__exp" + str(k) + "_performance.csv",
              "w") as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(performance_per_repetition)
    writeFile.close()

    print(differences_per_repetition)
    print(performance_per_repetition)
