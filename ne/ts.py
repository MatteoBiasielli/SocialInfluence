import graph.graph as g
import numpy as np
import csv
import tqdm


def ts_no_edge_experiment(net_size, stimulations, delta, use_features, experiment_index):
    # --- PARAMETERS --- #
    repetitions = 10
    save_subname = "feats" if use_features else "no_feats"

    if net_size == 100:
        true_graph = g.GraphScaleFree.create_graph100()
        est_graph = g.GraphScaleFree.create_graph100()
        budget = 400
    elif net_size == 1000:
        true_graph = g.GraphScaleFree.create_graph1000()
        est_graph = g.GraphScaleFree.create_graph1000()
        budget = 800
    else:
        print("No net of requested size. Shutting down.")
        return

    est_graph.init_estimates(estimator="ts")

    differences_per_repetition = []
    performance_per_repetition = []

    # --- EXPERIMENT COMPUTATION --- #
    print("Experiment Index: " + str(experiment_index))
    # Buy seed based on model
    seeds, remainder = true_graph.seeds_at_time_zero(budget)
    for i in tqdm.tqdm(range(repetitions)):
        print("Repetition " + str(i))
        for j in range(stimulations):
            # Witness cascade
            realizations_per_node = true_graph.prog_cascade(seeds)[0]
            # Update representation (est_graph) based on observations
            for record in realizations_per_node:
                est_graph.update_estimate(record[0], record[1], estimator="ts")

        est_graph.update_weights(estimator="ts", use_features=use_features)

        # Find the best seeds for next repetition
        seeds = est_graph.find_best_seeds(initial_seeds=[], budget=budget, delta=delta, randomized_search=True,
                                          randomized_search_number=100)
        performance_per_repetition.append(sum(true_graph.monte_carlo_sampling(1000, seeds)))
        differences_per_repetition.append(abs(np.subtract(est_graph.get_edges(), true_graph.get_edges())))

    # --- STORING EXPERIMENT RESULTS --- #
    with open("results/ts_" + save_subname + "_" + str(net_size) + "nodes_" + str(repetitions) + "repetitions" +
              str(stimulations) + "stimulations_delta" + str(delta) + "__exp" + str(experiment_index) + "_differences.csv",
              "w") as writeFile:
        writer = csv.writer(writeFile)
        for data in differences_per_repetition:
            writer.writerow(data)
    writeFile.close()

    with open("results/ts_" + save_subname + "_" + str(net_size) + "nodes_" + str(repetitions) + "repetitions" +
              str(stimulations) + "stimulations_delta" + str(delta) + "__exp" + str(experiment_index) + "_performance.csv",
              "w") as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(performance_per_repetition)
    writeFile.close()

    print(differences_per_repetition)
    print(performance_per_repetition)


# TEST PARAMETERS DOMAIN ---> net_size in [100, 1000], stimulations in [10, 100], delta in [0.2, 0.4, 0.6, 0.95],
#                             use_features in [True, False]

for i in range(1, 3):
    ts_no_edge_experiment(net_size=1000, stimulations=10, delta=0.8, use_features=False, experiment_index=i)
