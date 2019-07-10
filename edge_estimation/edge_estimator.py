from graph.graph import *
import tqdm
import matplotlib.pyplot as plt

real_graph = GraphScaleFree.create_graph100(max_n_neighbors=15)
estimated_graph = GraphScaleFree.create_graph100(max_n_neighbors=15)
estimated_graph.convert_to_complete(edges_value=1)

num_of_cascades = 200
num_of_iteration = 50
time = 0
cut_treshold = 0.099
edges_per_round = []
edges_per_round.append(9900)

seeds = estimated_graph.get_initial_random_nodes()
estimated_graph.init_estimates(approach="optimistic")
result = []

for i in tqdm.tqdm(range(num_of_iteration)):

    seeds.sort()
    print("iteration " + str(i))
    print("seeds for cascade: " + str(seeds))

    for seed in seeds:
        real_graph.nodes[seed].setSeed()

    result.append(seeds)

    print("seeds value on the estimated graph:")
    print(sum(estimated_graph.monte_carlo_sampling(1000, seeds)))

    for k in range(num_of_cascades):
        time += 1
        real_graph.activate_live_edges()
        cascade_layers = real_graph.propagate_cascade(use_for_edge_estimator=True)
        real_graph.deactivate_live_edges()
        if len(cascade_layers) == 1:
            time -= 1

        """ CODE that starts from levels of activation and compute, for each node, its samples."""
        for sub_list_id in range(len(cascade_layers) - 1):
            for list_elem in cascade_layers[sub_list_id]:
                realizations = [0] * 100
                for following_list_elem in cascade_layers[sub_list_id + 1]:
                    realizations[following_list_elem] = 1
                deletion_indexes = []
                for unattached_id in estimated_graph.nodes[list_elem].unattached_nodes_ids:
                    deletion_indexes.append(unattached_id)
                deletion_indexes.append(list_elem)
                deletion_indexes.sort(reverse=True)
                for deletion_index in deletion_indexes:
                    realizations.__delitem__(deletion_index)
                estimated_graph.update_estimate(list_elem, realizations, time)

    print("Updating weights...")
    estimated_graph.update_weights(normalize=False, exp_coeff=0.15)
    print("Done.")

    print("starting cut procedure:")
    estimated_graph.execute_cut_procedure(weight_treshold=cut_treshold, verbose=False)
    print("Done.")
    print("Total degree of the estimated graph: " + str(estimated_graph.tot_degree))
    edges_per_round.append(estimated_graph.tot_degree)

    real_graph.reset_all_seeds()
    print("Estimating new seeds...")
    seeds = estimated_graph.find_best_seeds(delta=0.4, initial_seeds=[], verbose=False)

file = open("new_seeds_099_6.txt", "w")
file.write(str(result))
file.close()
file = open("edges_per_round_3.txt", "w")
file.write(str(edges_per_round))
file.close()














