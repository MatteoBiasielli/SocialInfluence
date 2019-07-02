from graph.graph import *
import tqdm

real_graph = GraphScaleFree.create_graph100(max_n_neighbors=15)
estimated_graph = GraphScaleFree.create_graph100()
estimated_graph.convert_to_complete(edges_value=0)

num_of_cascades = 1000
num_of_iteration = 50
time = 0

seeds = estimated_graph.get_initial_random_nodes()
estimated_graph.init_estimates(approach="optimistic")

for _ in tqdm.tqdm(range(num_of_iteration)):

    seeds.sort()
    print("seeds for cascade: " + str(seeds))

    for seed in seeds:
        real_graph.nodes[seed].setSeed()

    print("seeds value on the estimated graph:")
    print(sum(estimated_graph.monte_carlo_sampling(500, seeds)))

    for k in range(num_of_cascades):
        time += 1
        real_graph.activate_live_edges()
        cascade_layers = real_graph.propagate_cascade(use_for_edge_estimator=True)
        real_graph.deactivate_live_edges()
        if len(cascade_layers) == 1:
            time -= 1
            continue

        """ CODE that starts from levels of activation and compute, for each node, its samples."""
        for sub_list_id in range(len(cascade_layers) - 1):
            for list_elem in cascade_layers[sub_list_id]:
                realizations = [0] * 100
                for following_list_elem in cascade_layers[sub_list_id + 1]:
                    realizations[following_list_elem] = 1
                realizations.__delitem__(list_elem)
                estimated_graph.update_estimate(list_elem, realizations, time)

    estimated_graph.update_weights(normalize=True)
    real_graph.reset_all_seeds()
    seeds = estimated_graph.find_best_seeds(delta=0.8, initial_seeds=[], verbose=False)

# estimated_graph.print_edges()








