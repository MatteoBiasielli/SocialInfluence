import matplotlib.pyplot as plt
from collections import Counter
import graph.graph as g
import numpy as np
import operator


def run_experiment(approach, repetitions, budget, B, clairvoyant):
    # INITIALIZATION
    est_graph.init_estimates(estimator="ucb1", approach=approach)
    time = 1
    regret = [clairvoyant for _ in range(repetitions)]

    # Buy seed based on model
    history_of_seeds = []
    seeds, remainder = true_graph.seeds_at_time_zero(budget)
    history_of_seeds += seeds

    for i in range(repetitions):
        print("\n# Repetition: " + str(i))
        # Witness cascade
        realizations_per_node, nodes_activated = true_graph.prog_cascade(seeds)
        print("n° of nodes activated: {}".format(nodes_activated))
        time += 1
        # Compute regret
        exp_activations = sum(true_graph.monte_carlo_sampling(500, seeds))
        regret[i] = regret[i] - exp_activations
        # Update representation (est_graph) based on observations
        for record in realizations_per_node:
            est_graph.update_estimate(record[0], record[1], time=time, estimator="ucb1")

        est_graph.update_weights(estimator="ucb1", use_features=False, exp_coeff=B)

        # Find the best seeds for next repetition
        seeds = est_graph.find_best_seeds(initial_seeds=[], delta=0.8, budget=budget, verbose=False)
        print("best seeds found: {}".format(seeds))
        history_of_seeds += seeds

    final_n_of_nodes_activated = true_graph.prog_cascade(seeds)[1]
    print("\nFinal num. of nodes activated: {}".format(final_n_of_nodes_activated))

    return regret, history_of_seeds


if __name__ == '__main__':

    # PARAMETERS
    approach = "pessimistic"
    repetitions = 100
    budget = 50
    B = 0.5  # exploration coefficient
    num_of_experiments = 10

    true_graph = g.GraphScaleFree.create_graph100()
    est_graph = g.GraphScaleFree.create_graph100()

    # CLAIRVOYANT (maybe node 6 with circa 50 exp activations?)
    clairvoyant_best_seeds = true_graph.find_best_seeds(initial_seeds=[], delta=0.05, budget=budget, verbose=False)
    exp_clairvoyant_activations = sum(true_graph.monte_carlo_sampling(500, clairvoyant_best_seeds))
    avg_regret = []
    total_seeds = []

    # RUN ALL EXPERIMENTS
    for i in range(num_of_experiments):
        exp_regret, selected_seeds = run_experiment(approach, repetitions, budget, B,
                                                    clairvoyant=exp_clairvoyant_activations)
        avg_regret.append(exp_regret)
        total_seeds += selected_seeds

    # PLOT REGRET
    regret_to_plot = [sum(x) / len(avg_regret) for x in zip(*avg_regret)]
    plt.plot([0 for _ in range(repetitions)], "--k")
    plt.plot(regret_to_plot)
    plt.show()

    # PLOT MOST SELECTED SEEDS
    seed_popularity = Counter(total_seeds)
    sorted_seeds = sorted(seed_popularity.items(), key=operator.itemgetter(1))
    sorted_seeds.reverse()
    sorted_seeds = sorted_seeds[:25]  # top 25 most selected
    pos = np.arange(len(sorted_seeds))
    plt.bar(pos, [x[1] for x in sorted_seeds])
    plt.xticks(pos, [x[0] for x in sorted_seeds])
    plt.show()
    """
    # DIFFERENCE IN PROBABILITIES ESTIMATION
    difference = abs(np.subtract(est_graph.get_edges(), true_graph.get_edges()))
    print("Differences in edges estimations and true values:")
    print(difference)
    print("Cumulative error: {}".format(np.sum(difference)))
    """
