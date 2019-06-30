import matplotlib.pyplot as plt
from collections import Counter
import graph.graph as g
import numpy as np
import operator
from joblib import Parallel, delayed


def run_experiment(approach, repetitions, budget, B, verbose=True):
    # INITIALIZATION
    true_graph = g.GraphScaleFree.create_graph100()
    est_graph = g.GraphScaleFree.create_graph100()

    est_graph.init_estimates(estimator="ucb1", approach=approach)
    time = 1
    exp_reward = []

    # Buy seed based on model
    history_of_seeds = []
    seeds, remainder = true_graph.seeds_at_time_zero(budget)
    history_of_seeds += seeds

    for i in range(repetitions):
        # Witness cascade
        realizations_per_node, nodes_activated = true_graph.prog_cascade(seeds)
        if verbose:
            print("\n# Repetition: {}\nn° of nodes activated: {}".format(i, nodes_activated))
        time += 1
        # Update representation (est_graph) based on observations
        for record in realizations_per_node:
            est_graph.update_estimate(record[0], record[1], time=time, estimator="ucb1")

        est_graph.update_weights(estimator="ucb1", use_features=False, exp_coeff=B)
        # Find the best seeds for next repetition
        seeds = est_graph.find_best_seeds(initial_seeds=[], delta=0.4, budget=budget, verbose=False)
        # Estimate expected reward with new seeds
        exp_reward.append(sum(true_graph.monte_carlo_sampling(100, seeds)))
        history_of_seeds += seeds
        if verbose:
            print("best seeds found: {}".format(seeds))

    return exp_reward  # , history_of_seeds


if __name__ == '__main__':
    # PARAMETERS
    approach = "pessimistic"
    repetitions = 10
    budget = 50
    B = 0.5  # exploration coefficient
    num_of_experiments = 4

    # CLAIRVOYANT
    # clairvoyant_best_seeds = true_graph.find_best_seeds(initial_seeds=[], delta=0.05, budget=budget, verbose=False)
    # exp_clairvoyant_activations = sum(true_graph.monte_carlo_sampling(500, clairvoyant_best_seeds))
    exp_clairvoyant_activations = 100  # placeholder
    total_seeds = []

    # RUN ALL EXPERIMENTS
    experiments_exp_rewards = Parallel(n_jobs=-1, verbose=11)(  # all cpu are used with -1 (beware of lag)
        delayed(run_experiment)(approach, repetitions, budget, B, verbose=True) for i in range(num_of_experiments))

    # PLOT REGRET
    avg_exp_reward = [sum(x) / len(experiments_exp_rewards) for x in zip(*experiments_exp_rewards)]
    cum_regret = np.cumsum(np.array(exp_clairvoyant_activations) - avg_exp_reward)
    plt.plot(cum_regret)
    plt.show()

    """
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
    """
    # DIFFERENCE IN PROBABILITIES ESTIMATION
    difference = abs(np.subtract(est_graph.get_edges(), true_graph.get_edges()))
    print("Differences in edges estimations and true values:")
    print(difference)
    print("Cumulative error: {}".format(np.sum(difference)))
    """
