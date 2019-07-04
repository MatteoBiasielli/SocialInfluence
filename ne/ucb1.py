import matplotlib.pyplot as plt
from collections import Counter
import graph.graph as g
import numpy as np
import operator
from joblib import Parallel, delayed


def run_experiment(approach, repetitions, stimulations, B, delta, use_features=False, verbose=True) -> dict:
    # INITIALIZATION
    true_graph = g.GraphScaleFree.create_graph100()
    est_graph = g.GraphScaleFree.create_graph100()

    est_graph.init_estimates(estimator="ucb1", approach=approach)
    time = 1

    history_cum_error = []
    history_prob_errors = []
    history_of_seeds = []

    # Buy seed based on model
    budget = true_graph.compute_budget(100)
    seeds, remainder = true_graph.seeds_at_time_zero(budget)

    for i in range(repetitions):
        # Multiple stimulations of the network
        for j in range(stimulations):
            # Witness cascade
            realizations_per_node, nodes_activated = true_graph.prog_cascade(seeds)
            time += 1
            # Update representation (est_graph) based on observations
            for record in realizations_per_node:
                est_graph.update_estimate(record[0], record[1], time=time, estimator="ucb1")

        # Update weights (edges probabilities)
        est_graph.update_weights(estimator="ucb1", use_features=use_features, exp_coeff=B, normalize=False)
        # Find the best seeds for next repetition
        seeds = est_graph.find_best_seeds(initial_seeds=[], delta=delta, budget=budget, verbose=False)
        # Update performance statistics (seeds selected, probabilities estimation)
        history_of_seeds.append(seeds)
        prob_errors = np.subtract(true_graph.get_edges(), est_graph.get_empirical_means())
        history_prob_errors.append(prob_errors)
        cum_prob_error = np.sum(abs(prob_errors))
        history_cum_error.append(cum_prob_error)
        if verbose:
            print("\n# Repetition: {}\nbest seeds found: {}\ncumulative error: {}".format(i, seeds, cum_prob_error))

    return {"cum_error": history_cum_error, "prob_errors": history_prob_errors, "seeds": history_of_seeds}


if __name__ == '__main__':
    true_g = g.GraphScaleFree.create_graph100()

    # PARAMETERS
    approach = "pessimistic"
    repetitions = 20  # should be at least 10
    stimulations = 100
    B = 0.2  # exploration coefficient
    delta = 0.4  # should be 0.2, 0.4, 0.8, 0.95
    num_of_experiments = 20  # should be 20
    use_features = False

    # CLAIRVOYANT
    clairvoyant_best_seeds = true_g.find_best_seeds(initial_seeds=[], delta=0.1, verbose=False)
    exp_clairvoyant_activations = sum(true_g.monte_carlo_sampling(1000, clairvoyant_best_seeds))

    total_seeds = []

    # RUN ALL EXPERIMENTS
    results = Parallel(n_jobs=-2, verbose=11)(  # all cpu are used with -1 (beware of lag)
        delayed(run_experiment)(approach, repetitions, stimulations, B, delta, use_features, verbose=True) for i in
        range(num_of_experiments))  # returns a list of results (each item is a dictionary of results)

    # PLOT CUMULATIVE ERROR
    cum_errors = [result["cum_error"] for result in results]
    avg_cum_error = [sum(x) / len(cum_errors) for x in zip(*cum_errors)]
    plt.plot(results[0]["cum_error"])
    plt.title("Cumulative error")
    plt.show()

    # PLOT CUMULATIVE REGRET (with respect to clairvoyant expected activations)
    exp_rewards = []
    for exp in range(len(results)):  # for each experiment compute list of rewards
        sel_seeds = results[exp]["seeds"]
        exp_rewards.append([sum(true_g.monte_carlo_sampling(1000, seeds)) for seeds in sel_seeds])
    avg_exp_rewards = [sum(x) / len(exp_rewards) for x in zip(*exp_rewards)]

    cum_regret = np.cumsum(np.array(exp_clairvoyant_activations) - avg_exp_rewards)
    plt.plot(cum_regret)
    plt.title("Cumulative activations regret")
    plt.show()

    # PLOT AVG N° OF ACTIVATED NODES
    plt.plot(avg_exp_rewards)
    plt.title("Avg. activated nodes")
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
