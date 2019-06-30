import matplotlib.pyplot as plt
from collections import Counter
import graph.graph as g
import numpy as np
import operator
from joblib import Parallel, delayed


def avg_errors(exp_results: list, name: str) -> list:
    pass


def run_experiment(approach, repetitions, n_cascades, budget, B, verbose=True) -> dict:
    # INITIALIZATION
    true_graph = g.GraphScaleFree.create_graph100()
    est_graph = g.GraphScaleFree.create_graph100()

    est_graph.init_estimates(estimator="ucb1", approach=approach)
    time = 1
    history_cum_error = []
    history_prob_errors = []

    # Buy seed based on model
    seeds, remainder = true_graph.seeds_at_time_zero(budget)
    history_of_seeds = [seeds]

    for i in range(repetitions):
        if verbose:
            print("\n# Repetition: {}".format(i))
        # multiple cascades for repetition
        for j in range(n_cascades):
            # Witness cascade
            realizations_per_node, nodes_activated = true_graph.prog_cascade(seeds)
            time += 1
            # Update representation (est_graph) based on observations
            for record in realizations_per_node:
                est_graph.update_estimate(record[0], record[1], time=time, estimator="ucb1")

            est_graph.update_weights(estimator="ucb1", use_features=False, exp_coeff=B)
        # Find the best seeds for next repetition
        seeds = est_graph.find_best_seeds(initial_seeds=[], delta=0.4, budget=budget, verbose=False)
        # Update performance statistics (seeds selected, probabilities estimation)
        history_of_seeds.append(seeds)
        prob_errors = np.subtract(est_graph.get_edges(), true_graph.get_edges())
        history_prob_errors.append(prob_errors)
        cum_prob_error = np.sum(abs(prob_errors))
        history_cum_error.append(cum_prob_error)
        if verbose:
            print("best seeds found: {}\ncumulative error: {}".format(seeds, cum_prob_error))

    return {"cum_error": history_cum_error, "prob_errors": history_prob_errors, "seeds": history_of_seeds}


if __name__ == '__main__':
    true_g = g.GraphScaleFree.create_graph100()

    # PARAMETERS
    approach = "pessimistic"
    repetitions = 15
    n_cascades = 50
    budget = true_g.compute_budget(100)
    B = 0.3  # exploration coefficient
    num_of_experiments = 1

    # CLAIRVOYANT
    """
    clairvoyant_best_seeds = true_graph.find_best_seeds(initial_seeds=[], delta=0.05, budget=budget, verbose=False)
    exp_clairvoyant_activations = sum(true_graph.monte_carlo_sampling(500, clairvoyant_best_seeds))
    """
    exp_clairvoyant_activations = 100  # placeholder
    total_seeds = []

    # RUN ALL EXPERIMENTS
    results = Parallel(n_jobs=1, verbose=11)(  # all cpu are used with -1 (beware of lag)
        delayed(run_experiment)(approach, repetitions, n_cascades, budget, B, verbose=True) for i in
        range(num_of_experiments))  # returns a list of results (each item is a dictionary of results)

    # PLOT CUMULATIVE ERROR
    plt.plot(results[0]["cum_error"])  # TODO multiple exps
    plt.title("Cumulative error")
    plt.show()

    # PLOT CUMULATIVE REGRET (with respect to clairvoyant expected activations)
    sel_seeds = results[0]["seeds"]  # TODO multiple exps
    exp_reward = [sum(true_g.monte_carlo_sampling(1000, seeds)) for seeds in sel_seeds]
    cum_regret = np.cumsum(np.array(exp_clairvoyant_activations) - exp_reward)
    plt.plot(cum_regret)
    plt.title("Cumulative activations regret")
    plt.show()

    # PLOT AVG N° OF ACTIVATED NODES
    plt.plot(exp_reward)  # TODO multiple exps
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
