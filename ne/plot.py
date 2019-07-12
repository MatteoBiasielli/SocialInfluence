import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import csv

LOAD_PATH = "./results/"
SAVE_PATH = "./plots/"


def read_from_csv(filename):
    data = []

    with open(LOAD_PATH + filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row:
                data.append(row)

    csv_file.close()
    return data


def plot_experiment_differences(nodes=1000, stimulations=10, delta=0.4, n_exp=3, features=False, save=False):
    feats_subname = "feats" if features else "no_feats"
    plt.title("Edge Probability Errors\n(" + str(nodes) + "n | " + feats_subname + " | " + str(stimulations)
              + "s | " + str(delta) + "d | " + str(n_exp) + "exp)")
    plt.xlabel("Repetition")
    plt.ylabel("Error Over Edge Probabilities")
    experiments_set_differences = []

    for i in range(n_exp):
        data = read_from_csv("ts_" + feats_subname + "_" + str(nodes) + "nodes_10repetitions" + str(stimulations) +
                             "stimulations_delta" + str(delta) + "__exp" + str(i) + "_differences.csv")
        experiments_set_differences.append(data)

    # Convert to float
    for i in range(len(experiments_set_differences)):
        for j in range(len(experiments_set_differences[0])):
            for k in range(len(experiments_set_differences[0][0])):
                experiments_set_differences[i][j][k] = float(experiments_set_differences[i][j][k])

    # Average over experiments
    experiments_set_differences = list(np.mean(experiments_set_differences, axis=0))

    # Plot results
    sb.boxplot(x=[i for i in range(1, 11)], y=experiments_set_differences)

    if save:
        plt.savefig(fname=SAVE_PATH + "plot_differences_" + str(nodes) + "_" + str(stimulations) + "_" + str(delta) + "_"
                          + feats_subname + ".png")

    plt.show()


def plot_experiment_performance(nodes=100, stimulations=10, delta=0.95, n_exp=20, features=False, save=False):
    feats_subname = "feats" if features else "no_feats"
    plt.title("Average Number Of Activations\n(" + str(nodes) + "n | " + feats_subname + " | " + str(stimulations)
              + "s | " + str(delta) + "d | " + str(n_exp) + "exp)")
    plt.xlabel("Repetition")
    plt.ylabel("Average Number of Activations")
    experiments_set_performance = []

    for i in range(n_exp):
        data = read_from_csv("ts_" + feats_subname + "_" + str(nodes) + "nodes_10repetitions" + str(stimulations) +
                             "stimulations_delta" + str(delta) + "__exp" + str(i) + "_performance.csv")
        experiments_set_performance += data

    # Convert to float
    for i in range(len(experiments_set_performance)):
        for j in range(len(experiments_set_performance[0])):
            experiments_set_performance[i][j] = float(experiments_set_performance[i][j])

    # Reorder in chunks per repetition
    processed_set_performance = []
    for i in range(len(experiments_set_performance[0])):
        tmp_repetition = []
        for j in range(len(experiments_set_performance)):
            tmp_repetition.append(experiments_set_performance[j][i])

        processed_set_performance.append(tmp_repetition)

    # Plot results
    if n_exp >= 5:
        sb.boxplot(x=[i for i in range(1, 11)], y=processed_set_performance)
    else:
        sb.pointplot(x=np.reshape([[i]*n_exp for i in range(1, 11)], -1), y=np.reshape(processed_set_performance, -1))

    if save:
        plt.savefig(fname=SAVE_PATH + "plot_performance_" + str(nodes) + "_" + str(stimulations) + "_" + str(delta) + "_"
                          + feats_subname + ".png")

    plt.show()


def plot_regret(nodes=100, stimulations=100, features=False, save=False):
    feats_subname = "feats" if features else "no_feats"
    n_exp = 20 if nodes == 100 else 3
    plt.title("Average Cumulative Regret\n(" + str(nodes) + "n | " + feats_subname + " | " + str(stimulations)
              + "s | " + str(n_exp) + "exp)")
    plt.xlabel("Repetition")
    plt.ylabel("Average Cumulative Regret")
    experiments_set_performance = []
    cum_regret_per_delta = []

    for delta in [0.2, 0.4, 0.8, 0.95]:
        for i in range(n_exp):
            data = read_from_csv("ts_" + feats_subname + "_" + str(nodes) + "nodes_10repetitions" + str(stimulations) +
                                 "stimulations_delta" + str(delta) + "__exp" + str(i) + "_performance.csv")
            experiments_set_performance += data

        # Convert to float
        for i in range(len(experiments_set_performance)):
            for j in range(len(experiments_set_performance[0])):
                experiments_set_performance[i][j] = float(experiments_set_performance[i][j])

        # Reorder in chunks per repetition
        processed_set_performance = []
        for i in range(len(experiments_set_performance[0])):
            tmp_repetition = []
            for j in range(len(experiments_set_performance)):
                tmp_repetition.append(experiments_set_performance[j][i])

            processed_set_performance.append(tmp_repetition)

        # Compute cumulative regret
        mean_set_performance = np.mean(processed_set_performance, axis=1)
        clairvoyant_performance = 45 if nodes == 100 else 330
        clr_set_performance = np.repeat(clairvoyant_performance, len(mean_set_performance))
        inst_regret = clr_set_performance - mean_set_performance
        cum_regret = np.cumsum(inst_regret)

        cum_regret_per_delta.append(cum_regret)

    # Plot results
    for i in range(len(cum_regret_per_delta)):
        plt.plot([i for i in range(1, 11)], cum_regret_per_delta[i])

    plt.legend(["0.2", "0.4", "0.8", "0.95"])

    if save:
        plt.savefig(
            fname=SAVE_PATH + "plot_regret_" + str(nodes) + "_" + str(stimulations) + "_"
                  + feats_subname + ".png")

    plt.show()


for nodes in [100, 1000]:
    for features in [True, False]:
        for stimulations in [10, 100]:
            plot_regret(nodes=nodes, stimulations=stimulations, features=features, save=True)
