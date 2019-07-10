import csv
import matplotlib.pyplot as plt

PATH = "./results/ucb1_exp_coeff/"


def plot_different_b_values():
    filename = "ucb1_no_feats_100nodes_10repetitions_100stimulations_delta0.4_B"

    b_values = [0.2, 0.4, 0.6, 0.8, 1]
    plots = []

    for B in b_values:
        with open(PATH + filename + str(B) + "_" + "performance.csv", "r") as csv_file:
            csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
            plots.append(next(csv_reader))

    for p in plots:
        plt.plot(p)

    plt.legend(b_values)
    plt.title("Avg. n° of activated nodes")
    plt.savefig("{}.png".format("performance_all"), dpi=200)
    plt.show()


def plot_pessimistic_vs_optimistic():
    filename = "ucb1_no_feats_100nodes_10repetitions_100stimulations_delta0.4_B0.2_"

    # pessimistic
    with open(PATH + filename + "differences.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        pessimistic = next(csv_reader)

    # optimistic
    with open(PATH + filename + "optimistic_" + "differences.csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        optimistic = next(csv_reader)

    plt.plot(pessimistic)
    plt.plot(optimistic)
    plt.legend(["pessimistic", "optimistic"])
    plt.title("Cumulative error on probs. estimation")
    plt.savefig("{}.png".format("p_vs_o_cum_error"), dpi=200)
    plt.show()


if __name__ == '__main__':
    plot_pessimistic_vs_optimistic()
