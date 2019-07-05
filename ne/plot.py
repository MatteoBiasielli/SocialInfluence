import seaborn as sb
import csv

PATH = "./results/"
N_EXP = 20


def read_from_csv(filename):

    with open(PATH + filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        data = [row for row in csv_reader]

    csv_file.close()

    return data[:-1]


it_nodes = [100]
it_stim = [10, 100]
it_delta = [0.2, 0.4, 0.8, 0.95]
it_exp = [i for i in range(N_EXP)]

for i in it_nodes:
    for j in it_stim:
        for k in it_delta:
            for l in it_exp:
                data = read_from_csv("ts_no_feats_" + str(i) + "nodes_10repetitions" + str(j) + "stimulations_delta" + str(k) +
                                     "__exp" + str(l) + "_differences.csv")

                print(data)

