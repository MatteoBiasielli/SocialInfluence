from graph.graph import *
from multiprocessing import Process, Manager
import tqdm

N_WORKERS = 1
N_NODES = 100
deltas = [0.95, 0.8, 0.4, 0.2]
num_of_experiment = 5
experiments_per_worker = num_of_experiment // N_WORKERS
count = 0

"""
def f(app_lists):
    i = 0
    for delta in deltas:
        tmp = []
        for _ in tqdm.tqdm(range(experiments_per_worker)):
            gr = GraphScaleFree.create_graph100()
            # greedy approach = cost_based to do
            best_seeds = gr.find_best_seeds(initial_seeds=[], verbose=False, greedy_approach="standard", delta=delta)
            print(best_seeds)
            best_activations_probs = gr.monte_carlo_sampling(1000, best_seeds)
            tmp.append(sum(best_activations_probs))
        app_lists[i].append(sum(tmp) / len(tmp))
        print(delta, app_lists[i][-1])

        i += 1


if __name__ == '__main__':
    man = Manager()
    app_lists = man.list()
    for delta in deltas:
        app_lists.append(man.list())

    # Create processes
    procs = [Process(target=f, args=(app_lists,)) for _ in range(N_WORKERS)]

    # Start all
    for p in procs:
        p.start()

    # join all
    for p in procs:
        p.join()

    for i in range(len(deltas)):
        print(deltas[i], list(app_lists[i]))
"""

res = []
for delta in deltas:
    tmp = []
    for _ in tqdm.tqdm(range(experiments_per_worker)):
        gr = GraphScaleFree.create_graph1000(max_n_neighbors=30)
        # greedy approach = cost_based to do
        best_seeds = gr.find_best_seeds(initial_seeds=[], verbose=True, greedy_approach="standard", delta=delta, randomized_search=True, randomized_search_number=100, budget=800)
        best_activations_probs = gr.monte_carlo_sampling(1000, best_seeds)
        tmp.append(sum(best_activations_probs))
        print(tmp[-1])
    res.append([delta, tmp])

print(res)
"""
res = []
for delta in deltas:
    tmp = []
    for _ in tqdm.tqdm(range(experiments_per_worker)):
        gr = GraphScaleFree.create_graph1000(max_n_neighbors=20)
        # greedy approach = cost_based to do
        best_seeds = gr.find_best_seeds(initial_seeds=[], verbose=True, greedy_approach="standard", delta=delta)
        best_activations_probs = gr.monte_carlo_sampling(1000, best_seeds)
        tmp.append(sum(best_activations_probs))
    res.append([delta, tmp])

print(res)"""


"""[[0.95, [67.98699999999998, 66.192, 67.05800000000002, 69.38000000000001, 65.887, 69.28999999999998, 71.36000000000001, 69.77100000000002, 65.8, 69.36]], [0.8, [64.74, 68.64699999999998, 65.93000000000002, 64.028, 66.61100000000003, 69.51899999999999, 68.157, 64.087, 65.31400000000002, 67.45700000000004]], [0.6, [67.90599999999998, 67.93299999999999, 68.10300000000002, 67.642, 66.031, 66.678, 67.63499999999998, 62.472, 65.53900000000002, 68.20400000000001]], [0.4, [65.02499999999999, 70.31699999999996, 68.853, 67.707, 68.12200000000001, 67.023, 66.61800000000001, 66.30700000000002, 66.14299999999999, 68.571]], [0.2, [66.101, 67.59100000000001, 68.08500000000002, 66.381, 68.34500000000001, 67.748, 69.29699999999998, 67.508, 69.74699999999999, 69.181]]]"""