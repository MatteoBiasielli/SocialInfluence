from graph.graph import *
import numpy as np
from multiprocessing import Process, Manager
import tqdm

N_WORKERS = 4
N_NODES = 100
deltas = [0.25, 0.2, 0.15, 0.1, 0.05]
num_of_experiment = 20
experiments_per_worker = num_of_experiment // N_WORKERS
count = 0


def f(app_lists):
    gr = GraphScaleFree.create_graph100()
    for delta in deltas:
        tmp = []
        for _ in tqdm.tqdm(range(experiments_per_worker)):
            # greedy approach = cost_based to do
            best_seeds = gr.find_best_seeds(initial_seeds=[], verbose=False, greedy_approach="standard", delta=delta)
            best_activations_probs = gr.monte_carlo_sampling(1000, best_seeds)
            tmp.append(sum(best_activations_probs))
        app_lists.append(sum(tmp) / len(tmp))


if __name__ == '__main__':
    app_lists = Manager().list()
    # Create processes
    procs = [Process(target=f, args=(app_lists,)) for _ in range(N_WORKERS)]

    # Start all
    for p in procs:
        p.start()

    # join all
    for p in procs:
        p.join()

    # todo the final print
    print(sum(app_lists) / len(app_lists))
