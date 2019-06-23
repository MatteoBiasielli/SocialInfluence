from graph.graph import *
from multiprocessing import Process, Manager
import tqdm

N_WORKERS = 8
N_NODES = 100
deltas = [0.9, 0.6]
num_of_experiment = 40
experiments_per_worker = num_of_experiment // N_WORKERS
count = 0


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
