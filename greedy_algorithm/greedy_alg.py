from graph.graph import *
import numpy as np
from multiprocessing import Process, Manager
import tqdm

N_WORKERS = 20
N_NODES = 100
appearence_times = np.zeros([N_NODES, 2])
num_of_experiment = 40 // N_WORKERS
count = 0


for i in appearence_times:
    i[1] = count
    count += 1


def f(app_lists):
    gr = GraphScaleFree.create_graph100()
    app_lis = np.zeros(N_NODES)
    for _ in tqdm.tqdm(range(num_of_experiment)):
        tmp = gr.find_best_seeds(initial_seeds=[], verbose=False)
        for id in tmp:
            app_lis[id] += 1
    app_lists.append(app_lis)


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
    for app_list in app_lists:
        for i in range(N_NODES):
            appearence_times[i][0] += app_list[i]
    appearence_times = appearence_times / [num_of_experiment * N_WORKERS, 1]
    appearence_times = list(appearence_times)
    appearence_times.sort(key=lambda tup: tup[0], reverse=True)
    print(appearence_times)
