from graph.graph import *
from multiprocessing import Process, Manager
import tqdm

N_WORKERS = 1
N_NODES = 100
deltas = [0.95, 0.8, 0.4, 0.2]
num_of_experiment = 5
experiments_per_worker = num_of_experiment // N_WORKERS
count = 0

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

print(res)