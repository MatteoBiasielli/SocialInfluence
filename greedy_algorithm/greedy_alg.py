from graph.graph import *
import numpy as np
import tqdm


gr = GraphScaleFree.create_graph100()
num_of_experiment = 100
appearence_times = np.zeros([len(gr.nodes), 2])
count = 0
for i in appearence_times:
    i[1] = count
    count += 1

for i in tqdm.tqdm(range(num_of_experiment)):
    tmp = gr.find_best_seeds(initial_seeds=[], verbose=False)
    for id in tmp:
        appearence_times[id] += 1

appearence_times = appearence_times / [num_of_experiment, 1]
appearence_times = list(appearence_times)
appearence_times.sort(key=lambda tup: tup[0], reverse=True)
print(appearence_times)
