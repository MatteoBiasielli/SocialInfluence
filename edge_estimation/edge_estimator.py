from graph.graph import *


gr = GraphScaleFree.create_graph100()
estimated_graph = GraphScaleFree.create_graph100()
estimated_graph.convert_to_complete()

gr.nodes[0].setSeed()
gr.nodes[3].setSeed()
gr.nodes[7].setSeed()
gr.nodes[10].setSeed()
gr.nodes[15].setSeed()
gr.nodes[21].setSeed()
gr.nodes[25].setSeed()
gr.activate_live_edges()

result = gr.propagate_cascade(use_for_edge_estimator=True)
print(result)

for i in range(1, len(result) - 1):
    for sequence_node in result[i]:
        for previous_node in result[i - 1]:
            print("AAAA") # todo