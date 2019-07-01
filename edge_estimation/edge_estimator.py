from graph.graph import *


gr = GraphScaleFree.create_graph100()
estimated_graph = GraphScaleFree.create_graph100()

gr.find_best_seeds(initial_seeds=[])