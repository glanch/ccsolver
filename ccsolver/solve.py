import networkx as nx
from gurobipy import Model, GRB, quicksum

import matplotlib.pyplot as plt

from game import Game


def build_graph(game):
    number_tiles = game.number_tiles
    color_tiles = game.color_tiles

    all_tiles = list(number_tiles.keys()) + list(color_tiles.keys())
    # Build graph
    vertices = (
        list(number_tiles.keys()) + list(color_tiles.keys()) + ["t", "s"]
    )  # + [(n, color_tiles[n]) for n in color_tiles.keys()] + [("s", "src"), ("t", "sink")]

    edges = []
    # Add all edges:

    # Source edges
    for (i, j) in color_tiles.keys():
        edges.append(("s", (i, j)))

    # Sink edges
    for coordinate in all_tiles:
        edges.append((coordinate, "t"))

    # Neighboring all tiles
    for coordinate in all_tiles:
        neighbors = game.get_neighbors(coordinate)
        for neighbor_coordinate in neighbors:
            edges.append((coordinate, neighbor_coordinate))

    # # Jump edges (new stroke)
    for coordinate in all_tiles:
        for color_coordinate in color_tiles.keys():
            edges.append((coordinate, color_coordinate))

    graph = nx.DiGraph()
    graph.add_nodes_from(vertices)
    graph.add_edges_from(edges)
    return graph


def solve_with_mtz_sec(game):
    graph = build_graph(game)
    graph.add_edge("t", "s")

    plt.subplot(211)
    nx.draw(
        graph,
        {v:v for v in graph.nodes()} | {"s": (-1, -1), "t": (-2, -2)},
        labels={
            key: f"{key}.{(game.color_tiles | game.number_tiles)[key]}"
            for key in (game.color_tiles | game.number_tiles).keys()
        }
        | {"s": "source", "t": "sink"},
    ) 
    # Create model
    model = Model("trivial_mip")

    x = {}

    # Introduce variable for every edge
    for (i, j) in graph.edges():
        x[(i, j)] = model.addVar(name=f"x_{i}_{j}", vtype=GRB.BINARY)

    # Every vertex should be visited exactly once
    # Every vertex has in-degree=out-degree = 1
    for vertex in graph.nodes():
            # in-degree
            model.addConstr(
                quicksum(x[(pred, vertex)] for pred in graph.predecessors(vertex)) == 1
            )

            # out-degree
            model.addConstr(
                quicksum(x[(vertex, succ)] for succ in graph.successors(vertex)) == 1
            )

    # Subtour Elimination Constraint: MTZ
    u = {}

    for vertex in graph.nodes():
        u[vertex] = model.addVar(name=f"u_{vertex}", lb=0, vtype=GRB.INTEGER)
    
    vertex_count = len(graph.nodes())

    model.addConstr(u["s"] == 1)
    for (i,j) in graph.edges():
        if j != "s":
            model.addConstr(u[i] - u[j] + vertex_count*x[(i, j)] <= vertex_count - 1)

    model.optimize()

    selected_edges = [(i, j) for (i, j) in x.keys() if x[(i, j)].x > 0.0001]

    # Show solution and initial game graph
    print("Selected edges")
    print(selected_edges)

    plt.subplot(212)
    nx.draw_networkx_edges(
        graph,
        {v:v for v in graph.nodes()} | {"s": (-10, -5), "t": (-2, -6)},
        
        edgelist=selected_edges
    ) 

    plt.show()
    while True:
        pass
    
# Alternative game_representations

# game_representation = """
# 1X
# 11
# """
# game_representation = """
# 11X332
# 112222
# """
game_representation = """
11X332
112222
122222
1111 2
1111X1
12X1X1
"""

# Build game
game = Game()
game.parse(game_representation)
solve_with_mtz_sec(game)
