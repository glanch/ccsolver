import networkx as nx
from gurobipy import Model, GRB, quicksum

import matplotlib.pyplot as plt

from game import Game, parse


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


def solve_with_mtz_sec(game, visualize=True, add_order_constraints=True, suppress_gurobi_output=True):
    graph = build_graph(game)

    # Add subloop so MTZ TSP formulation can be applied here
    graph.add_edge("t", "s")

    graph_positions = {
        v: (2 * v[0], 5 * v[1]) for v in graph.nodes() if v != "s" and v != "t"
    } | {"s": (-10, -5), "t": (-2, -6)}
    node_labels = {
        key: f"{key}.{(game.color_tiles | game.number_tiles)[key]}"
        for key in (game.color_tiles | game.number_tiles).keys()
    } | {"s": "source", "t": "sink"}

    node_size = 1000

    if visualize:
        plt.subplot(211)
        nx.draw(graph, pos=graph_positions, labels=node_labels, node_size=node_size)

    # Create model
    model = Model("cc_tsp_mtz")
    
    # Suppress output if wished 
    if suppress_gurobi_output:
        model.Params.LogToConsole = 0
    
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
    for (i, j) in graph.edges():
        if j != "s":
            model.addConstr(u[i] - u[j] + vertex_count * x[(i, j)] <= vertex_count - 1)

    # Optional: ordering of color edges so reduce possible solution set size
    if add_order_constraints:
        for coordinate, color in game.color_tiles.items():
            # Find all smaller color tiles - for every smaller color tile, add ordering constraint
            for pred in [
                other_coordinate
                for other_coordinate, other_color in game.color_tiles.items()
                if other_color <= color
            ]:
                model.addConstr(u[pred] <= u[coordinate])

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        return False, model

    # Based on variable assingment build edge list
    selected_edges = [(i, j) for (i, j) in x.keys() if x[(i, j)].x > 0.0001]

    # build subgraph induced by selected_edges
    solution_subgraph = graph.edge_subgraph(selected_edges)

    # Test some proprties
    # In-degree(v) = out-degree(v) = 1 for all nodes
    assert len([d for _, d in solution_subgraph.in_degree() if d != 1]) == 0
    assert len([d for _, d in solution_subgraph.out_degree() if d != 1]) == 0

    # solution_subgraph should be fully connected
    assert nx.is_strongly_connected(solution_subgraph)

    # Visualize solution if wished
    if visualize:
        plt.subplot(212)
        nx.draw_networkx_nodes(
            graph, graph_positions, nodelist=graph.nodes(), node_size=node_size
        )
        nx.draw_networkx_labels(graph, graph_positions, labels=node_labels)
        # We want to omit jump edges
        selected_edges_without_jump_edges = [
            (v, w) for (v, w) in selected_edges if w not in game.color_tiles
        ]
        nx.draw_networkx_edges(
            graph,
            graph_positions,
            edgelist=selected_edges_without_jump_edges,
            node_size=node_size,
        )

        plt.show()

    return True, graph, selected_edges, model

levels = [
    """
1X3
122
""",
    """
11X332
11222
""",
    """
11X332
112222
122222
1111 2
1111X1
12X1X1
""",
    """
 1  1   
 11111
1111X1X
 XXX11 
 111111
  1
  """,
]

runtimes = []
for level in levels:
    # Test 20 times
    level_runtimes = []

    for i in range(0, 20):
        # Build game
        game = parse(level)
        return_value = solve_with_mtz_sec(game, visualize=False)
        success = return_value[0]
        if success == True:
            graph, selected_edges, model = return_value[1:]
            level_runtimes.append(model.Runtime)
        else:
            model = return_value[1]
            print("Level unsolvable")
            level_runtimes.append(model.Runtime)
            break
    
    print(f"AVG: {sum(level_runtimes)/len(level_runtimes)}")
    print(f"SUM: {sum(level_runtimes)}")
    print("------")
    runtimes.append(level_runtimes)