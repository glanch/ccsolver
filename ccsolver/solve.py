import networkx as nx
from gurobipy import Model, GRB, quicksum

from ccsolver.game import Game, parse

from ccsolver.solvers.mip_tsp import MipTspMTZSolver

levels = [
    """
1X3
122
""",
    """
11X332
112222
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
        solver = MipTspMTZSolver()
        success, runtime, edges = solver.solve(game)
    
        level_runtimes.append(runtime)
        if success == False:
            print("Level unsolvable")
            break
    
    print(f"AVG: {sum(level_runtimes)/len(level_runtimes)}")
    print(f"SUM: {sum(level_runtimes)}")
    print("------")
    runtimes.append(level_runtimes)