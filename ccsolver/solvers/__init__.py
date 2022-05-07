from ccsolver.game import Game, Coordinate

SolveResult = tuple[bool, float, list[tuple[Coordinate, Coordinate]]]

class Solver:
   def solve(self, game: Game) -> SolveResult:
        pass