Coordinate = tuple[int, int]
TileColor = str

class Game:
    color_tiles: dict[Coordinate, int] = {}
    number_tiles: dict[Coordinate, int] = {}
    
    
    def __init__(self):
        pass 

    def get_neighbors(self, coordinate: Coordinate):
        # Color tiles do not have neighbors
        # if coordinate not in self.number_tiles:
        #     return []

        
        i, j = coordinate
        candidates = [(i + 1, j), (i, j + 1), (i - 1, j), (i, j - 1)]
        
        # Filter out to eligible neighbors
        neighbors = []
        for candidate in candidates:
            if candidate in self.number_tiles:
                # All number neighboring color tile and those number tiles higher or equal are considered neighbors 
                if coordinate in self.color_tiles or (coordinate in self.number_tiles and self.number_tiles[candidate] >= self.number_tiles[coordinate]):
                    neighbors.append(candidate)
            else:
                # Do not consider further
                continue

        return neighbors

                
def parse(representation: str):
    game = Game()
    game.color_tiles = {}
    game.number_tiles = {}
    current_color = 0
    j = len([x for x in representation.split("\n") if x != ""])
    for line in representation.split("\n"):
        # Ignore blank lines
        if line == "":
            continue

        i = 0
        for character in line:
            if character == " ":
                # Ignore blank character
                pass
            elif character == "X": 
                game.color_tiles[(i,j)] = "C" + str(current_color)
                current_color += 1
            elif character.isnumeric():
                game.number_tiles[(i,j)] = int(character)

            i += 1

        j -= 1

    return game