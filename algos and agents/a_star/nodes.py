import kuimaze

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None, cost=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0
        self.cost = cost

    def __eq__(self, other):
        return self.position == other.position