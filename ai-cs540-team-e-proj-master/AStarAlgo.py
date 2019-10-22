import math
import heapq as hq

cache = {}

class Node:
    def __init__(self, coordinate):
        self.coordinate = coordinate
        self.parent = None
        self.H = 0
        self.G = 0


    def __lt__(self, other):
        return self.score() - other.score() < 0
    def __gt__(self, other):
        return self.score() - other.score() > 0
    def __eq__(self, other):
        return self.score() - other.score() == 0
    def __le__(self, other):
        return self.score() - other.score() <= 0
    def __ge__(self, other):
        return self.score() - other.score() >= 0
    def __ne__(self, other):
        return self.score() - other.score() != 0


    def __cmp__(self,other):
        if self.G + self.H < other.G + other.H:
            return -1
        elif self.G + self.H > other.G + other.H:
            return 1
        else:
            return 0


    def move_cost(self, otherNode):
        # return 1
        x1, y1, z1 = self.coordinate
        x2, y2, z2 = otherNode.coordinate
        cost = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2))
        return cost


    def score(self):
        return self.G + self.H


def neighbourNodes(currentNode, state):
    x, y, z = currentNode.coordinate
    links = []
    for x_id in (x - 1, x, x + 1):
        for y_id in (y - 1, y, y + 1):
            for z_id in (z - 1, z, z + 1):
                if x_id==x and y_id==y and z_id==z:
                    continue
                if ((x_id < -50 or x_id > 50) or (y_id < 0 or y_id > 50) or (z_id < -50 or z_id > 50)):
                    continue
                if (x_id, y_id, z_id) in state:
                    continue
                links.append((x_id, y_id, z_id))
    return links


def heuristic(point1, point2, D = 10):
    x1, y1, z1 = point1.coordinate
    x2, y2, z2 = point2.coordinate
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dz = abs(z1 - z2)
    #Movement cost due to non-diagonal movement
    H_value = (D * (dx + dy + dz))
    #Savings due to 3D Diagonal movement
    H_value += ((D - 3 * D) * min(dx, dy, dz))
    #Savings due to 2D Diagonal movement
    H_value += ((D - 2 * D) * (min(dx, dy) - min(dx, dy, dz)))
    H_value += ((D - 2 * D) * (min(dy, dz) - min(dx, dy, dz)))
    H_value += ((D - 2 * D) * (min(dx, dz) - min(dx, dy, dz)))
    return H_value


def find_path(startNode, target, grid):
    targetNode = Node(target)
    #The open and closed sets
    openSet = {}
    heap = []
    # seen_before = {}
    closedSet = set()
    #Current point is the starting point
    current = Node(startNode)
    #Add the starting point to the open set
    openSet[current.coordinate] = 0
    hq.heappush(heap,(current.score(), current))
    #While the open set is not empty
    while openSet:
        # Find the item in the open set with the lowest G + H score
        # current = min(openSet)
        curr_s, current = hq.heappop(heap)
        if current.coordinate in openSet and openSet[current.coordinate] < current.G:
            continue
        # print(current.coordinate)
        #If it is the item we want, retrace the path and return it
        if current.coordinate == targetNode.coordinate:
            path = []
            while current.parent:
                path.append(current)
                current = current.parent
            path.append(current)
            # print(path)
            return path[::-1]
        #Remove the item from the open set
        if current.coordinate in openSet:
            del openSet[current.coordinate]
        #Add it to the closed set
        closedSet.add(current.coordinate)
        #Loop through the node's children/siblings
        for node_tup in neighbourNodes(current, grid):
            #If it is already in the closed set, skip it
            if node_tup in closedSet:
                continue
            node = Node(node_tup)
            #Otherwise if it is already in the open set
            if node.coordinate in openSet:
                #Check if we beat the G score 
                new_g = current.G + current.move_cost(node)
                if openSet[node.coordinate] > new_g:
                    #If so, update the node to have a new parent
                    node.G = new_g
                    # node.H = heuristic(node, targetNode)
                    node.parent = current
                    hq.heappush(heap,(node.score(),node))
                    openSet[node.coordinate] = node.G
            else:
                #If it isn't in the open set, calculate the G and H score for the node
                node.G = current.G + current.move_cost(node)
                node.H = heuristic(node, targetNode)
                #Set the parent to our current item
                node.parent = current
                #Add it to the set
                openSet[node.coordinate] = node.G
                hq.heappush(heap,(node.score(), node))
    #Throw an exception if there is no path
    raise ValueError('No Path Found')


#grid is 3 dimensional list of dimension 101*50*101
def aStar(initNode,targetNode,grid):
    global cache
    #Convert all the points to instances of Node
    # for x in range(len(grid)):
    #     for y in range(len(grid[x])):
    #         for z in range(len(grid[x][y])):
    #            grid[x][y][z] = Node((x,y,z))
    #Get the path
    if initNode+targetNode in cache:
        return cache[initNode+targetNode]
    path = find_path(initNode, targetNode, grid)
    #Output the path
    # print(len(path) - 1)
    path_nodes = []
    for node in path:
        x, y, z = node.coordinate
        # print(x, y, z)
        path_nodes.append((x,y,z))
    cache[initNode+targetNode] = path_nodes
    return path_nodes

def clearcache():
    global cache
    cache = {}
        

if __name__ == '__main__':
    state = {(-50, 0, -50): 'green',
 (-50, 0, 50): 'indigo',
 (0, 0, 0): 'drone',
 (1, 0, 3): 'blue',
 (5, 0, 10): 'red',
 (18, 20, 20): 'yellow',
 (18, 21, 20): 'yellow',
 (18, 22, 20): 'yellow',
 (18, 23, 20): 'yellow',
 (18, 24, 20): 'yellow',
 (19, 20, 20): 'yellow',
 (19, 21, 20): 'yellow',
 (19, 22, 20): 'yellow',
 (19, 23, 20): 'yellow',
 (19, 24, 20): 'yellow',
 (20, 0, 20): 'yellow',
 (20, 20, 18): 'yellow',
 (20, 20, 19): 'yellow',
 (20, 20, 20): 'yellow',
 (20, 21, 18): 'yellow',
 (20, 21, 19): 'yellow',
 (20, 21, 20): 'yellow',
 (20, 22, 18): 'yellow',
 (20, 22, 19): 'yellow',
 (20, 22, 20): 'yellow',
 (20, 23, 18): 'yellow',
 (20, 23, 19): 'yellow',
 (20, 23, 20): 'yellow',
 (20, 24, 18): 'yellow',
 (20, 24, 19): 'yellow',
 (20, 24, 20): 'yellow',
 (50, 0, -50): 'violet',
 (50, 0, 50): 'orange'}

    start_node = (10,22,10)
    dest_node = (22,22,22)
    path = aStar(start_node, dest_node, state)
    print(len(path))
    for elem in path:
        print(str(elem[0]) + ',' + str(elem[1]) + ',' + str(elem[2])+  " violet")
    # print(path)

