#!/usr/bin/env python
import scenario
import AStarAlgo
from math import *
import numpy as np

class experiment:

    lookup_table = {}
    lakes = {}
    fires = {}
    drone = {}

    def __init__(self):
        self.scene = scenario.scenario_cityfire()
        self.scene.generate_scenario()

    def create_lookup_table(self):
        allBlockIds = self.scene.simul.getBlockIds()
        experiment.lakes = {k:v['pos'] for k, v in allBlockIds.items() if v['color'] is 'blue'}
        experiment.fires = {k:v['pos'] for k, v in allBlockIds.items() if v['color'] is 'orange'}
        experiment.drone = {k:v['pos'] for k, v in allBlockIds.items() if v['color'] is 'drone'}

        for lakeId, lakePos in experiment.lakes.items():
            # find the distance from the lake to the drone
            for droneId, dronePos in experiment.drone.items():
                if droneId not in experiment.lookup_table:
                    experiment.lookup_table[droneId] = {}
                if lakeId not in experiment.lookup_table:
                    experiment.lookup_table[lakeId] = {}

                lakeX, lakeY, lakeZ = lakePos
                droneX, droneY, droneZ = dronePos

                experiment.lookup_table[droneId][lakeId] = experiment.lookup_table[lakeId][droneId] = \
                    len(AStarAlgo.aStar((droneX, droneY, droneZ), (lakeX, lakeY+1, lakeZ), self.scene.simul.state()))
 
            # find the distance from each lake to each fire
            for fireId, firePos in experiment.fires.items():
                if fireId not in experiment.lookup_table:
                    experiment.lookup_table[fireId] = {}

                lakeX, lakeY, lakeZ = lakePos
                fireX, fireY, fireZ = firePos

                experiment.lookup_table[fireId][lakeId] = \
                    len(AStarAlgo.aStar((lakeX, lakeY+1, lakeZ), (fireX, fireY+2, fireZ), self.scene.simul.state()))

                experiment.lookup_table[lakeId][fireId] = \
                    2*len(AStarAlgo.aStar((lakeX, lakeY+1, lakeZ), (fireX, fireY+2, fireZ), self.scene.simul.state()))
    
    def validate_solution(self, solution):
        # check that the solution is the right size
        if len(solution) != 2 * len(experiment.fires.keys()):
            print("Solution wrong size!  Expected:", len(experiment.fires.keys())*2, "Actual:", len(solution))
            return False

        # check that the solution has the right pattern
        # and that all the block IDs are valid
        for i in range(0,len(solution)):
            if i % 2 == 0 and solution[i] not in experiment.lakes:
                print("Solution has bad lake key:", solution[i])
                return False
            if i % 2 == 1 and solution[i] not in experiment.fires:
                print("Solution has bad fire key:", solution[i])
                return False

        # check that all fires are accounted for and not duplicated
        if sorted(list(solution[1::2])) != sorted(list(experiment.fires.keys())):
            print("Solution had duplicated fire keys!\n", \
                    "Expected:", sorted(list(experiment.fires.keys())), "\n", \
                    "Actual:", sorted(list(solution[1::2])))
            return False

        return True

    def evaluate_solution(self, solution):
        # check that the solution is valid
        if self.validate_solution(solution):
            totalCost = 0
            
            # determine the cost of the first move
            #totalCost += experiment.lookup_table[self.scene.simul.droneId][solution[0]] + 1
            totalCost += experiment.lookup_table[self.scene.simul.droneId][solution[0]]

            # determine the cost of all the other moves
            for i in range(0,len(solution)-1):
                #totalCost += experiment.lookup_table[solution[i]][solution[i+1]] + 1
                totalCost += experiment.lookup_table[solution[i]][solution[i+1]]

            return totalCost
        else:
            return None

    def find_neighbor_solutions(self, solution):
        neighbors = []

        lsolution = list(solution)
        
        # generate the lake permutations
        for i in range(0, len(lsolution), 2):
            for newLake in experiment.lakes.keys():
                if lsolution[i] == newLake:
                    continue
                newSolution = []
                newSolution.extend(lsolution)
                newSolution[i] = newLake
                tnewSolution = tuple(newSolution)
                if self.validate_solution(tnewSolution):
                    neighbors.append(tnewSolution)
        
        # generate the fire permutations
        for i in range(1, len(lsolution) - 2, 2):
            for j in range(i + 2, len(lsolution), 2):
                newSolution = []
                newSolution.extend(lsolution)
                newSolution[i], newSolution[j] = newSolution[j], newSolution[i]
                tnewSolution = tuple(newSolution)
                if self.validate_solution(tnewSolution):
                    neighbors.append(tnewSolution)
        return neighbors

    def find_best_neighbor(self, solution):
        if self.validate_solution(solution):
            neighbors = self.find_neighbor_solutions(solution)
            bestSolution = solution
            bestCost = self.evaluate_solution(solution)
            for n in neighbors:
                nCost = self.evaluate_solution(n)
                if nCost < bestCost:
                    bestCost = nCost
                    bestSolution = n
            return bestSolution, bestCost

    def random_solution(self):
        fireIds = list(experiment.fires.keys())
        np.random.shuffle(fireIds)
        lakeIds = experiment.lakes.keys()
        lakeIds = np.random.choice(list(lakeIds), len(fireIds))
        return tuple([val for pair in zip(lakeIds, fireIds) for val in pair])

    def convert_solution_to_path(self, solution):
        path = []
        for step in solution:
            if step in self.lakes:
                path.append((self.lakes[step][0],self.lakes[step][1]+1,self.lakes[step][2]))
            elif step in self.fires:
                path.append((self.fires[step][0],self.fires[step][1]+2,self.fires[step][2]))
        return path

if __name__ == "__main__":
    import time
    exp = experiment()
    start = time.time()
    exp.create_lookup_table()
    elapsed = time.time() - start
    tableSize = 0
    for k in experiment.lookup_table.keys():
        tableSize += len(experiment.lookup_table[k])
    print("Created", tableSize, "lookup table entries in", elapsed, "seconds.")

    randsol = exp.random_solution()
    print("Random Solution:", randsol)
    print("Valid Solution?:", exp.validate_solution(randsol))
    print("Total path cost:", exp.evaluate_solution(randsol))

    start = time.time()
    neighbors = exp.find_neighbor_solutions(randsol)
    elapsed = time.time() - start

    print("\nFound", len(neighbors), "neighbors in", elapsed, "seconds.")

    start = time.time()
    bestNeighbor, bestCost = exp.find_best_neighbor(randsol)
    elapsed = time.time() - start

    print("\nFound best neighbor of prev solution:", bestNeighbor)
    print("Elapsed Time:", elapsed)
    print("Cost of best neighbor:", bestCost, "\n")

    print("Path of best neighbor:", exp.convert_solution_to_path(bestNeighbor))
