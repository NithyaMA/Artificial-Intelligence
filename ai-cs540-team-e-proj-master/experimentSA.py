#!/usr/bin/env python
import experiment
import scenario
import numpy as np

class experimentSA(experiment.experiment):

    def __init__(self):
        self.scene = scenario.scenario_cityfire()
        self.scene.generate_scenario(seed=123123)
        self.create_lookup_table()
        self.costTrace = []

    def perform_simulated_annealing(self, startTemp=100, decay=0.99, cutoff=0.1):
        temperature = startTemp
        
        currSolution = self.random_solution()
        self.costTrace.append(self.evaluate_solution(currSolution))
        
        maxHit = 0

        while temperature > cutoff and maxHit < 10:
            rand_prob = (startTemp - temperature) / startTemp
            bfs_prob = 1 - rand_prob

            rand_choice = bool(np.random.choice(2, p=[rand_prob, bfs_prob]))

            if rand_choice:
                neighbors = self.find_neighbor_solutions(currSolution)
                idx = np.random.choice(len(neighbors))
                currSolution = neighbors[idx]
                self.costTrace.append(self.evaluate_solution(currSolution))
            else:
                currSolution, cost = self.find_best_neighbor(currSolution)
                self.costTrace.append(cost)
            
            if self.costTrace[-1] == self.costTrace[-2]:
                maxHit += 1
            else:
                maxHit = 0


            temperature *= decay
        return currSolution

if __name__ == "__main__":
    import time
    test = experimentSA()
    start = time.time()
    bestSol = test.perform_simulated_annealing()
    elapsed = time.time() - start

    print("Finished simulated annealing.  Performed", len(test.costTrace), "iterations in", elapsed, "seconds.")
    print("Best solution found:", bestSol)
    print("Cost trace:", test.costTrace)
