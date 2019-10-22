#!/usr/bin/env python
import experiment
import scenario
import numpy as np

class experimentMSA(experiment.experiment):

    def __init__(self):
        self.scene = scenario.scenario_cityfire()
        self.scene.generate_scenario()
        self.create_lookup_table()
        self.costTrace = []
        self.bestCostTrace = []

    def perform_multi_simulated_annealing(self, numAgents=5, startTemp=100, decay=0.99, cutoff=0.1):
        Debug = False
        temperature = []
        searchComplete = []
        solutions = []
        maxHits = []
        for i in range(numAgents):
            searchComplete.append(False)
            temperature.append(startTemp)
            solutions.append(self.random_solution())
            self.costTrace.append([])
            self.costTrace[i].append(self.evaluate_solution(solutions[i]))
            maxHits.append(0)

        globalBestSol = solutions[0]
        globalBestCost = self.evaluate_solution(solutions[0])
        self.bestCostTrace.append(globalBestCost)

        for i in range(1, numAgents):
            cost = self.evaluate_solution(solutions[i])
            if cost < globalBestCost:
                globalBestSol = solutions[i]
                globalBestCost = cost
                
        while False in searchComplete:
            for i in range(numAgents):
                if searchComplete[i]:
                    continue

                rand_prob = (startTemp - temperature[i]) / startTemp
                bfs_prob = 1 - rand_prob

                rand_choice = bool(np.random.choice(2, p=[rand_prob, bfs_prob]))

                if rand_choice:
                    if Debug: print("Agent", i, ": Random!")
                    neighbors = self.find_neighbor_solutions(solutions[i])
                    idx = np.random.choice(len(neighbors))
                    solutions[i] = neighbors[idx]
                    self.costTrace[i].append(self.evaluate_solution(solutions[i]))
                else:
                    if Debug: print("Agent", i, ": Best!")
                    solutions[i], cost = self.find_best_neighbor(solutions[i])
                    self.costTrace[i].append(cost)

                if self.costTrace[i][-1] == self.costTrace[i][-2]:
                    maxHits[i] += 1
                else:
                    maxHits[i] = 0

                temperature[i] *= decay

                if self.costTrace[i][-1] < globalBestCost:
                    if Debug:
                        print("Agent",i,"has found a new global maximum!")
                        print("Old:", globalBestCost, "   New:", self.costTrace[i][-1])
                    globalBestCost = self.costTrace[i][-1]
                    globalBestSol = solutions[i]
                    self.bestCostTrace.append(globalBestCost)

                if maxHits[i] > 10 or temperature[i] < cutoff:
                    if Debug: print("Agent",i,"has finished searching.")
                    searchComplete[i] = True

        return globalBestSol

if __name__ == "__main__":
    import time
    test = experimentMSA()
    start = time.time()
    bestSol = test.perform_multi_simulated_annealing()
    elapsed = time.time() - start

    iterations = 0
    for i in range(len(test.costTrace)):
        iterations += len(test.costTrace[i])

    print("Finished simulated annealing.  Performed", iterations, "iterations in", elapsed, "seconds.")
    print("Best solution found:", bestSol)
    print("Cost trace:", test.bestCostTrace)
