#!/usr/bin/env python
import experiment
import scenario
import numpy as np

class experimentHSA(experiment.experiment):

    def __init__(self):
        self.scene = scenario.scenario_cityfire()
        self.scene.generate_scenario()
        self.create_lookup_table()
        self.costTrace = []
        self.bestCostTrace = []

    def perform_hetero_simulated_annealing(self, numAgents=None, startTemps=None, decays=None, cutoffs=None):
        if numAgents is None or startTemps is None or decays is None or cutoffs is None:
            print("One or more arguments were undefined!")
            return None
        if len(startTemps) != numAgents or len(decays) != numAgents or len(cutoffs) != numAgents:
            print("One or more arguments have the wrong number of elements!")
            return None
        Debug = True
        temperature = []
        searchComplete = []
        solutions = []
        maxHits = []
        atPeak = []
        for i in range(numAgents):
            searchComplete.append(False)
            temperature.append(startTemps[i])
            solutions.append(self.random_solution())
            self.costTrace.append([])
            self.costTrace[i].append(self.evaluate_solution(solutions[i]))
            maxHits.append(0)
            atPeak.append(False)

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

                rand_prob = (startTemps[i] - temperature[i]) / startTemps[i]
                bfs_prob = 1 - rand_prob

                rand_choice = bool(np.random.choice(2, p=[rand_prob, bfs_prob]))

                if rand_choice:
                    if Debug: print("Agent", i, ": Random!")
                    neighbors = self.find_neighbor_solutions(solutions[i])
                    idx = np.random.choice(len(neighbors))
                    solutions[i] = neighbors[idx]
                    self.costTrace[i].append(self.evaluate_solution(solutions[i]))
                    atPeak[i] = False
                else:
                    if atPeak[i]:
                        if Debug: print("Agent", i, "got best while already at peak")
                        self.costTrace[i].append(self.costTrace[i][-1])
                    else:
                        if Debug: print("Agent", i, ": Best!")
                        newSol, cost = self.find_best_neighbor(solutions[i])
                        if solutions[i] == newSol:
                            atPeak[i] = True
                        else:
                            atPeak[i] = False
                            solutions[i] = newSol
                        self.costTrace[i].append(cost)

                if self.costTrace[i][-1] == self.costTrace[i][-2]:
                    maxHits[i] += 1
                else:
                    maxHits[i] = 0

                temperature[i] *= decays[i]

                if self.costTrace[i][-1] < globalBestCost:
                    if Debug:
                        print("Agent",i,"has found a new global maximum!")
                        print("Old:", globalBestCost, "   New:", self.costTrace[i][-1])
                    globalBestCost = self.costTrace[i][-1]
                    globalBestSol = solutions[i]
                    self.bestCostTrace.append(globalBestCost)

                if maxHits[i] >= 10 or temperature[i] < cutoffs[i]:
                    if Debug: print("Agent",i,"has finished searching.  maxHits:", maxHits[i], "Temperature:", temperature[i])
                    searchComplete[i] = True

        return globalBestSol

if __name__ == "__main__":
    import time
    test = experimentHSA()
    start = time.time()
    numAgents = 5
    temperatures = [100, 80, 50, 20, 1000]
    decays = [0.95, 0.92, 0.97, 0.95, 0.90]
    cutoffs = [0.1, 0.1, 0.1, 0.5, 0.1]
    bestSol = test.perform_hetero_simulated_annealing(numAgents, temperatures, decays, cutoffs)
    elapsed = time.time() - start

    iterations = 0
    for i in range(len(test.costTrace)):
        iterations += len(test.costTrace[i])

    print("Finished simulated annealing.  Performed", iterations, "iterations in", elapsed, "seconds.")
    print("Best solution found:", bestSol)
    print("Cost trace:", test.bestCostTrace)
