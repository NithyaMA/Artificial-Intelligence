#!/usr/bin/env python
import experiment
import scenario
import numpy as np

class Agent:
    def __init__(self, in_temp, in_af, in_qbr, in_sb, in_conferChance,  in_solution = None):
        self.initialTemp = in_temp
        self.currentTemp = in_temp
        self.adaptationFactor = in_af
        self.currentSolution = in_solution
        self.currentCost = None
        self.qualityBiasReduction = in_qbr
        self.selfBias = in_sb
        self.conferChance = in_conferChance

class experimentCISAT(experiment.experiment):

    def __init__(self, scene = None):
        self.converged = False
        self.bestSolution = None
        self.bestCost = None
        self.costTrace = []
        self.Debug = True
        if scene != None:
            self.scene = scene
            self.create_lookup_table()

    def initialize_scenario(self):
        self.scene = scenario.scenario_cityfire()
        self.scene.generate_scenario()
        self.create_lookup_table()

    def initialize_agents(self,agents):
        # Every agent gets a random solution
        for k in agents:
            if k.currentSolution is None:
                k.currentSolution = self.random_solution()
            k.currentCost = self.evaluate_solution(k.currentSolution)
        # Find the fitness of all current solutions
        F = [self.evaluate_solution(k.currentSolution) for k in agents]
        self.bestSolution = agents[0].currentSolution
        self.bestCost = F[0]
        self.update_global_best(agents, F)
        

    def update_global_best(self, agents, F):
        # Test if we've found a new global best
        bestOfF = np.argmin(F)
        if F[bestOfF] < self.bestCost:
            if self.Debug:
                print("Agent",bestOfF,"finds new global best.  Old:", self.bestCost, "New:",F[bestOfF])
            self.bestCost = F[bestOfF]
            self.bestSolution = agents[bestOfF].currentSolution

    def get_normalized_weight_vector(self, agents):
        # Find the fitness of all current solutions
        F = [self.evaluate_solution(k.currentSolution) for k in agents]
        self.update_global_best(agents, F)

        W = [-f + max(F) for f in F]
        if sum(W) != 0:
            W = [ w / sum(W) for w in W]
        return W

    def confer(self, agents):
        W = self.get_normalized_weight_vector(agents)

        conferred = []
        for i in range(len(agents)):
            k = agents[i]
            # Determine whether agent confers
            if np.random.choice(2, p=[k.conferChance, 1-k.conferChance]):
                conferred.append(False)
            else:
                conferred.append(True)

        for i in range(len(agents)):
            if not conferred[i]:
                if self.Debug: print("Agent", i, "did not confer.")
                continue

            k = agents[i]

            # Add the quality bias reduction
            W_k = [w + k.qualityBiasReduction for w in W]

            # Insert the self-bias
            W_k[i] += k.selfBias

            # Remove the agents who didn't confer
            for j in range(len(conferred)):
                if not conferred[j]:
                    W_k[j] = 0

            # Re-normalize the vector
            W_k = [w_k / sum(W_k) for w_k in W_k]
          
            # Select a random solution from the distribution
            newSolIdx = np.random.choice(len(agents), p=W_k)
            if self.Debug:
                if newSolIdx == i:
                    print("Agent", i,"stayed with its own solution")
                else:
                    print("Agent", i,"switched to solution of Agent", newSolIdx)
            k.currentSolution = agents[np.random.choice(len(agents), p=W_k)].currentSolution

    def update(self, agents):
        for i in range(len(agents)):
            k = agents[i]
            neighbors = self.find_neighbor_solutions(k.currentSolution)
            evalNeighbors = [self.evaluate_solution(n) for n in neighbors]
            bfs_chance = (k.initialTemp-k.currentTemp)/k.initialTemp
            rand_chance = 1-bfs_chance
            if np.random.choice(2, p=[rand_chance, bfs_chance]):
                # choose best option
                if self.Debug: print("Agent",i,"takes best neighbor. Temp =",k.currentTemp)
                idx = np.argmin(evalNeighbors)
                k.currentSolution = neighbors[idx]
            else:
                # choose random option
                if self.Debug: print("Agent",i,"takes random option. Temp =",k.currentTemp)
                idx = np.random.choice(len(neighbors))
                k.currentSolution = neighbors[idx]
            evalVariance = np.var(evalNeighbors)
            #newTemp = k.currentTemp*(1 - (k.currentTemp * k.adaptationFactor) / evalVariance)
            #k.currentTemp = newTemp
            k.currentTemp *= k.adaptationFactor

    def convergenceTest(self, agents):
        # If all agents have the same answer we've converged
        if agents.count(agents[0]) == len(agents):
            self.converged = True
        # If all agents have cooled
        if all(k.currentTemp < 1.0 for k in agents):
            self.converged = True

    def perform_CISAT(self, agents):
        self.initialize_agents(agents)

        while not self.converged:
            self.convergenceTest(agents)
            self.confer(agents)
            self.update(agents)

        return self.bestSolution, self.bestCost

if __name__ == "__main__":
    agents = []
    agents.append(Agent(100, 0.95, 0.5, 5, 0.8))
    agents.append(Agent(100, 0.92, 0.2, 10, 0.9))
    agents.append(Agent(100, 0.90, 0.8, 2, 1.0))
    agents.append(Agent(100, 0.88, 0.5, 5, 0.8))
    agents.append(Agent(100, 0.93, 0.5, 5, 0.2))
    
    test = experimentCISAT()
    test.initialize_scenario()
    import time
    start = time.time()
    bestPath, bestCost = test.perform_CISAT(agents)
    print("Best Path:", bestPath)
    print("Best Cost:", bestCost)
    elapsed = time.time() - start
    print("Analysis took", elapsed, "seconds")
