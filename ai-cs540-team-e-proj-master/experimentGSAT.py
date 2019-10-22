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
        self.atLocalBest = False

class experimentGSAT(experiment.experiment):

    def __init__(self, scene = None, in_Debug = False):
        self.converged = False
        self.bestSolution = None
        self.bestCost = None
        self.costTrace = []
        self.solutionsExplored = {}
        self.Debug = in_Debug
        self.satisficing = None
        self.minTemp = 1.0
        if scene != None:
            self.scene = scene
            self.create_lookup_table()

    def initialize_scenario(self):
        self.scene = scenario.scenario_cityfire()
        self.scene.generate_scenario()
        self.create_lookup_table()

    def initialize_agents(self,agents):
        # Every agent gets a random solution if not initialized
        for k in agents:
            if k.currentSolution is None:
                k.currentSolution = self.random_solution()
            k.currentCost = self.evaluate_solution(k.currentSolution)
            self.solutionsExplored[k.currentSolution] = k.currentCost
        # Find the fitness of all current solutions
        F = [k.currentCost for k in agents]
        self.update_global_best(agents, F)
        

    def update_global_best(self, agents, F):
        # Test if we've found a new global best
        bestOfF = np.argmin(F)
        if self.bestCost is None or F[bestOfF] < self.bestCost:
            if self.Debug:
                print("Agent",bestOfF,"finds new global best.  Old:", self.bestCost, "New:",F[bestOfF])
            self.bestCost = F[bestOfF]
            self.bestSolution = agents[bestOfF].currentSolution

    def get_normalized_weight_vector(self, agents):
        # Find the fitness of all current solutions
        F = [k.currentCost for k in agents]
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
            if newSolIdx != i:
                k.atLocalBest = False
                if self.Debug:
                    print("Agent", i,"switched to solution of Agent", newSolIdx)
            k.currentSolution = agents[np.random.choice(len(agents), p=W_k)].currentSolution

    def update(self, agents):
        for i in range(len(agents)):
            k = agents[i]
            if self.satisficing is not None and k.currentTemp < self.minTemp:
                continue
            if k.atLocalBest:
                # Don't need to repeatedly test neighbors if we're already at a local best
                if self.Debug: print("Agent",i,"is at a local best.  Temp =", k.currentTemp)
            else:
                neighbors = self.find_neighbor_solutions(k.currentSolution)
                bfs_chance = (k.initialTemp-k.currentTemp)/k.initialTemp
                rand_chance = 1-bfs_chance
                if np.random.choice(2, p=[rand_chance, bfs_chance]):
                    # choose best option
                    evalNeighbors = [self.evaluate_solution(n) for n in neighbors]
                    idx = np.argmin(evalNeighbors)
                    if(evalNeighbors[idx] < k.currentCost):
                        if self.Debug: print("Agent",i,"takes best neighbor. Temp =",k.currentTemp)
                        k.currentSolution = neighbors[idx]
                        k.currentCost = evalNeighbors[idx]
                    else:
                        if self.Debug: print("Agent",i,"is at a local best.  Temp =", k.currentTemp)
                        k.atLocalBest = True
                else:
                    # choose random option
                    if self.Debug: print("Agent",i,"takes random option. Temp =",k.currentTemp)
                    idx = np.random.choice(len(neighbors))
                    k.currentSolution = neighbors[idx]
                    k.currentCost = self.evaluate_solution(k.currentSolution)
                self.solutionsExplored[k.currentSolution] = k.currentCost

            # If satisficing is defined, reduce the temperature by multiplying the
            # adapatation factor if below the desired threshold
            # Otherwise decrease temperature by at most 5% of the initial temp
            if self.satisficing is None or k.currentCost < self.satisficing:
                k.currentTemp *= k.adaptationFactor
            else:
                k.currentTemp = max(k.currentTemp - k.initialTemp * 0.05, \
                        k.currentTemp * k.adaptationFactor)

    def convergenceTest(self, agents):
        # If all agents have the same answer we've converged
        if agents.count(agents[0]) == len(agents):
            self.converged = True
        # If all agents have cooled
        if all(k.currentTemp < self.minTemp for k in agents):
            self.converged = True

    def perform_CISAT(self, agents):
        self.converged = False
        self.initialize_agents(agents)
        while not self.converged:
            self.convergenceTest(agents)
            self.update(agents)
            self.confer(agents)

        return self.bestSolution, self.bestCost

    def perform_GSAT(self, base=10, iterations=3, in_satisficing=None):
        self.bestSolution = None
        self.bestCost = None
        self.costTrace = []
        self.solutionsExplored = {}
        previousSolutions = []
        evalPrevSolutions = []
        self.satisficing = in_satisficing
        for i in range(iterations):
            if self.Debug: print("Beginning iteration:",i)
            numAgents = base ** (iterations-i)
            agents = []
            for j in range(numAgents):
                # Temperature is always 100
                temperature = 100

                # Adaptation factor is greatly reduced for earlier generations
                # because we want the convergence to occur very rapidly when
                # there are a large number of agents.
                adaptationFactor = np.random.uniform(0.80,0.95) / ((iterations-i) ** 2)

                # Quality bias reduction is random number between 0.0 and 1.0
                qualityBiasReduction = np.random.uniform(0.0,1.0)

                # Self bias is random integer between 0 and 10
                selfBias = np.random.randint(11)

                # Conference chance is random number between 0.5 and 1.0
                # divided by ancestry level
                conferenceChance = np.random.uniform(0.5, 1.0) / (iterations-i)

                # Starting solution is populated if this is not the first iteration
                # and only for half of the agents, the rest are random
                # Give the best solutions to the next generation
                startingSolution = None
                if len(previousSolutions) > 0:
                    idx = np.argmin(evalPrevSolutions)
                    startingSolution = previousSolutions[idx]
                    if self.Debug: print("Created agent", j, "with starting solution", \
                            "with cost", evalPrevSolutions[idx])
                    del previousSolutions[idx]
                    del evalPrevSolutions[idx]
                elif self.Debug:
                    print("Created agent", j, "with no starting solution")

                agents.append(Agent(temperature, adaptationFactor, \
                        qualityBiasReduction, selfBias, \
                        conferenceChance, startingSolution))

            self.perform_CISAT(agents)
            previousSolutions = [a.currentSolution for a in agents]
            evalPrevSolutions = [a.currentCost for a in agents]

        return self.bestSolution, self.bestCost


if __name__ == "__main__":
    test = experimentGSAT()
    print("Generating lookup table...")
    test.initialize_scenario()
    print("Done!  Beginning GSAT analysis...")
    import time
    import math
    for base in range(2,11):
        for iterations in range(1,math.ceil(math.log(1000, base))+1):
            print("Base =", base, "Iterations =", iterations)
            start = time.time()
            _, bestCost = test.perform_GSAT(base, iterations)
            elapsed = time.time() - start
            print("Best Cost:", bestCost)
            print("Solutions explored", len(test.solutionsExplored))
            print("Analysis took", elapsed, "seconds or", elapsed/60.0, "minutes\n")

