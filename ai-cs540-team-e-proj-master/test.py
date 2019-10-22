#!/usr/bin/env python

from scenario import *
import AStarAlgo
import shortest
import genetic
import Divide_And_Conquer_Improved
import numpy as np
import random
import pickle
import time

import experiment
from experimentGSAT import *

def pathcost(path, drone, state, double=False):
    tot = 0
    p2 = AStarAlgo.aStar(drone, path[0], state)
    newp = []
    for x in p2:
        q = np.array([np.int(x[0]), np.int(x[1]), np.int(x[2])])
        newp.append(q)
    tot+=len(p2)
    for i in range(len(path) - 1):
        #print("start:", path[i], "end:", path[i+1])
        p2 = AStarAlgo.aStar(path[i], path[i+1], state)
        for x in p2:
            q = np.array([np.int(x[0]), np.int(x[1]), np.int(x[2])])
            newp.append(q)
        #print(p2, len(p2))
        if i % 2 == 0 and double:
            tot += len(p2)*2
        else:
            tot += len(p2)
    #scene.simul.trace(newp)
    return(tot, newp)

if __name__ == "__main__":
    double = True
    while True:
        # setup scenario
        randseed = random.randint(1, 10000)
        scene = scenario_cityfire(graphics=False)
        scene.generate_scenario(seed=randseed)
        state = scene.simul.state()
        scene.simul.delete_trace()

        # get the base path
        print("Computing base path")
        start = time.time()
        basepath = shortest.shortest(state)
        end = time.time() - start
        print("Getting base path cost - drone at:", scene.simul.drone)
        tot, newp = pathcost(basepath, scene.simul.drone, state, double=double)
        tofile = [end, tot, newp]
        pickle.dump(tofile, open(str(randseed)+"-basepath.pickle", "wb"))
        print("Base Path:", len(basepath), basepath)
        #scene.simul.trace(newp)
        print("Total cost of base path:", tot)
        
        # results of divide and conquer algorithm
        print("----------------------------------------------------------------------------")
        print("Computing the results of divide and conquer approach")
        print("Initial position of drone at:", scene.simul.drone)
        start = time.time()
        divpath = Divide_And_Conquer_Improved.shortest(state)
        end = time.time() - start
        print("Divide and Conquer Solution")
        print("Length of Div Path:", len(divpath))
        print("Div Path: ", divpath)
        print("Final position of Drone at:", scene.simul.drone)
        tot_div, newp_div = pathcost(divpath, scene.simul.drone, state)
        tofile = [end, tot_div, newp_div]
        pickle.dump(tofile, open(str(randseed)+"-divpath.pickle", "wb"))
        print("Total cost of div path:", tot_div)
        print("Elapsed Time: ", end)
        print("----------------------------------------------------------------------------")

        print("Running GSAT - drone at:", scene.simul.drone)
        start = time.time()
        gsat = experimentGSAT(scene=scene)
        end = time.time() - start
        gsat_path, gsat_cost = gsat.perform_GSAT(4,3)
        print(gsat_cost, gsat_path)
        p_gsat =  gsat.convert_solution_to_path(gsat_path)
        print("GSAT path:", len(p_gsat), p_gsat)
        print("Drone at:", scene.simul.drone)
        tot_gsat, newp_gsat = pathcost(p_gsat, scene.simul.drone, state)
        tofile = [end, tot_gsat, newp_gsat]
        pickle.dump(tofile, open(str(randseed)+"-cisatpath.pickle", "wb"))
        print("Total cost of GSAT path:", tot_gsat)
        #print(newp_cisat)
        #scene.simul.trace(newp_cisat)
        #raw_input("hit enter:")

        """
        basepath = shortest.shortest_two(state)
        tot, newp = pathcost(basepath, scene.simul.drone, state)
        #scene.simul.trace(newp)
        print("total cost of base path 2:", tot)
        """

        print("Computing genetic algorithm path")
        start = time.time()
        best_path, max_list, avg_list = genetic.genetic_algorithm(state, genetic.A_star_dist, genetic.fitness_d_water, pop_size=10000, num_generation=300, lamda=10000, mutation_prob=0.5)
        end = time.time() - start
        print("best answer: ", best_path)
        print("max list: ", max_list)
        print("avg list: ", avg_list)
        tot2, newp2 = pathcost(best_path, scene.simul.drone, state)
        tofile = [end, tot2, newp2]
        pickle.dump(tofile, open(str(randseed)+"-gapath.pickle", "wb"))
        print("Total cost of GA path:", tot2)
        print("Elapsed Time: ", end)
        if tot2 < tot:
            print("Woohoo! Genetic path cost shorter!")
        
        AStarAlgo.clearcache()
        #scene.simul.trace(newp)
