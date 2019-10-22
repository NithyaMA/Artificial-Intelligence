#!/usr/bin/env python

from scenario import *
import AStarAlgo
import shortest
import Divide_And_Conquer_Functional
import Divide_And_Conquer2
import genetic
import numpy as np
import random
import pickle
import time

import experiment
from experimentCISAT import *

def pathcost(path, drone, state):
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
        tot += len(p2)
    #scene.simul.trace(newp)
    return(tot, newp)


while True:
    # setup scenario
    randseed = random.randint(1, 10000)
    scene = scenario_cityfire(graphics=False)
    scene.generate_scenario(seed=randseed)
    state = scene.simul.state()
    env = scene.simul.getBlockIds()
    scene.simul.delete_trace()

    # get the base path
    print("Computing base path")
    start = time.time()
    basepath = shortest.shortest(state)
    end = time.time() - start
    print("Elapsed Time: ", end)
    #print("Getting base path cost - drone at:", scene.simul.drone)
    #tot, newp = pathcost(basepath, scene.simul.drone, state)
    #tofile = [end, tot, newp]
    #pickle.dump(tofile, open(str(randseed)+"-basepath.pickle", "wb"))
    print("Base Path:", len(basepath), basepath)
    #scene.simul.trace(newp)
    #print("Total cost of base path:", tot)

    print("Computing DIV Far path")
    start = time.time()
    divpath = Divide_And_Conquer_Functional.shortest(env)
    end = time.time() - start
    print("Elapsed Time: ", end)
    #print("Getting div path cost - drone at:", scene.simul.drone)
    #tot, newp = pathcost(divpath, scene.simul.drone, state)
    #tofile = [end, tot, newp]
    #pickle.dump(tofile, open(str(randseed)+"-basepath.pickle", "wb"))
    print("Div Path:", len(divpath), divpath)
    #scene.simul.trace(newp)
    #print("Total cost of div path:", tot)
    
    break