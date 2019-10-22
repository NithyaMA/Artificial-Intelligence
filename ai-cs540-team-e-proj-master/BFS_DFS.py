#!/usr/bin/python

import numpy as np
import copy
import time

from sim import *
from scenario import *
from AStarAlgo import aStar, clearcache

def dfs(vertices, edge, cur_path, shortest_path, cur_dist, shortest_dist, num_edges, level):
    if edge['node_id'] in cur_path:
        return (shortest_path, shortest_dist, num_edges)
    
    cur_path.append(edge['node_id'])
    cur_dist += edge['dist']
    num_edges += 1
    vertex = vertices[edge['node_id']]
    for edge in vertex:
        (shortest_path, shortest_dist, num_edges) = dfs(vertices, edge, cur_path, shortest_path, cur_dist, shortest_dist, num_edges, level + 1)
    
    if level == (len(vertices) - 1):
        if cur_dist < shortest_dist:
            #print(cur_path, cur_dist)
            shortest_path = copy.deepcopy(cur_path)
            shortest_dist = cur_dist
            print(shortest_path, shortest_dist)
    cur_path.pop()
    return (shortest_path, shortest_dist, num_edges)
        
        
def shortest(env):
    num_edges = 0
    fires = []
    lakes = []
    fires_block_id = []
    lakes_block_id = []
    for key, value in env.items():
        #print(key, value)
        x, y, z = value['pos']
        #x = np.float(x)
        #y = np.float(y)
        #z = np.float(z)
        if value['color'] == "orange":
            fires.append(np.array([x, y, z]))
            fires_block_id.append(key)
        if value['color'] == "blue":
            lakes.append(np.array([x, y, z]))
            lakes_block_id.append(key)
        if value['color'] == "drone":
            fires.insert(0, np.array([x, y, z]))
            fires_block_id.insert(0, key)
    
    #print("Hello")
    vertices = []   
    for i, fire1 in enumerate(fires):
        edges = []
        for j, fire2 in enumerate(fires):
            if i == j:
                continue
            edge = {}
            edge['start'] = fire1
            edge['dist'] = np.inf
            for k, lake in enumerate(lakes):
                dist1 = len(aStar(tuple(fire1), tuple(lake), env))
                dist2 = len(aStar(tuple(lake), tuple(fire2), env))
                num_edges += 1
                if (dist1 + dist2) < edge['dist']:
                    edge['dist'] = dist1 + dist2
                    #if (np.linalg.norm(fire1 - lake) + np.linalg.norm(lake - fire2)) < edge['dist']:
                    #    edge['dist'] = np.linalg.norm(fire1 - lake) + np.linalg.norm(lake - fire2)
                    edge['lake'] = lake
                    edge['lake_block_id'] = lakes_block_id[k]
                    edge['fire'] = fire2
                    edge['fire_block_id'] = fires_block_id[j]
                    edge['node_id'] = j
            edges.append(edge)
        vertices.append(edges)
    
    #print(vertices)
    #input('type enter')
    vertex = vertices[0]
    shortest_path = []
    cur_path = []
    cur_path.append(0)
    shortest_dist = np.inf
    cur_dist = 0
    for edge in vertex:
        (shortest_path, shortest_dist, num_edges) = dfs(vertices, edge, cur_path, shortest_path, cur_dist, shortest_dist, num_edges, 1)
    #print(shortest_dist, shortest_path, num_edges)
    final_path = []
    final_path_xyz = []
    #final_path.append(fires[0])
    i = 1
    prev_node_id = 0
    while i < len(shortest_path):
        node_id = shortest_path[i]
        #print(prev_node_id, node_id)
        if prev_node_id < node_id:
            final_path.append(vertices[prev_node_id][node_id - 1]['lake_block_id'])
            final_path.append(vertices[prev_node_id][node_id - 1]['fire_block_id'])
            final_path_xyz.append(vertices[prev_node_id][node_id - 1]['lake'])
            final_path_xyz.append(vertices[prev_node_id][node_id - 1]['fire'])
        else:
            final_path.append(vertices[prev_node_id][node_id]['lake_block_id'])
            final_path.append(vertices[prev_node_id][node_id]['fire_block_id'])
            final_path_xyz.append(vertices[prev_node_id][node_id]['lake'])
            final_path_xyz.append(vertices[prev_node_id][node_id]['fire'])
        #final_path.append(fires[node_id])
        prev_node_id = node_id
        i += 1
    #print(final_path)
    print("Distance:")
    print(shortest_dist)
    print("Number of edges analyzed:")
    print(num_edges)
    print("Path in Block ID:")
    print(final_path)
    print("Path in XYZ Coordinates:")
    print(final_path_xyz)
    return final_path

if __name__ == "__main__":
    scene = scenario_cityfire()
    scene.generate_scenario()
    clearcache()
    env = scene.simul.getBlockIds()
    #path = aStar((1, 1, 1), (10, 11, 11), env)
    #print(path)
    #print("Hello")
    start = time.time()
    p = shortest(env)
    elapsed = time.time() - start
    print("Elapsed Time")
    print(elapsed)