#!/usr/bin/python

import numpy as np
import copy
import time
import itertools

from sim import *
from scenario import *
from AStarAlgo import aStar, clearcache

def parse_env(env):
    fires = []
    lakes = []
    #print(env)
    for value in env:
        x, y, z = value
        x = np.float(x)
        y = np.float(y)
        z = np.float(z)
        #print(x, y, z)
        if env[value] == "orange":
            fires.append(np.array([x, y + 2, z]))
        if env[value] == "blue":
            lakes.append(np.array([x, y + 1, z]))
        if env[value] == "drone":
            fires.insert(0, np.array([x, y, z]))

    graph = []
    for i, fire1 in enumerate(fires):
        vertex = []
        for j, fire2 in enumerate(fires):
            edge = {}
            edge['start'] = fire1
            edge['dist'] = np.inf
            edge['end'] = fire1
            edge['path'] = []
            edge['path'].append(fire2)
            if i == j:
                edge['lake'] = None
                vertex.append(edge)
                continue

            for k, lake in enumerate(lakes):
                #print(i, j, k)
                #dist1 = len(aStar(tuple(fire1), tuple(lake), env))
                #dist2 = len(aStar(tuple(lake), tuple(fire2), env))
                dist1 = np.linalg.norm(fire1 - lake)
                dist2 = 2 * np.linalg.norm(lake - fire2)
                #num_edges += 1
                if (dist1 + dist2) < edge['dist']:
                    edge['dist'] = dist1 + dist2
                    edge['lake'] = lake
                    edge['fire'] = fire2
            #print(vertex)
            vertex.append(edge)
        graph.append(vertex)
    #print(graph, fires, lakes)
    return (graph, fires, lakes)

def pick_neighbors(graph, start_vertex_id, max_dist, max_num_vertices):
    vertex = graph[start_vertex_id]
    short_list = []
    num_vertices = 1
    short_list.append(start_vertex_id)
    for i, edge in enumerate(vertex):
        if edge['dist'] > max_dist:
            continue
        short_list.append(i)
        num_vertices += 1
        if num_vertices >= max_num_vertices:
            break
    return short_list

def pick_farthest_neighbor(graph, cur_vertex_id):
    farthest_vertex_id = 0
    farthest_dist = 0
    vertex = graph[cur_vertex_id]
    for i, edge in enumerate(vertex):
        if edge['dist'] == np.inf:
            continue
        if edge['dist'] > farthest_dist:
            farthest_dist = edge['dist']
            farthest_vertex_id = i
    #print(farthest_vertex_id)
    return farthest_vertex_id

def mark_vertices_covered(graph, short_list):
    for vertex_id in short_list:
        #vertex = graph[vertex_id]
        #for edge in vertex:
        #    edge['dist'] = np.inf
        for vertex in graph:
            vertex[vertex_id]['dist'] = np.inf
    #print(graph)
    if len(short_list) == 1:
        return graph
    for vertex_id in short_list[1:]:
        graph[vertex_id][vertex_id]['path'] = None
    return graph

def find_shortest_path(graph, short_list):
    shortest_path = []
    shortest_dist = np.inf
    if (short_list[0] == 0):
        start_vertex_id = short_list[0]
        for path in itertools.permutations(short_list[1:]):
            prev_vertex_id = start_vertex_id
            cur_dist = 0
            for vertex_id in path:
                cur_dist += graph[prev_vertex_id][vertex_id]['dist']
                prev_vertex_id = vertex_id
            if cur_dist < shortest_dist:
                shortest_path = copy.deepcopy(list(path))
                shortest_dist = cur_dist
        shortest_path.insert(0, start_vertex_id)
    else:
        for path in itertools.permutations(short_list):
            prev_vertex_id = path[0]
            cur_dist = 0
            for vertex_id in path[1:]:
                cur_dist += graph[prev_vertex_id][vertex_id]['dist']
                prev_vertex_id = vertex_id
            if cur_dist < shortest_dist:
                shortest_path = copy.deepcopy(list(path))
                shortest_dist = cur_dist
    return shortest_path

def populate_path_with_lakes(graph, path):
    path_with_lakes = []
    #final_path.append(fires[0])
    prev_vertex_id = path[0]
    path_with_lakes.extend(graph[prev_vertex_id][prev_vertex_id]['path'])
    for vertex_id in path[1:]:
        path_with_lakes.append(graph[prev_vertex_id][vertex_id]['lake'])
        path_with_lakes.extend(graph[prev_vertex_id][vertex_id]['path'])
        #print(path_with_lakes)
        prev_vertex_id = vertex_id
    #print(final_path)
    return path_with_lakes

def reconstruct_graph(graph, lakes):
    new_graph = []
    for i, v1 in enumerate(graph):
        if v1[i]['path'] == None:
            continue
        vertex = []
        for j, v2 in enumerate(graph):
            if v2[j]['path'] == None:
                continue
            edge = {}
            edge['start'] = v1[i]['start']
            edge['dist'] = np.inf
            edge['end'] = v1[i]['path'][-1]
            edge['path'] = copy.deepcopy(v2[j]['path'])
            if i == j:
                edge['lake'] = None
                vertex.append(edge)
                continue

            for k, lake in enumerate(lakes):
                #print(i, j, k)
                #dist1 = len(aStar(tuple(fire1), tuple(lake), env))
                #dist2 = len(aStar(tuple(lake), tuple(fire2), env))
                dist1 = np.linalg.norm(v1[i]['path'][-1] - lake)
                dist2 = 2 * np.linalg.norm(lake - v2[j]['path'][0])
                #num_edges += 1
                if (dist1 + dist2) < edge['dist']:
                    edge['dist'] = dist1 + dist2
                    edge['lake'] = lake
                    edge['fire'] = v2[j]['path'][0]
            #print(vertex)
            vertex.append(edge)
        new_graph.append(vertex)
    #print(graph, fires, lakes)
    return new_graph

def form_clusters(graph, max_dist, max_num_vertices, lakes):
    num_nodes_covered = 0
    num_clusters = 0
    start_vertex_id = 0
    while num_nodes_covered < len(graph):
        short_list = pick_neighbors(graph, start_vertex_id, max_dist, max_num_vertices)
        if (len(short_list) > 1):
            shortest_path = find_shortest_path(graph, short_list)
            shortest_path_with_lakes = populate_path_with_lakes(graph, shortest_path)
            #if num_clusters % 2 == 0:
            graph[shortest_path[0]][shortest_path[0]]['path'] = copy.deepcopy(shortest_path_with_lakes)
            #print("Clustered Nodes", num_clusters)
            #print(shortest_path)
            #print(graph[shortest_path[0]][shortest_path[0]]['path'])
            graph = mark_vertices_covered(graph, shortest_path)
            start_vertex_id = pick_farthest_neighbor(graph, shortest_path[-1])
            #else:
            #    graph[shortest_path[-1]][shortest_path[-1]]['path'] = copy.deepcopy(list(reversed(shortest_path_with_lakes)))
            #    graph = mark_vertices_covered(graph, short_list)
            #    start_vertex_id = pick_farthest_neighbor(graph, shortest_path[0])
        else:
            graph = mark_vertices_covered(graph, short_list)
            start_vertex_id = pick_farthest_neighbor(graph, short_list[0])
            #print("Clustered Nodes", num_clusters)
            #print(short_list)
            #print(graph[short_list[0]][short_list[0]]['path'])
        num_nodes_covered += len(short_list)
        num_clusters += 1
        #print(num_nodes_covered, num_clusters)
        #print(start_vertex_id)
    #print("Number of Clusters", num_clusters)
    graph = reconstruct_graph(graph, lakes)
    return (graph, num_clusters)

def shortest(env):
    orig_graph, orig_fires, orig_lakes = parse_env(env)
    #print("Orig_Graph", orig_graph)
    #print("Orig_Lakes", orig_lakes)
    #print("Orig_Fires", orig_fires)
    graph = copy.deepcopy(orig_graph)
    max_dist = 64
    max_num_vertices = 10
    while True:
        #print("Max Dist", max_dist)
        (graph, num_clusters) = form_clusters(graph, max_dist, max_num_vertices, orig_lakes)
        if (num_clusters == 1):
            break
        max_dist = max_dist * 2
    #print(graph[0][0]['path'])
    #prev_coord = graph[0][0]['path'][0]
    #total_dist = 0
    #for i, coord in enumerate(graph[0][0]['path'][1:]):
    #    if i % 2 == 0:
    #        total_dist += np.linalg.norm(prev_coord - coord)
    #    else:
    #        total_dist += 2 * np.linalg.norm(prev_coord - coord)
    #    prev_coord = coord
    #print("Total Distance: ", total_dist)
    convertedPath=[]
    for path in graph[0][0]['path'][1:]:
        x, y, z = path
        convertedPath.append(tuple((x, y, z)))
    #print(convertedPath)
    return convertedPath
    #print(orig_graph)

if __name__ == "__main__":
    scene = scenario_cityfire()
    scene.generate_scenario()
    clearcache()
    env = scene.simul.state()
    start = time.time()
    p = shortest(env)
    elapsed = time.time() - start
    print("Elapsed Time")
    print(elapsed)