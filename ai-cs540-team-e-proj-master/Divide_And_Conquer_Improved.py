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
    fire_graph_map = {}
    for i, fire1 in enumerate(fires):
        vertex = []
        for j, fire2 in enumerate(fires):
            edge = {}
            edge['start'] = fire1
            edge['start_id'] = i
            edge['dist'] = np.inf
            edge['end'] = fire1
            edge['end_id'] = i
            edge['path'] = []
            edge['path'].append(fire2)
            #edge['path_end_id'] = j
            edge['cluster'] = i
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
        fire_graph_map[tuple(fire1)] = i
    #print(graph, fires, lakes)
    return (graph, fires, lakes, fire_graph_map)

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
    farthest_vertex_id = -1
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

def mark_vertices_covered(graph, shortest_path, cluster_id):
    temp_path = []
    for vertex_id in shortest_path:
        temp_path.append(vertex_id)
        if graph[vertex_id][vertex_id]['start_id'] != graph[vertex_id][vertex_id]['end_id']:
            temp_path.append(graph[vertex_id][vertex_id]['end_id'])
    #print(temp_path)
    for vertex_id in temp_path:
        for vertex in graph:
            vertex[vertex_id]['dist'] = np.inf
        graph[vertex_id][vertex_id]['cluster'] = cluster_id

    if len(shortest_path) == 1:
        return graph
    
    graph[temp_path[-1]][temp_path[-1]]['path'] = copy.deepcopy(graph[shortest_path[-1]][shortest_path[-1]]['path'])
    for vertex_id in temp_path[1:-1]:
        for vertex in graph:
            vertex[vertex_id]['path'] = None

    for vertex in graph:
        vertex[temp_path[0]]['path'] = copy.deepcopy(graph[temp_path[0]][temp_path[0]]['path'])
        vertex[temp_path[-1]]['path'] = copy.deepcopy(graph[temp_path[-1]][temp_path[-1]]['path'])
    return graph

def find_shortest_path(graph, short_list):
    shortest_path = []
    shortest_dist = np.inf
    if (short_list[0] == 0):
        start_vertex_id = short_list[0]
        for path in itertools.permutations(short_list[1:]):
            prev_vertex_id = start_vertex_id
            cur_dist = 0
            temp_path = []
            cluster_list = []
            cluster_list.append(graph[prev_vertex_id][prev_vertex_id]['cluster'])
            for vertex_id in path:
                if graph[vertex_id][vertex_id]['cluster'] in cluster_list:
                    continue
                cluster_list.append(graph[vertex_id][vertex_id]['cluster'])
                cur_dist += graph[prev_vertex_id][vertex_id]['dist']
                prev_vertex_id = vertex_id
                temp_path.append(vertex_id)
            if cur_dist < shortest_dist:
                shortest_path = copy.deepcopy(list(temp_path))
                shortest_dist = cur_dist
        shortest_path.insert(0, start_vertex_id)
    else:
        for path in itertools.permutations(short_list):
            prev_vertex_id = path[0]
            cur_dist = 0
            temp_path = []
            temp_path.append(prev_vertex_id)
            cluster_list = []
            cluster_list.append(graph[prev_vertex_id][prev_vertex_id]['cluster'])
            for vertex_id in path[1:]:
                if graph[vertex_id][vertex_id]['cluster'] in cluster_list:
                    continue
                cluster_list.append(graph[vertex_id][vertex_id]['cluster'])
                cur_dist += graph[prev_vertex_id][vertex_id]['dist']
                prev_vertex_id = vertex_id
                temp_path.append(vertex_id)
            if cur_dist < shortest_dist:
                shortest_path = copy.deepcopy(list(temp_path))
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

def reconstruct_graph(graph, lakes, fire_to_graph):
    for i, vertex in enumerate(graph):
        if vertex[i]['path'] == None:
            continue
        for j, edge in enumerate(vertex):
            if edge['path'] == None:
                continue
            #if graph[i][i]['cluster'] == graph[j][j]['cluster']:
            #    continue
            edge['start'] = vertex[i]['start']
            edge['start_id'] = fire_to_graph[tuple(edge['start'])]
            edge['end'] = vertex[i]['path'][-1]
            edge['end_id'] = fire_to_graph[tuple(edge['end'])]
            if i == j:
                edge['lake'] = None
                continue

            for k, lake in enumerate(lakes):
                dist1 = np.linalg.norm(edge['end'] - lake)
                dist2 = 2 * np.linalg.norm(lake - edge['path'][0])
                #num_edges += 1
                if (dist1 + dist2) < edge['dist']:
                    edge['dist'] = dist1 + dist2
                    edge['lake'] = lake
                    edge['fire'] = edge['path'][0]
            #print(vertex)
    #print(graph, fires, lakes)
    return graph

def form_clusters(graph, max_dist, max_num_vertices, lakes, fire_to_graph):
    num_nodes_covered = 0
    num_clusters = 0
    start_vertex_id = 0
    while start_vertex_id >= 0:
        short_list = pick_neighbors(graph, start_vertex_id, max_dist, max_num_vertices)
        if (len(short_list) > 1):
            shortest_path = find_shortest_path(graph, short_list)
            shortest_path_with_lakes = populate_path_with_lakes(graph, shortest_path)
            graph[shortest_path[0]][shortest_path[0]]['path'] = copy.deepcopy(shortest_path_with_lakes)
            graph[shortest_path[-1]][shortest_path[-1]]['path'] = copy.deepcopy(list(reversed(shortest_path_with_lakes)))
            #print("Clustered Nodes", num_clusters)
            #print(shortest_path)
            #print(graph[shortest_path[0]][shortest_path[0]]['path'])
            graph = mark_vertices_covered(graph, shortest_path, num_clusters)
            start_vertex_id = pick_farthest_neighbor(graph, shortest_path[-1])
        else:
            graph = mark_vertices_covered(graph, short_list, num_clusters)
            start_vertex_id = pick_farthest_neighbor(graph, short_list[0])
            #print("Clustered Nodes", num_clusters)
            #print(short_list)
            #print(graph[short_list[0]][short_list[0]]['path'])
        num_nodes_covered += len(short_list)
        num_clusters += 1
    #print("Number of Clusters", num_clusters)
    graph = reconstruct_graph(graph, lakes, fire_to_graph)
    return (graph, num_clusters)

def shortest(env):
    orig_graph, orig_fires, orig_lakes, orig_fire_graph_map = parse_env(env)
    #print("Orig_Graph", orig_graph)
    #print("Orig_Lakes", orig_lakes)
    #print("Orig_Fires", orig_fires)
    #print(orig_fire_graph_map)
    graph = copy.deepcopy(orig_graph)
    max_dist = 64
    max_num_vertices = 8
    while True:
        #print("Max Dist", max_dist)
        (graph, num_clusters) = form_clusters(graph, max_dist, max_num_vertices, orig_lakes, orig_fire_graph_map)
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