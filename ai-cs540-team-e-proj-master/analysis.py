import numpy as np
from genetic import fitness_d_water as fit_func


def hamming_dist(seq1, seq2):
    tot_dist = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            tot_dist += 1
    return tot_dist


def hamming_dist_to_collection(seq1, seq_and_fit_list):
    coord_tups = []
    for seq, curr_fit in seq_and_fit_list:
        curr_dist = hamming_dist(seq1, seq)
        # curr_fit = fit_func(seq)
        coord_tups.append((curr_dist, curr_fit))
    return coord_tups


def plot_fit_and_dist(coord_tups):
    pass
