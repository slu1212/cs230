import numpy as np
import encoding_loader
from img_encoder import ModelType, ImageEncoder 
import time
import sys

def cos_sim_vec(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    dot = np.dot(v1, v2)
    cos_sim = dot / (norm1 * norm2)
    return cos_sim

def cos_sim(enc_matrix, img_enc):
    enc_norms = np.linalg.norm(enc_matrix, axis=1)
    img_norm = np.linalg.norm(img_enc)
    dot_prod = np.dot(enc_matrix, img_enc)
    cos_sim = dot_prod / enc_norms
    cos_sim = cos_sim / img_norm
    return cos_sim

def euclidian_dist(enc_matrix, img_enc):
    diffs = enc_matrix - img_enc
    dists = np.linalg.norm(diffs, axis=1)
    return dists

def update_best(best_files, best_sims, new_sim, new_file, metric):
    for i in range(len(best_sims)):
        if metric == 'cos' and  (best_sims[i] == None or new_sim > best_sims[i]):
                best_sims.insert(i, new_sim)
                best_files.insert(i, new_file)
                best_sims.pop(-1)
                best_files.pop(-1)
                break
        if metric == 'euclidean'and (best_sims[i] == None or new_sim < best_sims[i]):
                best_sims.insert(i, new_sim)
                best_files.insert(i, new_file)
                best_sims.pop(-1)
                best_files.pop(-1)
                break
    return best_files, best_sims

def display_progress(i, num_files = 97):
    sys.stdout.write('\r')
    percent = 100 * i / 97
    num = int((i * 20)/97)
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*num, percent))
    sys.stdout.flush()

def nearest_neighbors(enc_matrix, img_enc, metric='cos'):
    similarity = None
    sim_idx = -1
    if metric == 'cos':
        similarity = cos_sim(enc_matrix, img_enc)
        sim_idx = np.argmax(similarity)
    if metric == 'euclidean':
        similarity = euclidian_dist(enc_matrix, img_enc)
        sim_idx = np.argmin(similarity)
    return sim_idx, similarity[sim_idx]


