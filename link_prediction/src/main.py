import argparse
import pickle
from Wasserstein2vevc import Wasserstein2vec
from node2vec import node2vec
from hyper2vec import hyper2vec,convert_edgeemb_to_nodeemb
from hypergraph import *

import os
import time
import random
import numpy as np
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='?',default='../graph/imdb/edgelist.txt',
                        help='Input graph path')
    parser.add_argument('--save-model', nargs='?',
                        help='output model path')
    parser.add_argument('--dimensions', type=int, default=256,
                        help='Number of dimensions. Default is 32.')
    parser.add_argument('--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 20.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--window_size', type=int, default=20,
                        help='Context size for optimization. Default is 5.')
    parser.add_argument('--iter', default=20, type=int,
                        help='Number of epochs in Skipgram. Default is 20.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs in RMSprop. Default is 20.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=0.5,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=0.5,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--r', type=float, default=4,
                        help='r. Default is 0.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training. Default is 64.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    return parser.parse_args()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def load_or_create_graph(filename, args):
    cache_file = f"embeddings/{args.dataset}/{args.modelname}_cache.pkl"
     # Check if the cache file exists
    if os.path.exists(cache_file):
        print("Loading graph from cache...")
        with open(cache_file, 'rb') as f:
            G = pickle.load(f)
    else:
        print("Reading graph from file...")
        G = read_graph(filename, args.modelname)
        
        # Save the graph to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(G, f)
    
    return G
def read_graph(filename,model_name):
    """
    Read the input hypergraph.
    """
    G = Hypergraph(model_name)

    f = open(filename, 'r', encoding='utf8')
    lines = f.readlines()
    if args.weighted:
        for line in lines:
            line = line.split()
            edge_name = line[0]
            weight = line[1]
            G.add_edge(edge_name, line[2:], float(weight))
    else:
        for line in lines:
            line = line.split()
            edge_name = line[0]
            G.add_edge(edge_name, line[1:])
    f.close()
    return G

def save_or_load_embeddings(embeddings, file_path):
    if os.path.exists(file_path):
        print(f"Loading embeddings from {file_path}")
        # Load the embeddings from the file
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Saving embeddings to {file_path}")
        # Save the embeddings to the file
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        return embeddings

def main(args):
    set_seed(42)
    dataset = 'jingrong'
    args.dataset = dataset
    print("Dataset:", dataset)
    model_name = 'Nhne'
    args.modelname = model_name
    print('\n##### reading hypergraph...')
    filename = "graph/" + dataset + "/new_network.edgelist"
    # G = read_graph(filename,model_name)
    G = load_or_create_graph(filename, args)
    a = time.time()
    embs_file = f"embeddings/{dataset}/{model_name}_best_embeddings_1.pkl"
    if os.path.exists(embs_file):
        print(f"Loading existing embeddings from {embs_file}")
        with open(embs_file, 'rb') as f:
            embs = pickle.load(f)
    else:
        print("Generating new embeddings...")
        embs = hyper2vec(G, args)
        # Convert embeddings to numpy arrays if they aren't already
        if isinstance(embs, dict):
            embs = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in embs.items()}
        with open(embs_file, 'wb') as f:
            pickle.dump(embs, f)
    print(f"Time taken to generate embeddings: {time.time() - a} seconds")
  
 
    #训练对偶网络 
    fdual = "graph/" + dataset + "/dual_edgelist.txt"
    if not os.path.exists(fdual):
        print("\n##### generating dual hypergraph...")
        output_dual_hypergraph(G, fdual)

    Gd = read_graph(fdual,model_name)
    embs_edge_file = f"embeddings/{dataset}/{model_name}_best_embeddings_2.pkl"
    if os.path.exists(embs_edge_file):
        print(f"Loading existing dual embeddings from {embs_edge_file}")
        with open(embs_edge_file, 'rb') as f:
            embs_edge = pickle.load(f)
    else:
        print("Generating new dual embeddings...")
        embs_edge = hyper2vec(Gd, args)
        embs_dual = convert_edgeemb_to_nodeemb(G, embs_edge, args)
        # Convert dual embeddings to numpy arrays if they aren't already
        if isinstance(embs_dual, dict):
            embs_dual = {k: v.cpu().numpy() if torch.is_tensor(v) else v for k, v in embs_dual.items()}
        with open(embs_edge_file, 'wb') as f:
            pickle.dump(embs_dual, f)
 



    #

if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    main(args)
