import argparse
import pickle
from node2vec import node2vec
import torch
from hypergraph import *
import os
import time
from itertools import product
import json
from Wasserstein2vevc import Wasserstein2vec
from datetime import datetime
from trainer import evaluate_embeddings  # 导入trainer中的评估函数
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', nargs='?',default='',
                        help='Input graph path')
    parser.add_argument('--save-model', nargs='?',
                        help='output model path')
    parser.add_argument('--dimensions', type=int, default=256,
                        help='Number of dimensions. Default is 32.')
    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 20.')
    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')
    parser.add_argument('--window-size', type=int, default=20,
                        help='Context size for optimization. Default is 5.')
    parser.add_argument('--iter', default=20, type=int,  
                        help='Number of epochs in Skipgram. Default is 10.')
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of epochs in RMSprop. Default is 20.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=0.25,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('--r', type=float, default=4,
                        help='r. Default is 0.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training. Default is 64.')
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)
    parser.add_argument('--results_dir', type=str, default='graph/embeddings/WassersteinWalker/grid_search',
                        help='Directory to save results. Default is grid_search.')

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def read_graph(filename,args):
    """
    Read the input hypergraph.
    """
    G = Hypergraph(args.modelname)

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
def load_or_create_graph(filename, args):
    cache_file = f"graph/results/{args.dataset}/one/{args.dataset}_cache.pkl"
    
    # Check if the cache file exists
    if os.path.exists(cache_file):
        print("Loading graph from cache...")
        with open(cache_file, 'rb') as f:
            G = pickle.load(f)
    else:
        print("Reading graph from file...")
        G = read_graph(filename, args)
        
        # Save the graph to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(G, f)
    
    return G

def grid_search(G, args):
    """使用trainer中的评估方法进行网格搜索"""
    # 定义参数网格
    param_grid = {
        'num_walks': [10,20],
        'walk_length': [40,80],
        'dimensions': [128,256],
        'p': [0.25,0.5,1,2],
        'q': [0.25,0.5,1,2],
        # 'degree_weight': [1],  # 0: 不使用度数特征, 1: 使用度数特征
        # 'edge_weight': [1],     # 0: 不使用边权重特征, 1: 使用边权重特征
        # 'clustering_weight': [1], # 0: 不使用聚类系数特征, 1: 使用聚类系数特征
       'r':[2]
        # "r": [0.25, 0.5, 1, 2, 4, 8, 10],
       
    }
    
    # 获取参数组合
    param_combinations = [dict(zip(param_grid.keys(), v))  
                       for v in product(*param_grid.values())]
    # 存储结果
    results = []
    best_mrr = 0
    best_params = None
    best_embeddings = None
    best_embs_file = None
            
    print(f"Starting grid search with {len(param_combinations)} combinations")
    
    for idx, params in enumerate(param_combinations):
    # 加载或创建结果文件
        result_file = os.path.join(f'graph/results/{args.dataset}/one/', 'grid_search_results_{}.json'.format(args.modelname))
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                saved_results = json.load(f)
                results = saved_results.get('results', [])
                best_params = saved_results.get('best_params', None)
                best_mrr = saved_results.get('best_mrr', 0)

        # 检查是否已经评估过这组参数
        if any(r['params'] == params for r in results):
            print(f"Skipping already evaluated combination {idx+1}")
            continue
            
        try:
            print(f"\nEvaluating combination {idx+1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            # 更新当前参数
            current_args = argparse.Namespace(**vars(args))
            for k, v in params.items():
                setattr(current_args, k, v)
            result_file = os.path.join(f'graph/results/{args.dataset}/one', 'grid_search_results_{}.json'.format(args.modelname))
        
            # 构建嵌入文件名
            embs_file = (f"graph/results/{args.dataset}/one/"
                        f"{args.dataset}_grid_search_{idx}_"
                        f"{params['num_walks']}_{params['walk_length']}_{params['dimensions']}_{params['r']}_{args.modelname}.pkl")

            # 生成或加载嵌入
            if os.path.exists(embs_file):
                print(f"Loading cached embeddings from {embs_file}")
                with open(embs_file, 'rb') as f:
                    embs = pickle.load(f)
            else:
                print("Generating new embeddings...")
                start_time = time.time()
                embs = Wasserstein2vec(G, current_args)
                if isinstance(embs, dict):
                    embs = {k: v.cpu().numpy() if torch.is_tensor(v) else v 
                           for k, v in embs.items()}
                with open(embs_file, 'wb') as f:
                    pickle.dump(embs, f)
                print(f"Time taken: {time.time() - start_time:.2f} seconds")
            
            # 使用trainer中的评估函数
            print("Evaluating embeddings...")
            data_dir =f'graph/{args.dataset}/valid.edgelist'
            mrr, hits = evaluate_embeddings(embs_file, G, data_dir)
            print(f"MRR: {mrr:.4f}, Hits@1: {hits['hit@1']:.4f}, Hits@5: {hits['hit@5']:.4f}, Hits@10: {hits['hit@10']:.4f}")
            # 记录结果
            result = {
                'params': params,
                'mrr': mrr,
                'hits': hits,    
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            results.append(result)
            
            # 更新最佳结果
            if mrr > best_mrr:
                best_mrr = float(mrr)
                best_params = params
                best_embeddings = embs
                
                #保存最佳嵌入
                best_embs_file = (f"graph/results/{args.dataset}/one/"
                        f"{args.dataset}_grid_search_best_"
                        f"{params['num_walks']}_{params['walk_length']}_{params['dimensions']}_{params['r']}_{args.modelname}.pkl")
                with open(best_embs_file, 'wb') as f:
                    pickle.dump(best_embeddings, f)

            # 保存中间结果
            with open(result_file, 'w') as f:
                json.dump({
                    'results': results,
                    'best_params': best_params,
                    'best_mrr': best_mrr
                }, f, indent=4)
            
            print(f"MRR: {mrr:.4f}")
            print(f"Best score so far: {best_mrr:.4f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error with parameters {params}: {str(e)}")
            continue
        
        #删除非最佳的嵌入文件以节省空间
        if os.path.exists(embs_file) and embs_file != best_embs_file:
            os.remove(embs_file)

    return best_params, best_mrr, best_embeddings

def main(args):
    set_seed(42)

    # for dataset in ['dblp']:
    #     if dataset == 'FB-AUTO':
    #         args.dimensions = 256
    #         args.walk_length = 40
    #         args.num_walks = 10
    #         args.window_size = 10
    #         args.p = 0.25
    #         args.q = 2
    #         args.dataset = dataset
    #     elif dataset == 'dblp':
    #         args.dimensions = 256
    #         args.walk_length = 80
    #         args.num_walks = 20
    #         args.window_size = 20
    #         args.p = 1
    #         args.q = 0.25
    #         args.dataset = dataset
    #     else:
    #         args.dimensions = 128
    #         args.walk_length = 40
    #         args.num_walks = 10
    #         args.window_size = 10
    #         args.p = 0.25
    #         args.q = 0.5
    #         args.dataset = dataset
    dataset = 'FB-AUTO'
    args.dataset = dataset
    print("Dataset:", dataset)
    args.modelname = 'Wasserstein2vec'
    print('\n##### reading hypergraph...')
    filename = f"graph/{dataset}/edgelist_new.txt"
    G = load_or_create_graph(filename, args)

    print("\n##### Starting grid search...")
    best_params, best_mrr, best_embs = grid_search(G,args)
    print("\n##### Grid search completed")
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_mrr:.4f}")


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    main(args)



    #FBAUTO edgelist_new.txt valid.edgelist window_size = 20 iter = 10