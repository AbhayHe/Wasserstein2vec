import copy

import numpy as np
import random
import torch
from gensim.models import Word2Vec
import ot 
from scipy.stats import wasserstein_distance
from hypergraph import *
class WassersteinWalker(object):
    def __init__(self, G, p, q,args):
        self.G = G
        self.p = p
        self.q = q
        self.args = args
        self.Pr = get_Pr(G)  # 需要实现 get_Pr 函数
        self.alias_nodes = {}
        self.alias_edges = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

           # 添加缓存
        self.node_distributions = {}  # 缓存节点分布
    
     
    def preprocess_transition_probs(self):
        G = self.G
        nodes = list(G.nodes()) 
        """预处理转移概率"""
        print("Preprocessing node distributions...")
        # 预先计算所有节点的分布
        for node in nodes:
            if node not in self.node_distributions:
                self.node_distributions[node] = self.get_node_distribution(node)
        
        print("Computing node transition probabilities...")
        # 计算节点的转移概率
        for node in nodes:
            self.alias_nodes[node] = self.get_alias_node(node)
        
        print("Computing edge transition probabilities...")
        # 计算边的转移概率
        for v1 in nodes:
            for v2 in G.neighbors(v1):
                self.alias_edges[(v1, v2)] = self.get_alias_edge(v1, v2)  

    def simulate_walks(self, num_walks,walk_length):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.wasserstein_walk(walk_length=walk_length, start_node=node))
        return walks
    def wasserstein_walk(self, walk_length, start_node):
        """基于 Wasserstein 距离的随机游走"""
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
      
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    if alias_nodes[cur] is not None:
                        J, q = alias_nodes[cur]
                        next_node = cur_nbrs[alias_draw(J, q)]
                        walk.append(next_node)
                    else:
                        break
                else:
                    prev = walk[-2]
                    edge_key = (prev, cur)
                    if edge_key in alias_edges and alias_edges[edge_key] is not None:
                        J, q = alias_edges[edge_key]
                        next_node = cur_nbrs[alias_draw(J, q)]
                        walk.append(next_node)
                    else:
                        break
            else:
                break
        return walk
    

    
    def get_node_distribution(self, node):
        G = self.G
        features = {}
        
        # 1. 度数分布
        neighbors = list(G.neighbors(node))
        if neighbors:
            degrees = [G._nodes[nbr]['degree'] for nbr in neighbors]
            total_degree = sum(degrees)
            if total_degree > 0:
                degrees = np.array(degrees) / total_degree
            else:
                degrees = np.ones(len(neighbors)) / len(neighbors)
            features['degree'] = degrees
            
        # 2. 边权重分布
        edge_weights = []
        for e in G.incident_edges(node):
            edge_weights.append(G._edges[e]['weight'])
        if edge_weights:
            total_weight = sum(edge_weights)
            if total_weight > 0:
                edge_weights = np.array(edge_weights) / total_weight
            else:
                edge_weights = np.ones(len(edge_weights)) / len(edge_weights)
            features['edge_weight'] = edge_weights
            
        # 3. 邻居的聚类系数分布
        if neighbors:
            clustering = [G._nodes[nbr]['clustering'] for nbr in neighbors]
            clustering = np.array(clustering)
            if sum(clustering) > 0:
                clustering = clustering / sum(clustering)
            else:
                clustering = np.ones(len(neighbors)) / len(neighbors)
            features['clustering'] = clustering
            
        return features
       
    def node_wasserstein_distance(self, node1, node2):
        dist1 = self.node_distributions[node1]
        dist2 = self.node_distributions[node2]
        
        total_distance = 0
        weights = {
            'degree': 1,      # 从参数中获取权重
            'edge_weight':1,
            'clustering':1
        }
        #self.args.clustering_weight
        # self.args.clustering_weight
        total_weight = 0
        valid_features = 0  # 记录有效特征数量
        
        for feat_type in ['degree', 'edge_weight', 'clustering']:
            # 检查权重是否为0
            if weights[feat_type] <= 0:
                continue
                
            if feat_type in dist1 and feat_type in dist2:
                if len(dist1[feat_type]) == 0 or len(dist2[feat_type]) == 0:
                    continue
                    
                # 直接使用numpy数组计算
                d1 = np.array(dist1[feat_type])
                d2 = np.array(dist2[feat_type])
                
                # # 确保分布归一化
                # d1 = d1 / np.sum(d1)
                # d2 = d2 / np.sum(d2)
                
                try:
                    # 计算Wasserstein距离
                    distance = wasserstein_distance(d1, d2)
                    total_distance += distance * weights[feat_type]
                    total_weight += weights[feat_type]
                    valid_features += 1
                except Exception as e:
                    print(f"Error computing distance for {feat_type}: {e}")
                    continue
        
        # 如果没有有效特征，返回最大距离
        if valid_features == 0:
            return float('inf')  # 或者其他合适的最大距离值
            
        return total_distance / total_weight if total_weight > 0 else float('inf')
        
    def get_alias_node(self, dst):
    
        G = self.G
        Pr = self.Pr
        dst_id = G.node_id(dst)
        neighbors = list(G.neighbors(dst))
        
        if not neighbors:
            return None
            
        unnormalized_probs = []
        for dst_nbr in neighbors:
            w_distance = self.node_wasserstein_distance(dst, dst_nbr)
            w_similarity = self.args.r * np.exp(-w_distance)+1
            
           
            dst_nbr_id = G.node_id(dst_nbr)
            prob = w_similarity * Pr[dst_id, dst_nbr_id]
            unnormalized_probs.append(prob)

        norm_const = sum(unnormalized_probs)
        if norm_const == 0:
            normalized_probs = [1.0 / len(neighbors)] * len(neighbors)
        else:
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        
        return alias_setup(normalized_probs)

    def get_alias_edge(self, src, dst):
        """基于 Wasserstein 距离的边转移概率"""
        G = self.G
        p = self.p
        q = self.q
        Pr = self.Pr
        
        src_id = G.node_id(src)
        dst_id = G.node_id(dst)
        unnormalized_probs = []

        for dst_nbr in G.neighbors(dst):
            # 计算 Wasserstein 距离
            w_distance = self.node_wasserstein_distance(src, dst_nbr)
            w_similarity = self.args.r * np.exp(-w_distance)+1
            
            # 结合原有策略
            
            dst_nbr_id = G.node_id(dst_nbr)
            
            if dst_nbr == src:
                prob =   w_similarity * Pr[dst_id, dst_nbr_id] / p
            elif Pr[dst_nbr_id, src_id] > 0:
                prob = w_similarity * Pr[dst_id, dst_nbr_id]
            else:
                prob = w_similarity * Pr[dst_id, dst_nbr_id] / q
                
            unnormalized_probs.append(prob)

        norm_const = sum(unnormalized_probs) or 1.0
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return alias_setup(normalized_probs)
def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/ for details.
    """
    K = len(probs)
    if K == 0:
        return np.array([]), np.array([])
        
    # 确保概率和为1
    prob_sum = sum(probs)
    if prob_sum == 0:
        probs = [1.0 / K] * K
    else:
        probs = [p / prob_sum for p in probs]
        
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int64)
    
    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q


def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def learn_embeddings(walks, G, args):
    """
    Learn embeddings by the Skip-gram model.
    """
    walks = [list(map(str, walk)) for walk in walks]
    #0: cbow   1:skip gram
    if G.name == 'normal':
        sg = 0
    else:
        sg = 1
    word2vec = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,  min_alpha=0.0001, 
                        sample=1e-4, epochs=args.iter, negative=5)

    embs = {}
    for word in map(str, list(G.nodes())):
        embs[word] = word2vec.wv[word]

    return embs


def convert_edgeemb_to_nodeemb(G, embs_edge, args):
    embs_dual = {}

    for node in G.nodes():
        cnt = 0
        emb = [0] * args.dimensions
        for e in G.incident_edges(node):
            cnt += 1
            e_emb = embs_edge[e]
            for i in range(args.dimensions):
                emb[i] += float(e_emb[i])

        emb = np.divide(emb, cnt)
        embs_dual[node] = emb

    return embs_dual


def Wasserstein2vec(G, args):
    print('\n##### initializing hypergraph...')
    walker = WassersteinWalker(G, args.p, args.q,args)

    print('\n##### preprocessing transition probs...')
    walker.preprocess_transition_probs()

    print('\n##### walking...')
    walks = walker.simulate_walks(args.num_walks, args.walk_length)

    print("\n##### embedding...")
    embs = learn_embeddings(walks, G, args)
    return embs
