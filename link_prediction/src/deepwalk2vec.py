import torch
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm

class HyperDeepWalk:
    def __init__(self, G, embedding_dim=128, walk_length=80, 
                 walks_per_node=10, window_size=10, workers=4):
        self.G = G
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.window_size = window_size
        self.workers = workers
        
    def generate_walks(self):
        """生成随机游走序列"""
        print("Generating random walks...")
        walks = []
        nodes = list(self.G.nodes())
        
        for _ in tqdm(range(self.walks_per_node)):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = [node]
                for _ in range(self.walk_length - 1):
                    curr = walk[-1]
                    neighbors = list(self.G.neighbors(curr))
                    if neighbors:
                        walk.append(np.random.choice(neighbors))
                    else:
                        break
                walks.append([str(node) for node in walk])
        
        return walks
    
    def train(self):
        """训练词向量"""
        # 生成随机游走序列
        walks = self.generate_walks()
        
        print("Training word2vec model...")
        # 使用gensim的Word2Vec训练
        model = Word2Vec(
            walks,
            size=self.embedding_dim,
            window=self.window_size,
            min_count=0,
            sg=1,  # 使用skip-gram
            workers=self.workers,
            iter=5
        )
     

        
        # 将词向量转换为字典格式
        embeddings = {}
        for node in self.G.nodes():
            node_str = str(node)
            if node_str in model.wv:
                embeddings[node] = model.wv[node_str]
        
        return embeddings

def train_node_embeddings(G, args):
    """训练节点嵌入的主函数"""
    model = HyperDeepWalk(
        G,
        embedding_dim=args.dimensions,
        walk_length=args.walk_length,
        walks_per_node=args.num_walks,
        window_size=args.window_size,
        workers=args.workers
    )
    
    embeddings = model.train()
    return embeddings

from trainer import read_graph

# 读取超图
G = read_graph("graph/FB-AUTO/edgelist_new.txt",'deepwalk2vec')

# 设置参数
class Args:
    def __init__(self):
        self.dimensions = 256
        self.walk_length = 40
        self.num_walks = 10
        self.window_size = 10
        self.workers = 4

args = Args()

# 训练词向量
embeddings = train_node_embeddings(G, args)

# 保存词向量
import pickle
with open('graph/results/FB-AUTO/deepwalk2vec_search_results_256_40_10.pkl', 'wb') as f:
    pickle.dump(embeddings, f)