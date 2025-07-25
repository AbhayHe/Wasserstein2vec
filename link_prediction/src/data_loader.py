import random
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils import read_edgelist

class EdgeDataset(Dataset):
    def __init__(self, G, data_dir=''):
        self.G = G
        self.max_edge_degree = G.max_edge_degree
        self.X = []  # 查询（不完整的超边）
        self.y = []  # 正确答案在候选集中的索引
        self.candidates = []  # 候选节点集合
        self.PAD_TOKEN = len(G.nodes())
        data_dir = Path(data_dir)
        edges = read_edgelist(data_dir)
        
        # 计算所有可能的候选节点数量
        all_nodes = set(G.nodes())
        self.max_candidates = len(all_nodes) - self.max_edge_degree  # 最大可能的候选数量
        
        for edge_name, nodes in edges:
            # 随机移除一个节点作为预测目标
            nodes = list(nodes)
            target_idx = random.randrange(len(nodes))
            target_node = nodes.pop(target_idx)
            
            # 保存查询和正确答案
            processed_nodes = self._process_edge(nodes)
            self.X.append(processed_nodes)
            
            # 获取所有可用的负样本节点
            available_nodes = list(all_nodes - set(nodes) - {target_node})
            
            # 确保所有样本的候选集大小相同
            if len(available_nodes) > self.max_candidates - 1:  # -1 是为了留出位置给正确答案
                available_nodes = random.sample(available_nodes, self.max_candidates - 1)
            else:
                # 如果候选数量不足，通过重复来填充
                while len(available_nodes) < self.max_candidates - 1:
                    available_nodes.append(random.choice(available_nodes))
            
            # 随机选择一个位置插入正确答案
            insert_pos = random.randrange(len(available_nodes) + 1)
            available_nodes.insert(insert_pos, target_node)
            
            # 转换为节点ID
            candidates_ids = [self.G.node_id(node) for node in available_nodes]
            self.candidates.append(candidates_ids)
            
            # 存储正确答案在候选集中的索引
            self.y.append(insert_pos)
    
    def _process_edge(self, nodes):
        """处理边,填充到固定长度"""
         # 使用PAD_TOKEN进行填充
        x = [self.PAD_TOKEN] * (self.max_edge_degree - len(nodes)-1)  # 填充
        x.extend([self.G.node_id(node) for node in nodes])
        return x
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),  # 查询
            torch.tensor(self.candidates[idx], dtype=torch.long),  # 候选集
            torch.tensor(self.y[idx], dtype=torch.long)  # 正确答案的实际索引
        )