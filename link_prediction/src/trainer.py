import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from data_loader import EdgeDataset
from utils import calculate_metrics
import random
from hypergraph import *
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_embeddings(embedding_path):
    """加载预训练的图嵌入向量"""
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings



def evaluate_link_prediction(embeddings, test_loader, device,dual_embeddings=None):
    """评估链路预测的性能"""
    mrr_sum = 0
    hits_sum = {f'hit@{k}': 0 for k in [1, 5, 10]}
    num_batches = 0

    with torch.no_grad():
       for query, candidates, y_true in test_loader:
            # 获取查询节点的嵌入
            query_nodes = query.to(device)  # [batch_size, k-1]
            if dual_embeddings is  None: 
                query_emb = embeddings[query_nodes]  # [batch_size, k-1, dim]
                query_emb = torch.mean(query_emb, dim=1)  # [batch_size, dim]
            else:
                query_emb1 = embeddings[query_nodes]  # [batch_size, k-1, dim]
                query_emb2 = dual_embeddings[query_nodes]  # [batch_size, k-1, dim]
                # 取平均得到查询表示
                query_emb = (torch.mean(query_emb1, dim=1) + torch.mean(query_emb2, dim=1)) / 2
            
            # 获取候选节点的嵌入
            candidates = candidates.to(device)  # [batch_size, num_candidates]
            batch_size, num_candidates = candidates.shape
            candidates = candidates.view(-1)  # [batch_size * num_candidates]
            if dual_embeddings is  None:
                candidate_emb = embeddings[candidates]  # [batch_size * num_candidates, dim]
                candidate_emb = candidate_emb.view(batch_size, num_candidates, -1)  # [batch_size, num_candidates, dim]
            else:
                candidate_emb1 = embeddings[candidates]  # [batch_size * num_candidates, dim]
                candidate_emb2 = dual_embeddings[candidates]  # [batch_size * num_candidates, dim]
                candidate_emb = (candidate_emb1 + candidate_emb2) / 2
                candidate_emb = candidate_emb.view(batch_size, num_candidates, -1) 
            
            # 计算余弦相似度
            # 归一化嵌入向量
            query_emb = query_emb / query_emb.norm(dim=1, keepdim=True)
            candidate_emb = candidate_emb / candidate_emb.norm(dim=2, keepdim=True)
            
            # 计算相似度分数
            scores = torch.bmm(candidate_emb, query_emb.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_candidates]
            
            # 计算指标
            batch_mrr, batch_hits = calculate_metrics(scores.cpu(), y_true)
            
            mrr_sum += batch_mrr
            for k, v in batch_hits.items():
                hits_sum[k] += v
            num_batches += 1

    mrr = mrr_sum / num_batches
    hits = {k: v / num_batches for k, v in hits_sum.items()}
    
    return mrr, hits
    
def read_graph(filename,model_name):
    """
    Read the input hypergraph.
    """
    G = Hypergraph(model_name)

    f = open(filename, 'r', encoding='utf8')
    lines = f.readlines()
    weighted= False
    if weighted:
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


def predict_nodes(node1, node2, embeddings, G, top_k=10,dual_embeddings=None):
    """
    预测与给定两个节点最相关的其他节点
    
    参数:
    node1, node2: 输入的两个节点名称
    embeddings: 节点嵌入矩阵 [num_nodes, dim]
    G: 超图对象
    top_k: 返回的候选节点数量
    dual_embeddings: 可选的第二个嵌入矩阵
    
    返回:
    top_nodes: 最可能的top_k个节点及其分数
    """
    device = embeddings.device
    
    # 获取节点的ID
    node1_id = G.node_id(node1)
    node2_id = G.node_id(node2)
    
    # 获取节点的嵌入
    if dual_embeddings is None:
        query_emb = embeddings[[node1_id, node2_id]]  # [2, dim]
    else:
        query_emb1 = embeddings[[node1_id, node2_id]]  # [2, dim]
        query_emb2 = dual_embeddings[[node1_id, node2_id]]  # [2, dim]
        query_emb = (query_emb1 + query_emb2) / 2  # [2, dim]
    
    # 计算查询的平均嵌入
    query_emb = torch.mean(query_emb, dim=0)  # [dim]
    
    # 计算所有节点的分数
    scores = torch.matmul(embeddings, query_emb)  # [num_nodes]
    
    # 获取top_k的节点（排除输入的两个节点）
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[node1_id] = False
    mask[node2_id] = False
    scores[~mask] = float('-inf')
     
    # 获取前k个最高分的索引和分数
    top_scores, top_indices = torch.topk(scores, k=top_k)
    
    # 转换回节点名称
    results = []
    for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()):
        # 找到对应ID的节点名称
        for node in G.nodes():
            if G.node_id(node) == idx:
                node_name = node
                break
        results.append((node_name, float(score)))
    
    return results

def test_prediction(graph_path,embedding_path,dual_embedding_path):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载图和数据
    G = read_graph(graph_path)
   
    n = len(G.nodes())
    # 加载预训练的图嵌入
   
    embeddings = load_embeddings(embedding_path)
    
    # 初始化 dual_embeddings 为 None
    dual_embeddings = None
    dual_embedding_matrix = None
    
    # 转换embeddings为tensor
    embedding_matrix = np.zeros((n, embeddings[list(embeddings.keys())[0]].shape[0]))
    for _id, vect in embeddings.items():
        embedding_matrix[G.node_id(_id)] = embeddings.get(_id)
    embeddings = torch.FloatTensor(embedding_matrix).to(device)
    
    # 检查是否存在对偶嵌入路径 如果存在对偶嵌入，则转换对偶嵌入为tensor
    if dual_embedding_path is not None:
        print("加载对偶嵌入...")
        dual_embeddings = load_embeddings(dual_embedding_path)
        dual_embedding_matrix = np.zeros((n, dual_embeddings[list(dual_embeddings.keys())[0]].shape[0]))
        for _id, vect in dual_embeddings.items():
            dual_embedding_matrix[G.node_id(_id)] = dual_embeddings.get(_id)
        dual_embeddings = torch.FloatTensor(dual_embedding_matrix).to(device)
   
    test_pairs = [
        ("1161", "482"), 
        ("3770", "3134"), 
        ("2762", "558"),  # 替换为实际的节点名称
        # 可以添加更多测试对
    ]
    
    # 不使用对偶嵌入的预测
    print("\n使用原始嵌入进行预测:")
    for node1, node2 in test_pairs:
        print(f"\n预测与节点 '{node1}' 和 '{node2}' 最相关的节点:")
        results = predict_nodes(node1, node2, embeddings, G)
        for i, (node, score) in enumerate(results, 1):
            print(f"{i}. {node}: {score:.4f}")
    
    # 使用对偶嵌入的预测（如果存在）
    if dual_embeddings is not None:
        print("\n使用原始嵌入和对偶嵌入进行预测:")
        for node1, node2 in test_pairs:
            print(f"\n预测与节点 '{node1}' 和 '{node2}' 最相关的节点:")
            results = predict_nodes(node1, node2, embeddings, G, dual_embeddings=dual_embeddings)
            for i, (node, score) in enumerate(results, 1):
                print(f"{i}. {node}: {score:.4f}")

def evaluate_embeddings(embeddings_path, G, data_dir, dual_embeddings_path=None):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载第一个向量
    embeddings = load_embeddings(embeddings_path) 
    n = len(G.nodes())
    embedding_matrix = np.zeros((n+1, embeddings[list(embeddings.keys())[0]].shape[0]))
    for _id, vect in embeddings.items():
        embedding_matrix[G.node_id(_id)] = embeddings.get(_id)
    valid_embeddings = embedding_matrix[:-1]
    padding_vector = np.mean(valid_embeddings, axis=0)
    embedding_matrix[-1] = padding_vector
    embeddings = torch.FloatTensor(embedding_matrix).to(device)
    
    # 加载第二个向量（如果提供）
    dual_embeddings = None
    if dual_embeddings_path:
        dual_embs = load_embeddings(dual_embeddings_path)
        dual_matrix = np.zeros((n+1, dual_embs[list(dual_embs.keys())[0]].shape[0]))
        for _id, vect in dual_embs.items():
            dual_matrix[G.node_id(_id)] = dual_embs.get(_id)
        valid_dual = dual_matrix[:-1]
        dual_padding = np.mean(valid_dual, axis=0)
        dual_matrix[-1] = dual_padding
        dual_embeddings = torch.FloatTensor(dual_matrix).to(device)
    
    # 创建测试数据集和加载器
    test_dataset = EdgeDataset(G, data_dir=data_dir)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    
    # 评估链路预测性能
    mrr, hits = evaluate_link_prediction(embeddings, dual_embeddings=dual_embeddings, 
                                       test_loader=test_loader, device=device)
    
    return mrr, hits
    
    
   

def main():
    set_seed(42)
    # 设置设备
    model_name = 'jingrong'
    # 加载图和数据
    graph_path = "graph/jingrong/new_network.edgelist"
    embedding_path1 = f'embeddings/{model_name}/Nhne_best_embeddings_1.pkl'
    embedding_path2 = f'embeddings/{model_name}/Nhne_best_embeddings_2.pkl'
    # 加载图
    
    G = read_graph(graph_path,model_name)
    n = len(G.nodes())
    data_dir =f'graph/{model_name}/new_valid.edgelist'
    # 只使用一个向量：
    mrr, hits = evaluate_embeddings(embedding_path1, G=G, data_dir=data_dir)
    print(f"MRR: {mrr:.4f}, Hits@1: {hits['hit@1']:.4f}, Hits@5: {hits['hit@5']:.4f}, Hits@10: {hits['hit@10']:.4f}")
    # 使用两个向量：
    mrr, hits = evaluate_embeddings(embedding_path1, G=G, data_dir=data_dir, dual_embeddings_path=embedding_path2)
    print(f"MRR: {mrr:.4f}, Hits@1: {hits['hit@1']:.4f}, Hits@5: {hits['hit@5']:.4f}, Hits@10: {hits['hit@10']:.4f}")


    

if __name__ == '__main__':  
    main()
  

