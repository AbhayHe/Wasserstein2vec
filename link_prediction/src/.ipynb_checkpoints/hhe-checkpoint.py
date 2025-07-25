import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import json
import datetime
from data_loader import EdgeDataset
from utils import calculate_metrics
from trainer import set_seed
import os
from trainer import read_graph
import sys

root_path = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(root_path)
sys.path.append(root_path + '/link_prediction')
class NeuralHypergraphEmbedding(nn.Module):
    def __init__(self, G, args):
        """
        简化版HHE模型
        Args:
            G: 超图对象
            args: 参数配置
        """
        super(NeuralHypergraphEmbedding, self).__init__()
        self.G = G
        self.num_nodes = len(G.nodes())
        self.embedding_dim = args.dimensions
        self.device = args.device

        # 仅保留基础的节点嵌入层
        self.node_embeddings = nn.Embedding(self.num_nodes + 1, self.embedding_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)

        # 简单的超边转换层
        self.edge_transform = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.to(self.device)

    def encode_hyperedge(self, node_embeddings):
        """
        简单的超边编码：平均节点嵌入后进行线性变换
        """
        # 计算节点嵌入的平均值
        edge_embedding = torch.mean(node_embeddings, dim=0)
        # 通过线性层转换
        return self.edge_transform(edge_embedding)

    def forward(self, query_nodes, candidate_nodes):
        """
        前向传播
        Args:
            query_nodes: 查询超边中的节点 [batch_size, max_edge_size]
            candidate_nodes: 候选节点 [batch_size, num_candidates]
        """
        batch_size = query_nodes.size(0)
        num_candidates = candidate_nodes.size(1)

        # 1. 处理查询超边
        query_mask = (query_nodes != len(self.G.nodes()))
        query_edge_embeds = []
        
        for i in range(batch_size):
            # 获取有效节点
            valid_nodes = query_nodes[i][query_mask[i]]
            # 获取节点嵌入
            node_embeds = self.node_embeddings(valid_nodes)
            # 编码超边
            edge_embed = self.encode_hyperedge(node_embeds)
            query_edge_embeds.append(edge_embed)
        
        query_edge_embeds = torch.stack(query_edge_embeds)  # [batch_size, embedding_dim]

        # 2. 处理候选节点
        candidate_embeds = self.node_embeddings(candidate_nodes.view(-1)).view(
            batch_size, num_candidates, -1)  # [batch_size, num_candidates, embedding_dim]

        # 3. 计算相似度分数：简单的点积
        scores = torch.bmm(
            candidate_embeds,                    # [batch_size, num_candidates, embedding_dim]
            query_edge_embeds.unsqueeze(-1)     # [batch_size, embedding_dim, 1]
        ).squeeze(-1)  # [batch_size, num_candidates]

        return scores

def train_nhe(G, args):
    """
    训练NHE模型
    """
    print("\nInitializing NHE model...")
    model = NeuralHypergraphEmbedding(G, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3) 
    # 创建数据加载器
    train_dataset = EdgeDataset(G, data_dir="../graph/dblp/new_network.edgelist")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    test_dataset = EdgeDataset(G, data_dir="../graph/dblp/new_valid.edgelist")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    # 创建保存目录
    # save_dir = os.path.join(root_path, '../embeddings/imdb')
    save_dir ='../embeddings/dblp'
    os.makedirs(save_dir, exist_ok=True)
    # 训练循环
    print("\nStarting training...")
    model.train()
    best_model_state = None
    best_metrics = {'mrr': 0, 'hits': {}}
    # 创建训练历史记录列表
    training_history = []
    for epoch in range(args.epochs):
        # 训练阶段
        total_loss = 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in progress_bar:
            query, candidates, labels = [b.to(args.device) for b in batch]
            scores = model(query, candidates)
            loss = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()         
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({'batch_loss': f'{loss.item():.4f}'})
        epoch_loss = total_loss / batch_count
        print(f'\nEpoch {epoch+1} Average Loss: {epoch_loss:.4f}')
        
        # 评估阶段
        if (epoch + 1) % args.eval_freq == 0:
            mrr, hits = evaluate_model(model, test_loader)
            print(f"\nEpoch {epoch+1}")
            print(f"Average Loss: {epoch_loss:.4f}")
            print(f"MRR: {mrr:.4f}")
            print(f"Hit@1: {hits['hit@1']:.4f}")
            print(f"Hit@5: {hits['hit@5']:.4f}")
            print(f"Hit@10: {hits['hit@10']:.4f}")
            
            # 保存最佳模型（基于MRR）
            if mrr > best_metrics['mrr']:
                best_metrics['mrr'] = mrr
                best_metrics['hits'] = hits
                best_model_state = model.state_dict().copy()
                #torch.save(best_model_state, os.path.join(save_dir, 'hhe_best_model.pt'))
                print(f"New best model saved! MRR: {mrr:.4f}")
                best_result = {
                    "best_performance": {
                        "params": {
                            "embedding_dim": args.dimensions,
                            "num_nodes": len(G.nodes()),
                            "epochs": epoch,
                            "batch_size": args.batch_size,
                            "best_epoch": None  # 将在获得最佳结果时更新
                        },
                        "mrr": best_metrics['mrr'],
                        "hits": best_metrics['hits'],
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }}
                       
            # 记录当前epoch的结果
            epoch_result = {
                "params": {
                    "embedding_dim": args.dimensions,
                    "num_nodes": len(G.nodes()),
                    "epochs": epoch,
                    "batch_size": args.batch_size,
                    "epoch_loss": epoch_loss
                },
                "mrr": float(mrr),
                "hits": {
                    "hit@1": float(hits['hit@1']),
                    "hit@5": float(hits['hit@5']),
                    "hit@10": float(hits['hit@10'])
                },
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            training_history.append(epoch_result)
    
    # 保存训练历史
    history_file = os.path.join(save_dir, 'training_history.json')
    with open(history_file, 'a') as f:
        json.dump(training_history, f, indent=4)
    
    with open(history_file, 'a') as f:
        json.dump(best_result, f, indent=4)
    
    
   
    
    
    
def evaluate_model(model, test_loader):
    """
    评估模型性能
    Args:
        model: 训练好的模型
        G: 图对象
        test_loader: 测试数据加载器
        device: 计算设备
    Returns:
        mrr: Mean Reciprocal Rank
        hits: Dictionary of Hit@K scores
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mrr_sum = 0
    hits_sum = {f'hit@{k}': 0 for k in [1, 5, 10]}
    num_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            # 正确的数据转移方式
            query, candidates, labels = [b.to(device) for b in batch]
            
            # 计算分数
            scores = model(query, candidates)
            # 计算指标
            batch_mrr, batch_hits = calculate_metrics(scores.cpu(), labels.cpu())
            
            # 累加批次结果
            mrr_sum += batch_mrr
            for k, v in batch_hits.items():
                hits_sum[k] += v
            num_batches += 1

    # 计算平均值
    mrr = mrr_sum / num_batches
    hits = {k: v / num_batches for k, v in hits_sum.items()}
    
    return mrr, hits


def load_and_predict(model_path, G, query_nodes, num_candidates=10):
    """
    加载训练好的模型并进行预测
    
    Args:
        model_path: 模型保存路径
        G: 超图对象
        query_nodes: 查询节点列表 (list of node names)
        num_candidates: 返回的候选节点数量
    
    Returns:
        list of tuples: (node_name, score) 按分数排序的候选节点列表
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型参数对象
    class Args:
        def __init__(self):
            self.dimensions =256  # 需要与训练时使用的维度相同
            self.device = device
    
    # 初始化模型
    args = Args()
    model = NeuralHypergraphEmbedding(G, args)
    
    # 加载训练好的模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        # 将节点名称转换为ID
        query_ids = [G.node_id(node) for node in query_nodes]
        
        # 填充到最大边大小
        max_edge_size = 10  # 根据实际情况调整
        while len(query_ids) < max_edge_size:
            query_ids.append(len(G.nodes()))  # 添加padding token
        
        # 准备输入数据
        query_tensor = torch.tensor(query_ids).unsqueeze(0).to(device)  # [1, max_edge_size]
        
        # 准备所有可能的候选节点
        all_nodes = list(range(len(G.nodes())))
        # 排除查询节点
        candidates = [node for node in all_nodes if node not in query_ids]
        candidate_tensor = torch.tensor(candidates).unsqueeze(0).to(device)  # [1, num_candidates]
        
        # 获取预测分数
        scores = model(query_tensor, candidate_tensor)
        scores = scores.squeeze(0)  # 移除batch维度
        
        # 获取top-k的结果
        top_scores, top_indices = torch.topk(scores, min(num_candidates, len(candidates)))
        
        # 转换结果
        results = []
        for idx, score in zip(top_indices.cpu().numpy(), top_scores.cpu().numpy()):
            candidate_id = candidates[idx]
            # 找到对应ID的节点名称
            for node in G.nodes():
                if G.node_id(node) == candidate_id:
                    node_name = node
                    break
            results.append((node_name, float(score)))
    
    return results
def test_model_prediction():
    """
    测试模型预测的示例函数
    """
    # 设置路径
    model_path = "../embeddings/dblp/hhe_best_model.pt"
    graph_path = "../graph/dblp/edgelist_filtered.txt"
    
    # 加载图
    G = read_graph(graph_path, 'hhe2vec')
    
    # 测试用的查询节点
    test_queries = [
        ["14", "15", "16",'17'],  # 示例查询1
        ["5549", "6634", "2729", "5631"], 
        ['34','35'],
        ['54','51']# 示例查询2
         #5549 6634 2729 5808 5631
        #34 35 36
        #14 15 16 17 18
    ]
    
    # 对每个查询进行预测
    for query in test_queries:
        print(f"\n预测与节点 {query} 最可能形成超边的节点:")
        results = load_and_predict(model_path, G, query)
        for i, (node, score) in enumerate(results, 1):
            print(f"{i}. 节点: {node}, 分数: {score:.4f}")
if __name__ == "__main__":
    from main import parse_args
    set_seed(42)
    args = parse_args()
    args.epochs = 20
    args.eval_freq = 1  # 每5个epoch评估一次
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.dimensions = 256
    args.batch_size = 256
    # 读取超图
    from trainer import read_graph
    G = read_graph("../graph/dblp/new_network.edgelist", 'hhe2vec')
    
    # 训练模型
    train_nhe(G, args)


    #预测模型
    # test_model_prediction()
   
  