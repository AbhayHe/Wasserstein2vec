import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm
from trainer import set_seed
class LINE(nn.Module):
    def __init__(self, num_nodes, embedding_dim, order=1):
        """
        LINE模型实现
        Args:
            num_nodes: 节点数量
            embedding_dim: 嵌入维度
            order: 相似度阶数 (1 or 2)
        """
        super(LINE, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.order = order

        # 初始化嵌入层
        self.u_embeddings = nn.Embedding(num_nodes, embedding_dim)
        if order == 2:
            self.v_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        self._init_weights()

    def _init_weights(self):
        """初始化嵌入权重"""
        initrange = 0.5 / self.embedding_dim
        nn.init.uniform_(self.u_embeddings.weight, -initrange, initrange)
        if self.order == 2:
            nn.init.uniform_(self.v_embeddings.weight, -initrange, initrange)

    def forward(self, pos_u, pos_v, neg_v):
        """
        前向传播
        Args:
            pos_u: 源节点
            pos_v: 正样本目标节点
            neg_v: 负样本目标节点
        """
        emb_u = self.u_embeddings(pos_u)
        
        if self.order == 1:
            emb_pos_v = self.u_embeddings(pos_v)
            emb_neg_v = self.u_embeddings(neg_v)
        else:
            emb_pos_v = self.v_embeddings(pos_v)
            emb_neg_v = self.v_embeddings(neg_v)
            
        # 计算正样本得分
        pos_score = torch.sum(torch.mul(emb_u, emb_pos_v), dim=1)
        pos_score = F.logsigmoid(pos_score)
        
        # 计算负样本得分
        neg_score = torch.sum(torch.mul(emb_u.unsqueeze(1), emb_neg_v), dim=2)
        neg_score = F.logsigmoid(-neg_score)
        
        return -(torch.mean(pos_score) + torch.mean(neg_score))

    def get_embeddings(self):
        """获取节点嵌入"""
        if self.order == 1:
            return self.u_embeddings.weight.data
        else:
            return (self.u_embeddings.weight.data + self.v_embeddings.weight.data) / 2

class HyperedgeSampler:
    def __init__(self, hyperedges, num_nodes, sampling_rate=1e-4):
        """
        超边采样器
        Args:
            hyperedges: 超边列表 [(edge_id, [node1, node2, ...]), ...]
            num_nodes: 节点数量
            sampling_rate: 负采样率
        """
        self.hyperedges = hyperedges
        self.num_nodes = num_nodes
        self.sampling_rate = sampling_rate
        
        # 构建节点-超边关系表
        self.node_to_edges = defaultdict(list)
        self.node_degrees = np.zeros(num_nodes)
        
        # 记录每个节点参与的超边
        for edge_id, nodes in hyperedges:
            for node in nodes:
                self.node_to_edges[node].append(edge_id)
                self.node_degrees[node] += 1
            
        # 计算节点采样概率
        self.node_distribution = np.power(self.node_degrees, 0.75)
        self.node_distribution /= np.sum(self.node_distribution)

    def sample(self, batch_size, num_negative):
        """
        采样训练batch
        Args:
            batch_size: batch大小
            num_negative: 每个正样本对应的负样本数量
        """
        pos_u = []
        pos_v = []
        neg_v = []
        
        # 随机选择batch_size个超边
        selected_edges = random.sample(self.hyperedges, batch_size)
        
        for _, nodes in selected_edges:
            # 从超边中随机选择两个节点作为正样本对
            if len(nodes) >= 2:
                u, v = random.sample(nodes, 2)
                pos_u.append(u)
                pos_v.append(v)
                
                # 负采样
                neg = np.random.choice(
                    self.num_nodes,
                    size=num_negative,
                    p=self.node_distribution,
                    replace=True
                )
                neg_v.append(neg)
            
        return (torch.LongTensor(pos_u),
                torch.LongTensor(pos_v),
                torch.LongTensor(neg_v))

def train_line_hypergraph(hyperedges, num_nodes, args):
    """
    训练LINE模型(超图版本)
    Args:
        hyperedges: 超边列表
        num_nodes: 节点数量
        args: 训练参数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = LINE(num_nodes, args.dim, args.order).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 初始化超边采样器
    edge_sampler = HyperedgeSampler(hyperedges, num_nodes)
    
    # 训练循环
    print("\nStarting training...")
    model.train()
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = len(hyperedges) // args.batch_size
        
        with tqdm(total=num_batches) as pbar:
            for _ in range(num_batches):
                # 采样batch
                pos_u, pos_v, neg_v = edge_sampler.sample(
                    args.batch_size,
                    args.negative
                )
                
                # 移动数据到设备
                pos_u = pos_u.to(device)
                pos_v = pos_v.to(device)
                neg_v = neg_v.to(device)
                
                # 前向传播和损失计算
                loss = model(pos_u, pos_v, neg_v)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.update(1)
                pbar.set_description(
                    f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}'
                )
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.save_path)
    
    return model

def evaluate_line_prediction(embedding_path, test_edges_path, device='cuda'):
    """
    超图链路预测评估：对每条超边随机删除一个节点，预测该节点在所有候选节点中的排名
    Args:
        embedding_path: LINE嵌入文件路径(.npy格式)
        test_edges_path: 测试集超边文件路径
        device: 计算设备
    Returns:
        mrr: Mean Reciprocal Rank
        hits: Hit@1, Hit@5, Hit@10指标
    """
    # 加载嵌入
    embeddings = torch.from_numpy(np.load(embedding_path)).to(device)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化
    num_nodes = embeddings.shape[0]
    
    # 加载测试集超边
    test_edges = []
    with open(test_edges_path, 'r') as f:
        for line in f:
            items = line.strip().split()
            nodes = [int(x) for x in items[1:]]  # 跳过超边ID
            if len(nodes) >= 2:  # 只保留长度大于等于2的超边
                test_edges.append(nodes)
    
    mrr_sum = 0
    hits_sum = {'hit@1': 0, 'hit@5': 0, 'hit@10': 0}
    num_edges = 0
    
    print("\nEvaluating link prediction...")
    for edge in tqdm(test_edges):
        # 1. 随机选择一个节点作为缺失节点
        target_idx = random.randrange(len(edge))
        target_node = edge[target_idx]
        
        # 2. 获取剩余节点作为已知节点
        context_nodes = edge[:target_idx] + edge[target_idx+1:]
        
        # 3. 计算已知节点的平均嵌入
        context_emb = embeddings[context_nodes].mean(dim=0)
        context_emb = F.normalize(context_emb.unsqueeze(0), p=2, dim=1)
        
        # 4. 计算平均嵌入与所有节点的相似度
        all_scores = torch.mm(embeddings, context_emb.t()).squeeze()
        
        # 5. 将已知节点的分数设为负无穷（因为我们要在其他节点中找缺失节点）
        all_scores[context_nodes] = float('-inf')
        
        # 6. 获取所有节点的排序（降序）
        sorted_scores, sorted_indices = torch.sort(all_scores, descending=True)
        
        # 7. 找到缺失节点在排序中的位置
        rank = (sorted_indices == target_node).nonzero().item() + 1
        
        # 8. 更新评估指标
        mrr_sum += 1.0 / rank
        hits_sum['hit@1'] += 1 if rank <= 1 else 0
        hits_sum['hit@5'] += 1 if rank <= 5 else 0
        hits_sum['hit@10'] += 1 if rank <= 10 else 0
        num_edges += 1
        
        # 调试信息
        if num_edges <= 5:  # 打印前5个预测的详细信息
            print(f"\nDebug info for edge {num_edges}:")
            print(f"Edge: {edge}")
            print(f"Target node: {target_node}")
            print(f"Context nodes: {context_nodes}")
            print(f"Target node rank: {rank}")
            top_k = 5
            top_nodes = sorted_indices[:top_k].cpu().numpy()
            top_scores = sorted_scores[:top_k].cpu().numpy()
            print(f"Top {top_k} predictions: {list(zip(top_nodes, top_scores))}")
    
    # 计算平均值
    mrr = mrr_sum / num_edges
    hits = {k: v / num_edges for k, v in hits_sum.items()}
    
    print(f"\nEvaluation Results (compared with all {num_nodes} nodes):")
    print(f"Total evaluated edges: {num_edges}")
    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hits['hit@1']:.4f}")
    print(f"Hit@5: {hits['hit@5']:.4f}")
    print(f"Hit@10: {hits['hit@10']:.4f}")
    
    return mrr, hits

def predict_links(embedding_path, query_nodes, top_k=10, device='cuda'):
    """
    使用LINE嵌入预测最可能与查询节点形成超边的其他节点
    Args:
        embedding_path: LINE嵌入文件路径(.npy格式)
        query_nodes: 查询节点列表
        top_k: 返回前k个最可能的节点
        device: 计算设备
    Returns:
        predictions: 预测节点列表及其相似度分数
    """
    # 加载并归一化嵌入向量
    embeddings = torch.from_numpy(np.load(embedding_path)).to(device)
    embeddings = F.normalize(embeddings, p=2, dim=1)  # L2归一化
    
    # 计算并归一化查询节点的平均嵌入
    query_emb = embeddings[query_nodes].mean(dim=0)
    query_emb = F.normalize(query_emb.unsqueeze(0), p=2, dim=1).squeeze()
    
    # 计算与所有节点的相似度
    similarities = torch.mm(
        embeddings,
        query_emb.unsqueeze(1)
    ).squeeze()
    
    # 获取top-k预测
    top_values, top_indices = torch.topk(similarities, k=top_k+len(query_nodes))
    
    # 过滤掉查询节点
    predictions = []
    query_nodes_set = set(query_nodes)
    for idx, score in zip(top_indices.cpu().numpy(), top_values.cpu().numpy()):
        if idx not in query_nodes_set:
            predictions.append((idx, float(score)))
            if len(predictions) == top_k:
                break
    
    print(predictions)
    return predictions

def main():
    """主函数"""
    import argparse
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',default='graph/dblp/new_network.edgelist', help='输入超边列表文件')
    parser.add_argument('--output',default='embeddings/dblp/line.npy', help='输出嵌入文件')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--order', type=int, default=2, help='相似度阶数(1 or 2)')
    parser.add_argument('--negative', type=int, default=5, help='负采样数量')
    parser.add_argument('--batch-size', type=int, default=128, help='batch大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--save-path', default='model.pt', help='模型保存路径')
    parser.add_argument('--test-edges', default='graph/dblp/new_valid.edgelist', help='测试集超边文件路径')
    args = parser.parse_args() 
    
    for args.dataset in ['jingrong']:  
        print(f"开始训练{args.dataset}模型")
        args.input = f"graph/{args.dataset}/new_network.edgelist"
        args.output = f"embeddings/{args.dataset}/line.npy"
        args.test_edges = f"graph/{args.dataset}/new_valid.edgelist"
        # 读取超边列表
        hyperedges = []
        node_set = set()
        with open(args.input, 'r') as f:
            for line in f:
                items = line.strip().split()
                edge_id = items[0]
                nodes = [int(x) for x in items[1:]]
                hyperedges.append((edge_id, nodes))
                node_set.update(nodes)
        
            num_nodes = max(node_set) + 1
            
            # 训练模型
            model = train_line_hypergraph(hyperedges, num_nodes, args)
            
        # 保存嵌入
        embeddings = model.get_embeddings().cpu().numpy()
        np.save(args.output, embeddings)

        evaluate_line_prediction(args.output, args.test_edges)
    
    


if __name__ == "__main__":
    main()
    # imdb edgelist.txt,valid.edgelist
    # dblp new_network.edgelist,new_valid.edgelist 128
    # FB-AUTO new_network.txt,new_valid.edgelist