import random


def read_edgelist(filename):
    edges = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            edge_name = line[0]
            nodes = line[1:]
            edges.append((edge_name, nodes))
    return edges

    # 读取原始数据
    edges = read_edgelist(input_file)
    
  
    
    # 随机打乱边的顺序
    random.shuffle(edges)
    
    # 计算划分点
    total_edges = len(edges)
    network_split = int(total_edges * 0.8)
    train_split = int(total_edges * 0.9)
    
    # 划分数据集
    network_edges = edges[:network_split]
    train_edges = edges[network_split:train_split]
    valid_edges = edges[train_split:]
    
    # 写入网络数据
    with open(network_file, 'w', encoding='utf8') as f:
        for edge_name, nodes in network_edges:
            f.write(f"{edge_name} {' '.join(nodes)}\n")
    
    # 写入训练集
    with open(train_file, 'w', encoding='utf8') as f:
        for edge_name, nodes in train_edges:
            f.write(f"{edge_name} {' '.join(nodes)}\n")
    
    # 写入验证集
    with open(valid_file, 'w', encoding='utf8') as f:
        for edge_name, nodes in valid_edges:
            f.write(f"{edge_name} {' '.join(nodes)}\n")

def split_dataset(input_file, network_file, valid_file):
    """
    划分数据集，确保network包含所有valid数据集中的节点
    Args:
        input_file: 输入文件路径
        network_file: 网络结构文件路径
        valid_file: 验证集文件路径
    """
    # 读取原始数据
    edges = read_edgelist(input_file)
    
    # 获取所有唯一节点
    all_nodes = set()
    for _, nodes in edges:
        all_nodes.update(nodes)
    
    # 随机打乱边的顺序
    random.shuffle(edges)
    
    # 初始化数据集
    network_edges = []
    valid_edges = []
    
    # 计算目标划分比例
    total_edges = len(edges)
    target_valid = int(total_edges * 0.1)  # 20% 用于验证
    
    # 首先选择验证集的边
    valid_edges = edges[:target_valid]
    
    # 收集验证集中的所有节点
    valid_nodes = set()
    for edge_name, nodes in valid_edges:
        valid_nodes.update(nodes)
    
    # 构建network_edges，确保包含所有valid节点
    remaining_edges = edges[target_valid:]
    network_nodes = set()
    network_edges = []
    
    # 添加包含验证集节点的边
    for edge in remaining_edges:
        edge_name, nodes = edge
        if any(node in valid_nodes for node in nodes):
            network_edges.append(edge)
            network_nodes.update(nodes)
    
    # 确保network_edges包含所有valid_nodes
    for node in valid_nodes:
        if node not in network_nodes:
            for edge in remaining_edges:
                edge_name, nodes = edge
                if node in nodes:
                    network_edges.append(edge)
                    network_nodes.update(nodes)
                    break
    
    # 如果还有剩余的边，添加到network中直到达到目标大小
    remaining_target = int(total_edges * 0.9) - len(network_edges)
    if remaining_target > 0:
        for edge in remaining_edges:
            if edge not in network_edges and len(network_edges) < int(total_edges * 0.9):
                network_edges.append(edge)
    
    # 从valid中获取边以确保所有valid_nodes在network中
    for edge in valid_edges:
        edge_name, nodes = edge
        if any(node not in network_nodes for node in nodes):
            network_edges.append(edge)
            network_nodes.update(nodes)
    
    # 检查去重后的valid节点是否都在network中
    assert valid_nodes.issubset(network_nodes), "Some valid nodes are not in the network dataset!"
    
    print(f"Dataset split stats:")
    print(f"Total edges: {total_edges}")
    print(f"Network edges: {len(network_edges)}")
    print(f"Valid edges: {len(valid_edges)}")
    print(f"Network nodes: {len(network_nodes)}")
    print(f"Valid nodes: {len(valid_nodes)}")
    
    # 写入网络数据
    with open(network_file, 'w', encoding='utf8') as f:
        for edge_name, nodes in network_edges:
            f.write(f"{edge_name} {' '.join(nodes)}\n")
    
    # 写入验证集
    with open(valid_file, 'w', encoding='utf8') as f:
        for edge_name, nodes in valid_edges:
            f.write(f"{edge_name} {' '.join(nodes)}\n")

def calculate_metrics(scores, y_true, k_list=[1, 5, 10]):
    """
    scores: [batch_size, num_candidates] 模型对所有候选的预测分数
    y_true: [batch_size] 
    """
    batch_size = scores.size(0)
    mrr = 0
    hits = {f'hit@{k}': 0 for k in k_list}
    
    for i in range(batch_size):
        # 获取排序后的索引
        _, indices = scores[i].sort(descending=True)
       
        rank = (indices == y_true[i]).nonzero().item() + 1
        
        # 计算MRR
        mrr += 1.0 / rank
        
        # 计算Hits@K
        for k in k_list:
            if rank <= k:
                hits[f'hit@{k}'] += 1
    
    # 计算平均值
    mrr = mrr / batch_size
    hits = {k: v / batch_size for k, v in hits.items()}
    
    return mrr, hits


def write_data():
    import pandas as pd
    # 读取原始数据
    df = pd.read_csv('D:\garbage_chrome_data\Political Blogs\Political-Blogs.csv')
    # 按超边分组并将节点组合成列表
    df['edge_ids'] = df['edge_ids'].str.replace('e', '').astype(int)

    # 按edge_id分组
    grouped = df.groupby('edge_ids')['node_ids'].apply(list)

    # 过滤掉节点数小于2或大于20的超边
    filtered_edges = {edge_id: nodes for edge_id, nodes in grouped.items() 
                    if 2 <= len(nodes) <= 20}

   # 重新编号超边
    new_edges = {}
    for new_id, (_, nodes) in enumerate(sorted(filtered_edges.items()), start=1):
        new_edges[new_id] = nodes

    # 写入edgelist.txt格式
    with open('edgelist.txt', 'w') as f:
        for edge_id, nodes in new_edges.items():
            # 将节点列表转换为空格分隔的字符串
            nodes_str = ' '.join(map(str, nodes))
            # 写入格式: edge_id node1 node2 node3 ...
            f.write(f"{edge_id} {nodes_str}\n")


def process_hypergraph_files(input_file, output_dir):
    # 读取原始超图数据
    edges = []
    rel = []
    nodes = []
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip().split()
            edge_id = line[0]
            node_list = line[1:]
            edges.append((edge_id, node_list))
            nodes.extend(node_list)
            rel.append(edge_id)
    # 为节点创建ID映射
    unique_nodes = []
    seen = set()
    for node in nodes:
        if node not in seen or node not in unique_nodes:
            seen.add(node)
            unique_nodes.append(node)

    node2id = {node: str(idx+1) for idx, node in enumerate(unique_nodes)}
    
   
    unique_rel = []
    seens = set()
    for re in rel:
        if re not in seens or re not in unique_rel:
            seens.add(re)
            unique_rel.append(re)
    # 为超边创建ID映射 (保持原始顺序)
    rel2id = {edge: str(num) for num, edge in enumerate(unique_rel)}
    
    # # 写入node2id.txt
    # with open(f"{output_dir}/node2id.txt", 'w', encoding='utf8') as f:
    #     for node, node_id in node2id.items(): 
    #         f.write(f"{node}\t{node_id}\n")
    
    # # 写入edge2id.txt
    # with open(f"{output_dir}/edge2id.txt", 'w', encoding='utf8') as f:
    #     for edge, edge_id in rel2id.items():
    #         f.write(f"{edge}\t{edge_id}\n")
    
    # 写入edgelist.txt
    with open(f"{output_dir}/test.edgelist", 'w', encoding='utf8') as f:
        for edge, nodes in edges:  # 保持原始顺序
            edge_id = rel2id[edge]
            # 过滤掉空节点并转换为ID
            valid_nodes = [node for node in nodes if node.strip()]  # 去除空字符串
            if len(valid_nodes) >= 2:  # 确保至少有两个节点
                node_ids = [node2id[node] for node in valid_nodes]
                f.write(f"{edge_id} {' '.join(node_ids)}\n")


def process_edgelist(input_file, output_file):
    """
    处理 edgelist.txt 文件，将第一列改为递增的数字
    """
    # 读取原始文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理并写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            parts = line.strip().split()
            # 将第一列替换为递增的数字（从0开始）
            new_line = f"{i} {' '.join(parts[1:])}\n"
            f.write(new_line)

def convert_using_node2id(test_file, node2id_file, output_file):
    """
    使用已有的node2id文件将test.txt转换为edgelist格式
    """
    # 读取node2id映射关系
    node2id = {}
    with open(node2id_file, 'r', encoding='utf-8') as f:
        for line in f:
            node, id_ = line.strip().split()
            node2id[node] = id_
    
    # 转换并写入新的边列表
    with open(test_file, 'r', encoding='utf-8') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for i, line in enumerate(fin):  # 从1开始编号
                parts = line.strip().split()
                # 跳过第一列，转换其余的节点
                if not all(node in node2id for node in parts[1:]):
                    continue  
                converted_nodes = [node2id[node] for node in parts[1:] if node in node2id]
                # 写入：序号 + 转换后的节点
                fout.write(f"{i} {' '.join(converted_nodes)}\n")
# 使用示例

def count_nodes_per_line(edgelist_file):
    """
    统计每行节点的个数及其频率
    """
    # 用字典存储节点数量的统计
    count_stats = {}
    
    with open(edgelist_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 分割并计算节点数（减去第一列的超边ID）
            nodes = line.strip().split()
            num_nodes = len(nodes) - 1
            
            # 更新统计
            count_stats[num_nodes] = count_stats.get(num_nodes, 0) + 1
    
    # 按节点数量排序输出结果
    for num_nodes, frequency in sorted(count_stats.items()):
        print(f"节点数量为 {num_nodes} 的超边有 {frequency} 行")
    
    return count_stats

def filter_hyperedges(input_file, output_file, max_nodes=5):
    """
    删除节点数大于max_nodes的行，生成新的文件
    """
    filtered_count = 0
    kept_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                nodes = line.strip().split()
                # 计算节点数（减去第一列的超边ID）
                num_nodes = len(nodes) - 1
                
                if num_nodes <= max_nodes:
                    # 保留该行
                    fout.write(line)
                    kept_count += 1
                else:
                    # 跳过该行
                    filtered_count += 1
    
    print(f"处理完成：")
    print(f"保留了 {kept_count} 行")
    print(f"删除了 {filtered_count} 行")
    
    return kept_count, filtered_count

import os
def load_id_mapping(file_path):
    id_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            key, value = line.strip().split()
            id_mapping[key] = value
    return id_mapping

def convert_test_file(test_file, node2id, edge2id, output_file):
    with open(test_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            # Convert the second column using edge2id
            edge_id = edge2id.get(parts[1], parts[1])
            # Convert the remaining columns using node2id
            node_ids = [node2id.get(node, node) for node in parts[2:]]
            fout.write(f"{edge_id} {' '.join(node_ids)}\n")

def process_test_files(test_dir, node2id_file, edge2id_file, output_dir):
    node2id = load_id_mapping(node2id_file)
    edge2id = load_id_mapping(edge2id_file)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(test_dir):
        if filename.startswith('test'):
            test_file = os.path.join(test_dir, filename)
            output_file = os.path.join(output_dir, filename)
            convert_test_file(test_file, node2id, edge2id, output_file)
            print(f"Processed {filename} and saved to {output_file}")



if __name__ == "__main__":
    # input_file = "graph/dblp/edgelist_filtered.txt"
    # network_file = "graph/dblp/network.edgelist"
    # train_file = "graph/dblp/train.edgelist"
    # valid_file = "graph/dblp/valid.edgelist"
    
    #split_dataset(input_file, network_file, train_file, valid_file)
    # write_data()
    # input_file = "graph/FB-AUTO/test.txt"
    # output_dir = "graph/FB-AUTO/"
    # process_hypergraph_files(input_file, output_dir)

    input_file = "link_prediction/graph/imdb/edgelist.txt"
    network_file = "link_prediction/graph/imdb/new_network.edgelist"
    valid_file = "link_prediction/graph/imdb/new_valid.edgelist"
    split_dataset(input_file, network_file, valid_file)

    # input_path = "graph/JF17K/processed_file/test_6.txt"
    # output_path = "graph/JF17K/final_file/valid_6.edgelist"
    # process_edgelist(input_path, output_path)   

    # test_path = "graph/FB-AUTO/test.txt"
    # node2id_path = "graph/FB-AUTO/node2id.txt"
    # output_path = "graph/FB-AUTO/test.edgelist"
    # convert_using_node2id(test_path, node2id_path, output_path)

    # edgelist_path = "graph/dblp/edgelist.txt"
    # stats = count_nodes_per_line(edgelist_path)

    # input_path = "graph/dblp/edgelist.txt"
    # output_path = "graph/dblp/edgelist_filtered.txt"
    # kept, filtered = filter_hyperedges(input_path, output_path, max_nodes=6)

    # Example usage
    # test_dir = "C:/Users/14522/Desktop/博士idea/zip/link_prediction/graph/JF17K/original_file"
    # node2id_file = "C:/Users/14522/Desktop/博士idea/zip/link_prediction/graph/JF17K/processed_file/node2id.txt"
    # edge2id_file = "C:/Users/14522/Desktop/博士idea/zip/link_prediction/graph/JF17K/processed_file/edge2id.txt"
    # output_dir = "C:/Users/14522/Desktop/博士idea/zip/link_prediction/graph/JF17K/processed_file"

    # process_test_files(test_dir, node2id_file, edge2id_file,output_dir)
