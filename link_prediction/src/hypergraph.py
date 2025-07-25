class Hypergraph(object):
    def __init__(self,model_name):
        self._edges = {}
        self._nodes = {}
        self.ID = 0
        self.node_ID = 0
        self.max_edge_degree = 0
        self.name = "normal"
        self.clustering_coefficients = {} 
        self.model_name = model_name

    def nodes(self):
        return list(self._nodes.keys())

    def edges(self):
        es = []
        for e in self._edges.values():
            es.append(e['edge'])
        return es

    def add_edge(self, edge_name, edge, weight=1.):
        edge = tuple(sorted(edge))
        edge_dict = {}
        edge_dict['edge'] = edge
        edge_dict['weight'] = weight
        edge_dict['id'] = self.ID
        self.ID += 1
        self._edges[edge_name] = edge_dict
        self.max_edge_degree = max(len(edge), self.max_edge_degree)
        
        
        affected_nodes = set(edge)  # 将超边中的所有节点加入affected_nodes
        for v in edge:
            node_dict = self._nodes.get(v, {})
            edge_set = node_dict.get('edge', set())
            edge_set.add(edge_name)
            node_dict['edge'] = edge_set

            node_weight = node_dict.get('weight', 1.)
            node_dict['weight'] = node_weight

            node_degree = node_dict.get('degree', 0)
            node_degree += weight
            node_dict['degree'] = node_degree

            node_id = node_dict.get('id', -1)
            if node_id == -1:
                node_dict['id'] = self.node_ID
                self.node_ID += 1

            neighbors = node_dict.get('neighbors', set())
            for v0 in edge:
                if v0 != v:
                    neighbors.add(v0)
                    affected_nodes.add(v0)  # 同时也将邻居加入affected_nodes
            node_dict['neighbors'] = neighbors
            
            if 'clustering' not in node_dict:
                node_dict['clustering'] = 0.0
                
            self._nodes[v] = node_dict
        if self.model_name == 'WassersteinWalker':
            self._update_clustering_coefficients(affected_nodes)

    def neighbors(self, n):
        return self._nodes[n]['neighbors']

    #节点对应的所有超边
    def incident_edges(self, n):
        return self._nodes[n]['edge']
    #超边n 的所有节点
    def incident_nodes(self, e):
        return self._edges[e]['edge']

    def node_id(self, n):
        return self._nodes[n]['id']
    
    def _update_clustering_coefficients(self, nodes):
        for node in nodes:
            if node not in self._nodes:
                continue               
            neighbors = self.neighbors(node)
            if len(neighbors) < 2:
                self._nodes[node]['clustering'] = 0.0
                continue
            
            # 获取包含当前节ww点的所有超边
            node_edges = self.incident_edges(node)
            connected_pairs = set()
            
            # 对于每条包含当前节点的超边
            for edge_name in node_edges:
                # 获取超边中的所有节点
                edge_nodes = set(self.incident_nodes(edge_name))
                # 如果超边大小小于3，跳过（因为需要至少3个节点才能形成三角形）
                if len(edge_nodes) < 3:
                    continue
                    
                # 在当前超边中寻找邻居节点对
                for n1 in neighbors:
                    if n1 in edge_nodes:
                        for n2 in neighbors:
                            # 确保不重复计算节点对
                            if n1 < n2 and n2 in edge_nodes:
                                # 如果两个邻居节点都在同一个超边中，则它们形成一个连接对
                                connected_pairs.add((n1, n2))
            
            # 计算可能的总邻居对数量
            total_possible = len(neighbors) * (len(neighbors) - 1) / 2
            # 计算实际连接的邻居对数量
            connections = len(connected_pairs)
            # 更新聚类系数
            self._nodes[node]['clustering'] = connections / total_possible if total_possible > 0 else 0.0
    def get_node_clustering(self, node):
  
        return self._nodes[node]['clustering']
  
       
    
def output_dual_hypergraph(G, fdual):
    f = open(fdual, 'w', encoding='utf8')
    for v1 in G.nodes():
        f.write(str(v1))
        for e in G.incident_edges(v1):
            f.write(' ' + e)
        f.write('\n')
    f.close()


def get_Pr(G):
    from scipy.sparse import lil_matrix
    n = len(G.nodes())
    P = lil_matrix((n, n))

    for v1 in G.nodes():
        for e in G.incident_edges(v1):
            for v2 in G.incident_nodes(e):
                if v1 != v2:
                    P[G.node_id(v1), G.node_id(v2)] += G._edges[e]['weight'] / (
                                G._nodes[v1]['degree'] * len(G.incident_nodes(e)))
    return P
