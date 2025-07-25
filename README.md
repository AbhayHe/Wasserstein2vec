## 项目简介
**Wasserstein2vec** 是论文《Wasserstein2vec: Hypergraph Embedding for Link Prediction based on Wasserstein Distance》的官方实现。这是一种新颖的超图嵌入框架，旨在解决超图表示学习中难以充分捕捉节点邻居分布特征的挑战。

## 核心特点
- 采用Wasserstein距离，基于超节点的局部和全局网络特征分布（如度、边权重、聚类系数）衡量节点相似度
- 有效保留节点间的一阶和二阶邻近性
- 适用于复杂系统中的超图结构表示学习

## 实验结果
- 在三个真实世界网络的链接预测任务中，性能优于现有SOTA基线方法
- 在投资合作超图应用中，实现53.1%的前10推荐命中率，验证了方法的稳健性和可靠性，可支持企业战略投资决策

## 引用

## 联系方式
如有问题，请联系：[your.email@example.com]

## 许可证
本项目基于MIT许可证开源 - 详见[LICENSE](LICENSE)文件。
