"""
Step7 — 知识图谱质量评估（文献来源说明）
=======================================

本评估模块参考并简化实现了以下权威文献提出的知识图谱（Knowledge Graph, KG）
质量评估框架：

【主要来源】
Zaveri, A. et al. (2016). 
"A Practical Framework for Evaluating the Quality of Knowledge Graphs."
Journal of Data and Information Quality (JDIQ), 7(3). 
DOI: 10.1145/2723576

该框架将 KG 质量分为多个维度，其中工程界最常使用的是：
    1. Accuracy（准确性）
    2. Completeness（完备性）
    3. Consistency（⼀致性）
    4. Conciseness（简洁性）
并提出了一系列可落地的结构与语义指标。

本模块所实现的指标对应如下：

1) Accuracy（准确性）
    - 文献定义：三元组 correctness / trustworthiness。
    - 本实现采用 triple 的抽取 confidence 作为非标注场景下的近似指标。

2) Completeness（结构完备性）
    - 文献建议使用 graph-structural proxies：
        * average degree（平均度数）
        * clustering coefficient（聚类系数）
        * size of largest connected component（最大连通块比例）
    - 本实现采用上述全部指标作为完备性的结构估计。

3) Consistency（⼀致性）
    - 文献定义：KG 内部不应有逻辑矛盾或冲突。
      包括重复事实、非法自环、schema 冲突等。
    - 本实现采用：
        * duplicate edge ratio（重复边率）
        * self-loop ratio（自环率）
      作为一致性的简化指标。

4) Conciseness（简洁性）
    - 文献要求 KG 遵循 Minimality 原则，不冗余、不重复。
    - 本实现采用：
        * conciseness = 1 - duplicate_ratio

本模块旨在提供一个工程可用、与经典 KG 质量框架对齐的轻量级评估方案，
适用于文档抽取型 KG、初构型 KG 的质量巡检与比较。

"""