import numpy as np
import torch
from sklearn.cluster import SpectralClustering

def find(node, colors, graph):
    """检查是否可以为当前节点分配某个颜色"""
    st = set([colors[neighbor] for neighbor in graph[node]])
    if len(st) == 0:
        return 1
    for color in range(1,max(st)+2):
        if color not in st:
            return color

def dfs(node, color, graph):
    """递归分配颜色"""
    if color[node] != 0:
        return
    color[node] = find(node, color, graph)
    for child in graph[node]:
        dfs(child, color, graph)

def verify(n, K, constraints):
    """
    n: 人数
    k: 组数
    m: 关系数
    constraints: 列表，每个元素是 (i, j)，表示 i 和 j 不能在同一组
    """
    # 构建图
    graph = {i: [] for i in range(n)}
    for i, j, k in constraints:
        graph[i].append(j)
        graph[j].append(i)

    # 用来存储每个节点的颜色
    colors = [0] * n  # 初始未分配颜色
    # 尝试分配
    for i in range(n):
        dfs(i, colors, graph)

    return max(colors) <= K

def spectral(n, K, dist):
    if K >= n:
        return np.array([i for i in range(0, n)])
    edges = []
    dist = np.array(dist)
    for i in range(0, n):
        for j in range(i+1, n):
            edges.append((i, j, dist[i, j].item()))
            edges.append((j, i, dist[i, j].item()))
    edges.sort(key=lambda x: x[2])
    l = 0
    r = len(edges)
    ans = None
    while l <= r:
        mid = (l + r) >> 1
        done = verify(n, K, edges[-mid:])
        if done:
            ans = mid
            l = mid + 1
        else:
            r = mid - 1

    # for i in range(1, ans+1):
    #     u, v, w = edges[-i]
    #     dist[u, v] = dist[v, u] = 0

    sigma = np.median(dist[dist > 0])
    if np.isnan(sigma) or sigma == 0:
        affinity_matrix = np.ones_like(dist)
    else:
        affinity_matrix = np.exp(-dist ** 2 / (2 * sigma ** 2))

    # for i in range(1, ans+1):
    #     u, v, w = edges[-i]
    #     affinity_matrix[u, v] = affinity_matrix[v, u] = 0

    for i in range(n):
        affinity_matrix[i, i] = 1
    np.set_printoptions(2)
    # print(affinity_matrix)
    clustering = SpectralClustering(n_clusters=K, assign_labels='discretize', affinity='precomputed', random_state=0)

    labels = clustering.fit_predict(affinity_matrix)
    return labels


if __name__ == '__main__':
    # dis = torch.Tensor([[0.0000, 0.0137, 0.0155, 0.0671, 0.0066],
    #  [0.0137, 0.0000, 0.0191, 0.0658, 0.0071],
    #  [0.0155, 0.0191, 0.0000, 0.0506, 0.0209],
    #  [0.0671, 0.0658, 0.0506, 0.0000, 0.1199],
    #  [0.0066, 0.0071, 0.0209, 0.1199, 0.0000]])
    dis = torch.tensor([[0.00, 0.22, 0.15, 0.02, 0.03, 0.12],
        [0.22, 0.00, 0.01, 0.23, 0.14, 0.07],
        [0.15, 0.01, 0.00, 0.19, 0.18, 0.09],
        [0.02, 0.23, 0.19, 0.00, 0.02, 0.11],
        [0.03, 0.14, 0.18, 0.02, 0.00, 0.03],
        [0.12, 0.07, 0.09, 0.11, 0.03, 0.00]])

    print(type(spectral(6,2,dis)))
