import torch
import numpy as np
import torch.nn.functional as F


# def euclidean_dist(x, y):
#     # x: N x D
#     # y: M x D
#     n = x.size(0)
#     m = y.size(0)
#     d = x.size(1)
#     n_shot = y.size(1)
#     assert d == y.size(2)  # 判断：如果d(x的列数)不等于的列数，报错，原来是size(1)
#
#     x = x.unsqueeze(1).expand(n, n_shot, d)  # 把zq即query扩展成300x10x2048
#
#     x = x.unsqueeze(1).expand(n, m, n_shot, d)
#     y = y.unsqueeze(0).expand(n, m, n_shot, d)
#     # 把x和y都扩展成300x60x10x2048的尺寸
#     return torch.reshape(torch.pow(x - y, 2).sum(3), (n, -1))

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


# torch.pow(x - y, 2)表示，对x-y的每个元素都求平方
# torch.sum(2)表示对tensor的第2维（即第三个维度）求和。
# unsqueeze可以增加一个纬度，但是维度的size只是1而已，而expand就可以将数据进行复制，将数据变为n


def cosine_distance(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    distance = F.cosine_similarity(x, y, dim=2)
    return distance
