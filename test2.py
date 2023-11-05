import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist


def cosine_distance(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    # distance = cdist(x, y, 'cosine')
    distance = F.cosine_similarity(x, y, dim=2)
    return distance


a = torch.rand(300, 2048)
b = torch.rand(60, 2048)


# s = cosine_distance(a, b)
# print(s, s.size())


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


d = cosine_distance(a, b)
# print(d, d.size())
s = euclidean_dist(a, b)
# d = cdist(a, b, 'mahalanobis')
print('d:', d.size(), d)
print('s:', s.size(), s)
