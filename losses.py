import torch
from torch import nn


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine

class CombinedMargin(nn.Module):
    def __init__(self, s=64.0, m2=0.3, m3=0.2):
        super(CombinedMargin, self).__init__()
        self.s = s
        self.m2 = m2
        self.m3 = m3

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot2 = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot2.scatter_(1, label[index, None], self.m2)
        m_hot3 = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot3.scatter_(1, label[index, None], self.m3)
        cosine.acos_()
        cosine[index] += m_hot2
        cosine.cos_()
        cosine[index] -= m_hot3
        ret = cosine * self.s
        return ret
