from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass


class Fro(Regularizer):
    def __init__(self, weight: float):
        super(Fro, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.norm(f, 2) ** 2
                )
        return norm / factors[0][0].shape[0]
class F2(Regularizer):
    def __init__(self, weight: float):
        super(F2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(f ** 2)
        return norm / factors[0].shape[0]

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 3
                ) / f.shape[0]
        return norm


class ER_abs(Regularizer):
    def __init__(self, weight: float):
        super(ER_abs, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        a = 1;
        b = 1  # can be adjusted
        rate = 0.455
        scale = 0.6  # can be adopted from{0.5-1.5}

        for factor in factors:
            h, r, t = factor
            norm += rate * torch.sum(t ** 2 + h ** 2)
            t = torch.abs(t);
            h = torch.abs(h);
            r = torch.abs(r)
            if a == b:
                norm += scale * torch.sum(
                    a * h ** 2 * r ** 2 + a * t ** 2 * r ** 2 + b * h ** 2 * r ** 2 + b * t ** 2 * r ** 2)
            else:
                norm += scale * torch.sum(
                    a * h ** 2 * r ** 2 + a * 2 * h * r * t * r + a * t ** 2 * r ** 2 + b * h ** 2 * r ** 2 - b * 2 * h * r * t * r + b * t ** 2 * r ** 2)
        return self.weight * norm / h.shape[0]


class ER(Regularizer):
    def __init__(self, weight: float):
        super(ER, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        a = 1;
        b = 1  # can be adjusted
        rate = 0.455  # can be adopted from{0.5-1.5}  0.5
        scale = 0.6  # can be adopted from{0.5-1.5}  0.75

        for factor in factors:
            h, r, t = factor
            norm += rate * torch.sum(t ** 2 + h ** 2)
            # t=torch.abs(t);h=torch.abs(h);r=torch.abs(r)
            if a == b:
                norm += scale * torch.sum(
                    a * h ** 2 * r ** 2 + a * t ** 2 * r ** 2 + b * h ** 2 * r ** 2 + b * t ** 2 * r ** 2)
            else:
                norm += scale * torch.sum(
                    a * h ** 2 * r ** 2 + a * 2 * h * r * t * r + a * t ** 2 * r ** 2 + b * h ** 2 * r ** 2 - b * 2 * h * r * t * r + b * t ** 2 * r ** 2)
        return self.weight * norm / h.shape[0]


class ER1(Regularizer):
    def __init__(self, weight: float):
        super(ER_W1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        a = 1;
        b = 1  # can be adjusted
        rate = 0.455  # can be adopted from{0.5-1.5}
        scale = 0.6  # can be adopted from{0.5-1.5}
        for factor in factors:
            h, r, t = factor
            norm += rate * torch.sum(t ** 3 + h ** 3)
            if a == b:
                norm += scale * torch.sum(
                    a * h ** 2 * r ** 2 + a * t ** 2 * r ** 2 + b * h ** 2 * r ** 2 + b * t ** 2 * r ** 2)
            else:
                norm += scale * torch.sum(
                    a * h ** 2 * r ** 2 + a * 2 * h * r * t * r + a * t ** 2 * r ** 2 + b * h ** 2 * r ** 2 - b * 2 * h * r * t * r + b * t ** 2 * r ** 2)
        return self.weight * norm / h.shape[0]


class ER2(Regularizer):
    def __init__(self, weight: float):
        super(ER_W2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        a = 1;
        b = 1  # can be adjusted
        rate = 0.455  # can be adopted from{0.5-1.5}
        scale = 0.6  # can be adopted from{0.5-1.5}
        for factor in factors:
            h, r, t = factor
            norm += rate * torch.sum(torch.abs(t) ** 3 + torch.abs(h) ** 3)
            if a == b:
                norm += scale * torch.sum(
                    a * h ** 2 * r ** 2 + a * t ** 2 * r ** 2 + b * h ** 2 * r ** 2 + b * t ** 2 * r ** 2)
            else:
                norm += scale * torch.sum(
                    a * h ** 2 * r ** 2 + a * 2 * h * r * t * r + a * t ** 2 * r ** 2 + b * h ** 2 * r ** 2 - b * 2 * h * r * t * r + b * t ** 2 * r ** 2)
        return self.weight * norm / h.shape[0]


class ER_RESCAL(Regularizer):
    def __init__(self, weight: float):
        super(ER_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        a = 1;
        b = 1  # can be adjusted. e.g., a=1 b=1.02
        rate = 1  # can be adopted from{0-1.5}  1  1.05
        scale = 0.5  # can be adopted from{0-1.5}  0.5  0.455
        for factor in factors:
            h, r, t = factor
            norm += rate * torch.sum(h ** 2 + t ** 2)
            if a == b:
                norm += scale * torch.sum(
                    a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + a * torch.bmm(r, t.unsqueeze(
                        -1)) ** 2 + b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + b * torch.bmm(r,
                                                                                                           t.unsqueeze(
                                                                                                               -1)) ** 2)
            else:
                norm += scale * torch.sum(
                    a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + 2 * a * torch.bmm(r.transpose(1, 2),
                                                                                               h.unsqueeze(
                                                                                                   -1)) * torch.bmm(r,
                                                                                                                    t.unsqueeze(
                                                                                                                        -1)) + a * torch.bmm(
                        r, t.unsqueeze(-1)) ** 2 + b * torch.bmm(r.transpose(1, 2),
                                                                 h.unsqueeze(-1)) ** 2 - 2 * b * torch.bmm(
                        r.transpose(1, 2), h.unsqueeze(-1)) * torch.bmm(r, t.unsqueeze(-1)) + b * torch.bmm(r,
                                                                                                            t.unsqueeze(
                                                                                                                -1)) ** 2)
        return self.weight * norm / h.shape[0]


class ER_RESCAL1(Regularizer):
    def __init__(self, weight: float):
        super(ER_RESCAL1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        a = 1;
        b = 1.02  # can be adjusted from {0-2}. e.g., a=1 b=1.02
        rate = 1  # can be adopted from{0-1.5}  1
        scale = 0.5  # can be adopted from{0-1.5}  0.5
        for factor in factors:
            h, r, t = factor
            norm += torch.sum(h ** 3 + t ** 3)
            if a == b:
                norm += scale * torch.sum(
                    a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + a * torch.bmm(r, t.unsqueeze(
                        -1)) ** 2 + b * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + b * torch.bmm(r,
                                                                                                           t.unsqueeze(
                                                                                                               -1)) ** 2)
            else:
                norm += scale * torch.sum(
                    a * torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + 2 * a * torch.bmm(r.transpose(1, 2),
                                                                                               h.unsqueeze(
                                                                                                   -1)) * torch.bmm(r,
                                                                                                                    t.unsqueeze(
                                                                                                                        -1)) + a * torch.bmm(
                        r, t.unsqueeze(-1)) ** 2 + b * torch.bmm(r.transpose(1, 2),
                                                                 h.unsqueeze(-1)) ** 2 - 2 * b * torch.bmm(
                        r.transpose(1, 2), h.unsqueeze(-1)) * torch.bmm(r, t.unsqueeze(-1)) + b * torch.bmm(r,
                                                                                                            t.unsqueeze(
                                                                                                                -1)) ** 2)
        return self.weight * norm / h.shape[0]

class DURA(Regularizer):
    def __init__(self, weight: float):
        super(DURA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0

        for factor in factors:
            h, r, t = factor

            norm += torch.sum(t**2 + h**2)
            norm += torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]


class DURA_RESCAL(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += torch.sum(h ** 2 + t ** 2)
            norm += torch.sum(
                torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)
        return self.weight * norm / h.shape[0]


class DURA_RESCAL_W(Regularizer):
    def __init__(self, weight: float):
        super(DURA_RESCAL_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor
            norm += 2.0 * torch.sum(h ** 2 + t ** 2)
            norm += 0.5 * torch.sum(
                torch.bmm(r.transpose(1, 2), h.unsqueeze(-1)) ** 2 + torch.bmm(r, t.unsqueeze(-1)) ** 2)
        return self.weight * norm / h.shape[0]


class DURA_W(Regularizer):
    def __init__(self, weight: float):
        super(DURA_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor

            norm += 0.5 * torch.sum(t**2 + h**2)
            norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]


class L2(Regularizer):
    def __init__(self, weight: float):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 2
                )
        return norm / factors[0][0].shape[0]


class L1(Regularizer):
    def __init__(self, weight: float):
        super(L1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f)**1
                )
        return norm / factors[0][0].shape[0]


class NA(Regularizer):
    def __init__(self, weight: float):
        super(NA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        return torch.Tensor([0.0]).cuda()





