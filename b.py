import torch as t
from torch.autograd import Variable as v
m = v(t.FloatTensor([[2, 3]]), requires_grad=True)
n = v(t.zeros(1, 2))
n[0, 0] = m[0, 0] ** 2
n[0, 1] = m[0, 1] ** 3
n.backward(t.FloatTensor([[1, 1]]))
print(m.grad.data)