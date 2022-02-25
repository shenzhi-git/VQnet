# import paddle
# import numpy as np
# a = np.random.randint(4,size = 36)
# a = a.reshape([4,3,3])
# print(a)
# # the axis is a int element
# x = paddle.to_tensor(a,stop_gradient=False)

# result1 = paddle.max(x,axis = 1)
# print(result1)
# result1.backward()
# print(x.grad)

from pyvqnet.tensor import tensor
from pyvqnet.tensor import QTensor
import numpy as np 
t = tensor.arange(2, 30, 0.05)
print(t)
print('======================')
print(np.arange(2, 30, 0.05))