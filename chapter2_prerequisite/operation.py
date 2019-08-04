from mxnet import nd
x = nd.arange(12)
x1 = x.reshape((3, 4))
x2 = nd.arange(4)
print("x1:", x1)
print("x1.sum():", x1.sum().asscalar())
print("x2:", x2)
x3 = x1 * x2
print("x3:", x3)
print("x2.exp: ", x2.exp())
print("x1 dot x1:", nd.dot(x1, x1.T))
y1 = nd.array([1, 2, 4, 3, 9])
y2 = nd.array([2, 2, 3, 4, 9])
flag = y1 < 6
print('y1 < 6', flag.asnumpy())