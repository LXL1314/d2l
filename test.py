from mxnet import nd

a = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
a = nd.array(a)
print("a.mean:", a.mean(axis=0))

print("nd.array([0]", nd.array([0]))

y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([2, 2], dtype='int32')
flag = (y_hat.argmax(axis=1) == y.astype('float32'))
print(flag.mean())
a = nd.array([0.1, 0.01])
print(nd.log(nd.array([0.1])))
print('log:', a.log())

y_hat = nd.ones([5, 1])*2
y = nd.ones([5,])
print("y_hat", y_hat)
print("y ", y)
h = y_hat - y
print("y_hat - y", h)
print("h.sum()", h.sum())

a = nd.array([[1, 2], [3, 4]])
b = a.copy()
print(a * b)


