from mxnet import nd, autograd

x = nd.arange(4).reshape([4, 1])
print(x)
x.attach_grad()
print("before:")
print(autograd.is_recording())
print(autograd.is_training())
print("ing:")
with autograd.record():
    print(autograd.is_recording())
    print(autograd.is_training())
    y = 2 * nd.dot(x.T, x)
print("after:")
print(autograd.is_recording())
print(autograd.is_training())
y.backward()
print('x.grad', x.grad)
print(x.grad == 4*x)

