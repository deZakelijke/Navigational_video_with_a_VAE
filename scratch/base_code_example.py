
import XXX
from XXX import Variable, grad_dict, rand
model = XXX.nn.Linear(1, 1)
loss_fn = XXX.nn.MSELoss(size_average=False)
optimizer = XXX.optim.SGD(lr=0.01)
for t in range(10000):
|   x = Variable(rand((1,1)), XXX.float32)
|   y = x * 3
|   y_pred = model(x)
|   loss = loss_fn(y_pred, y)
|   param_grads = XXX.autograd.grad_dict(outputs=loss, inputs=model.parameters())
|   optimizer.update_(param_grads)
|   print(loss.data[0])
