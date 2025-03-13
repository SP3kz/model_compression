import torch

def curvature_backprop(model, loss_fn, optimizer, data, labels, beta=0.9):
    model.zero_grad()
    output = model(data)
    loss = loss_fn(output, labels)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    with torch.no_grad():
        for param, grad in zip(model.parameters(), grads):
            param_update = beta * grad + (1 - beta) * param.grad
            param -= param_update

    return loss.item()
