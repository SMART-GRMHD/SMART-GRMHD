import torch
from torch import nn


class ResPINN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden=5, hidden_width=100, activation=nn.Tanh()):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_dims = [input_size, *(n_hidden*[hidden_width]), output_size]
        for i, j in zip(layer_dims, layer_dims[1:]):
            self.layers.append(nn.Linear(i, j))
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.layers[0](x))
        for layer in self.layers[1:-1]:
            out = self.activation(layer(out) + out)
        out = self.activation(self.layers[-1](out))
        return out


def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)


def init_glorot_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=2/(module.in_features + module.out_features))


def get_model(input_size, output_size, n_hidden=5, hidden_width=100, activation=nn.Tanh(), res=False):
    if res: 
        model = ResPINN(input_size, output_size, n_hidden, hidden_width, activation)
    else: 
        layers = nn.ModuleList()
        layer_dims = [input_size, *(n_hidden*[hidden_width])]
        for i, j in zip(layer_dims, layer_dims[1:]):
            layers.append(nn.Linear(i, j))
            layers.append(activation)
        layers.append(nn.Linear(layer_dims[-1], output_size))

        model = nn.Sequential(*layers)
        # model.apply(init_xavier)
        model.apply(init_glorot_normal)
    
    return model


def get_grads(model): 
    return [p.grad for p in model.parameters()]

def set_grads(model, grads): 
    for p, g in zip(model.parameters(), grads): 
        p.grad = g

def grad_dot(grads1, grads2): 
    return torch.dot(
        torch.cat([t.flatten() for t in grads1]),
        torch.cat([t.flatten() for t in grads2]),
    )


def calc_dpm_grad(overall_grads, domain_grads, delta): 
    numerator = -grad_dot(overall_grads, domain_grads) + delta
    denominator = grad_dot(domain_grads, domain_grads)
    factor = numerator / denominator
    return [factor * t_f + t_o for (t_f, t_o) in zip(domain_grads, overall_grads)]