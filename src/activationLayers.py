import torch
import torch.nn as nn

class TrainableTanh(nn.Module):
    def __init__(self):
        super(TrainableTanh, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  

    def forward(self, x):
        return torch.tanh(self.beta * x)

_modv = lambda a,b : 2/torch.pi*(torch.arctan(torch.sqrt(a**2+b**2))/torch.sqrt(a**2+b**2))

class FinalActivation(nn.Module):
    def __init__(self):
        super(FinalActivation, self).__init__()
    @staticmethod
    def _finalActivationHelper(inputs=None):

        if inputs is None:
            return 5
        rhoprime, pprime, a, b, B_y = inputs 
        return [
            torch.exp(rhoprime), 
            torch.exp(pprime), 
            a*_modv(a,b), 
            b*_modv(a,b), 
            B_y
        ]  # Define your functions here

    def forward(self, unconstrained_primitives):
        # Ensure input tensor is in the right shape and the number of functions matches the number of elements in the tensor
        if unconstrained_primitives.numel() != 5:
            print(unconstrained_primitives.numel())
            print(unconstrained_primitives.size())
            raise ValueError("The number of functions must match the number of elements in the tensor {}".format(unconstrained_primitives.numel()))

        # Apply each function to its corresponding element
        output = torch.empty_like(unconstrained_primitives)
        return self._finalActivationHelper(inputs=unconstrained_primitives)

