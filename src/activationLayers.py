import torch
import torch.nn as nn

class TrainableTanh(nn.Module):
    def __init__(self):
        super(TrainableTanh, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  

    def forward(self, x):
        return torch.tanh(self.beta * x)

_modv = lambda a,b,c : 2/torch.pi*(torch.arctan(torch.sqrt(a**2+b**2+c**2))/torch.sqrt(a**2+b**2+c**2))

# TODO: This needs to be fixed and made more efficient
class FinalActivation(nn.Module):
    def __init__(self):
        super(FinalActivation, self).__init__()
    @staticmethod
    def _finalActivationHelper(inputs=None):

        if inputs is None:
            return 7
        rhoprime, a, b, c, B_y, B_z, pprime = [inputs[:, i] for i in range(7)]
        ans = torch.stack((
            torch.exp(rhoprime), 
            a*_modv(a,b,c), 
            b*_modv(a,b,c), 
            c*_modv(a,b,c),
            B_y,
            B_z,
            torch.exp(pprime), 
        )).t()
        return ans  # Define your functions here

    def forward(self, unconstrained_primitives):
        # Ensure input tensor is in the right shape and the number of functions matches the number of elements in the tensor

        #if unconstrained_primitives[1] != 7:
        #    print(unconstrained_primitives.numel())
        #    print(unconstrained_primitives.shape)
        #    raise ValueError("The number of functions must match the number of elements in the tensor {}".format(unconstrained_primitives.numel()))

        # Apply each function to its corresponding element
        #output = torch.empty_like(unconstrained_primitives)
        return self._finalActivationHelper(inputs=unconstrained_primitives)

