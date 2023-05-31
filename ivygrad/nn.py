import random
from ivygrad.engine import Value
#  import ivygrad as ig
from typing import List

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1, 1), label=f'w{i}') for i in range(nin)]
        self.b = Value(np.random.uniform(-1, 1), label='b')
        
    def __call__(self, x):
        out = sum(xi*wi for xi, wi in zip(x, self.w)) + self.b
        activated = out.tanh()
        return activated
        
    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
class Layer(Module):
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        
    def __call__(self, x):
        activateds = [n(x) for n in self.neurons]
        return activateds[0] if len(activateds) == 1 else activateds
    
    def parameters(self):
        params = [] 
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params
    

class MLP(Module):

    def __init__(self, nin: int, nouts: List):
        sizes = [nin] + nouts
        self.layers = []
        
        for i in range(len(nouts)):
            layer = Layer(sizes[i], sizes[i + 1])
            self.layers.append(layer)
    
    def __call__(self, x):
        for layer in self.layers: 
            x = layer(x)
        return x
    
    def parameters(self):
        params = [] 
        for layer in self.layers:
            ps = layer.parameters()
            params.extend(ps)
        return params 
