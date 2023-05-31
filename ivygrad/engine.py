class Value:
    """
    This class will be used to store the value of the node in the neural network
    """
    
    def __init__(self, data, _children=(), operation="", label=""):
        self.data = data
        self._prev = set(_children)
        self._operation = operation
        self._label = label
        
        # we assume that the value doesn't affect the 
        # loss function by default, so the slope is 0.0
        self.grad = 0.0 
        
        # This will store a function which will calculate the 
        # local derivative according to the operation and store
        # in the `self.grad` variable.
        self._backward = lambda : None
        
    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Value(t, (self,), "tanh")
        
        def _backward():
            self.grad += (1 - t ** 2) * out.grad # chain rule here too!
        out._backward = _backward
        return out
    
    def relu(self):
        x = self.data
        out = Value(0 if x < 0 else x, (self,), 'ReLU')
        return out
    
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

            
    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    
    def __truediv__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out =  Value(self.data / other.data, (self, other), '/')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    
    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    
    def __sub__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.data - other.data, (self, other), '-')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return self - other
    
    def __rtruediv__(self, other):
        return self / other
    
    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad # chain rule
        out._backward = _backward
        return out
        
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only int/float can be used"
        out = Value(self.data ** other, (self, ), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad # chain rule
        out._backward = _backward
        return out
    
    def __repr__(self):
        return f"Value(data={self.data})"