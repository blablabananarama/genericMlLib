
class Unit:
    def zero_grad(self):
        for p in self.params():
            p.grad = 0
            
    def params(self):
        return []

class Neuron(Unit):
    def __init__(self, n_inputs, activation_fun):
        # weight initialization
        self.weights = [Var(random.uniform(-1,1)) for i in range(n_inputs)]
        self.bias = Var(0)
        self.activation_fun = activation_fun
    
    def params(self):
        return self.weights + [self.bias]
    
    def __call__(self, inputs):
        weighted_sum = sum((weight_i * input_i for weight_i, input_i in zip(self.weights, inputs))) + self.bias
        activation = weighted_sum.relu() if self.activation_fun == "relu" else weighted_sum.sigmoid()
        return activation
    
    def __repr__(self):
        return f"relu Neuron with {len(self.weights)} inputs"
        
class Layer(Unit):
    def __init__(self, n_neurons, n_inputs, activation_fun="sigmoid"):
        self.neurons = [Neuron(n_inputs, activation_fun) for i in range(n_neurons)]
    
    def __repr__(self):
        return self.neurons.__repr__()

    def params(self):
        return [p for n in self.neurons for p in n.params()]
    
class NeuralNetwork(Unit):
    def __init__(self):
        self.layers = [Layer(2,4),Layer(4,2)]
    
    def forward(self, inputs):
        result = inputs
        for i in self.layers:
            result = [j(result) for j in i.neurons]
        return result
        
    def params(self):
        return [p for layer in self.layers for p in layer.params()]
        
    def __repr__(self):
        return ",".join([i.__repr__() for i in self.layers])
    

    

def regularization(alpha, params, norm="l2"):
    if norm == "l2":
        reg_val = Var(sum([param.value**2 for param in params]))
    else:
        reg_val = Var(math.sqrt(sum([param.value**2 for param in params])))
    return reg_val
    
def loss(prediction, ground_truth):
    intermediate = [(i - j)**2 for i, j in zip(prediction, ground_truth)]
    temp = Var(0)
    for i in intermediate:
        temp = temp + i
    return temp
