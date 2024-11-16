from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List
from enum import Enum

# stupid wrapper because numpy array printing is stupid
def printarr(arr):
    print("[" + ", ".join(str(item) for item in arr.flatten()) + "]")

def union(dict1, dict2):
    combined_dict = {**dict1, **dict2}
    return dict(sorted(combined_dict.items()))

class Tape():

    VARIABLE_INDICES = 0

    input_nodes = {}
    output_node = None

    topological_order = None
    reverse_topological_order = None

    def __init__(self, inputs, output):
        self.input_nodes = inputs
        self.output_node = output
    
    def Const(value):
        const_node = Const(value)
        return Tape(dict(), const_node)
    
    def Var():
        var_node = Var(Tape.VARIABLE_INDICES)
        Tape.VARIABLE_INDICES += 1
        return Tape({var_node.input_index: var_node}, var_node)

    def __generate_topo_order_if_necessary(self):
        if self.topological_order == None:
            self.topological_order = []
            self.__rec_make_topo_order(self.output_node, set())
            self.reverse_topological_order = [t for t in self.topological_order]
            self.reverse_topological_order.reverse()

    def __rec_make_topo_order(self, to_search, seen_nodes):
        
        for source in to_search.sources:
            if source not in seen_nodes:
                seen_nodes.add(source)
                self.__rec_make_topo_order(source, seen_nodes)
            
        self.topological_order.append(to_search)

    def forwards(self, inputs) -> float:

        self.__generate_topo_order_if_necessary()

        for node in self.topological_order:
            node.run(inputs)

        return self.output_node.result
    
    def backwards_from_last_forwards(self):

        self.__generate_topo_order_if_necessary()

        for node in self.topological_order:
            node.clear_gradient()

        self.output_node.gradient = 1
        
        for node in self.reverse_topological_order:
            node.propagate_gradient()

        return np.array([inp.gradient for inp in self.input_nodes.values()])
    
    def check(self, x):

        v = np.random.randn(len(x))

        max_iters = 32
        h = np.zeros(max_iters)
        err0 = np.zeros(max_iters)
        err1 = np.zeros(max_iters)

        for i in range(max_iters):
            h[i] = 2**(-i) # halve our stepsize every time

            fv = np.array(self.forwards(x + h[i]*v))

            T0 = np.array(self.forwards(x))
            T0_grad = np.array(self.backwards_from_last_forwards())

            T1 = T0 + h[i]*np.matmul(T0_grad.transpose(), v)

            err0[i] = np.linalg.norm(fv - T0) # this error should be linear
            err1[i] = np.linalg.norm(fv - T1) # this error should be quadratic

            # print('h = ', h, ', err0 = ', err0[i], ', err1 = ', err1[i])

            #print('h: %.3e, \t err0: %.3e, \t err1: %.3e' % (h[i], err0[i], err1[i]))

        plt.loglog(h, err0, linewidth=3)
        plt.loglog(h, err1, linewidth=3)
        plt.legend(['$\|f(x) - T_0(x)\|$', '$\|f(x)-T_1(x)\|$'], fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

    def __add__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = Add(self.output_node, other.output_node)
        return Tape(new_inputs, new_output)
    
    def __sub__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = Add(self.output_node, Mult(Const(-1), other.output_node))
        return Tape(new_inputs, new_output)

    def __mul__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = Mult(self.output_node, other.output_node)
        return Tape(new_inputs, new_output)
    
    def __div__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = Div(self.output_node, other.output_node)
        return Tape(new_inputs, new_output)
    
    def tanh(self):
        new_output = Tanh(self.output_node)
        return Tape(self.input_nodes, new_output)

    def sigmoid(self):
        new_output = Sigmoid(self.output_node)
        return Tape(self.input_nodes, new_output)
    
    def relu(self):
        new_output = Relu(self.output_node)
        return Tape(self.input_nodes, new_output)
    
    def __str__(self) -> str:
        return str(self.output_node)

class AbstractOperationNode(ABC):

    sources = []
    listeners = []

    result: float
    gradient: float

    def __init__(self, *args) -> None:

        for source in args:
            source.listeners.append(source)

        self.sources = args
        self.listeners = []

        super().__init__()

    def run(self, inputs):
        self.result = self.inner_run(inputs)

    def clear_gradient(self):
        self.gradient = 0

    def find_variable_node_dependencies(self):
        variables = []
        searched = [self]
        to_search = deque()
        to_search.append(self)

        # if this fails to terminate, you probably have a dependency loop
        while len(to_search) > 0:
            next_search = to_search.popleft()
            
            if next_search.is_variable():
                variables.append(next_search)
            else:
                for source in next_search.sources:
                    if source not in searched:
                        to_search.append(source)
                        searched.append(source)

        variables.sort()
        return variables

    @abstractmethod
    def inner_run(self, inputs) -> float:
        pass

    @abstractmethod
    def propagate_gradient(self): # assume that grad is completely setup
        pass

    @abstractmethod
    def __str__(self) -> str:
        return ""
    
    def is_variable(self) -> bool:
        return False

class Const(AbstractOperationNode):

    c: float

    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def inner_run(self, inputs) -> float:
        return self.c

    def propagate_gradient(self): # assume that grad is completely setup
        pass
    
    def __str__(self) -> str:
        return str(self.c)

class Var(AbstractOperationNode):

    input_index: int

    def __init__(self, input_index) -> None:
        super().__init__()
        self.input_index = input_index

    def inner_run(self, inputs) -> float:
        return inputs[self.input_index]

    def propagate_gradient(self): # assume that grad is completely setup
        pass
    
    def __str__(self) -> str:
        return "[" + str(self.input_index) + "]"
    
    def __lt__(self, other):
        return self.input_index < other.input_index
    
    def is_variable(self) -> bool:
        return True

class Add(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode, b: AbstractOperationNode):
        super().__init__(a, b)

    def inner_run(self, inputs) -> float:
        return self.sources[0].result + self.sources[1].result

    def propagate_gradient(self): # assume that grad is completely setup
        self.sources[0].gradient += self.gradient
        self.sources[1].gradient += self.gradient

    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + " + " + str(self.sources[1]) + ")"

class Mult(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode, b: AbstractOperationNode):
        super().__init__(a, b)

    def inner_run(self, inputs) -> float:
        return self.sources[0].result * self.sources[1].result

    def propagate_gradient(self): # assume that grad is completely setup
        self.sources[0].gradient += self.sources[1].result * self.gradient
        self.sources[1].gradient += self.sources[0].result * self.gradient
    
    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + " * " + str(self.sources[1]) + ")"
        
class Div(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode, b: AbstractOperationNode):
        super().__init__(a, b)

    def inner_run(self, inputs) -> float:
        return self.sources[0].result / self.sources[1].result

    def propagate_gradient(self): # assume that grad is completely setup
        self.sources[0].gradient += self.gradient / self.sources[1].result
        self.sources[1].gradient += (-self.gradient * self.sources[0].result / (self.sources[1].result * self.sources[1].result))
    
    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + " / " + str(self.sources[1]) + ")"
    
class Tanh(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode):
        super().__init__(a)

    def inner_run(self, inputs) -> float:
        return np.tanh(self.sources[0].result)
    
    def propagate_gradient(self): # assume that grad is completely setup
        self.sources[0].gradient += (1 - np.tanh(self.sources[0].result)**2) * self.gradient
    
    def __str__(self) -> str:
        return "tanh(" + str(self.sources[0]) + ")"
    
class Sigmoid(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode):
        super().__init__(a)

    @classmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def inner_run(self, inputs) -> float:
        return Sigmoid.sigmoid(self.sources[0].result)
    
    def propagate_gradient(self): # assume that grad is completely setup
        x = self.sources[0].result
        self.sources[0].gradient += Sigmoid.sigmoid(x) * (1 - Sigmoid.sigmoid(x)) * self.gradient
    
    def __str__(self) -> str:
        return "sigmoid(" + str(self.sources[0]) + ")"

class Relu(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode):
        super().__init__(a)

    def inner_run(self, inputs) -> float:
        return np.maximum(0, self.sources[0].result)
    
    def propagate_gradient(self): # assume that grad is completely setup
        self.sources[0].gradient += np.where(self.sources[0].result >= 0, 1, 0) * self.gradient
    
    def __str__(self) -> str:
        return "relu(" + str(self.sources[0]) + ")"

# some other helper functions
def make_var_matrix(n, m):
    return np.array([[Tape.Var() for _ in range(m)] for _ in range(n)])

def make_var_array(n):
    return np.array([Tape.Var()for _ in range(n)])

def to_const_matrix(arr):
    return np.array([[Tape.Const(col) for col in row] for row in arr])

def all_forwards(arr, inputs):
    return [x.forwards(inputs) for x in arr]

def all_backwards(arr):
    return np.array([x.backwards_from_last_forwards() for x in arr])

def all_tanh(arr):
    return np.array([x.tanh() for x in arr])

def all_sigmoid(arr):
    return np.array([x.sigmoid() for x in arr])

def all_relu(arr):
    return np.array([x.relu() for x in arr])

def all_mult(arr, multiplier):
    print(arr)
    multiplier = Tape.Const(multiplier)
    return np.array([x * multiplier for x in arr])

def all_subs(arr1, arr2):
    return np.array([arr1[r] - arr2[r] for r in range(len(arr1))])

def all_square(arr):
    return np.array([x * x for x in arr])

def all_sum(arr):
    acc = Tape.Const(0)
    for x in arr:
        acc = acc + x
    return acc

class AbstractNetworkLayer:

    @abstractmethod
    def forwards(self, inputs: List[Tape]) -> List[Tape]:
        return inputs
    
    @abstractmethod
    def initialize(self, parameter_count: int) -> List[float]:
        return np.zeros(parameter_count)

class ActivationFunction(Enum):
    TANH = 0
    SIGMOID = 1
    RELU = 2

class InitializationType(Enum):
    NORMAL = 0
    UNIFORM = 1

class SimpleLayer(AbstractNetworkLayer):

    output_vars: int
    activation_type: ActivationFunction
    init_type: InitializationType

    def __init__(self, output_vars: int, activation_type: ActivationFunction = ActivationFunction.TANH, init_type: InitializationType = InitializationType.UNIFORM) -> None:
        super().__init__()
        self.output_vars = output_vars
        self.activation_type = activation_type
        self.init_type = init_type

    def forwards(self, inputs: List[Tape]) -> List[Tape]:

        new_weights = make_var_matrix(self.output_vars, len(inputs))
        new_biases = make_var_matrix(1, self.output_vars)[0]

        ret = new_weights @ inputs + new_biases

        if self.activation_type == ActivationFunction.TANH:
            ret = all_tanh(ret)
        elif self.activation_type == ActivationFunction.SIGMOID:
            ret = all_sigmoid(ret)
        elif self.activation_type == ActivationFunction.RELU:
            ret = all_relu(ret)

        return ret
    
    def initialize(self, parameter_count: int) -> List[float]:
        if self.init_type == InitializationType.UNIFORM:
            return np.random.rand(parameter_count)
        else:
            return np.random.randn(parameter_count)

class NeuralNetwork:

    input_count: int
    layers: List[AbstractNetworkLayer]
    learning_rate: float

    layer_parameter_sizes: List[int]
    total_parameters: int
    output_count: int

    input_layer = []
    output_layer = []
    expected_outputs = []
    cost = None

    current_parameters = []

    def __init__(self, input_count: int, layers: List[AbstractNetworkLayer], learning_rate: float) -> None:
        self.input_count = input_count
        self.layers = layers
        self.learning_rate = learning_rate

        # this is jank, we should find a better way of doing this
        Tape.VARIABLE_INDICES = 0
        old_variable_indices = 0
        self.layer_parameter_sizes = []

        tapes = np.array([Tape.Const(0) for _ in range(self.input_count)])
        self.input_layer = tapes

        for layer in self.layers:
            tapes = layer.forwards(tapes)

            self.layer_parameter_sizes.append(Tape.VARIABLE_INDICES - old_variable_indices)
            old_variable_indices = Tape.VARIABLE_INDICES

        self.output_layer = tapes
        self.total_parameters = sum(self.layer_parameter_sizes)
        self.output_count = len(tapes)

        self.expected_outputs = np.array([Tape.Const(0) for _ in range(self.output_count)])
        self.cost = all_sum(all_square(all_subs(tapes, self.expected_outputs))) * Tape.Const(0.5)

        self.reset_parameters()

    def reset_parameters(self):
        self.current_parameters = []
        for i in range(len(self.layers)):
            self.current_parameters.extend(self.layers[i].initialize(self.layer_parameter_sizes[i]))

    def set_inputs(self, inputs: List[float]):
        for i in range(len(self.input_layer)):
            self.input_layer[i].output_node.c = inputs[i]
    
    def set_expected_outputs(self, expected_outputs: List[float]):
        for i in range(len(self.expected_outputs)):
            self.expected_outputs[i].output_node.c = expected_outputs[i]

    def forwards(self, inputs: List[float]) -> List[float]:
        self.set_inputs(inputs)
        return all_forwards(self.output_layer, self.current_parameters)

    def get_cost(self, inputs: List[int], expected_output: List[int]) -> float:
        self.set_inputs(inputs)
        self.set_expected_outputs(expected_output)
        return self.cost.forwards(self.current_parameters)

    def train(self, inputs: List[int], expected_outputs: List[int]):

        self.get_cost(inputs, expected_outputs)
        gradients = self.cost.backwards_from_last_forwards()
        self.current_parameters -= self.learning_rate * gradients
    
    def train_batch(self, inputs: List[List[float]], expected_outputs: List[List[float]]):
        batch_size = len(inputs)
        
        total_gradients = 0
        for i in range(0, batch_size):

            self.get_cost(inputs[i], expected_outputs[i])
            gradients = self.cost.backwards_from_last_forwards()
            total_gradients += gradients

        self.current_parameters -= self.learning_rate * total_gradients / batch_size
        
        
