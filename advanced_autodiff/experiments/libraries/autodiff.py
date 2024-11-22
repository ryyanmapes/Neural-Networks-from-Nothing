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
from typing import Type
from typing import Tuple
from collections import ChainMap
import scipy.sparse as sp
from numpy.lib.stride_tricks import as_strided

arrayf = Type[np.ndarray[float]]

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

    input_parameter_count = 0

    topological_order = None
    reverse_topological_order = None

    indices = []

    def __init__(self, inputs, output):
        self.input_nodes = inputs
        self.output_node = output

        self.input_parameter_count = sum(input_node.variable_count for input_node in self.input_nodes.values())
        self.indices = [None for _ in range(output.get_output_length())]
    
    def Const(value):
        const_node = Const(value)
        return Tape(dict(), const_node)
    
    def Var(size: Tuple[int, int]):
        var_node = Var(Tape.VARIABLE_INDICES, size)
        Tape.VARIABLE_INDICES += var_node.variable_count
        return Tape({var_node.input_start_index: var_node}, var_node)
    
    def Combine(tapes):
        new_inputs = dict(ChainMap(*[tape.input_nodes for tape in tapes]))
        new_output = Combine([tape.output_node for tape in tapes])
        return Tape(new_inputs, new_output)
    
    # todo this is inefficient
    def sum_all_variables(self, f = lambda x: x):
        acc = Tape.Const(np.array([[0]]))
        for input_node in self.input_nodes.values():
            acc = acc + f(Tape({input_node.input_start_index: input_node}, input_node)).sum()
        return acc

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
            node.clear_grad()

        self.output_node.grad = 1
        
        for node in self.reverse_topological_order:
            node.propagate_grad()

        grads = np.zeros(self.input_parameter_count)
        for inp in self.input_nodes.values():
            grads[inp.input_start_index : inp.input_start_index + inp.variable_count] = inp.grad

        return grads
    
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
        plt.show()

    def scale(self, scalar: float):
        new_output = Scale(scalar, self.output_node)
        return Tape(self.input_nodes, new_output)

    def __add__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = Add(self.output_node, other.output_node)
        return Tape(new_inputs, new_output)
    
    def __sub__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = Add(self.output_node, Scale(-1, other.output_node))
        return Tape(new_inputs, new_output)

    def __mul__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = Mult(self.output_node, other.output_node)
        return Tape(new_inputs, new_output)
    
    def __div__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = Div(self.output_node, other.output_node)
        return Tape(new_inputs, new_output)
    
    def __pow__(self, exponent: float):
        new_output = Pow(self.output_node, exponent)
        return Tape(self.input_nodes, new_output)
    
    def tanh(self):
        new_output = Tanh(self.output_node)
        return Tape(self.input_nodes, new_output)

    def sigmoid(self):
        new_output = Sigmoid(self.output_node)
        return Tape(self.input_nodes, new_output)
    
    def relu(self):
        new_output = Relu(self.output_node)
        return Tape(self.input_nodes, new_output)

    def __matmul__(self, other):
        new_inputs = union(self.input_nodes, other.input_nodes)
        new_output = MVMult(self.output_node, other.output_node)
        return Tape(new_inputs, new_output)
    
    def __getitem__(self, index):
        if (self.indices[index] == None):
            new_output = Index(self.output_node, index)
            new_tape = Tape(self.input_nodes, new_output)
            self.indices[index] = new_tape
            return new_tape
        else:
            return self.indices[index]
    
    def split(self):
        output_dims = self.output_node.get_output_dimensions()
        assert(output_dims[1] == 1)
        return [self[i] for i in range(output_dims[0])]

    def square(self):
        return self**2
    
    def sqrt(self):
        return self**0.5

    def sum(self):
        new_output = Sum(self.output_node)
        return Tape(self.input_nodes, new_output)
    
    def convolve(self, stencil, strides: Tuple[int, int]):
        new_inputs = union(self.input_nodes, stencil.input_nodes)
        new_output = Convolve(self.output_node, stencil.output_node, strides)
        return Tape(new_inputs, new_output)
    
    def reshape(self, new_size):
        new_output = Reshape(self.output_node, new_size)
        return Tape(self.input_nodes, new_output)
    
    def __str__(self) -> str:
        return str(self.output_node)
    
    def __len__(self):
        return self.output_node.get_output_length()
    
    def get_output_dimensions(self):
        return self.output_node.get_output_dimensions()

class AbstractOperationNode(ABC):

    sources = []
    listeners = []

    result: arrayf # vector of length output length
    grad: arrayf # vector of length output length

    def __init__(self, *args) -> None:

        for source in args:
            source.listeners.append(source)

        self.sources = args
        self.listeners = []

        super().__init__()

    def run(self, inputs):
        self.result = self.inner_run(inputs)

    def clear_grad(self):
        self.grad = np.zeros(self.get_output_length())

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
    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        pass

    @abstractmethod
    def propagate_grad(self): # assume that grad is completely setup
        pass

    def is_variable(self) -> bool:
        return False
    
    @abstractmethod
    def get_output_dimensions(self) -> Tuple[int, int]:
        return (0, 0)

    def get_output_length(self) -> Tuple[int, int]:
        dims = self.get_output_dimensions()
        return dims[0] * dims[1]
    
    def get_result_matrix(self) -> arrayf:
        return self.result.reshape(self.get_output_dimensions()) # should never copy!
    
    def get_grad_matrix(self) -> arrayf: 
        return self.grad.reshape(self.get_output_dimensions()) # should never copy!

    @abstractmethod
    def __str__(self) -> str:
        return ""

class Const(AbstractOperationNode):

    val: arrayf
    size: Tuple[int, int]

    def __init__(self, c: arrayf):
        super().__init__()
        self.val = c.flatten()
        self.size = tuple(c.shape)
        if (len(self.size) == 1):
            self.size = (self.size[0], 1)

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return self.val

    def propagate_grad(self): # assume that grad is completely setup
        pass
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.size

    def __str__(self) -> str:
        return str(self.val)

class Var(AbstractOperationNode):

    input_start_index: int
    size: Tuple[int, int]
    variable_count: int

    def __init__(self, input_start_index, size) -> None:
        super().__init__()
        self.input_start_index = input_start_index
        self.size = tuple(size)
        self.variable_count = size[0] * size[1]

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return np.array(inputs[self.input_start_index : self.input_start_index + self.variable_count])

    def propagate_grad(self): # assume that grad is completely setup
        pass

    def is_variable(self) -> bool:
        return True
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.size

    def __str__(self) -> str:
        return "[" + str(self.input_start_index) + ":" + str(self.input_start_index + self.variable_count) + "]"
    
    def __lt__(self, other):
        return self.input_start_index < other.input_start_index

class Scale(AbstractOperationNode):

    scalar: float

    def __init__(self, scalar: float, a: AbstractOperationNode):
        super().__init__(a)
        self.scalar = scalar

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return self.scalar * self.sources[0].result

    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += self.scalar * self.grad

    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.sources[0].get_output_dimensions()

    def __str__(self) -> str:
        return "(" + str(self.scalar) + str(self.sources[0]) + ")"

class Add(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode, b: AbstractOperationNode):
        super().__init__(a, b)
        assert(tuple(a.get_output_dimensions()) == b.get_output_dimensions())

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return self.sources[0].result + self.sources[1].result

    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += self.grad
        self.sources[1].grad += self.grad

    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.sources[0].get_output_dimensions()

    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + " + " + str(self.sources[1]) + ")"

class Mult(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode, b: AbstractOperationNode):
        super().__init__(a, b)
        assert(a.get_output_dimensions() == b.get_output_dimensions())

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return self.sources[0].result * self.sources[1].result

    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += self.sources[1].result * self.grad
        self.sources[1].grad += self.sources[0].result * self.grad
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.sources[0].get_output_dimensions()

    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + " * " + str(self.sources[1]) + ")"
        
class Div(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode, b: AbstractOperationNode):
        super().__init__(a, b)
        assert(a.get_output_dimensions() == b.get_output_dimensions())

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return self.sources[0].result / self.sources[1].result

    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += self.grad / self.sources[1].result
        self.sources[1].grad += (-self.grad * self.sources[0].result / (self.sources[1].result * self.sources[1].result))
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.sources[0].get_output_dimensions()

    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + " / " + str(self.sources[1]) + ")"
    
class Tanh(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode):
        super().__init__(a)

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return np.tanh(self.sources[0].result)
    
    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += (1 - np.tanh(self.sources[0].result)**2) * self.grad
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.sources[0].get_output_dimensions()

    def __str__(self) -> str:
        return "tanh(" + str(self.sources[0]) + ")"
    
class Sigmoid(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode):
        super().__init__(a)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return Sigmoid.sigmoid(self.sources[0].result)
    
    def propagate_grad(self): # assume that grad is completely setup
        x = self.sources[0].result
        self.sources[0].grad += Sigmoid.sigmoid(x) * (1 - Sigmoid.sigmoid(x)) * self.grad
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.sources[0].get_output_dimensions()

    def __str__(self) -> str:
        return "sigmoid(" + str(self.sources[0]) + ")"

class Relu(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode):
        super().__init__(a)

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return np.maximum(0, self.sources[0].result)
    
    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += np.where(self.sources[0].result >= 0, 1, 0) * self.grad
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.sources[0].get_output_dimensions()

    def __str__(self) -> str:
        return "relu(" + str(self.sources[0]) + ")"

class Pow(AbstractOperationNode):

    exponent: float

    def __init__(self, a: AbstractOperationNode, exponent: float):
        super().__init__(a)
        self.exponent = exponent

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return np.power(self.sources[0].result, self.exponent)
    
    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += self.exponent * np.power(self.sources[0].result, self.exponent - 1) * self.grad
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.sources[0].get_output_dimensions()

    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + ")**" + str(self.exponent)

# matrix-vector multiplication
class MVMult(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode, b: AbstractOperationNode):
        super().__init__(a, b)
        assert(a.get_output_dimensions()[1] == b.get_output_dimensions()[0])

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return (self.sources[0].get_result_matrix() @ self.sources[1].get_result_matrix()).flatten()

    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += np.outer(self.grad, self.sources[1].get_result_matrix()).flatten()
        self.sources[1].grad += self.sources[0].get_result_matrix().transpose() @ self.grad
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return (self.sources[0].get_output_dimensions()[0], self.sources[1].get_output_dimensions()[1])

    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + " * " + str(self.sources[1]) + ")"
    
class Index(AbstractOperationNode):

    index: int

    def __init__(self, a: AbstractOperationNode, index: int):
        super().__init__(a)
        self.index = index
        assert(a.get_output_dimensions()[1] == 1)

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return self.sources[0].result[self.index]

    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad[self.index] += self.grad[0]
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return (1, 1)

    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + "[" + str(self.index) + "])"
    
class Combine(AbstractOperationNode):

    def __init__(self, combines: List[AbstractOperationNode]):
        super().__init__(*combines)
        for combine in combines:
            assert(combine.get_output_dimensions() == (1,1))

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return np.vstack([source.result for source in self.sources])

    def propagate_grad(self): # assume that grad is completely setup
        for i in range(len(self.sources)):
            self.sources[i].grad[0] += self.grad[i]
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return (len(self.sources), 1)

    def __str__(self) -> str:
        return "(" + str(self.sources) + ")"

class Sum(AbstractOperationNode):

    def __init__(self, a: AbstractOperationNode):
        super().__init__(a)
        # assert(a.get_output_dimensions()[1] == 1)

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return sum(self.sources[0].result)

    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += self.grad
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return (1, 1)

    def __str__(self) -> str:
        return "sum(" + str(self.sources[0]) + ")"

# matrix-vector multiplication
class Convolve(AbstractOperationNode):
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    output_size: Tuple[int, int]
    convolution_shape: Tuple[int, int, int, int]

    last_strided = None

    def __init__(self, image: AbstractOperationNode, stencil: AbstractOperationNode, strides: Tuple[int, int]):
        super().__init__(image, stencil)
        self.strides = strides
        input_size = image.get_output_dimensions()
        self.kernel_size = stencil.get_output_dimensions()
        self.output_size = ((input_size[0] - self.kernel_size[0]) // strides[0] + 1, (input_size[1] - self.kernel_size[1]) // strides[1] + 1)
        self.convolution_shape = (self.output_size[0], self.output_size[1], self.kernel_size[0], self.kernel_size[1])


    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        image = self.sources[0].get_result_matrix()
        stencil = self.sources[1].get_result_matrix()
        image_strides = (image.strides[0] * self.strides[0], image.strides[1] * self.strides[1], image.strides[0] * 1, image.strides[1] * 1)
        self.last_strided = as_strided(image, shape=self.convolution_shape, strides=image_strides)
        result = np.einsum('ijkl,kl->ij', self.last_strided, stencil)
        return result.flatten()

    def propagate_grad(self): # assume that grad is completely setup
        grad_reshaped = self.get_grad_matrix()

        # image grad
        image_grad = self.sources[0].get_grad_matrix()
        stencil = self.sources[1].get_result_matrix()
        grad_strides = (image_grad.strides[0] * self.strides[0], image_grad.strides[1] * self.strides[1], image_grad.strides[0] * 1, image_grad.strides[1] * 1)
        grad_strided = as_strided(image_grad, shape=self.convolution_shape, strides=grad_strides, writeable=True)
        
        change = np.einsum('ij,kl->ijkl', grad_reshaped, stencil)
        batches_needed_x = (self.kernel_size[0] - self.strides[0] + 1)
        batches_needed_y = (self.kernel_size[1] - self.strides[1] + 1)
        for x in range(batches_needed_x):
            for y in range(batches_needed_y):
                grad_strided[x::batches_needed_x, y::batches_needed_y] += change[x::batches_needed_x, y::batches_needed_y]
        # doesn't work because of concurrency issues
        # grad_strided += np.einsum('ij,kl->ijkl', grad_reshaped, stencil)

        # stencil grad
        self.sources[1].grad += (np.einsum('ijkl,ij->kl', self.last_strided, grad_reshaped)).flatten()
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.output_size

    def __str__(self) -> str:
        return "(" + str(self.sources[0]) + " * " + str(self.sources[1]) + ")"

class Reshape(AbstractOperationNode):

    output_size: Tuple[int, int]

    def __init__(self, a: AbstractOperationNode, output_size: Tuple[int, int]):
        super().__init__(a)
        self.output_size = output_size
        if (np.prod(a.get_output_dimensions()) != np.prod(output_size)):
            print("expected area: " + str(np.prod(a.get_output_dimensions())))
            print("desired area: " + str(np.prod(output_size)))
            assert(np.prod(a.get_output_dimensions()) == np.prod(output_size))

    def inner_run(self, inputs) -> arrayf: # returns a vector of length output length
        return self.sources[0].result
    
    def propagate_grad(self): # assume that grad is completely setup
        self.sources[0].grad += self.grad
    
    def get_output_dimensions(self) -> Tuple[int, int]:
        return self.output_size

    def __str__(self) -> str:
        return "{" + str(self.sources[0]) + "}"

class AbstractNetworkLayer:

    @abstractmethod
    def forwards(self, inputs: Tape) -> Tape:
        return inputs
    
    @abstractmethod
    def initialize(self, parameter_count: int) -> List[float]:
        return np.zeros(parameter_count)

class ActivationFunction(Enum):
    TANH = 0
    SIGMOID = 1
    RELU = 2
    NONE = 3

def apply_activation_function(activation_type: ActivationFunction, inputs: Tape) -> Tape:
    if activation_type == ActivationFunction.TANH:
        return inputs.tanh()
    elif activation_type == ActivationFunction.SIGMOID:
        return inputs.sigmoid()
    elif activation_type == ActivationFunction.RELU:
        return inputs.relu()
    else:
        return inputs

class InitializationType(Enum):
    NORMAL = 0
    UNIFORM = 1
    ZEROES = 2

def get_initialization_variables(init_type: InitializationType, parameter_count: int) -> np.ndarray:
    if init_type == InitializationType.UNIFORM:
        return np.random.rand(parameter_count)
    elif init_type == InitializationType.NORMAL:
        return np.random.randn(parameter_count)
    elif init_type == InitializationType.ZEROES:
        return np.zeros(parameter_count)
    else:
        return np.zeros(parameter_count)

class SimpleLayer(AbstractNetworkLayer):

    output_vars: int
    activation_type: ActivationFunction
    init_type: InitializationType

    def __init__(self, output_vars: int, activation_type: ActivationFunction = ActivationFunction.TANH, init_type: InitializationType = InitializationType.NORMAL) -> None:
        super().__init__()
        self.output_vars = output_vars
        self.activation_type = activation_type
        self.init_type = init_type

    def forwards(self, inputs: Tape) -> Tape:

        weights = Tape.Var([self.output_vars, len(inputs)])
        biases = Tape.Var([self.output_vars, 1])

        return apply_activation_function(self.activation_type, weights @ inputs + biases)
    
    def initialize(self, parameter_count: int) -> List[float]:
        return get_initialization_variables(self.init_type, parameter_count)


class ConvolutionLayer(AbstractNetworkLayer):

    image_shape: Tuple[int, int]
    stencil_shape: Tuple[int, int]
    strides: Tuple[int, int]
    activation_type: ActivationFunction
    init_type: InitializationType

    def __init__(self, image_shape: Tuple[int, int], stencil_shape: Tuple[int, int], strides: Tuple[int, int], activation_type: ActivationFunction = None, init_type: InitializationType = InitializationType.NORMAL) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.stencil_shape = stencil_shape
        self.strides = strides
        self.activation_type = activation_type
        self.init_type = init_type

    def forwards(self, inputs: Tape) -> Tape:

        stencil = Tape.Var([self.stencil_shape[0], self.stencil_shape[1]])

        convolved = inputs.convolve(stencil, self.strides)
        return apply_activation_function(self.activation_type, convolved)
    
    def initialize(self, parameter_count: int) -> List[float]:
        return get_initialization_variables(self.init_type, parameter_count)

class ReshapeLayer(AbstractNetworkLayer):
    
    output_size: Tuple[int, int]

    def __init__(self, output_size: Tuple[int, int]):
        super().__init__()
        self.output_size = output_size

    def forwards(self, inputs: Tape) -> Tape:
        return inputs.reshape(self.output_size)
    
    def initialize(self, parameter_count: int) -> List[float]:
        return []

class NeuralNetwork:

    input_size: Tuple[int, int]
    layers: List[AbstractNetworkLayer]
    learning_rate: float
    weight_decay_rate: float
    weight_decay_exp: float

    layer_parameter_sizes: List[int]
    total_parameters: int
    output_count: int

    input_layer: Tape
    output_layer: Tape
    expected_outputs: Tape
    cost = None

    current_parameters = []

    def __init__(self, input_size: Tuple[int, int], layers: List[AbstractNetworkLayer], learning_rate: float, weight_decay_rate: float = 0, weight_decay_exp: float = 1) -> None:
        self.input_size = input_size
        self.layers = layers
        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.weight_decay_exp = weight_decay_exp

        # this is jank, we should find a better way of doing this
        Tape.VARIABLE_INDICES = 0
        old_variable_indices = 0
        self.layer_parameter_sizes = []

        tape = Tape.Const(np.zeros(self.input_size))
        self.input_layer = tape

        for layer in self.layers:
            tape = layer.forwards(tape)

            self.layer_parameter_sizes.append(Tape.VARIABLE_INDICES - old_variable_indices)
            old_variable_indices = Tape.VARIABLE_INDICES

        self.output_layer = tape
        self.total_parameters = sum(self.layer_parameter_sizes)
        self.output_count = len(tape)

        self.expected_outputs = Tape.Const(np.zeros(self.output_count))
        self.cost = (tape - self.expected_outputs).square().sum().scale(0.5)

        if self.weight_decay_rate != 0:
            self.cost = self.cost + self.cost.sum_all_variables(lambda x: x**weight_decay_exp).sum().scale(self.weight_decay_rate / self.weight_decay_exp)

        self.reset_parameters()

    def reset_parameters(self):
        self.current_parameters = []
        for i in range(len(self.layers)):
            self.current_parameters.extend(self.layers[i].initialize(self.layer_parameter_sizes[i]))

    def set_inputs(self, inputs: arrayf):
        self.input_layer.output_node.val = inputs
    
    def set_expected_outputs(self, expected_outputs: arrayf):
        self.expected_outputs.output_node.val = expected_outputs

    # sanity check
    def check(self, inputs: arrayf):
        self.set_inputs(inputs)
        self.cost.check(self.current_parameters)

    def forwards(self, inputs: arrayf) -> arrayf:
        self.set_inputs(inputs)
        return self.output_layer.forwards(self.current_parameters)

    def get_cost(self, inputs: arrayf, expected_output: arrayf) -> float:
        self.set_inputs(inputs)
        self.set_expected_outputs(expected_output)
        return self.cost.forwards(self.current_parameters)

    def train(self, inputs: arrayf, expected_outputs: arrayf):

        self.get_cost(inputs, expected_outputs)
        grads = self.cost.backwards_from_last_forwards()
        self.current_parameters -= self.learning_rate * grads
    
    def train_batch(self, inputs: List[arrayf], expected_outputs: List[arrayf]):
        batch_size = len(inputs)
        
        total_grads = 0
        for i in range(0, batch_size):

            self.get_cost(inputs[i], expected_outputs[i])
            grads = self.cost.backwards_from_last_forwards()
            total_grads += grads

        self.current_parameters -= self.learning_rate * total_grads / batch_size
        
        
