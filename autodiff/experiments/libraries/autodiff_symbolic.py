# Symbolic Autodiff
import numpy as np
from abc import ABC, abstractmethod

class AbstractOperation(ABC):

    @abstractmethod
    def run(self, ctx) -> float:
        pass

    @abstractmethod
    def diff(self, wrt): # returns AbstractOperation
        pass

    def terms(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        return ""

class Const(AbstractOperation):

    c: float

    def __init__(self, c: float):
        super().__init__()
        self.c = c

    def run(self, ctx):
        return self.c
    
    def diff(self, wrt):
        return Const(0)
    
    def terms(self):
        return 1
    
    def __str__(self) -> str:
        return str(self.c)

class Variable(AbstractOperation):

    name: str

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def run(self, ctx) -> float:
        return ctx[self.name]
    
    def diff(self, wrt):
        if wrt == self.name:
            return Const(1)
        else:
            return Const(0)

    def terms(self):
        return 1
    
    def __str__(self) -> str:
        return "<" + self.name + ">"

class Add(AbstractOperation):

    a: AbstractOperation
    b: AbstractOperation

    def __init__(self, a: AbstractOperation, b: AbstractOperation):
        super().__init__()
        self.a = a
        self.b = b

    def run(self, ctx) -> float:
        return self.a.run(ctx) + self.b.run(ctx)
    
    def diff(self, wrt):
        return Add(self.a.diff(wrt), self.b.diff(wrt))

    def terms(self):
        return self.a.terms() + self.b.terms()

    def __str__(self) -> str:
        return "(" + str(self.a) + " + " + str(self.b) + ")"

class Sub(AbstractOperation):

    a: AbstractOperation
    b: AbstractOperation

    def __init__(self, a: AbstractOperation, b: AbstractOperation):
        super().__init__()
        self.a = a
        self.b = b

    def run(self, ctx) -> float:
        return self.a.run(ctx) - self.b.run(ctx)
    
    def diff(self, wrt):
        return Sub(self.a.diff(wrt), self.b.diff(wrt))

    def terms(self):
        return self.a.terms() + self.b.terms()
    
    def __str__(self) -> str:
        return "(" + str(self.a) + " - " + str(self.b) + ")"

class Mult(AbstractOperation):

    a: AbstractOperation
    b: AbstractOperation

    def __init__(self, a: AbstractOperation, b: AbstractOperation):
        super().__init__()
        self.a = a
        self.b = b

    def run(self, ctx) -> float:
        return self.a.run(ctx) * self.b.run(ctx)
    
    def diff(self, wrt):
        return Add(Mult(self.a.diff(wrt), self.b), Mult(self.a, self.b.diff(wrt)))

    def terms(self):
        return self.a.terms() + self.b.terms()
    
    def __str__(self) -> str:
        return "(" + str(self.a) + " * " + str(self.b) + ")"

class Div(AbstractOperation):

    a: AbstractOperation
    b: AbstractOperation

    def __init__(self, a: AbstractOperation, b: AbstractOperation):
        super().__init__()
        self.a = a
        self.b = b

    def run(self, ctx) -> float:
        return self.a.run(ctx) / self.b.run(ctx)
    
    def diff(self, wrt):
        return Div(Add(Mult(self.a.diff(wrt), self.b), Mult(self.a, self.b.diff(wrt))), Mult(self.b, self.b))

    def terms(self):
        return self.a.terms() + self.b.terms()
    
    def __str__(self) -> str:
        return "(" + str(self.a) + " / " + str(self.b) + ")"

class Square(AbstractOperation):

    a: AbstractOperation

    def __init__(self, a: AbstractOperation):
        super().__init__()
        self.a = a

    def run(self, ctx) -> float:
        return self.a.run(ctx) ** 2
    
    def diff(self, wrt):
        return Mult(Mult(Const(2), self.a.diff(wrt)), self.a.diff(wrt))

    def terms(self):
        return self.a.terms()

    def terms(self):
        return 1 + self.a.terms()
    
    def __str__(self) -> str:
        return "(" + str(self.a) + ")**2"

class Tanh(AbstractOperation):

    a: AbstractOperation

    def __init__(self, a: AbstractOperation):
        super().__init__()
        self.a = a

    def run(self, ctx) -> float:
        return np.tanh(self.a.run(ctx))
    
    def diff(self, wrt):
        return Mult(Sub(Const(1), Mult(Tanh(self.a), Tanh(self.a))), self.a.diff(wrt))

    def terms(self):
        return 1 + self.a.terms()
    
    def __str__(self) -> str:
        return "tanh(" + str(self.a) + ")"