# Forwards-mode Autodiff
import numpy as np

class DualFloat():

    real_part: float
    dual_part: float

    def __init__(self, real_part: float, dual_part: float) -> None:
        self.real_part = real_part
        self.dual_part = dual_part

    def __add__(self, other):
        return DualFloat(self.real_part + other.real_part, self.dual_part + other.dual_part)
    
    def __sub__(self, other):
        return DualFloat(self.real_part - other.real_part, self.dual_part - other.dual_part)
    
    def __mul__(self, other):
        return DualFloat(self.real_part * other.real_part, self.real_part * other.dual_part + self.dual_part * other.real_part)
    
    def __div__(self, other):
        return DualFloat(self.real_part / other.real_part, ((self.dual_part * other.real_part) - (self.real_part * other.dual_part)) / (other.real_part * other.real_part))

    def tanh(self):
        return DualFloat(np.tanh(self.real_part), self.dual_part * (1 - np.tanh(self.real_part) ** 2))

    def __str__(self) -> str:
        return str(self.real_part) + " + " + str(self.dual_part) + "E"