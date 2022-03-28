from math import exp
from math import cos
def Mymorlet(t):
    y = exp(-(t**2)/2) * cos(1.75*t)
    return y
