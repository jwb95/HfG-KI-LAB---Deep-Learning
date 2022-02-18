# Funktion, deren Nullstellen die gesuchte sqrt(2) und -sqrt(2) sind
def f(x):
    y = x**2 - 2
    return y

# Ableitung von f(x)
def dfx_dx(x):
    y = 2*x
    return y

num_steps = 5
x = 3.0
from math import sqrt
print(sqrt(2))

for step in range(num_steps):
    x = x - f(x) / dfx_dx(x)
    print(x)



