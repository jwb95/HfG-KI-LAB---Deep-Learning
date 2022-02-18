from misc.visualise import plot_gradient_descent



# -------------------------------------------------------------------------------------------------------------------------------

def f(x):
    y = x**2
    return y

def dfx_x(x):
    y = 2*x
    return y

from random import uniform
learning_rate = 0.03
x = uniform(-4,4)

def optimization_step(x, learning_rate):
    x = x - dfx_x(x) * learning_rate
    error = f(x)
    return x, error

num_steps = 20

xs, sqrd_errors = [x,], [f(x),]
for i in range(num_steps):
    x, sqrd_error = optimization_step(x, learning_rate)
    print(sqrd_error)
    xs.append(x)
    sqrd_errors.append(sqrd_error)

plot_gradient_descent(xs, sqrd_errors)