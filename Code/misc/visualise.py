import matplotlib.pyplot as plt
from cv2 import imread, cvtColor, COLOR_BGR2RGB

def plot_gradient_descent(xs, errors):
    plt.scatter(xs, errors)
    plt.plot(xs, errors)
    xlim = max([abs(x) for x in xs])
    plt.xlim([-xlim,xlim])
    plt.ylim([0,max(errors)])
    plt.xlabel('x')
    plt.ylabel('squared_error')
    plt.show()

def plot_losses(losses):
    plt.plot(losses)
    plt.xlabel('trainingsschritt')
    plt.ylabel('squared_error')
    plt.show()