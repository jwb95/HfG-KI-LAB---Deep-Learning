from misc.visualise import plot_losses



# -------------------------------------------------------------------------------------------------------------------------------

# Fehlerfunktion
# = Das Quadrat der Differenz zw. Prediction und Label.
def lossfunction(x, y):
    loss = (x - y)**2
    return loss

# 1. Ableitung der Fehlerfunktion (= f(x)) in Abhängigkeit zu x.
# = Analytisch:     Das Verhältnis zw. der winzigen Veränderung in f(x) (= f(x+dx) - f(x)) 
#                   und der sich 0 annähernden winzigen Veränderung in x (= dx).
# = Geometrisch:    Die Steigung der Tangente, die den Graphen von f(x) am Punkt (x, f(x)) berührt.
# = Symbolisch:     lim x -> 0: (f(x + dx) - f(x)) / dx 
def dlossfunction_dx(x, y):
    z = 2*(x - y)
    return z

# Datenset
# [[a0, a1], y/label]
Dataset = [
    [[0,0], 0],
    [[0,1], 1],
    [[1,1], 1],
    [[1,0], 0]
]

# Gewichte definieren/initialisieren
from random import uniform
w0 = uniform(-1, 1)#0.3
w1 = uniform(-1, 1)#-0.2
print('Initialisiert    : w0 = '+str(w0)+', w1 = '+str(w1))

# Forwardschritt definieren
def forward(a0, a1):
    x = a0*w0 + a1*w1
    return x

# Funktion zur Anpassung der Gewichte definieren.
def optimieren(w0, w1, x, y, a0, a1, learning_rate):
    z = dlossfunction_dx(x, y)
    w0 = w0 - z * a0 * learning_rate
    w1 = w1 - z * a1 * learning_rate
    return w0, w1

# Trainingsloop
num_epochs = 3
learning_rate = 0.5

fehlerhistorie = []
for i_epoch in range(num_epochs):
    for trainingsbeispiel in Dataset:
        a0, a1  = trainingsbeispiel[0]
        y       = trainingsbeispiel[1]
        x       = forward(a0, a1)
        fehlerhistorie.append(lossfunction(x, y))
        w0, w1  = optimieren(w0, w1, x, y, a0, a1, learning_rate)

print('Fehlerhistorie   :', fehlerhistorie)
print('Nach Training    : w0 = '+str(w0)+', w1 = '+str(w1))
plot_losses(fehlerhistorie)


