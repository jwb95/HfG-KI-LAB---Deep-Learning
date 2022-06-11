In dieser Lektion implementieren wir das Neuron aus Lektion 3 mittels der Deep Learning-Library Tensorflow.
Die wesentliche Funktionalität von Deep Learning-Libraries besteht darin, dass sie alle Ableitungen, die wir für die Optimierung unseres Models benötigen, automatisiert berechnen können. Von dieser Funktion werden wir bereits Gebrauch machen, sodass nur noch die Lossfunction, jedoch nicht mehr deren Ableitungen in Abhängigkeit zu den  trainierbaren Variablen des Models von uns gecoded werden muss.
Auch das Anpassen aller trainierbaren Variablen des Models mittels dieser Ableitungen reduziert sich auf 2 Lines Code.

Übliche Alternativen zu Tensorflow sind Pytorch und Jax.
