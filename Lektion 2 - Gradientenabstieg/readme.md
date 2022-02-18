Die schwarze Linie ist die Hypotenuse des rechtwinkligen Steigungsdreiecks in dem Interval, welches von der roten (x0) und der blauen (x0+h) Linie eingefasst wird. Dieses Interval hat die Größe/Breite h. Lassen wir h nun kleiner werden, nähert sich die Hypotenuse der Tangente, die den grünen Graphen am Punkt x0 berührt, an. Ziehen wir einen ausreichend kleinen Anteil der Steigung dieser Tangente von x0 ab, so müsste sich x0 in Richtung 0 (wo sich das Minimum des grünen Graphen befindet) bewegen. Nach diesem Prinzip lässt sich das globale Minimum jeder konvexen Funktion finden.

Desmos-Graph: https://www.desmos.com/calculator/wrkow3r3iv

![](https://github.com/jwb95/HfG-KI-LAB/blob/main/Lektion%202%20-%20Gradientenabstieg/media/tangente.png)

In dieser Lektion implementieren wir den beschriebenen Optimierungs-Algorithmus, um ein zufälliges zwischen -4 und 4 initialisiertes x in Richtung des Minimums von f(x)=x^2 zu optimieren.

![](https://github.com/jwb95/HfG-KI-LAB/blob/main/Lektion%202%20-%20Gradientenabstieg/media/optimierung.png)

Was ist ein Quotient?

Der Quotient drückt ein Verhältnis von zwei Größen als Resultat der Division der einen Größe durch die andere aus.
Ist das Verhältnis von Apfelkuchen zu Äpfeln 1 zu 3, so ist der Quotient 1:3 oder als Bruch geschrieben 1/3.

Steigung ist eine Form eines Quotienten, die das Verhältnis von (räumlicher) vertikaler Zunahme zu horizontaler Zunahme beschreibt.
Legt eine Bergsteigerin in einem Interval ihrer Wanderung 1km parallel zur (vereinfachend) flachen Erdscheibe
und währenddessen 100 Höhenmeter zurück, so betrug die Steigung in diesem Interval 100m:1000m, also 1/10.
Im Sinne des folgenden Schaubilds wäre dy=100m und dx=1000m.

![](https://github.com/jwb95/HfG-KI-LAB/blob/main/Lektion%202%20-%20Gradientenabstieg/media/steigung.png)

