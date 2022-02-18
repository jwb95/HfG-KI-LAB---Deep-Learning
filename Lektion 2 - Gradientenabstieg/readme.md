Die schwarze Linie ist die Hypotenuse des rechtwinkligen Steigungsdreiecks in dem Interval, welches von der roten (x0) und der blauen (x0+h) Linie eingefasst wird. Dieses Interval hat die Größe/Breite h. Lassen wir h nun kleiner werden, nähert sich die Hypotenuse der Tangente, die den grünen Graphen von f(x) am Punkt (x0, f(x0)) berührt, an. Ziehen wir einen ausreichend kleinen Anteil der Steigung dieser Tangente von x0 ab, so müsste sich x0 in Richtung 0 (wo sich das Minimum des grünen Graphen befindet) bewegen. Nach diesem Prinzip lässt sich x dem globalen Minimum jeder konvexen Funktion schrittweise annähern.


Desmos-Graph: https://www.desmos.com/calculator/wrkow3r3iv

![](https://github.com/jwb95/HfG-KI-LAB/blob/main/Lektion%202%20-%20Gradientenabstieg/media/tangente.png)

In dieser Lektion implementieren wir den beschriebenen Optimierungs-Algorithmus, um ein zufälliges zwischen -4 und 4 initialisiertes x in Richtung des Minimums von f(x)=x^2 zu optimieren.

![](https://github.com/jwb95/HfG-KI-LAB/blob/main/Lektion%202%20-%20Gradientenabstieg/media/optimierung.png)

Was ist Steigung?

Steigung ist ein spezifischer Quotient.
Ein Quotient drückt ein Verhältnis von zwei Größen als Resultat der Division der einen Größe durch die andere aus.
Ist das Verhältnis von Apfelkuchen zu Äpfeln 1 zu 3, so ist der Quotient 1:3 oder als Bruch geschrieben 1/3.

Steigung ist jener Quotient, der das Verhältnis von (räumlicher) vertikaler Zunahme zu horizontaler Zunahme beschreibt.
Legt eine Bergsteigerin in einem Interval ihrer Wanderung 1km parallel zur (vereinfachend) flachen Erdscheibe
und währenddessen 100 Höhenmeter zurück, so betrug die Steigung in diesem Interval 100m:1000m, also 1/10.

Im Sinne des folgenden Schaubilds wäre Δy=100m und Δx=1000m.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Slope_picture.svg/170px-Slope_picture.svg.png)

Was ist eine Ableitung und wozu brauchen wir sie?

Die allgemeine Definition der Ableitung für eine Funktion f(x) ist:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Clim_%7Bh%5Cto%5C0%7D%20%5Cfrac%7Bf%28x&plus;h%29%20-%20f%28x%29%7D%7Bh%7D)

Bemerke, dass der folgende Ausdruck nichts anderes bedeutet als die Steigung der Funktion f(x) im Interval x bis x+h:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cfrac%7Bf%28x&plus;h%29%20-%20f%28x%29%7D%7Bh%7D)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Clim_%7Bh%5Cto%5C0%7D) bedeutet lediglich, dass wir betrachten wollen, was der Ausdruck, welchem es voransteht, also ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cfrac%7Bf%28x&plus;h%29%20-%20f%28x%29%7D%7Bh%7D) annähert, während sich h 0 annähert.
Die Definition der Ableitung kann folglich gelesen werden als: Jene Funktion, welcher wir uns nähern, wenn wir die Steigung von f(x) im Interval x bis x+h betrachten und nun die Größe des Intervals zunehmend kleiner werden lassen.
Bemerke, dass die Definition der Ableitung für h = 0 nicht definiert ist, da Division durch 0 nicht definiert ist. Wir können also nicht ohne Weiteres h = 0 setzen, um die Ableitung für ein f(x) auszurechnen. Trotzdem liefert es Erkenntnis zu betrachten, was passiert, wenn h einen Wert, der sehr nah an 0 ist, annimmt.

Wenn wir im Desmos-Graphen h zunehmend kleiner werden lassen und letztlich eine winzige Zahl (bspw. 0.0001) eintragen, sehen wir, dass die schwarze Linie sich zunehmend jener Tangente annähert, die den grünen Graphen von f(x) am Punkt (x0, f(x0)) berührt.
Bewegen wir anschließend den Fader von x0 sehen wir, dass die schwarze Linie für jedes x0 die entsprechende Tangente quasi abbildet.

Die Ableitung einer Funktion gibt also die Steigung der den Graphen berührenden Tangente am Punkt (x, f(x) für jedes x an.
