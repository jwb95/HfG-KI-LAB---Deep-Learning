In dieser Lektion lernen wir den Gradient Descent, den grundlegenden Algorithmus kennen, der das Training Neuronaler Netze ermöglicht.
Dazu machen wir zuerst ein paar mathematische Beobachtungen und implementieren anschließend einen Trainings-Algorithmus, um ein zufälliges zwischen -4 und 4 initialisiertes x0 in Richtung des Minimums von ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20f%28x%29%3Dx%5E%7B2%7D) zu optimieren.
Parallel zeigen wir auch eine Lösung via Trial and Error, die für dieses Problem genauso gut funktioniert. Wir werden jedoch zeitnah sehen, dass Trial and Error für große Deep-Learning-Systeme nicht funktionieren wird und wir Gradient Descent benötigen.

## Was ist Steigung?

Steigung ist ein spezifischer Quotient.
Ein Quotient drückt ein Verhältnis von zwei Größen als Resultat der Division der einen Größe durch die andere aus.
Ist das Verhältnis von Apfelkuchen zu Äpfeln 1 zu 3, so ist der Quotient 1:3 oder als Bruch geschrieben 1/3.

Steigung ist jener Quotient, der das Verhältnis von (räumlicher) vertikaler Zunahme zu horizontaler Zunahme beschreibt.
Legt eine Bergsteigerin in einem Interval ihrer Wanderung 1km parallel zur (vereinfachend) flachen Erdscheibe
und währenddessen 100 Höhenmeter zurück, so betrug die Steigung in diesem Interval 100m:1000m, also 1/10.

Im Sinne des folgenden Schaubilds wäre Δy=100m und Δx=1000m.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Slope_picture.svg/170px-Slope_picture.svg.png)

## Eine geometrische Beobachtung

Die schwarze Linie ist die Hypotenuse des rechtwinkligen Steigungsdreiecks in dem Interval, welches von der roten (x0) und der blauen (x0+h) Linie eingefasst wird. Dieses Interval hat die Größe/Breite h. Bewegen wir h nun in Richtung 0, so nähert sich die Hypotenuse der Tangente, die den grünen Graphen von f(x) am Punkt (x0, f(x0)) berührt, an. Ziehen wir einen ausreichend kleinen Anteil der Steigung dieser Tangente von x0 ab, so müsste sich x0 in Richtung 0 (wo sich das Minimum des grünen Graphen befindet) bewegen. Nach diesem Prinzip lässt sich x0 dem globalen Minimum jeder konvexen Funktion schrittweise annähern.


Anhand des [Desmos-Graphen](https://www.desmos.com/calculator/weppwasdzw) lässt sich das gut sehen.

![](https://github.com/jwb95/HfG-KI-LAB---Deep-Learning/blob/main/Lektion%2002%20-%20Gradientenabstieg/media/tangente.png)


## Was ist eine Ableitung und wozu brauchen wir sie?

Die allgemeine Definition der Ableitung ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdf%7D%7Bdx%7D) für eine Funktion f(x) in Abhängigkeit von x ist:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cfrac%7Bdf%7D%7Bdx%7D%3D%5Clim_%7Bdx%5Cto%5C0%7D%5Cfrac%7Bf%28x&plus;dx%29-f%28x%29%7D%7Bdx%7D)

Bemerke, dass der folgende Ausdruck nichts anderes bedeutet als die Steigung der Funktion f(x) im Interval x bis x+dx:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cfrac%7Bf%28x&plus;dx%29%20-%20f%28x%29%7D%7Bdx%7D)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Clim_%7Bdx%5Cto%5C0%7D) bedeutet lediglich, dass wir betrachten wollen, was sich der Ausdruck, welchem es voransteht, also ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Cfrac%7Bf%28x&plus;dx%29%20-%20f%28x%29%7D%7Bdx%7D) annähert, während sich dx 0 annähert.
Die Definition der Ableitung kann folglich gelesen werden als: Jene Funktion, welcher wir uns nähern, wenn wir die Steigung von f(x) im Interval x bis x+dx betrachten und nun die Größe des Intervals zunehmend 0 annähern.
Daher beschreibt die Ableitung in Abhängigkeit von x eine Steigung und ist folglich auch als Quotient ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdf%7D%7Bdx%7D) schreibbar, wobei dx eine Veränderung in x darstellt, die eine Veränderung von df im Funktionswert von f nach sich zieht, wobei wir explizit den Quotienten meinen, dem sich ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdf%7D%7Bdx%7D) annähert während sich dx 0 annähert.

Bemerke, dass die Definition der Ableitung für dx = 0 nicht definiert ist, da Division durch 0 nicht definiert ist. Wir können also nicht ohne Weiteres dx = 0 setzen, um die Ableitung für ein f(x) auszurechnen. Jedoch liefert die geometrische Betrachtung dessen, was passiert wenn dx einen Wert, der sehr nah an 0 ist, annimmt, Erkenntnis:
Wenn wir im [Desmos-Graphen](https://www.desmos.com/calculator/weppwasdzw) h (was stellvertretend für dx steht) zunehmend kleiner werden lassen und letztlich eine winzige Zahl (bspw. 0.0001) eintragen, sehen wir, dass die schwarze Linie sich zunehmend jener Tangente annähert, die den grünen Graphen von f(x) am Punkt (x0, f(x0)) berührt.
Bewegen wir anschließend den Fader von x0 sehen wir, dass die schwarze Linie für jedes x0 die entsprechende Tangente quasi abbildet.

Die Ableitung einer Funktion von x in Abhängigkeit von x gibt also die Steigung der den Graphen berührenden Tangente am Punkt (x, f(x)) für jedes x an.
An den Ablauf des Optimierungsverfahrens erinnernd, nach welchem wir bei jedem Optimierungsschritt von x0 einen ausreichend kleinen Teil der Steigung der den Graphen von f(x) am Punkt (x0, f(x0)) berührenden Tangente abziehen, benötigen wir also die Ableitung von f(x) in Abhängigkeit von x.

Finden wir nun algebraisch die entsprechende Ableitung für ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20f%28x%29%3Dx%5E%7B2%7D).
Dazu ersetzen wir in der allgemeinen Definition der Ableitung einfach den allgemeinen Ausdruck ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20f%28x%29) durch ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%5E%7B2%7D) und erhalten:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Clim_%7Bh%5Cto%5C0%7D%5Cfrac%7B%28x&plus;h%29%5E%7B2%7D%20-%20x%5E%7B2%7D%7D%7Bh%7D)

Die 1. binomische Formel nutzend:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Clim_%7Bh%5Cto%5C0%7D%5Cfrac%7Bx%5E%7B2%7D%20&plus;%202xh%20&plus;%20h%5E%7B2%7D-%20x%5E%7B2%7D%7D%7Bh%7D)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%5E%7B2%7D) und ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20-x%5E%7B2%7D) löschen sich aus:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Clim_%7Bh%5Cto%5C0%7D%5Cfrac%7B2xh%20&plus;%20h%5E%7B2%7D%7D%7Bh%7D)

Aus ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%202xh) und ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20h%5E%7B2%7D) lässt sich jeweils ein ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20h) herauskürzen:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Clim_%7Bh%5Cto%5C0%7D%5C2x&plus;h)

Für diesen Ausdruck ist es nun völlig legitim ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20h%3D0) zu setzen, sodass wir letztlich die Ableitung von ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20f%28x%29%3Dx%5E%7B2%7D) erhalten:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%202x)


Den Optimierungsschritt können wir also folgendermaßen schreiben, wobei x_n die Schätzung von x0 nach n Schritten ist, entsprechend x_n+1 x0 nach n+1 Schritten darstellt und ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20%5Calpha) für die 'learningrate' steht, welche ein vom User heuristisch festgelegter kleiner Faktor ist, der sicherstellen soll, dass wir von x_n nicht mehr als einen ausreichend kleinen Teil der Ableitung von f(x_n) abziehen.

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20x_%7Bn&plus;1%7D%20%3D%20x_n%20-%202x_n%5Calpha)
