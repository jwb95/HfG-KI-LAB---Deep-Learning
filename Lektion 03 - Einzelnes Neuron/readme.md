Das Schaubild zeigt eine Form eines einfachen Neurons mit 2 Inputs (x0, x1) und einem Output x.
In dieser Lektion optimieren wir mittels Gradient Descent die zu Beginn zufällig initialisierten Gewichte des Models (w0, w1) derartig, dass es anschließend ein spezifisches Problem lösen kann.

![](https://github.com/jwb95/HfG-KI-LAB---Deep-Learning/blob/main/Lektion%2003%20-%20Einzelnes%20Neuron/media/nn.jpg)

Ziel des Trainings eines Neuronalen Netzes ist es, die Gewichte des Netzes so anzupassen, dass es anschließend
sehr gut darin ist, für einen Input den richtigen Output auszugeben.
Betrachte ein Datenset (der Form [x0, x1] -> y) bestehend aus den folgenden 4 Trainingsbeispielen:

[0, 0] -> 0, 
[0, 1] -> 1, 
[1, 0] -> 0, 
[1, 1] -> 1

Wir sehen: Immer dann, wenn x1 = 1, soll das Model den Wert 1, ansonsten 0 ausgeben.
Eindeutig würde das optimierte Model wie im folgenden Schaubild aussehen:

![](https://github.com/jwb95/HfG-KI-LAB---Deep-Learning/blob/main/Lektion%2003%20-%20Einzelnes%20Neuron/media/nn_optimized.png)

Der Zweck des Optimierens Neuronaler Netze besteht darin, dass eine annehmbare Lösung eines Problems in Form der richtigen Anpassung der Gewichte automatisiert für komplizierte Probleme, für die im Gegensatz zu unserem Beispiel eine Lösung nicht unbedingt ersichtlich ist, gefunden werden kann. Nutzen wir dennoch das Beispiel, um uns mit dem gängigen Optimierungs-Algorithmus für Neuronale Netze vertraut zu machen.

Zunächst benötigen wir eine konvexe Fehlerfunktion (= lossfunction) L(x,y). Hierbei ist x die Ausgabe des Models für einen spezifischen Input und y das sog. Label - jener Wert, den das Model unserer Ansicht nach für den spezifischen Input ausgeben soll. Dafür bietet sich der 'Squared Error' an:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20L%28x%2Cy%29%20%3D%20%28x-y%29%5E%7B2%7D)

Der Squared Error besitzt eine Eigenschaft, die ihn für unser Problem als Fehlerfunktion qualizifiert. Betrachten wir 2 Trainingsbeispiele mit jeweils dem Label ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a) und nehmen wir an, das Model gibt für das erste Beispiel ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a&plus;b) und für das zweite ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a-b) aus, dann ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20L%28a&plus;b%2Ca%29%3DL%28a-b%2Ca%29).
Übersetzt: Es ist egal, ob das Model für ein Trainingsbeispiel ein x ausgibt, welches zu hoch oder zu niedrig ist. Wesentlich für den Wert, den die Lossfunction annimmt ist nur der absolute Abstand von x zum Label y.

## Test

Zunächst initialisieren wir das Model mit zufälligen Gewichten, z.B. w0 = 0.3 und w1 = -0.2

Ziehen wir nun ein Trainingsbeispiel, z.B. [1, 1] -> 1 und errechnen für dieses die Ausgabe des Models:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%28w_0%3D0.3%2Cx_0%3D1%2C%20w_1%3D-0.2%2Cx_1%3D1%29%29%20%3D%201%5Ccdot%200.3&plus;1%5Ccdot%20%28-0.2%29%3D0.1)

Errechnen wir dann den Squared Error:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20L%280.1%2C1%29%3D%280.1-1%29%5E%7B2%7D%3D%28-0.9%29%5E%7B2%7D%3D0.81)

Wir sehen: Die Performance des Models lässt zu wünschen übrig und es gilt die Gewichte des Models so anzupassen, dass L sinkt.

## Herleitung des Optimierungsalgorithmus

Damit L sinkt, müsste x verändert werden. Wir erinnern uns, dass wir gemäß Gradient Descent von x einen ausreichenden kleinen Teil der Ableitung von L(x,y) in Abhängigkeit von x abziehen müssten, damit sich x so verändert, dass L sinkt. Diese Ableitung gibt die Steigung der Tangente an, die für ein fixes y den Graphen von L(x,y) am Punkt (x, L(x,y)) berührt. Wir erinnern uns, dass sie auch als Quotient ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%2Cy%29%7D%7Bdx%7D) schreibbar ist.

Nun ist x keine 'trainierbare' Variable, die wir direkt verändern können sondern hängt selbst wiederum von w0, w1 und den Inputs (x0, x1) ab.
x ist eine Funktion von w0, x0, w1 und x1:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%28w_0%2Cw_1%2Cx_0%2Cx_1%29%20%3D%20w_0x_0&plus;w_1x_1)

Die Inputs (x0, x1) können wir nicht verändern. Wir müssen also folgende Frage für jedes Gewicht stellen: Wie müsste bspw. w0 verändert werden, damit x sich derartig verändert, dass L sinkt?
Dazu betrachten wir die Ableitung von x(w0, x0, w1, x1) in Abhängigkeit von w0. Als Quotient ausgedrückt: ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdx%7D%7Bdw_0%7D)

Stellen wir uns vor, wir veränderten den Wert von w0 um einen kleinen Betrag, ca. dw0 (wobei dw0 von 0 kaum zu unterscheiden ist), sodass sich gemäß ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdx%7D%7Bdw_0%7D) der Funktionswert von x um etwa dx verändere, wobei dx von 0 ebenfalls kaum zu unterscheiden ist. Gemäß ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%2Cy%29%7D%7Bdx%7D) würde sich dann der Funktionswert von L um etwa dL verändern. Daher ist der Quotient, der das Verhältnis zwischen der Veränderung von L zu einer sich 0 annähernden Veränderung in w0 angibt, ausdrückbar durch:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%7D%7Bdw_0%7D%3D%5Cfrac%7BdL%7D%7Bdx%7D%5Ccdot%20%5Cfrac%7Bdx%7D%7Bdw_0%7D)

Weiter unten findet sich für diese sogenannte Kettenregel noch ein algebraischer Beweis.

Wir haben also ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%2Cy%29%7D%7Bdw_0%7D) bzw. ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%28w_0%2Cx_0%2Cw_1%2Cx_1%29%2Cy%29%7D%7Bdw_0%7D) gefunden und diese Ableitung von L in Abhängigkeit von w0 beschreibt die Steigung der Tangenten, die für ein fixes x0, w1, x1 und y den Graphen von L am Punkt (w0, L(x(w0,x0,w1,x1),y)) berührt.
Gemäß Gradient Descent wird L, wenn wir von w0 einen ausreichend kleinen Teil von ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%2Cy%29%7D%7Bdw_0%7D) abziehen, sinken.
Analog sinkt L, wenn wir von w1 einen ausreichend kleinen Teil von ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%7D%7Bdw_1%7D) abziehen.

### Trainingsloop

Das 'Training' unseres Neurons besteht aus folgender Schleife, die wir ausführen bis für jedes Trainingsbeispiel ein akzeptabler Loss erzielt wird:
  1. Aus einem Datenset ziehen wir ein Trainingsbeispiel [[x0, x1], y]
  2. Wir geben die Inputs des Trainingsbeispiels (x0, x1) in das Model und errechnen die Ausgabe x.
  4. Wir optimieren die Gewichte (w0, w1) mittels Gradient Descent, d.h. wir überschreiben die Gewichte folgendermaßen:

     ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20w_0%20%3D%20w_0%20-%20%5Calpha%5Ccdot%20%5Cfrac%7BdC%7D%7Bdw_0%7D)
     
     ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20w_1%20%3D%20w_1%20-%20%5Calpha%5Ccdot%20%5Cfrac%7BdC%7D%7Bdw_1%7D)
     
Da wir in jeder Iteration des Loops die beiden obigen Ableitungen benötigen berechnen wir sie vorab:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdC%7D%7Bdw_0%7D%20%3D%202%28x-y%29%20x_0)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdC%7D%7Bdw_1%7D%20%3D%202%28x-y%29%20x_1)

## Beweise/Ableitungen

#### Chain Rule: ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdf%28g%28x%29%29%7D%7Bdx%7D%3D%5Cfrac%7Bdf%28g%28x%29%29%7D%7Bdg%28x%29%7D%5Ccdot%20%5Cfrac%7Bdg%28x%29%7D%7Bdx%7D)

Beginnend mit der allgemeinen Definition der Ableitung für f(g(x)) in Abhängigkeit von x:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdf%28g%28x%29%29%7D%7Bdx%7D%3D%5Clim_%7Bdx%5Cto%5C%5C0%7D%20%5Cfrac%7Bf%28g%28x&plus;dx%29%29-f%28g%28x%29%29%7D%7Bdx%7D)

Bruch erweitern, was gemäß der Gesetze für Limits zulässig ist:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%5Clim_%7Bdx%5Cto%5C%5C0%7D%20%5Cfrac%7Bf%28g%28x&plus;dx%29%29-f%28g%28x%29%29%7D%7Bg%28x&plus;dx%29-g%28x%29%7D%5Ccdot%20%5Cfrac%7Bg%28x&plus;dx%29-g%28x%29%7D%7Bdx%7D)

Substitution im ersten Bruch: g(x+dx)-g(x) = dg(x), dann müssen wir besagten Bruch im ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Clim_%7Bdg%28x%29%5Cto%5C%5C0%7D) betrachten.

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%5Clim_%7Bdg%28x%29%5Cto%5C%5C0%7D%20%5Cfrac%7Bf%28g%28x%29&plus;dg%28x%29%29-f%28g%28x%29%29%7D%7Bdg%28x%29%7D%5Ccdot%20%5Clim_%7Bdx%5Cto%5C%5C0%7D%20%5Cfrac%7Bg%28x&plus;dx%29-g%28x%29%7D%7Bdx%7D)

Per Definition der Ableitung:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%5Cfrac%7Bdf%28g%28x%29%29%7D%7Bdg%28x%29%7D%5Ccdot%20%5Cfrac%7Bdg%28x%29%7D%7Bdx%7D)


#### Ableitung von ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20L%28x-y%29%3D%28x-y%29%5E%7B2%7D) in Abhängigkeit von x:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%7D%7Bdx%7D%20%3D%20%5Clim_%7Bdx%5Cto%5C%5C0%7D%5Cfrac%7B%28x&plus;dx-y%29%5E%7B2%7D-%28x-y%29%5E%7B2%7D%7D%7Bdx%7D)

Terme im Zähler ausmultiplizieren:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%20%5Clim_%7Bdx%5Cto%5C%5C0%7D%5Cfrac%7Bx%5E%7B2%7D&plus;xdx-xy&plus;xdx&plus;dx%5E%7B2%7D-dxy-xy-dxy&plus;y%5E%7B2%7D-x%5E%7B2%7D&plus;2xy-y%5E%7B2%7D%7D%7Bdx%7D)

Zähler vereinfachen:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%20%5Clim_%7Bdx%5Cto%5C%5C0%7D%5Cfrac%7B2xdx&plus;dx%5E%7B2%7D-2dxy%7D%7Bdx%7D)

dx kürzen:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%20%5Clim_%7Bdx%5Cto%5C%5C0%7D2x&plus;dx-2y)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Clim_%7Bdx%5Cto%5C%5C0%7D) auflösen:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%202x-2y%20%3D%202%28x-y%29)

#### Ableitung von x = x0*w0 + x1*w1 in Abhängigkeit von w0:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdx%28x_0%2Cw_0%2Cx_1%2Cw_1%29%7D%7Bdw_0%7D%3D%5Clim_%7Bdw_0%5Cto%5C%5C0%7D%5Cfrac%7Bx_0%28w_0&plus;dw_0%29&plus;x_1w_1-%28x_0w_0&plus;x_1w_1%29%7D%7Bdw_0%7D)

Distributivgesetz:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%5Clim_%7Bdw_0%5Cto%5C%5C0%7D%5Cfrac%7Bx_0w_0&plus;x_0dw_0&plus;x_1w_1-x_0w_0-x_1w_1%7D%7Bdw_0%7D)

Zähler vereinfachen:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3D%5Clim_%7Bdw_0%5Cto%5C%5C0%7D%5Cfrac%7Bx_0dw_0%7D%7Bdw_0%7D)

dw0 kürzen, dann ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Clim_%7Bdw_0%5Cto%5C%5C0%7D) auflösen:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%3Dx_0)
