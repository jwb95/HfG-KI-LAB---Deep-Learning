Das Schaubild zeigt eine Form eines sehr einfachen Neuronalen Netzes mit 2 Inputs (x0, x1) und einem Output y.
In dieser Lektion optimieren wir mittels Gradient Descent die zu Beginn zufällig initialisierten Gewichte des Netzes (w0, w1) derartig, dass das Netz anschließend ein spezifisches Problem lösen kann.

![](https://github.com/jwb95/HfG-KI-LAB/blob/main/Lektion%203%20-%20Simples%20Neuronales%20Netz/media/nn.jpg)

Ziel des Trainings eines Neuronalen Netzes ist es, die Gewichte des Netzes so anzupassen, dass es anschließend
sehr gut darin ist, für einen Input den richtigen Output auszugeben.
Betrachte ein Datenset (der Form [x0, x1] -> y) bestehend aus den folgenden 4 Trainingsbeispielen:

[0, 0] -> 0, 
[0, 1] -> 1, 
[1, 0] -> 0, 
[1, 1] -> 1

Wir sehen: Immer dann, wenn x1 = 1, soll das Netz den Wert 1, ansonsten 0 ausgeben.
Eindeutig würde das optimierte Netz wie im folgenden Schaubild aussehen:

![](https://github.com/jwb95/HfG-KI-LAB/blob/main/Lektion%203%20-%20Simples%20Neuronales%20Netz/media/nn_optimized.png)

Der Zweck des Optimierens Neuronaler Netze besteht darin, dass eine annehmbare Lösung eines Problems in Form der richtigen Anpassung der Gewichte automatisiert für komplizierte Probleme, für die im Gegensatz zu unserem Beispiel eine Lösung nicht unbedingt ersichtlich ist, gefunden werden kann. Nutzen wir dennoch das Beispiel, um uns mit dem gängigen Optimierungs-Algorithmus für Neuronale Netze vertraut zu machen.

Zunächst benötigen wir eine konvexe Fehlerfunktion (= lossfunction) L(x,y). Hierbei ist x die Ausgabe des Models für einen spezifischen Input und y das sog. Label - jener Wert, den das Netz unserer Ansicht nach für den spezifischen Input ausgeben soll. Dafür bietet sich der 'Squared Error' an:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20L%28x%2Cy%29%20%3D%20%28x-y%29%5E%7B2%7D)

Der Squared Error besitzt eine Eigenschaft, die ihn für unser Problem als Fehlerfunktion qualizifiert. Betrachten wir 2 Trainingsbeispiele mit jeweils dem Label ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a) und nehmen wir an, das Netz gibt für das erste Beispiel ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a&plus;b) und für das zweite ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a-b) aus, dann ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20L%28a&plus;b%2Ca%29%3DL%28a-b%2Ca%29).

Übersetzt: Es ist egal, ob das Netz für ein Trainingsbeispiel ein x ausgibt, welches zu hoch oder zu niedrig ist. Wesentlich für den Wert, den die Lossfunction annimmt ist nur der absolute Abstand von x zum Label y.

Zunächst initialisieren wir das Netz mit zufälligen Gewichten, z.B. w0 = 0.3 und w1 = -0.2

Ziehen wir nun ein Trainingsbeispiel, z.B. [1, 1] -> 1 und errechnen für dieses die Ausgabe des Netzes:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20NN%281%2C1%29%3D1%5Ccdot%200.3&plus;1%5Ccdot%20%28-0.2%29%3D0.1)

Errechnen wir dann den Squared Error:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20L%280.1%2C1%29%3D%280.1-1%29%5E%7B2%7D%3D%28-0.9%29%5E%7B2%7D%3D0.81)

Wir sehen: Die Performance des Netzes lässt zu wünschen übrig und es gilt die Gewichte des Netzes so anzupassen, dass L sinkt.
Damit L sinkt, müsste x verändert werden. Um festzutellen, ob x erhöht oder verringert werden müsste, betrachten wir die Ableitung von L(x,y) in Abhängigkeit von x. Diese gibt die Steigung der Tangente an, die für ein fixes y den Graphen von L(x,y) am Punkt (x, L(x,y)) berührt. Wir erinnern uns, dass diese auch als Quotient ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%2Cy%29%7D%7Bdx%7D) schreibbar ist.

Nun ist x keine 'trainierbare' Variable, die wir direkt verändern können sondern hängt selbst wiederum von w0, w1 und den Inputs (x0, x1) ab.
x ist eine Funktion von w0, x0, w1 und x1:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%28w_0%2Cw_1%2Cx_0%2Cx_1%29%20%3D%20w_0x_0&plus;w_1x_1)

Wir müssen also die selbe Frage erneut für jedes Gewicht stellen: Wie müsste bspw. w0 verändert werden, damit x steigt bzw. sinkt?
Dazu betrachten wir die Ableitung von x(w0, x0, w1, x1) in Abhängigkeit von w0. Als Quotient ausgedrückt: ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdx%7D%7Bdw_0%7D)

Stellen wir uns vor, wir veränderten den Wert von w0 um einen kleinen Betrag, ca. dw0 (wobei dw0 von 0 kaum zu unterscheiden ist), sodass sich gemäß ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7Bdx%7D%7Bdw_0%7D) der Funktionswert von x um etwa dx verändere. Gemäß ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%2Cy%29%7D%7Bdx%7D) würde sich dann der Funktionswert von L um etwa dL verändern. Daher ist der Quotient, der das Verhältnis der Veränderung von L in Bezug zu einer sich 0 annähernden Veränderung in w0 angibt, ausdrückbar durch:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%7D%7Bdw_0%7D%3D%5Cfrac%7BdL%7D%7Bdx%7D%5Ccdot%20%5Cfrac%7Bdx%7D%7Bdw_0%7D)

Da in der Erklärung nur von 'etwa' gesprochen wird ist das Schließen auf eine Gleichung, obwohl sie 'etwa stimmen müsste' nicht korrekt.
Doch sie stimmt tatsächlich. Weiter unten wird ein Beweis geliefert.

Wir haben also ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%2Cy%29%7D%7Bdw_0%7D) bzw. ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%28w_0%2Cx_0%2Cw_1%2Cx_1%29%2Cy%29%7D%7Bdw_0%7D) gefunden und diese Ableitung von L in Abhängigkeit von w0 beschreibt die Steigung der Tangenten, die für ein fixes x1, w1, x1 und y den Graphen von L am Punkt (w0, L((w0,x0,w1,x1),y)) berührt.
Gemäß Gradient Descent wird L, wenn wir von w0 einen ausreichend kleinen Teil von ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%28x%2Cy%29%7D%7Bdw_0%7D) abziehen, sinken.
Analog sinkt L, wenn wir von w1 einen ausreichend kleinen Teil von ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdL%7D%7Bdw_1%7D) abziehen.

Das 'Training' unseres Neuronalen Netzes besteht aus folgender Schleife, die wir ausführen bis für jedes Trainingsbeispiel ein akzeptabler Loss erzielt wird:
  1. Aus einem Datenset ziehen wir ein Trainingsbeispiel [[x0, x1], y]
  2. Wir geben die Inputs des Trainingsbeispiels (x0, x1) in das Netz und errechnen die Ausgabe x.
  4. Wir optimieren die Gewichte (w0, w1) mittels Gradient Descent, d.h. wir überschreiben die Gewichte folgendermaßen:

     ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20w_0%20%3D%20w_0%20-%20%5Calpha%5Ccdot%20%5Cfrac%7BdC%7D%7Bdw_0%7D)
     
     ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20w_1%20%3D%20w_1%20-%20%5Calpha%5Ccdot%20%5Cfrac%7BdC%7D%7Bdw_1%7D)
     
Da wir also in jeder Iteration des Loops die beiden obigen Ableitungen benötigen berechnen wir sie vorab. Beweise für alle Ableitungen stehen am Ende.

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdC%7D%7Bdw_0%7D%20%3D%202%28x-y%29%20x_0)

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cfrac%7BdC%7D%7Bdw_1%7D%20%3D%202%28x-y%29%20x_1)
