Das Schaubild zeigt eine Form eines sehr einfachen Neuronalen Netzes mit 2 Inputs (x0, x1) und einem Output y.
In dieser Lektion optimieren wir mittels Gradient Descent die Gewichte des Netzes (w0, w1) derartig, dass das Netz anschließend ein spezifisches Problem lösen kann.

![](https://github.com/jwb95/HfG-KI-LAB/blob/main/Lektion%203%20-%20Simples%20Neuronales%20Netz/media/nn.jpg)

Ziel des Trainings eines Neuronalen Netzes ist es, die Gewichte des Netzes so anzupassen, dass es anschließend
sehr gut darin ist, für einen Input den richtigen Output auszugeben.
Betrachte ein Datenset (der Form [x0, x1] -> y) bestehend aus den folgenden 4 Trainingsbeispielen:

[0, 0] -> 0

[0, 1] -> 1

[1, 0] -> 0

[1, 1] -> 1

Wir sehen, immer dann, wenn x1 = 1, soll das Netz den Wert 1, ansonsten 0 ausgeben.


Mittels welchem Algorithmus können wir die Gewichte entsprechend anpassen?

Zunächst benötigen wir eine konvexe Fehlerfunktion (= lossfunction) L(x,y). Hierbei ist x die Ausgabe des Models für einen spezifischen Input und y das sog. Label, jener Wert, den das Netz unserer Ansicht nach für den spezifischen Input ausgeben soll. Dafür bietet sich der 'Squared Error' an:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20L%28x%2Cy%29%20%3D%20%28x-y%29%5E%7B2%7D)

Bemerke: In der Lektion 2 optimierten wir die gleiche lossfunction wobei das Label für jeden Input x einfach 0 war.

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B150%7D%20L%28x%29%3D%28x-0%29%5E%7B2%7D%20%3D%20x%5E%7B2%7D)

Der Squared Error besitzt eine Eigenschaft, die ihn als Fehlerfunktion qualizifiert. Betrachten wir 2 Trainingsbeispiele mit jeweils dem Label ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a) und nehmen wir an, das Netz gibt für das erste Beispiel ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a&plus;b) und für das zweite ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20a-b) aus, dann ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20L%28a&plus;b%2Ca%29%3DL%28a-b%2Ca%29).

Übersetzt: Es ist egal, ob das Netz für ein Trainingsbeispiel eine Voraussage macht, die zu hoch oder zu niedrig ist. Für den Fehler wesentlich ist lediglich der Abstand zum Label.

Das 'Training' eines Neuronalen Netzes besteht aus folgendem Loop, den wir ausführen, bis sich
  1. Aus einem Datenset ziehen wir ein Trainingsbeispiel. Wir geben die Inputs des Trainingsbeispiels (x0, x1) in das Netz und errechnen die Ausgabe y.
  2. Wir evaluieren die Fehlerfunktion für das Trainingsbeispiel.
  3. Wir optimieren die Gewichte mittels Gradient Descent.

