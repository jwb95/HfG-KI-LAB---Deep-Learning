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

Wir sehen, immer dann, wenn x1 = 1, soll das Netz 1 ansonsten 0 ausgeben-


Wie können wir die Gewichte entsprechend anpassen?

Zunächst benötigen wir eine konvexe Fehlerfunktion (= lossfunction) L. Wir wählen ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B300%7D%20L%28x%2Cy%29%20%3D%20%28x-y%29%5E%7B2%7D)

Das 'Training' eines Neuronalen Netzes besteht aus folgendem Loop, den wir ausführen, bis sich
  1. Aus einem Datenset ziehen wir ein Trainingsbeispiel. Wir geben die Inputs des Trainingsbeispiels (x0, x1) in das Netz und errechnen die Ausgabe y.
  2. Wir evaluieren die Fehlerfunktion für das Trainingsbeispiel.
  3. Wir optimieren die Gewichte mittels Gradient Descent.

