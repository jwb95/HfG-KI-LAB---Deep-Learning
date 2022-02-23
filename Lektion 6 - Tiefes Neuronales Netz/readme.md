In dieser Lektion implementieren und trainieren wir ein Tiefes Neuronales Netz, um ein Mutliclass-Classification-Problem zu lösen. Konkret trainieren wir das Model auf dem MNIST-Datenset, sodass es nach dem Training in der Lage ist Bilder jeweils einer handgeschriebenen Ziffer (0, 1, ... 9) der jeweils zugehörigen Ziffer zuzuordnen.

![](https://miro.medium.com/max/1400/1*LyRlX__08q40UJohhJG9Ow.png)

Das Neuron, das wir in Lektion 3 und 4 betrachteten, bestand nur aus dem Input- und dem Output-Layer. Tiefe Neuronale Netze haben dagegen (hidden) Layers dazwischen, welche möglicherweise mehr als die Anzahl der Netz-Inputs an Neuronen beinhalten.

![](https://miro.medium.com/max/1400/1*KHs1Chs6TCJDTIIQVyIJxg.png)

Die Funktionalität NNs besteht darin, dass, wenn wir sie zur Lösung eines spezfischen Problems erfolgreich optimieren können, sie als eine mathematische Funktion fungieren, die dieses Problem löst - bspw. Input-Daten auf die richtigen Labels zu mappen. Das Universal Approximation Theorem besagt, dass ein NN mit nur einem Fully-Connected-Layer zwischen Input- und Outputlayer theoretisch jede kontinuierliche Funktion, die Inputs einer begrenzten Range nimmt, annehmen kann, insofern genügend Neuronen im Layer vorhanden sind. Außerdem müssen die Neuronen dafür zwei weitere Eigenschaften besitzen:

1.) Ein 'Neuron' verfügt üblicherweise über einen 'Bias'. Das ist eine weitere trainierbare Variable, welche die Summe, die ein Neuron produziert, um eine Konstante erweitert.

Verfügte das Neuron aus Lektion 3/4 über einen Bias b, so wäre die Funktion x folgendermaßen definiert: ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%28x_0%2Cw_0%2Cx_1%2Cw_1%2Cb%29%20%3D%20x_0w_0&plus;x_1w_1&plus;b)

Im Desmos-Graphen (https://www.desmos.com/calculator/7lotzx4thg) findet sich eine Erklärung dafür, warum ein Bias die Funktionalität unseres Neurons erweitern wird. Die Beobachtung generalisiert, sodass es wahr ist, dass auch ein Netz mit vielen Neuronen und möglicherweise Layers nicht jede lineare Funktion annehmen kann. Dann kann es schon gar kein 'Universal Function Approximator' sein.

2.) Der Vorgang bei dem eine Input-Variable zuerst mit einem Gewicht multipliziert und dann mit einem Bias addiert wird ist nichts anderes als eine Lineare Funktion. Die Summe von beliebig vielen Linearen Funktionen ergibt immer eine Lineare Funktion und beliebig viele verschachtelte Lineare Funktionen
f(g(...(h(x))) ergeben immer eine Lineare Funktion. Jede existierende Lineare Funktion ist durch 2 Zahlen ausdrückbar, das ist nicht sonderlich kompliziert.
Funktionen, die etwas interessantes tun sollen, sind kompliziert. So kompliziert, dass wir sie nicht selbst bauen wollen, sondern auf die Idee kommen, sie von einem NN lernen zu lassen. Solche Funktionen sind nicht linear, daher ist es notwendig, dass Non-Linearities im Netwerk verbaut sind.
Die letzte Zutat für unser Neuron besteht also aus einer sog. Non-Linearity. Das ist eine Funktion, die einen nicht-linearen Graph besitzt. Wir kennen z.b schon f(x)=x^2. Üblicherweise werden Non-Linearities auf den Output eines Neurons angewandt. Etwa so: Es sei ϕ eine beliebige nicht-lineare Funktion, dann können wir unser Neuron aus Lektion 3/4 erweitern, sodass: ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%28x_0%2Cw_0%2Cx_1%2Cw_1%2Cb%29%20%3D%20%5Cphi%20%28x_0w_0&plus;x_1w_1&plus;b%29)

Siehe im Desmos-Graphen (https://www.desmos.com/calculator/icdrdyrpq4), welche Funktionalität eine Non-Linearity unserem Neuron hinzufügt.
Wichtig ist, dass diese Non-Linearity ableitbar ist, damit alle Ableitungen der Lossfunction in Abhängigkeit zu jeweils jeder trainierbaren Variable berechnet werden können. Zu den häufigsten Non-Linearities gehören ![Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) und ![ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)). Dennoch ist prinzipiell jede ableitbare Non-Linearity denkbar, denn jede kontinuierliche Funktion für eine begrentze Range ist als Summe von kontinuierlichen nicht-linearen Funktionen beliebiger Art ausdrückbar. In ![diesem Artikel](https://towardsdatascience.com/can-neural-networks-really-learn-any-function-65e106617fc6) baut der Autor, um sich den Verhalt zu veranschaulichen, ein Polynom aus einer Summe von nicht-linearen Funktionen.
