### Tiefe Neuronale Netze

Das Netzwerk, das wir in dieser Lektion betrachteten, hatte zwischen dem Input-Layer und dem Output-Layer kein weiteres Layer und verfügte nur über ein einziges Neuron. Neuronale Netze, die sehr komplexe Probleme lösen sollen, haben dagegen viele (hidden) Layer dazwischen, welche möglicherweise mehr als die Anzahl der Netz-Inputs an Neuronen beinhalten. Hinzu kommen außerdem zwei weitere Eigenschaften:

1.) Ein 'Neuron' verfügt üblicherweise über einen 'Bias'. Das ist eine weitere trainierbare Variable, welche die Summe, die ein Neuron produziert, um eine Konstante erweitert.

Verfügte das Neuron unseres Netzes über einen Bias b, so wäre die Funktion x folgendermaßen definiert: ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%28x_0%2Cw_0%2Cx_1%2Cw_1%2Cb%29%20%3D%20x_0w_0&plus;x_1w_1&plus;b)

2.) Nach der Berechnung der Summe eines Neurons folgt in der Regel eine differenzierbare 'Non-linearity'. Also eine Funktion, die ableitbar und nicht linear ist. Wir kennen beispielsweise schon f(x)=x^2

Übliche Non-linearities in tiefen neuronalen Netzen sind ![Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) oder ![ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

![](https://miro.medium.com/max/1400/1*KHs1Chs6TCJDTIIQVyIJxg.png)
