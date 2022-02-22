Das Neuron, das wir in Lektion 3 und 4 betrachteten, bestand nur aus dem Input- und dem Output-Layer.m Neuronale Netze, die sehr komplexe Probleme lösen sollen, haben dagegen viele (hidden) Layer dazwischen, welche möglicherweise mehr als die Anzahl der Netz-Inputs an Neuronen beinhalten.

Neuronale Netze sind sog. 'Universal Function Approximators'. Ihre Funktionalität besteht darin, dass, wenn wir sie zur Lösung eines spezfischen Problems erfolgreich optimieren können, sie als eine mathematische Funktion fungieren, die dieses Problem löst - bspw. Input-Daten auf die richtigen Labels zu mappen. 'Universal' ein Neuronales Netz mit nur einem Fully-Connected-Layer zwischen Input- und Outputlayer theoretisch jede mathematische Funktion annehmen kann, insofern genügend Neuronen im Layer vorhanden sind. Außerdem müssen die Neuronen zwei weitere Eigenschaften erfüllen:

1.) Ein 'Neuron' verfügt üblicherweise über einen 'Bias'. Das ist eine weitere trainierbare Variable, welche die Summe, die ein Neuron produziert, um eine Konstante erweitert.

Verfügte das Neuron aus Lektion 3/4 über einen Bias b, so wäre die Funktion x folgendermaßen definiert: ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20x%28x_0%2Cw_0%2Cx_1%2Cw_1%2Cb%29%20%3D%20x_0w_0&plus;x_1w_1&plus;b)

What it do?
Im ![Desmos-Graphen](https://www.desmos.com/calculator/w76gukcw33) sei x = der einzige Input zu einem Neuron f(x), w das zugehörige Gewicht und b der Bias. Den Bias auf 0 zu belassen ist gleichwertig damit keinen Bias zu benutzen. Stellen wir uns vor, wir wollten das Neuron derartig optimieren, dass es eine Funktion findet, die im Graphen die beiden Punkte voneinander räumlich abtrennt.
Wir stellen fest, dass, wenn wir w verändern immer nur eine Lineare Funktion gefunden wird, die durch den Ursprung (0,0) verläuft. Benutzen wir auch den Bias können wir eine gewünschte Funktion optimieren können.


2.) Nach der Berechnung der Summe eines Neurons folgt in der Regel eine differenzierbare 'Non-linearity'. Also eine Funktion, die ableitbar und nicht linear ist. Wir kennen beispielsweise schon f(x)=x^2

Übliche Non-linearities in tiefen neuronalen Netzen sind ![Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) oder ![ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

If we only allow linear activation functions in a neural network, the output will just be a linear transformation of the input, which is not enough to form a universal function approximator


A simpler way to understand what the bias is: it is somehow similar to the constant b of a linear function

y = ax + b

It allows you to move the line up and down to fit the prediction with the data better.

Without b, the line always goes through the origin (0, 0) and you may get a poorer fit.


![](https://miro.medium.com/max/1400/1*KHs1Chs6TCJDTIIQVyIJxg.png)

