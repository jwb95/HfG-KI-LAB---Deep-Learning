In dieser Lektion lernen wir ein neues Layer, das eine Alternative zum Fully-Connected-Layer darstellt, kennen: das Convolutional Layer.
Es bildet die Basis für alle SotA-Models im Image-Bereich.

Während ein FC-Layer bei der Erstellung neben der Art der Weight-Initialisierung nur die Anzahl der Neuronen im Layer benötigt, übergeben wir dem Convolutional Layer die Anzahl der 'Filter' und eine Kernel-Größe. Stellen wir uns ein Convolutional Layer mit 64 Filtern und einem 2-dimensionalen Kernel der Größe 3x3 vor. Dann besitzt das Convolutonal Layer über 64 3x3 Matrizen bzw. Filter wobei jeder Eintrag einer Matrix ein Gewicht ist. Der Input zu diesem Layer sei ein schwarz-weiß-Bild der Form 32x32x1. Dann läuft jede der jeweils 64 Matrizen von links nach rechts und von oben nach unten über die Matrix des Bildes und produziert für jeden pixelweisen Schritt
einen Scalar, der die Summe der punktweisen Multiplikation der Filter-Matrix mit dem jeweiligen Bild-Ausschnitts ist. Üblicherweise wird auf diese Summe noch ein Bias addiert.
Da wir 64 Filter festgelegt haben, werden also 64 Summen für jeden 3x3-Bild-Ausschnitt produziert. Dann wäre die Form der Output-Matrix des Layers 30x30x64, da für eine Seite der Länge von 32 nur 30 Ausschnitte der Länge 3 existieren. Man spricht bei den Summen die ein Filter über allen Image-Patches produziert von einer Featuremap. Der Output 30x30x64 besteht also aus 64 Feauturemaps. 

![](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_17A-ConvolutionalNeuralNetworks-WHITEBG.png)

Um die Seitendimensionen nicht zu verändern ist es üblich die Seiten mit Nullen aufzufüllen. Ein 3x3-Filter, der über eine 5x5x1-Matrix läuft, würde eine Matrix der Form 3x3x1 produzieren. Wir könnten folgendermaßen 0-en hinzufügen, sodass aus der 5x5x1-Matrix eine 7x7x1-Matrix wird und ein 3x3-Filter entsprechend eine 5x5x1 Matrix produziert. Das nennt man (Zero-)Padding.

![](https://miro.medium.com/max/1838/1*GE2sny83f_u_o0jf6_wNRQ.png)

Wozu Convolutions?

Betrachte diese Matrix:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%20%26%200%5C%5C%200%20%26%200%20%26%200%20%26%200%5Cend%7Bbmatrix%7D)

Um eine Repräsentation davon zu lernen, welche Image-Patches equivalent mit Matrix 

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%5C%5C%200%20%26%200%20%5Cend%7Bbmatrix%7D)

sind, ist lediglich ein einzelner 2x2-Filter mit eben dieser Matrix notwendig. Der würde folgenden Output produzieren:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D)

Ein FC-Layer kann nicht eine derartig sparsame Extraktion von Information vollziehen. Convolutions nutzen den Umstand aus, dass das selbe 'Feauture', z.B. eine Matrix wie ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B50%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D) wiederholt in den Daten vorkommen kann, was für natürliche Bild- und Audiodaten bzw. im Grunde alle Daten, die sich in Frequenzen unterteilen lassen, die Regel ist.
