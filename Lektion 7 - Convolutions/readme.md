In dieser Lektion lernen wir ein neues Layer, das eine Alternative zum Fully-Connected-Layer darstellt, kennen: das Convolutional Layer.
Es bildet die Basis für alle SotA-Models im Image-Bereich.

Während ein FC-Layer bei der Erstellung neben der Art der Weight-Initialisierung nur die Anzahl der Neuronen im Layer benötigt, übergeben wir dem Convolutional Layer die Anzahl der 'Filter' und eine Kernel-Größe. Stellen wir uns ein Convolutional Layer mit 64 Filtern und einem 2-dimensionalen Kernel der Größe 3x3 vor. Dann besitzt das Convolutonal Layer über 64 3x3 Matrizen bzw. Filter wobei jeder Eintrag einer Matrix ein Gewicht ist. Der Input zu diesem Layer sei ein schwarz-weiß-Bild der Form 32x32x1. Dann läuft jede der jeweils 64 Matrizen von links nach rechts und von oben nach unten über die Matrix des Bildes und produziert für jeden pixelweisen Schritt
einen Scalar, der die Summe der punktweisen Multiplikationen der Filter-Matrix und des jeweiligen Bild-Ausschnitts ist. Üblicherweise wird auf diese Summe noch ein Bias addiert.
Da wir 64 Filter festgelegt haben, werden also 64 Summen für jeden 3x3-Bild-Ausschnitt produziert. Dann wäre die Form der Output-Matrix des Layers 30x30x64, da für eine Seite der Länge von 32 nur 30 Ausschnitte der Länge 3 existieren.

![](https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_17A-ConvolutionalNeuralNetworks-WHITEBG.png)

Um die Seitendimensionen nicht zu verändern ist es üblich die Seiten mit Nullen aufzufüllen. Ein 2x2-Filter, der über eine 4x4x1-Matrix läuft, würde eine Matrix der Form 3x3x1 produzieren. Wir könnten folgendermaßen 0-en hinzufügens, sodass aus der 4x4x1-Matrix eine 5x5x1-Matrix wird und ein 2x2-Filter entsprechend eine 4x4x1 MAtrix produziert. Das nennt man (Zero-)Padding.

![](https://classic.d2l.ai/_images/conv-pad.svg)


