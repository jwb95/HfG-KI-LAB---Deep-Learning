Cats vs Dogs Dataset: 1tsZ2a_KHg3yp1FHxArWgllCs_pgvWtXg

In dieser Lektion lernen wir ein neues Layer, das eine Alternative zum Fully-Connected-Layer darstellt, kennen: das Convolutional Layer.
Es bildet die Basis für alle SotA-Models generativer Image- und Audio-Models. Im Notebook implementieren wir ein Conv-Net für Binary-Image-Classification (Hund oder Katze?).

Während ein FC-Layer bei der Erstellung neben der Art der Weight-Initialisierung nur die Anzahl der Neuronen im Layer benötigt, übergeben wir dem Convolutional Layer die Anzahl der 'Filter' und eine Kernel-Größe. Stellen wir uns ein Convolutional Layer mit 64 Filtern und einem 2-dimensionalen Kernel der Größe 3x3 vor. Dann besitzt das Convolutonal Layer über 64 3x3 Matrizen bzw. Filter wobei jeder Eintrag einer Matrix ein Gewicht ist. Der Input zu diesem Layer sei ein schwarz-weiß-Bild der Form 32x32x1. Dann läuft jede der jeweils 64 Matrizen von links nach rechts und von oben nach unten über die Matrix des Bildes und produziert für jeden pixelweisen Schritt
einen Scalar, der die Summe der punktweisen Multiplikation der Filter-Matrix mit dem jeweiligen Bild-Ausschnitts ist. Üblicherweise wird auf diese Summe noch ein Bias addiert.
Da wir 64 Filter festgelegt haben, werden also 64 Summen für jeden 3x3-Bild-Ausschnitt produziert. Dann wäre die Form der Output-Matrix des Layers 30x30x64, da für eine Seite der Länge von 32 nur 30 Ausschnitte der Länge 3 existieren. Man spricht bei den Summen die ein Filter über allen Image-Patches produziert von einer Featuremap. Der Output 30x30x64 besteht also aus 64 Feauturemaps. 

<img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_02_17A-ConvolutionalNeuralNetworks-WHITEBG.png" width="1000" />

Um die Seitendimensionen nicht zu verändern ist es üblich die Seiten mit Nullen aufzufüllen. Ein 3x3-Filter, der über eine 5x5x1-Matrix läuft, würde eine Matrix der Form 3x3x1 produzieren. Wir könnten folgendermaßen 0-en hinzufügen, sodass aus der 5x5x1-Matrix eine 7x7x1-Matrix wird und ein 3x3-Filter entsprechend eine 5x5x1 Matrix produziert. Das nennt man (Zero-)Padding.

<img src="https://miro.medium.com/max/1838/1*GE2sny83f_u_o0jf6_wNRQ.png" width="1000" />

## Wozu Convolutions?

Betrachte diese Matrix der Form 4x4x1.

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%20%26%200%5C%5C%200%20%26%200%20%26%200%20%26%200%5Cend%7Bbmatrix%7D)

Um eine Repräsentation davon zu lernen, welche Image-Patches equivalent mit Matrix 



![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%5C%5C%200%20%26%200%20%5Cend%7Bbmatrix%7D)

sind, ist lediglich ein einzelner 2x2x1-Filter mit eben dieser Matrix notwendig. Der würde folgenden Output produzieren:

![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D)

Ein FC-Layer kann nicht eine derartig sparsame Extraktion von Information vollziehen. Convolutions nutzen den Umstand aus, dass das selbe 'Feauture', z.B. eine Matrix wie ![](https://latex.codecogs.com/png.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%5C%5C%200%20%26%200%20%5Cend%7Bbmatrix%7D) wiederholt in den Daten vorkommen kann, was für natürliche Bild- und Audiodaten bzw. im Grunde alle Daten, die sich in Frequenzen unterteilen lassen, die Regel ist.

Bemerke: eine 2-dimensionale Convolution (also eine solche, bei der die Filter von links nach rechts und von oben nach unten wandern) hat eine 3-dimensionalen Filtergröße. Wenn wir beispielsweise eine Convolution über ein RGB-Bild vollziehen wollen, so benötigen wir einen Filter der Form (x, y, 3). Denn wie bereits erwähnt ist das Ergbenis des Filters über einem Bild-Ausschnitt die Summe der punktweisen Multiplikationen. Haben wir einen Bildausschnitt der Größe (7, 7, 3), so brauchen wir per Definition auch eine Filter-Matrix mit dieser Form. Die Anzahl der Filter, die wir in einem Layer verwenden wollen, wird jedoch vom User spezifiziert. D.h. definieren wir bspw. auf einem Input der Form (x, y, 20) eine Convolution mit 100 Filtern mit jeweils Filtergröße 3x3, so erhalten wir 100 Matrizen der Form (3, 3, 20) oder auch eine 4-dimensionale Matrix der Form (100, 3, 3, 20). ;)
Da wir nun wissen, dass die Convolutions, die sich für Bilder eignen, 2-dimensionale Convolutions genannt werden, können wir uns denken, dass es auch 1-dimensionale, 3-dimensionale bishin zu theoretisch unendlich-dimensionalen Convolutions gibt.

## Conv-Nets

Convolutional Neural Networks sind solche tiefen Neuronalen Netze, die anstelle von Fully-Connected Layers hauptsächlich Convolutional Layers verwenden.
Üblicherweise werden diese wie im folgenden Schaubild konstruiert. Ignoriere hierbei vorerst die Indikation des Detection Layers, sondern betrachte die 3-dimensionalen CNN-Layers. Mit fortschreitenden Layers erhöht sich die Anzahl der Filter im jeweiligen Layer. Das erste Layer hat 192 Filter und das letzte 1024. Gleichzeitig schrumpfen die Seitendimensionen, die ursprünglich durch das Image definiert wurden (hier 448x488). Die von jedem großen Quader eingefassten kleinen Quader geben einen Hinweis auf die Filter-Größe jedes Layers. (7x7x3 für das erste Layer, anschließend 3x3x192 usw.).

<img src="https://blogs.sas.com/content/subconsciousmusings/files/2018/11/YOLOnetworkarchitecture.png" width="1000" />

Welchen Zweck hat das Vermindern der Seitendimensionen und warum tritt es auf?
Folgende Beobachtungen sind wichtig:

1.) Die Zeit, die ein Filter benötigt, um über ein quadratisches Bild zu wandern wächst quadratisch mit der Seitenlänge des Bildes.
So benötigt eine Convolution über eine Matrix der Form (8, 8, f) 64 Zeiteinheiten, über eine Matrix der Form (16, 16, f) bereits 256 Zeiteinheiten.
Es stellt sich also eine Frage der Effizienz, denn das Training eines Conv-Nets ist unter umständen bereits langwierig.

2.) In Klassifikationsproblemen ist die Anzahl der Skalare im Output-Layer weitaus geringer als die des Input-Layers. Für ein Problem, bei dem ein Neuronales Netz voraussagen soll, ob ein RGB-Bild mit Seitenlängen 64x64 einen Hund oder eine Katze abbildet, besteht das Input-Layer 64x64x3 = 12288 Skalaren, das Output-Layer benötigt jedoch (je nach Design) nur 1 oder 2 Skalare. Also muss die Anzahl der Variablen an irgendeiner oder mehreren Stellen im Neural Network verringert werden.
Die Intuition hinter einer annähernd kontinuierlichen Verringerung der Anzahl der Skalare der internen Repräsentation des Neuronalen Netzwerkes von Layer zu Layer ist wie folgt:
Betrachten wir ein Neuronales Netz mit mehreren Convolutional Layers und Filtern der Größe 3x3x1, so kann das erste Layer nur Features lernen, die durch eine Matrix der Größe 3x3 abbildbar sind, also bspw. Linien, die sich über 3 Pixel erstrecken. Das folgende Layer kann mittels der durch Layer 1 produzierten Repräsentation bereits kompliziertere Features 'erkennen', die sich aus den benachbarten 3x3-Matrizen zusammensetzen lassen (siehe für Details den folgenden Abschnitt 'Receptive Field'). Mit dem Fortschreiten der Layer kann die jeweilige Repräsentation zunehmend kompliziertere und größere Bereiche des Ursprungsbildes einnehmende Features abdecken, bis irgendwann ein solches Feature erkannt werden kann, welches beispielsweise eindeutig auf einen Elephanten schließen lässt. Siehe dazu das folgende Schaubild:

<img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Deep_Learning.jpg" width="1000" />

## Receptive Field

Betrachten wir ein eindimensionales, die Serienlänge erhaltendes Conv-Net mit 3 Layern mit jeweils einem Kernel der Größe 3, n Filtern und Zero-Padding. Nach 3 Convolutions haben wir anstelle einer Serie mit Shape (x, 3) eine Featuremap mit Shape (x, n). Dann encodiert für jeden 'Pixel' die ihm zugeordnete Feature-Achse n Features, die eine Beschreibung des dem jeweiligen Pixel zugeordneten Ausschnitts der Serie darstellen. Wie groß ist ein solcher Ausschnitt?
Betrachte das folgende Schaubild: Jeder Pixel im untersten Layer wird von 3 benachbarten Pixeln der vorangestellten Featuremap beeinflusst - das ist logisch, denn ein jedes Feature des Pixels ist ja, gemäß der Kernel-Größe 3, die Summe der punktweisen Multiplikation der jeweiligen 3x1-Filtermatrix mit dem entsprechenden 3x1-Feauturemap-Ausschnitt. Wir sehen, jedes weitere Conv-Layer mit Kernel-Größe 3x1 erhöht die Anzahl der Pixel, die den Output-Pixel beeinflussen um 2.

<img src="https://github.com/jwb95/HfG-KI-LAB---Deep-Learning/blob/main/Lektion%2007%20-%20Convolutions/media/receptive_field1.jpg" width="1000" />

Das Receptive Field unseres Conv-Nets ist also =7. Das ist gleichbedeutend mit: Der längste Ausschnitt, den das Netzwerk zum 'Lernen' betrachten kann, beträgt 7 Pixel. Damit könnte, insofern die Input-Daten aus Serien, die länger als 7 Pixel sind, keinen globalen, sondern nur lokale Zusammenhänge aus den Daten gelernt werden.
Analog für ein 2-dimensionales Conv-Net mit entsprechenden 3x3-Filtern wären die größten Bildausschnitte aus denen das Netz lernen könnte 7x7-Bildausschnitte.
Wollten wir also ein Netz aus ausschließlich Convolutions bauen, welches den globalen Kontext für verhältnismäßig kleine Bilder mit bspw. 64x64 Pixeln betrachten können soll, so benötigten wir dafür mindestens 32 Convolutional Layers. Wie bereits gezeigt macht es vom Standpunkt der Effizienz also Sinn die Seitenlängen innerhalb des Netzwerkes zu verringern. Wie sieht es dann mit dem Receptive Field aus?

<img src="https://github.com/jwb95/HfG-KI-LAB---Deep-Learning/blob/main/Lektion%2007%20-%20Convolutions/media/receptive_field2.jpg?raw=true" width="1000" />

Wie wir sehen lässt sich mit einem Pooling-Layer mit Poolsize =2 (siehe für Details die folgende Sektion), das in irgendeiner Form die Informationen aus 2 benachbarten Pixeln zusammenfasst, das Receptive Field sehr "kostengünstig" verdoppeln. Jedoch besteht der Haken, dass wir ein Informationsbottleneck einbauen, wenn wir bspw. 2 Skalare zu einem zusammenfassen - es ist letzten Endes eine Kompression der Daten.

## Wie werden die Seitendimensionen verringert?

Wie wir bereits wissen, verringert sich, insofern wir auf Zero-Padding verzichten, die Seitendimensionen für jedes Layer automatisch. Jedoch mit einem 3x3x?-Filter um gerademal 2 Pixel. Möchte man also die Seitendimensionen drastischer verringern, müssen andere Techniken in Betracht gezogen werden. Es gibt 2 Varianten:

1.) Strides. Definieren wir für eine 2-dimensionale Convolution bspw. einen Stride von (2, 2), so überspringt jeder Filter jeden zweiten Image-Patch. Entsprechend wäre der Output für eine Matrix der Form 16x16x?, über die wir eine Convolution mit Zero-Padding, besagtem Stride und 64 Filtern definieren, von der Form 8x8x64.

2.) Pooling. Pooling Layers agieren wie Convolutional Layers ohne Filter. Sie haben auch einen Stride, jedoch statt einer Kernel-Größe und Anzahl an Filters eine sog. Poolsize. Ein Pooling Layer mit einer Poolsize von (4, 4) und einem Stride von (2, 2) errechnet für jeden Image-Patch der Größe 4x4 des Bildes einen Scalar, wobei jeder 2. Pixel gemäß des Strides übersprungen wird. Die Art der Berechenung des Skalars kann verschiedene Formen annehmen. So gibt es Max-Pooling (der höchste Wert des Image-Patches wird gewählt), Mean-Pooling (Der Durchschnitt aller Werte des Image-Patches wird berechnet) etc. pp. Die Pooling-Operation wird für jedes Feature der Featuremap unabhängig vollzogen, folglich resultiert aus einer Matrix der Form 16x16x64 mit Padding eine Matrix 8x8x64, wenn wir ein Pooling-Layer mit einer Poolsize und einem Stride von jeweils (2, 2) anwenden.

Final noch wie versprochen die Erläuterung, was es mit dem Detection Layer im vorletzten Schaubild auf sich hat: Am Ende der Convolutions wird die Dimensionalität der letzten Featuremap zu 1D reduziert. Das heißt aus einer Matrix der Dimension 16x16x512 produzieren wir einen Vektor mit 16x16x512 = 131072 Skalaren, der die identische Information entählt. Dann können wir diesen Vektor in ein finales Fully-Connected Layer geben, welches den Output- bzw. die Output-Skalare produziert.
Ein FC-Layer, welches weniger Output-Neuronen als Input-Neuronen besitzt ist ergo auch eine Möglichkeit um die Anzahl der Skalare durch ein Layer zu reduzieren.
