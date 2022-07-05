In dieser Lektion machen wir uns mit der Abbildung von Daten in einen komprimierten Feature-Space vertraut. Die NNs, die wir für Klassikations-Probleme einsetzten, haben genau so etwas prinzipiell schon getan - nämlich hoch-dimensionale (Bild-)Daten auf eine stark komprimierte Repräsentation (z.B. einen Vektor mit 9 Skalaren, der angibt, welche handgeschriebene Ziffer auf einem Bild dargestellt ist) abgebildet.

## Injektivität

Wir bauen in dieser Lektion eine neue Netzwerk-Architektur, den Autoencoder. Dieser besteht aus zwei Teilen bzw. Funktionen, die durch jeweils ein NN gelernt werden sollen. Eine davon soll ebenfalls einen hoch-dimensionalen Datenpunkt (wie ein Bild) auf eine komprimierte Repräsentation abbilden. Diese Funktion soll im Gegensatz zur Klassifikation injektiv sein. Injektiv bedeutet, dass jeder Funktionswert/Output, den die Funktion annehmen kann, maximal einem Input zugeordnet wird.

Eine 'klassifizierende' Funktion wie wir sie bereits bauten, ist nicht injektiv, denn mehrere Bilder werden durch sie auf die gleiche Klasse abgebildet. So würden möglicherweise zwei verschiedene Bilder der Ziffer 4 den gleichen Vektor [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] produzieren.

Weitere Beispiele:

![](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20x%5E2) ist nicht injektiv, denn beispielsweise gibt es für den Funktionswert ![](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%204) zwei Lösungen, nämlich ![](https://latex.codecogs.com/gif.latex?x%3D2) und ![](https://latex.codecogs.com/gif.latex?x%3D-2).

![](https://latex.codecogs.com/gif.latex?f%28x%29%20%3D%20x&plus;2) ist injektiv, denn für jeden Funktionswert f(x) gibt es nur genau ein x, welches man in die Funktion geben müsste, um dieses f(x) zu erhalten.


Nehmen wir an wir haben eine endliche Liste von einmaligen 2-dimensionalen Matrizen (z.B. Bildern), dann könnten wir eine Funktion definieren, die jedes Element dieser Liste auf eine einmalige, komprimierte Repräsentation (also eine solche Matrix, die weniger Skalare besitzt als die ursprüngliche) abbildet. Dadurch, dass jede komprimierte Repräsentation einmalig ist, existiert zu dieser Funktion auch eine Umkehrfunktion, die die komprimierte Repräsentation zurück auf die Ursprungsmatrix abbildet. Die Funktion ist also injektiv und dadurch theoretisch umkehrbar.

Ein Autoencoder ist ein NN, welches beide diese Funktionen für ein Datenset lernen soll. Er besteht aus einem Encoder, welcher ein Input-Image auf einen sog. Latentcode abbildet und einem Decoder, welcher diesen Latentcode zurück auf das Image mappen soll. Der Latentcode besitzt üblicherweise weniger Skalare als das Image. Der Encoder vollzieht also eine Kompression, der Decoder versucht diese Kompression umzukehren.

<img src="https://www.compthree.com/images/blog/ae/ae.png" width="1000" />

Welche Eigenschaften müsste ein perfekter trainierter Autoencoder erfüllen?
1. Der Encoder produziert für jedes Image einen einzigartigen Latentcode.
2. Der Decoder ist in der Lage auf Basis eines Latentcodes das zugehörige Bild verlustfrei zu rekonstruieren.

## Welchen Nutzen haben Autoencoder?

Der naheliegendste Grund ist Datenkompression. Nehmen wir an, wir haben eine Liste mit w Bildern. Diese Liste nimmt auf der Festplatte x Bits Speicher ein.
Trainieren wir nun einen Autoencoder mit y sog. Latentdimensions (also ein Bild wird durch den Encoder auf einen aus y Skalaren bestehenden Vektor abgebildet). Üblicherweise wiegt jeder Parameter des Netzwerkes 32 bits, denn er ist idR ein float32. Nehmen wir an der trainierte Autoencoder ist nun perfekt. Die Speichergröße in Bits, die der Decoder auf der Festplatte einnimmt, nennen wir z. Listen wir das nochmal auf:

w = Anzahl der Bilder

x = Größe aller Bilder in Bits

y = Anzahl der Latentdimensions

z = Größe des Decoders in Bits

Dann stellt der Decoder eine verlustfreie komprimierte Repräsentation unseres Datensets dar, genau dann wenn ![](https://latex.codecogs.com/gif.latex?z%20&plus;%20w*y*32%20%3C%20x)

In Prosa: Wenn der Speicher, der benötigt wird um unseren Decoder und die Liste der unseren Bildern zugehörigen Latentcodes abzuspeichern kleiner ist als der Speicher, den wir benötigten, um unsere Bilder direkt abzuspeichern, hätten wir eine effektive Möglichkeit um Festplattenspeicher zu sparen.

## Wieso brauchen wir einen Encoder?

Genau genommen brauchen wir für den Encoder-Teil eigentlich kein Neuronales Netz. Schließlich könnten wir uns auch einfach eine Funktion ausdenken, die für jedes Bild einen einmaligen Latentcode produziert. Dann müssten wir nur einen Decoder trainieren, der jeden Code auf das entsprechende Bild abbildet. Man könnte annehmen, dass man vielleicht so noch mehr Speicher sparen könnte, da wir mit einem Computerprogramm aus einer Zeile Code bereits eine Liste von einmaligen Latentcodes produzieren könnten, z.B:

```latentcodes = [np.float32(np.zeros(number_of_latentdimensions) + i) for i in range(number_of_images)]```

Wir müssten dann also nicht mal mehr die Codes speichern, sondern lediglich das Programm, welches diese produziert.
Der Grund für ein Encoder-Netzwerk ist, dass ein solches Encoding die Aufgabe des Decoders maximal hart macht. Da es zwischen der Information in einem solchen willkürlich festgelegten Latentcode und der Information, die im zugeordneten Image enthalten ist, quasi keine Überschneidung gibt, gibt es für den Encoder zur Erfüllung seiner Aufgabe nur die Möglichkeit jeden Code und jedes zugehörige Image vollständig auswendig zu lernen.

Die Vermutung ist nun, dass wenn wir einen Encoder und einen Decoder mit dem selben Objektiv (nämlich der akkuraten Rekonstruktion der Bilder) trainieren, der Encoder eine Funktion lernt, die die Aufgabe des Decoders so weit wie möglich vereinfacht. D.h. wenn der Encoder ein Bild auf bspw. gerade mal 20 Skalare abbilden soll, wird er versuchen in diesen 20 Skalaren so viel Information wie möglich unterzubringen, um es dem Decoder so leicht wie möglich zu machen das zugehörige Bild zu rekonstruieren.
Deshalb geht man davon aus, dass Autoencoder nützlich sind um eine komprimierte und gewissermaßen 'sinnvolle' Repräsentation unserer Daten zu produzieren. 

## Warum beschäftigen wir uns damit? - Die Verbindung zum Generative Modeling

Alle generativen Models basieren in irgendeiner Form auf den Prinzipien von Encodern und Decodern. Die Auseinandersetzung mit Autoencodern gibt uns deshalb eine exzellente Möglichkeit Wichtiges über die Dynamiken und Problemstellungen in der Optimierung von solchen Models zu lernen.
Im Grunde ist das Konzept des Decoders jenes, welches Generative Modeling überhaupt impliziert: Wäre es deine Aufgabe ein Portrait deiner Freundin anzufertigen,
so könntest du dieses mit traditionellen Mitteln selbst herstellen (Malen, Zeichen, Photographie, 3D-Modelling etc.). Wir wissen, dass es dann unendlich verschiedene zufriedenstellende Versionen eines Portraits deiner Freundin geben kann. Wir können den Herstellungsprozess, um zu einer dieser Versionen zu gelangen als Informationsmaximierungs-Prozess beschreiben. Wir beginnen im Zeichnen bspw. mit dem blanken Papier (welches keine Informationen enthält) und fügen nach und nach Partikel hinzu, sodass Informationen entstehen, die unser kognitiver Apparat als 2-dimensionale Abbildung unserer Freudin begreift. Je mehr Zeit und Aufwand wir investieren, desto mehr Information/Details können wir hinzufügen - bis zum kleinsten Objekt, welches wir mit unserer Bildauflösung sehen können (Hautporen, Haarsträhnen etc.). Theoretisch müsste es eine Art Limit geben, ab welchem keine weitere Information mehr hinzugefügt werden könnte.
Ein funktionierender Generative Modeling-Algorithmus generiert aus einem Input, der wenig bis keine Information enthält einen Output mit idealerweise maximaler Information. Bspw. generiert StyleGAN aus einem Vektor zufälliger Skalare ein quasi photorealistisches, einmaliges Photo. Eine Liste mit zufälligen Zahlen beinhaltet aus unserer Perspektive quasi keine Information, ein realistisches Photo dagegen sehr viele. Die Abbildung einer Datenstruktur, die isoliert betrachtet wenig bis keine Informationen enthält, auf eine zugehörige Datenstruktur mit maximal viel Informationen ist genau das, was der Decoder in unserem Autoencoder tun soll und ist gleichzeitig eine der primären Funktionalitäten, die ein Generative Modeling-Algorithmus erfüllen muss.
