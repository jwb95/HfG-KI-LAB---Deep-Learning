In dieser Lektion machen wir uns mit der Abbildung von Daten in einen komprimierten Feature-Space vertraut und lernen daran angelehnt eine neue Netzwerk-Architektur (den Autoencoder) kennen. Da wir schon Klassifikations-Probleme gelöst haben, haben wir so etwas schon in der Praxis getan. Bei der Klassifikation von handgeschriebenen Ziffern optimierten wir eine Funktion in Form eines NNs, die eine Matrix der Form (28, 28, 1) in einen Vektor der Form (9,) überführt. Die Funktion ist jedoch nicht injektiv, d.h. mehrere Bilder werden durch sie auf die gleiche Klasse abgebildet. So würden möglicherweise zwei verschiedene Bilder der Ziffer 4 den gleichen Vektor [0, 0, 0, 0, 1, 0, 0, 0, 0, 0] produzieren.

Nehmen wir an wir haben eine endliche Liste von einmaligen 2-dimensionalen Matrizen (z.B. Bildern), dann könnten wir eine Funktion definieren, die jedes Element dieser Liste auf eine einmalige, komprimierte Repräsentation (also eine solche Matrix, die weniger Skalare besitzt als die ursprüngliche) abbildet. Dadurch, dass jede komprimierte Repräsentation einmalig ist, existiert zu dieser Funktion auch eine Umkehrfunktion, die die komprimierte Repräsentation zurück auf die Ursprungsmatrix abbildet.

Ein Autoencoder ist ein NN, welches beide diese Funktionen für ein Datenset lernen soll. Er besteht aus einem Encoder, welcher ein Input-Image auf einen sog. Latentcode abbildet und einem Decoder, welcher diesen Latentcode zurück auf das Image mappen soll. Der Latentcode besitzt ist üblicherweise weniger Skalare als das Image. Der Encoder vollzieht als eine Kompression, der Decoder versucht diese Kompression umzukehren.

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

Genau genommen brauchen wir für den Encoder-Teil eigentlich kein Neuronales Netz. Schließlich könnten wir uns auch einfach eine Funktion ausdenken, die für jedes Bild einen einmaligen Latentcode produziert. Dann müssten wir nur einen Decoder trainieren, der jeden Code auf das entsprechende Bild abbildet. So könnten wir sogar noch mehr Speicher sparen, da wir mit einem Computerprogramm aus einer Zeile Code bereits eine Liste von einmaligen Latentcodes produzieren könnten, z.B:

```latentcodes = [np.float32(np.zeros(number_of_latentdimensions) + i) for i in range(number_of_images)]```

Wir müssten dann also nicht mal mehr die Codes speichern, sondern lediglich das Programm, welches diese produziert.
Der Grund für ein Encoder-Netzwerk ist, dass ein solches Encoding die Aufgabe des Decoders maximal hart macht. Da es zwischen der Information in einem solchen willkürlich festgelegten Latentcode und dem zugeordneten Image quasi keine Überschneidung von Information gibt, gibt es für den Encoder zur Erfüllung seiner Aufgabe nur die Möglichkeit jeden Code und jedes zugehörige Image vollständig auswendig zu lernen.

## Warum beschäftigen wir uns damit?

Alle generativen Models basieren in irgendeiner Form auf Encodern und Decodern. Die Auseinandersetzung mit Autoencodern gibt uns deshalb eine exzellente Möglichkeit wichtiges über die Dynamiken und Problemstellungen in der Optimierung von solchen Models zu lernen.
Tatsächlich sind Autoencoder nicht sonderlich spannend. Im Generative Modeling werden wir jedoch immer wieder mit Encodern und Decodern zu tun haben.

