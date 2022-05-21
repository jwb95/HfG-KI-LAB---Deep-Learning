Diese Lektion liefert lediglich Anhaltspunkte in Form von Deep Learning-Papers und Projekten, die im Bereich des generative Modeling herausragende Ergebnisse gezeigt haben.
Es gibt eine der Forschung inhärente Tendenz in Richtung Bildanwendungen. Das liegt daran, dass Bilddatensets relativ leicht sammelbar sind bzw. schon einige brauchbare existieren und Resultate ganz wörtlich anschaubar und damit verhätlnismäßig leicht zu evaluieren sind. Grundsätzlich ist das Arbeiten mit Daten jeder Art absolut möglich. Man darf  annehmen, dass Algorithmen, die auf Bildern gute Ergebnisse erzielen grundsätzlich auch auf andere Datentypen anwendbar sein müssten oder mindestens Fortschritte im Feld der Bild-Synthese auch Fortschritte für die Synthese anderer Datentypen (Text, Video, Roboter-Aktionen, Audio, 3D-Modelle etc. pp.) bedeuten müssten.

#### Unconditional Image-Synthesis

'Uncoditional' bedeutet, dass wir einen generativen Algorithmus mit einem Datenset einer Klasse trainieren und das Model neue Daten generiert, die aus diesem Datenset
stammen könnten. Das 'Conditioning' für den Algorithmus besteht aus dem Datenset.

Wir haben StyleGAN2 (<https://github.com/NVlabs/stylegan2>) bereits kennengelernt. Mittlerweile gibt es auch schon StyleGAN3 (<https://github.com/NVlabs/stylegan3>).
StyleGANs gehören zur Model-Klasse der Generative Adversarial Networks.

![](https://github.com/NVlabs/stylegan3/raw/main/docs/stylegan3-teaser-1920x1006.png)

Alternativen zu dieser Klasse sind VAEs (= Variational Autoencoders) wie NVAE (<https://github.com/NVlabs/NVAE>). 
VAEs tendieren dazu, Bilder schlechterer Qualität als GANs, jedoch dafür vielseitigere zu generieren.

![](https://github.com/NVlabs/NVAE/raw/master/img/celebahq.png)

Und Diffusion-Models: <https://github.com/hojonathanho/diffusion>, die in Versatilität wie Qualität gut sind, deren Inferenz jedoch langsam ist.

![](https://github.com/hojonathanho/diffusion/raw/master/resources/samples.png)

#### Conditional Image-Synthesis

Conditional bedeutet, dass das generative Model mit Daten aus mehreren Klassen derartig trainiert wurde, sodass wir bei der Synthese dem Model einen Code übergeben können, der dem Model bedeutet aus welcher Klasse ein Bild generiert werden soll. Wie wir im StyleGAN2-Tutorial gesehen haben gibt es Conditional-StyleGAN2-Models (Wikiart).
Diffusion Models sind momentan SotA: <https://cascaded-diffusion.github.io/>

![](https://cascaded-diffusion.github.io/assets/img/header_small.png)

#### Paired Domain-Translation

Das erste Model für paired-image-2-image war Pix2Pix: <https://github.com/affinelayer/pix2pix-tensorflow>

![](https://github.com/affinelayer/pix2pix-tensorflow/raw/master/docs/examples.jpg)

Für das Training braucht man ein Datenset mit Paaren von Bildern, z.B. Paaren aus jeweils einem Photo von einer Handtasche und einer primitiven Zeichnung dieser. Dann soll das Model die gezeichnete auf die photo-realistische Version der Handtasche mappen. Bei paired-domain-translation geht es also um die Umwandlung der Repräsentationsform bzw. Domain der selben Sache. Es gibt auch Pix2PixHD, welches größer ist und bessere Ergebnisse erzielt: <https://github.com/NVIDIA/pix2pixHD>
Da Video nur Listen von Bildern sind lassen sich Pix2Pix-artige Models auch auf Videos anwenden: <https://github.com/NVIDIA/vid2vid>

![](https://github.com/NVIDIA/vid2vid/raw/master/imgs/teaser.gif)

Ein klassischer Fall von paired-image-2-image ist Super-Resolution, also die Erhöhung der Pixelauflösung von Bildern, wofür es GAN- (<https://github.com/xinntao/ESRGAN>) wie auch Diffusion-basierte Models (<https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement>) gibt.

![](https://iterative-refinement.github.io/images/cascade_fig.svg)

Unter paired-domain-translation fällt auch semantische Image-Synthese. Also die Synthese von Daten auf Basis von semantischen Karten, die der User erstellt, um zu festzulegen, wo gewisser Inhalt im Bild sichtbar sein soll. <https://github.com/NVlabs/SPADE>
Auf ganpaint.io kann man ein trainiertes Model dieser Art in Aktion erleben: <https://ganpaint.io/>

![](https://camo.githubusercontent.com/f7e852bab5b53dae22f795d500f1cb480a9f436d70fc8cba3f71568759a448de/68747470733a2f2f6e766c6162732e6769746875622e696f2f53504144452f2f696d616765732f6f6365616e2e676966)

Für Domain-Translation ist es übrigens auch denkbar mit Domains zweier verschiedenere Datentype zu arbeiten: Z.B. Text2Image. <https://github.com/crowsonkb/v-diffusion-pytorch> Das zugehörige Datenset hat 12Mio Text-Image-Paare: <https://github.com/google-research-datasets/conceptual-12m>
Unter Text2Image fallen auch Projekte wie Disco Diffusion, für welche es auf Reddit eine ganze Community gibt (https://www.reddit.com/r/DiscoDiffusion).


![](https://i.redd.it/2m3aybg0in091.jpg)

![](https://preview.redd.it/h4uqesxpxm091.png?width=960&crop=smart&auto=webp&s=83895c72006bd22a81c3bb4e08b4b6c51e2837f4)

#### Unpaired Domain-Translation

... bezeichnet das gleiche Problem, nur mit dem Unterschied, dass die Trainingsbeispiele nicht in Paarform vorliegen. Solche Problem sind "ill-posed", da für sie keine einzigartige Lösung existiert. Nehmen wir z.B. das Problem, dass wir Bilder von Pferden in Bilder von Zebras verwandeln wollten, dann könnten wir einem Pferde-Bild theoretisch jedes erdenkliche Zebra-Bild zuordnen. Da sich Zebras und Pferde in ihrer Form recht stark ähneln wäre es jedoch interessant zu erproben, ob man aus dem Bild eines Pferdes das Bild eines Zebras in gleicher Pose generieren könnte. CycleGAN (<https://junyanz.github.io/CycleGAN/>) war das Pioneer-Paper auf diesem Feld.

![](https://junyanz.github.io/CycleGAN/images/teaser.jpg)

Mittlerweile gibt es GAN-basierte Models, die weitaus bessere Ergebnisse erzielen: MUNIT (<https://github.com/NVlabs/MUNIT>) und FUNIT (<https://github.com/NVlabs/FUNIT>).

![](https://github.com/NVlabs/MUNIT/raw/master/results/animal.jpg)

![](https://github.com/NVlabs/FUNIT/raw/master/docs/images/animal.gif)

Tatsächlich waren das jetzt alles Image-Projekte. Die Auflistung hatte aber den Zweck aufzulisten was im Bereich des 'generative Modelling' grundsätzlich denkbar ist.
Existieren Daten und eine Architektur, die die Daten verarbeiten kann, ist jeder andere Datentyp genauso denkbar. Nach allen Lektionen werden wir eine Intuition dafür besitzen, wie diese Models prinzipiell funktionieren und werden in die Lage versetzt sein, dieses Wissen auf eigenen Probleme und Daten anzuwenden.


