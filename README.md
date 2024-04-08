# ProjectStatistiekVariantieReductie

Deze github bevat de volgende bestanden

## Antithetic Variates.py
Dit programma bevat de code om een Monte Carlo simulatie met en zonder antithetic variates toe te passen. De functies berekenen de call-optie schatter en variantie van de Monte Carlo simulatie met en zonder antithetic variates toegepast.

## AR sampling.py
Dit programma bevat de code om het Acceptance-Rejection algoritme uit te voeren. Verder bevat het een stukje dat het sample gegenereerd door Acceptance-Rejection visualiseert om te bekijken hoe nauwkeurig de mu en sigma2 schatters zijn van het sample.

## Basisproject.py

## Control Variates.py
Dit programma bevat de code om een Monte Carlo simulatie met en zonder control variates toe te passen. De functies berekenen de call-optie schatter en variantie van de Monte Carlo simulatie met en zonder control variates toegepast.

## Correlaties.py
Dit programma bevat twee functies om de correlaties van het gesimuleerde object en de control variate bij control variates te bepalen. Dit is gedaan om makkelijk te onderzoeken welke invloed parameters in de vergelijking hebben op de correlatie tussen het gesimuleerde object en de control variate

## Runtime Basisproject.py
Dit programma bevat de statements en setup van de module timeit. In de setup staat het bestand Basisproject.py

## Runtime Uitbreiding.py
Dit programma bevat de statements en setup van de module timeit. In de setup staat het bestand Uitrbeiding Project.py

## Uitbreiding Project.py
Dit programma bevat de code om een Monte Carlo simulatie met en zonder importance sampling uit te voeren. De functies berekenen de call-optie schatter en variantie van de Monte Carlo simulatie met en zonder importance sampling toegepast. Importance sampling is geïmplementeerd aan de hand van Exponential Tilting. 
