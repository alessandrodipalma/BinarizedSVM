# BinarizedSVM
Questo progetto si propone di riprodurre il lavoro di Carrizosa et al. in [1].

## Struttura del repository
### Datasets
La cartella datasets contiene i file dei singoli dataset.
Ciascun dataset può essere caricato usando la relativa funzione all'interno del modulo `load_dataset.py`.
Per caricare tutti i dataset in una lista, utilizzare la funzione `load_all()`.
### Implementazione BinarizedSVM
Il modulo `BinarizedSVM.py` contiene la classe che implementa il classificatore proposto. La classe utilizza il metodo per 
la generazione di colonne, implementato in `column_generation.py`. Questo metodo utilizza a sua volta l-algoritmo per la ricerca
delle soglie ottimali, implementato nel modulo `algorithm1.py`.
Il classificatore implementato segue lo stile di `sklearn`, quindi mette a disposizione un metodo `fit` e un `predict`.
Inoltre, per visualizzare le soglie in maniera analoga a quanto proposto in [1], la classe `BinarizedSVM` include un metodo `visualizza_soglie`
che produce un grafico che mostra le soglie per ogni singola feature (predictor variable) ed i relativi pesi. 
### Implementazione column generation
Per la risoluzione del duale del problem 6F, è stata utilizzata la libreria `cvxpy`, che permette di utilizzare vari ottimizzatori, tra cui CPLEX.

## Esecuzione
Innanzi tutto, assicurarsi di soddisfare i requisiti lanciando il comando

```pip install -r requirements.txt```.

Il notebook `main.ipynb` contiene un esempio di utilizzo della libreria e contiene il codice necessario a produrre un confronto tra la BinarizedSVm e il SVC di sklearn.

I notebook `bupa.ipynb` e `wdbc.ipynb` contengono delle grid search più estensive sui singoli dataset.

# Riferimenti
[1] DOI:10.1287/ijoc.1090.0317
