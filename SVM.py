from column_generation import column_generation
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable

class BinarizedSVM:

    def __init__(self, C=1, verbose=False):
        self.C = C
        self.verbose = verbose

    def fit(self, X, y, predictor_variables_names=None, max_iter=100):
        """

        :param X: bidimensional numpy array of shape (n,m), containing the n training samples
        :param y: unidimensional numpy array of len n, containing the labels of the n training samples. The labels of the two classes must be [1,-1]
        :param predictor_variables_names: iterable of m string, containing the names of the predictor variables names of X
        :return:
        """

        if predictor_variables_names is None:
            predictor_variables_names = [str(i) for i in range(X.shape[1])]
        elif len(predictor_variables_names) != X.shape[1]:
            print(f"Predictor variables names should match input shape {X.shape}")
            return

        if X.shape[0] != y.shape[0]:
            print("Esempi ed etichette devono avere la stessa lunghezza.")
            return

        if self.verbose:
            print(f"Nomi delle variabili: {predictor_variables_names}")

        self.predictor_variables_names = predictor_variables_names
        self.n_predictor_vars = len(self.predictor_variables_names)
        self.lambda_star, self.F, self.soglie, stats = column_generation(X, y, self.C, verbose=self.verbose, max_iter=max_iter)

        if self.verbose:
            print(f"\nAddestramento completato in {stats['iter_count']} iterate")

        self.omega_star = {}

        for l in self.soglie.keys():
            self.omega_star[l] = np.zeros(len(self.soglie[l]))
            for b, phi in enumerate(self.soglie[l]):
                for i in range(len(X)):
                    c = 1 if X[i, l] > phi else -1
                    self.omega_star[l][b] += self.lambda_star[i] * c


        if self.verbose:
            table = PrettyTable()
            table.field_names = ["Predictor var", "soglia","peso"]
            print("Le soglie selezionate sono:")
            for l in self.soglie.keys():
                for b, phi in enumerate(self.soglie[l]):
                    if b==0:
                        table.add_row([f"{l}, {len(self.soglie[l]) } features", self.soglie[l][b], self.omega_star[l][b]])
                    else:
                        table.add_row(["", self.soglie[l][b], self.omega_star[l][b]])

            print(table)

    def visualizza_soglie(self, out_dir=None):

        for predictor_variable, thresholds in self.soglie.items():

            sorted_soglie = sorted(list(zip(self.soglie[predictor_variable], self.omega_star[predictor_variable])),
                                   reverse=False)
            soglies = [v for (v, l) in sorted_soglie]
            omegas = [l for (v, l) in sorted_soglie]


            plt.step(soglies, omegas)
            plt.xlabel('soglia')
            plt.ylabel('Weight')
            plt.title(f"Pesi per var {predictor_variable}")
            if out_dir is None:
                plt.show()
            else:
                plt.savefig(f"{out_dir}{predictor_variable}")

    def predict(self, X):
        return np.array(list(map(self.score_function, X)))

    def score_function(self, x):
        sum_ = 0
        n_predictor_vars = len(self.soglie)
        for l in range(n_predictor_vars):
            for j, phi in enumerate(self.F[l]):
                sum_ += self.omega_star[l][j] * phi(x[l])

        score = sum_  # + B
        # print(score)
        if score > 0:
            return 1
        else:
            return -1



