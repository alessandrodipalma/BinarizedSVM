from column_generation import column_generation

class BinarizedSVM:

    def __init__(C=1):
        self.C = C

    def fit(self, X, y, predictor_variables_names=None):
        if len(predictor_variables_name) != X.shape[1]:
            print(f"Predictor variables names should match input shape {X.shape}")
            return
        
        self.predictor_variables_names = predictor_variables_names
        self.n_predictor_vars = len(self.predictor_variables_names)
        
        self.lambda_star, self.F, self.soglie = column_generation(X, y, self.C)
        print("Fitting completed")

        omega_star = {}

        for l in soglie.keys():
            omega_star[l] = np.zeros(len(soglie[l]))
            for b, phi in enumerate(soglie[l]):
                for i in range(len(X)):
                    c = 1 if X[i,l] > phi else -1
                    omega_star[l][b] += lambda_star[i] * c

        self.omega_star = omega_star        

    def visualizza_soglie(self, out_dir=None):
        for predictor_variable, thresholds in self.soglie.items():
            sorted_soglie = sorted(list(zip(soglie[predictor_variable], omega_star[predictor_variable])), reverse=False)
            soglies = [v for (v, l) in sorted_soglie]
            omegas = [l for (v, l) in sorted_soglie]

            print(soglies, omegas)
            plt.step(soglies, omegas)
            plt.xlabel('Threshold')
            plt.ylabel('Weight')
            plt.title(f"Pesi per le soglie della feature {predictor_variable}")
            if out_dir is None:
                plt.show()
            else:
                plt.savefig(f"{out_dir}{predictor_variable}")

    
    def predict(self, X):
        return np.array(list(map(self.score_function, x)))trn 

    
    def score_function(self, x):
        sum_ = 0
        n_predictor_vars = len(self.soglie)
        for l in range(n_predictor_vars):
            for j, b in enumerate(soglie[l]):
                if x[l] >= b:
                    sum_ += omega_star[l][j]

        score = sum_ #+ B
        # print(score)
        if score > 0:
            return 1
        else:
            return -1

