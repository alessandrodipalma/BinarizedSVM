import numpy as np
from alogrithm1 import scegli_soglie
import cvxpy

def solve_6F_dual_with_cvpy(x, c, C, F, verbose=False):
    I = len(x)
    lambda_ = cvxpy.Variable(I)
    objective = cvxpy.Maximize(cvxpy.sum(lambda_))
    constraints = []
    for l in F.keys():
        for phi in F[l]:
            coeff = np.array([c[u] * phi(x[u,l]) for u in range(I)])
            constraints.extend([coeff.T @ lambda_ >= -1.0, coeff.T @ lambda_ <= 1.0])

    constraints.append(c.T @ lambda_ == 0.0)
    constraints.extend([lambda_ >= 0.0, lambda_ <= C])

    problem = cvxpy.Problem(objective, constraints)

    problem.solve(solver='ECOS', verbose=verbose, abstol=1e-8)

    return lambda_.value


def mediane(x):
    # Step 0: inizializzazione di F0
    F0 = {}
    number_of_predictor_variables = len(x[0])
    soglie = {}
    for l in range(number_of_predictor_variables):
        soglie[l] = []
        F0[l] = []

    for l in range(number_of_predictor_variables):
        b_star_l = np.median(x[:, l])
        phi_star_l = lambda x: 1 if x >= b_star_l else 0
        soglie[l].append(b_star_l)
        F0[l].append(phi_star_l)

    return F0, soglie


def gamma(x, b, lambda_star, labels):
    phi = lambda x: 1 if x >= b else 0
    sum_ = 0

    for u in range(x.shape[0]):
        sum_ += lambda_star[u] * labels[u] * phi(x[u])

    return sum_, phi


def column_generation(x, labels, C, max_iter=1000, verbose=False):
    number_of_predictor_variables = len(x[0])

    # inizializzazione di F
    if verbose: print("initializing F...")
    F, soglie = mediane(x)

    iter_count = 0
    F_modified = True

    while F_modified and max_iter>iter_count:  # Step 3: se F non è stato modificato, abbiamo trovato la soluzione ottima di (6)
        if verbose: print(f"iteration {iter_count}" + '-' * 100)
        lambda_star = solve_6F_dual_with_cvpy(x, labels, C, F, False)
        # Step 2: calcolo delle φlb^+_l e φlb^-_l e aggiornamento di F se necessario
        F_modified = False

        for l in range(number_of_predictor_variables):
            if verbose: print(f"scelgo le soglie per {l}...")

            b_plus_l, b_minus_l = scegli_soglie(x[:, l], labels, lambda_star)

            if verbose: print(f"soglie trovate: b_plus:{b_plus_l}\t b_minus:{b_minus_l}")

            gamma_plus_l, phi_plus_l = gamma(x[:, l], b_plus_l, lambda_star, labels)
            if gamma_plus_l > 1:
                F[l].append(phi_plus_l)
                soglie[l].append(b_plus_l)
                F_modified = True
                if verbose: print(f"\taggiungo soglia+ per {l}...")

            gamma_minus_l, phi_minus_l = gamma(x[:, l], b_minus_l, lambda_star, labels)
            if gamma_minus_l < -1:
                F[l].append(phi_minus_l)
                soglie[l].append(b_minus_l)
                F_modified = True
                if verbose: print(f"\taggiungo soglia- per {l}...")

        iter_count += 1

    stats = {"iter_count": iter_count}

    return lambda_star, F, soglie, stats
