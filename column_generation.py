import cplex
import numpy as np
from alogrithm1 import scegli_soglie
def solve_6F(F):
    # Step 1: risoluzione del problema (6-F)
    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)

    # definizione delle variabili di decisione
    omega_plus = model.variables.add(obj=[omega_plus_lb[b] for b in B for lb in range(p)], lb=[0] * (p * len(B)))
    omega_minus = model.variables.add(obj=[omega_minus_lb[b] for b in B for lb in range(p)], lb=[0] * (p * len(B)))
    xi = model.variables.add(obj=[C] * m, lb=[0] * m)
    beta_var = model.variables.add(obj=[0])

    # definizione dei vincoli
    for u in range(m):
        lhs = [[(omega_plus_lb[b] - omega_minus_lb[b]) * cost_matrix[u, lb] * F[lb][u],
                omega_plus[lb * len(B) + b] - omega_minus[lb * len(B) + b]] for lb in range(p) for b in B]
        lhs.append([1, beta_var[0]])
        lhs.append([-1, xi[u]])
        model.linear_constraints.add(lin_expr=[lhs], senses=['G'], rhs=[1])

    # risoluzione del modello
    model.solve()

    # estrazione della soluzione ottima
    omega_star = np.zeros((p, len(B)))
    for lb in range(p):
        for b in B:
            omega_star[lb, b] = model.solution.get_values(omega_plus[lb * len(B) + b]) - model.solution.get_values(
                omega_minus[lb * len(B) + b])
    beta_star = model.solution.get_values(beta_var[0])
    lambda_star = model.solution.get_dual_values()

    return beta_star, lambda_star, omega_star
def mediane(x):
    # Step 0: inizializzazione di F0
    F0 = []
    number_of_predictor_variables = len(x[0])

    for l in range(number_of_predictor_variables):
        b_star_l = np.median(x[:,l])
        phi_star_l = np.zeros(len(x))
        for i in range(len(x)):
            if x[i,l] >= b_star_l:
                phi_star_l[i] = 1
            else:
                phi_star_l[i] = -1
        F0.append(phi_star_l)

    return F0
def column_generation(x):

    # inizializzazione di F
    F = mediane(x)

    # inizializzazione di β
    beta = 0

    while True:
        beta_star, lambda_star, omega_star = solve_6F(F)
        # Step 2: calcolo delle φlb^+_l e φlb^-_l e aggiornamento di F se necessario
        F_modified = False
        for l in range(p):
            phi_plus_l, phi_minus_l = scegli_soglie(predictor_variable[:,l], omega_star[l], lambda_star)
            if gamma(phi_plus_l) > 1:
                F.append(phi_plus_l)
                F_modified = True
            if gamma(phi_minus_l) < -1:
                F.append(phi_minus_l)
                F_modified = True

        # Step 3: se F non è stato modificato, abbiamo trovato la soluzione ottima di (6)
        if not F_modified:
            break

    # calcolo della funzione obiettivo e degli omega_lb ottimi
    omega_lb_star = np.zeros((p, len(B)))
    for lb in range(p):
        for b in B:
            omega_lb_star[lb,b] = sum([omega_star[lb,b_prime] for b_prime in range(b, len(B))])

    f_star = sum([sum([omega_lb_star[l,b] * F[l][

    return lambda_star