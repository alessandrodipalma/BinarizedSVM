import cplex
import numpy as np
from alogrithm1 import scegli_soglie
import cvxpy


def solve_6F(x, labels, F, C):
    # Step 1: risoluzione del problema (6-F)

    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_error_stream(None)
    model.set_warning_stream(None)
    model.set_results_stream(None)

    p = number_of_predictor_variables = len(x[0])

    # definizione delle variabili di decisione
    omega_plus = model.variables.add(obj=[omega_plus_lb[b] for b in F for lb in range(p)], lb=[0] * (p * len(F)))
    omega_minus = model.variables.add(obj=[omega_minus_lb[b] for b in F for lb in range(p)], lb=[0] * (p * len(F)))
    xi = model.variables.add(obj=[C] * m, lb=[0] * m)
    beta_var = model.variables.add(obj=[0])

    # definizione dei vincoli
    for u in range(m):
        lhs = [[(omega_plus_lb[b] - omega_minus_lb[b]) * labels[u] * F[l][b][u],
                omega_plus[lb * len(F) + b] - omega_minus[lb * len(F) + b]] for lb in range(p) for b in F]
        lhs.append([1, beta_var[0]])
        lhs.append([-1, xi[u]])
        model.linear_constraints.add(lin_expr=[lhs], senses=['G'], rhs=[1])

    # risoluzione del modello
    model.solve()

    # estrazione della soluzione ottima
    omega_star = np.zeros((p, len(F)))
    for lb in range(p):
        for b in F:
            omega_star[lb, b] = model.solution.get_values(omega_plus[lb * len(F) + b]) - model.solution.get_values(
                omega_minus[lb * len(F) + b])
    beta_star = model.solution.get_values(beta_var[0])
    lambda_star = model.solution.get_dual_values()

    return lambda_star, omega_star, beta_star


def solve_6F_dual(x, y, C, F):
    model = cplex.Cplex()

    # Add the decision variables
    I = range(len(x))

    lambdas = model.variables.add(names=[f"lambda_{u}" for u in I], lb=[0.0] * len(x))

    # Set the objective function
    model.objective.set_sense(model.objective.sense.maximize)
    # model.objective.set_linear([(lambdas[u], 1.0) for u in I])

    # Add the constraints
    # primo vincolo: -1 <= sum(lambda_u * c_u * phi(x_u)) <= 1 per ogni phi in F
    for phi in F:
        expr = cplex.SparsePair(ind=[f"lambda_{u}" for u in I],
                                val=[y[u] * phi[u] for u in I])
        model.linear_constraints.add(lin_expr=[expr, expr], senses=["L", "G"], rhs=[1.0, -1.0])

    # secondo vincolo: sum(lambda_u * c_u) = 0
    expr = cplex.SparsePair(ind=[f"lambda_{u}" for u in I], val=y)
    model.linear_constraints.add(lin_expr=[expr], senses=["E"], rhs=[0.0])

    # terzo vincolo
    expr = cplex.SparsePair(ind=[f"lambda_{u}" for u in I], val=np.ones(len(x)))
    model.linear_constraints.add(lin_expr=[expr, expr], senses=["G", "L"], rhs=[0.0, C])

    # Solve the model
    model.solve()

    # Print the solution
    print("Solution status:", model.solution.get_status())
    print("Objective value:", model.solution.get_objective_value())
    print("Dual variables:")
    # for u in I:
    #     print("  lambda_" + str(u) + " = " + str(model.solution.get_values(f"lambda_{u}")))

    lambda_star = [model.solution.get_values(f"lambda_{u}") for u in I]
    return lambda_star


def solve_6F_dual_with_cvpy(x, y, C, F):
    n = len(x)
    lambda_ = cvxpy.Variable(n)
    objective = cvxpy.Maximize(cvxpy.sum(lambda_))
    constraints = []
    for phi in F:
        coeff = np.array([y[i] * phi[i] for i in range(n)])
        constraints.extend([coeff.T @ lambda_ >= -1.0, coeff.T @ lambda_ <= 1.0])

    constraints.append(y.T @ lambda_ == 0)
    constraints.extend([lambda_ >= 0, lambda_ <= C])

    problem = cvxpy.Problem(objective, constraints)

    problem.solve(solver='ECOS', verbose=True, abstol=1e-4)

    return lambda_.value


def mediane(x):
    # Step 0: inizializzazione di F0
    F0 = []
    soglie = []
    number_of_predictor_variables = len(x[0])

    for l in range(number_of_predictor_variables):
        b_star_l = np.median(x[:, l])
        phi_star_l = np.zeros(len(x))
        soglie.append(b_star_l)
        for i in range(len(x)):
            if x[i, l] >= b_star_l:
                phi_star_l[i] = 1
            else:
                phi_star_l[i] = -1
        F0.append(phi_star_l)

    return F0, soglie


def gamma(phi, lambda_star, labels):
    sum_ = 0

    for u in range(len(phi)):
        sum_ += lambda_star[u] * labels[u] * phi[u]

    return sum_


def column_generation(x, labels, C):
    number_of_predictor_variables = len(x[0])

    # inizializzazione di F
    F, soglie = mediane(x)

    iter_count = 0
    F_modified = True

    while F_modified:  # Step 3: se F non è stato modificato, abbiamo trovato la soluzione ottima di (6)
        print(f"iteration {iter_count}" + '-' * 100)
        lambda_star = solve_6F_dual_with_cvpy(x, labels, C, F)
        # Step 2: calcolo delle φlb^+_l e φlb^-_l e aggiornamento di F se necessario
        F_modified = False
        for l in range(number_of_predictor_variables):
            print(f"scelgo le soglie per {l}...")
            b_plus_l, b_minus_l = scegli_soglie(x[:, l], labels, lambda_star)
            print(f"soglie trovate: {b_plus_l} {b_minus_l}")
            phi_plus_l = [1 if x[i, l] >= b_plus_l else 0 for i in range(len(x))]
            phi_minus_l = [1 if x[i, l] >= b_minus_l else 0 for i in range(len(x))]

            if gamma(phi_plus_l, lambda_star, labels) > 1:
                F.append(np.array(phi_plus_l))
                soglie.append(b_plus_l)
                F_modified = True
            if gamma(phi_minus_l, lambda_star, labels) < -1:
                F.append(np.array(phi_minus_l))
                soglie.append(b_minus_l)
                F_modified = True

        iter_count += 1

    return lambda_star, F, soglie


def score_function(x, omega, soglie, B):
    sum_ = 0

    for predictor_var in range(len(x)):
        for b in soglie[predictor_var]:
            if x[predictor_var] > b:
                sum_ += omega[predictor_var]

    score = sum_ + B

    if score > 0:
        return 1
    else:
        return -1
