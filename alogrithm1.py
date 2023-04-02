import numpy as np
from tqdm import tqdm

def ci_sono_valori_ripetuti(values, t, i):
    return len(np.unique(values[i - t:])) < len(values[i - t :])


def hanno_classi_diverse(labels, t, i):
    for j in range(t):
        if labels[i] != labels[i - j]:
            return True


def ci_sono_soglie_migliori(values, labels, i):
    for t in range(i):
        if ci_sono_valori_ripetuti(values, t, i) and hanno_classi_diverse(labels, t, i):
            return True
    return False


def scegli_soglie(x_l, c, lambda_star):
    # Passo 0: ordinamento decrescente degli oggetti in base a x
    sorted_x = sorted(list(zip(x_l, c)), reverse=True)
    x_l = [v for (v, l) in sorted_x]
    c = [l for (v, l) in sorted_x]
    # Passo 1: inizializzazione delle variabili
    sum_ = max_ = min_ = 0
    i_plus = i_minus = i = 0

    # Passo 2-4: ciclo per la scelta delle soglie
    for i in range(len(x_l) - 1):  # Passo 4: passa al successivo
        # Passo 2
        sum_ = sum_ + lambda_star[i] * c[i]

        # Passo 3.1: se x(i) = x(i+1), passa al successivo
        if x_l[i] != x_l[i + 1]:
            # Passo 3.2: se x(i) ≠ x(i+1), controlla se ci sono soglie migliori
            if ci_sono_soglie_migliori(x_l, c, i):
                # Passo 3.2.a: ci sono soglie migliori
                if sum_ > max_:
                    max_ = sum_
                    i_plus = i
                elif sum_ < min_:
                    min_ = sum_
                    i_minus = i
            else:
                # Passo 3.3: ci sono soglie migliori
                if c[i] == 1 and c[i + 1] == -1 and sum_ > max_:
                    max_ = sum_
                    i_plus = i
                elif c[i] == -1 and c[i + 1] == 1 and sum_ < min_:
                    min_ = sum_
                    i_minus = i

    # Restituisce le soglie scelte
    b_plus = x_l[i_plus]
    b_minus = x_l[i_minus]

    return b_plus, b_minus
