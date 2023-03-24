import numpy as np


def ci_sono_valori_ripetuti(values, t, i):
    return len(np.unique(values[i - t - 1:])) < len(values[i - t - 1:])


def hanno_classi_diverse(labels, t, i):
    for j in range(1, t):
        if labels[i] != labels[i - j]:
            return True


def ci_sono_soglie_migliori(values, labels, i):
    t = 1
    while True:
        if ci_sono_valori_ripetuti(values, t, i) and hanno_classi_diverse(labels, t, i):
            return True
        t += 1


def scegli_soglie(x, c, lambda_star):
    # Passo 0: ordinamento decrescente degli oggetti in base a x
    sorted_x = sorted(list(zip(x, c)), reverse=True)
    values = [v for (v, l) in sorted_x]
    labels = [l for (v, l) in sorted_x]
    # Passo 1: inizializzazione delle variabili
    sum_ = max_ = min_ = 0
    i_plus = i_minus = i = 1

    # Passo 2-4: ciclo per la scelta delle soglie
    while i < len(values):
        # Passo 2
        sum_ = sum_ + lambda_star[i] * labels[i]

        # Passo 3.1: se x(i) = x(i+1), passa al successivo
        if values[i] != values[i + 1]:
            # Passo 3.2: se x(i) ≠ x(i+1), controlla se ci sono soglie migliori
            if ci_sono_soglie_migliori(values, labels):
                # Passo 3.2.a: ci sono soglie migliori
                if sum_ > max_:
                    max_ = sum_
                    i_plus = i
                elif sum_ < min_:
                    min_ = sum_
                    i_minus = i
            else:
                # Passo 3.3: ci sono soglie migliori
                if labels[i] == 1 and labels[i + 1] == -1 and sum_ > max_:
                    max_ = sum_
                    i_plus = i
                elif labels[i] == -1 and labels[i + 1] == 1 and sum_ < min_:
                    min_ = sum_
                    i_minus = i

        # Passo 4: passa al successivo
        i += 1

    # Restituisce le soglie scelte
    b_plus = i_plus
    b_minus = i_minus

    return b_plus, b_minus
