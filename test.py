def lagrange_term(term, x_datapoints, i, x, degree):
    for j in range(degree+1):
        if j != i:
            term *= (x - x_datapoints[j]) / (x_datapoints[i] - x_datapoints[j])
    return term

def lagrange_evalfunct(x_datapoints, y_datapoints, x, degree) -> float:
    value = 0
    for i in range(degree+1):
        term = lagrange_term(y_datapoints[i], x_datapoints, i, x, degree)
        value += term
    return value

x = [0, 1, 2, 5]
y = [2, 3, 12, 147]

print(lagrange_evalfunct(x, y, 3, 3))