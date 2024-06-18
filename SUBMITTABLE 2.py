import math
import numpy as np
import plotly.graph_objects as go

# Gaussian quadrature constants 
# Notation: (w_i, x_i)
GAUSS_LAGUERRE = [
    (0.458964, 0.222847),
    (0.417, 1.188932),
    (0.113373, 2.992736),
    (0.0103992, 5.775144),
    (0.000261017, 9.837467),
    (0.000000898548, 15.982874)
]

GAUSS_LEGENDRE = [
    (0.360762, -0.661209),
    (0.467914, -0.238619),
    (0.171324, -0.932469),
    (0.171324, 0.932469),
    (0.467914, 0.238619),
    (0.360762, 0.661209)
]

# Weibull distribution function -- uses Gauss-Laguerre quadratures
def f_Weibull (t, k, mu):
    denomArea = 0
    for node in GAUSS_LAGUERRE:
        denomArea += node[0] * (node[1]**(1/k))

    lbda = mu / denomArea

    return (k/lbda) * ((t/lbda)**(k-1)) * (math.e ** (-(t/lbda)**k))

# Integral of Weibull function -- uses Gauss-Legendre quadratures
def cdf (t, k, mu):
    pdf_area = 0
    for node in GAUSS_LEGENDRE:
        u = (t/2)*node[1] + t/2
        du = t/2
        pdf_area += node[0] * f_Weibull(u, k, mu) * du

    return pdf_area

# Inverse survivability -- solved as root-finding problem using bisection method
def bisection (a, b, alpha, mu, k):
    f = lambda t : 1 - cdf(t, k, mu) - alpha

    tol = 1.0e-9
    fa = f(a)
    fb = f(b)

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if np.sign(fa) == np.sign(fb):
        return None
    
    n = int (math.ceil (math.log(abs(b-a)/tol) / math.log(2.0)))

    for i in range(n):
        c = 0.5 * (a + b)
        fc = f(c)

        if fc == 0.0:
            return c
        if np.sign(fa) != np.sign(fc):
            b = c
            fb = fc
        elif np.sign(fb) != np.sign(fc):
            a = c
            fa = fc

    return 0.5 * (a+b)

# Plotting function
def plot (x05, x1, x2, y05, y1, y2, xtitle, ytitle, title):
    fig = go.Figure()
    fig.add_traces([
        go.Scatter(x=x05, y=y05, mode='lines', marker = {'color' : 'blue'}, name="k = 0.5"),
        go.Scatter(x=x1, y=y1, mode='lines', marker = {'color' : 'red'}, name="k = 1"),
        go.Scatter(x=x2, y=y2, mode='lines', marker = {'color' : 'magenta'}, name="k = 2")
    ])
    fig.update_layout(
        title_text=title,
        height=1080*0.5,
        width=1920*0.6,
        xaxis_title=xtitle,
        yaxis_title=ytitle
    )
    fig.show()

# Lists for values storing
x_values, y_t_05, y_t_1, y_t_2 = [], [], [], []

# Average life expectancy
mu = 78

# For alpha in (0,1) with 0.01 step
for alpha in range(1, 100, 1):
    alpha /= 100
    x_values.append(alpha)
    y_t_05.append(bisection(1, 250, alpha, mu, 0.5))
    y_t_1.append(bisection(1, 250, alpha, mu, 1))
    y_t_2.append(bisection(1, 250, alpha, mu, 2))

# Plotting proper
plot(x_values, x_values, x_values, y_t_05, y_t_1, y_t_2, "Probability (alpha)", "Time (t)", "Inverse survival (mu = 78)")