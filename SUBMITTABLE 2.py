import math
import numpy as np
import plotly.graph_objects as go

# Gaussian quadrature constants 
# Notation: (w_i, x_i)
GAUSS_LAGUERRE = [
    (0.458964, 0.222847), (0.417, 1.188932), (0.113373, 2.992736), (0.0103992, 5.775144),
    (0.000261017, 9.837467), (0.000000898548, 15.982874)
]

GAUSS_LEGENDRE = [
    (0.2491470458134028, -0.1252334085114689), (0.2491470458134028, 0.1252334085114689),
    (0.2334925365383548, -0.3678314989981802), (0.2334925365383548, 0.3678314989981802),
    (0.2031674267230659, -0.5873179542866175), (0.2031674267230659, 0.5873179542866175),
    (0.1600783285433462, -0.7699026741943047), (0.1600783285433462, 0.7699026741943047),
    (0.1069393259953184, -0.9041172563704749), (0.1069393259953184, 0.9041172563704749),
    (0.0471753363865118, -0.9815606342467192), (0.0471753363865118, 0.9815606342467192)
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

# Inverse survivability -- solved as root-finding problem using regula falsi method
def regula_falsi(a, b, alpha, mu, k):
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
    
    while True:
        c = b - fb * (b-a)/(fb-fa)
        fc = f(c)

        if fc == 0.0 or abs(fc) < tol:
            return c
        elif fa*fc < 0:
            b = c
        else:
            a = c 

# Plotting function
def plot (x05, x1, x2, y05, y1, y2, xtitle, ytitle, title, printAsymptote = False):
    fig = go.Figure()
    
    fig.add_traces([
        go.Scatter(x=x05, y=y05, mode='lines', marker = {'color' : 'blue'}, name="k = 0.5"),
        go.Scatter(x=x1, y=y1, mode='lines', marker = {'color' : 'red'}, name="k = 1"),
        go.Scatter(x=x2, y=y2, mode='lines', marker = {'color' : 'magenta'}, name="k = 2")
    ])

    if printAsymptote:
        fig.add_traces([go.Scatter(x=[i/100 for i in range(0, 101, 1)], y=[78 for i in range(0, 101, 1)], line_dash ='dash', marker = {'color' : 'orange'}, name="Life expectancy")])
    
    fig.update_layout(
        title_text=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        height=1080*0.5,
        width=1920*0.6,
        font_family="CMU Serif",
        font_size=15,
        title_font_size=25,
        font_color="#0e0f11",
        margin=dict(t=120, b=80)
    )
    
    fig.show()

# Lists for values storing
x_values, y_t_05, y_t_1, y_t_2 = [], [], [], []

# Average life expectancy
mu = 78

# For alpha in (0,1) with 0.01 step
for alpha in range(1, 101, 1):
    alpha /= 100
    x_values.append(alpha)
    y_t_05.append(regula_falsi(1, 121, alpha, mu, 0.5))
    y_t_1.append(regula_falsi(1, 121, alpha, mu, 1))
    y_t_2.append(regula_falsi(1, 121, alpha, mu, 2))

# Plotting proper
plot(x_values, x_values, x_values, y_t_05, y_t_1, y_t_2, "Probability (alpha)", "Time (t)", "Inverse survival function", True)