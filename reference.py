from math import pi, e
from math import sqrt, pow
from matplotlib import pyplot

### LEGENDRE-GAUSS VALUES (N = 13)
### NOTATION: (weight, abscissa)
LG_VALUES = [
    (0.2325515532308739, 0.0),
    (0.2262831802628972, -0.2304583159551348),
    (0.2262831802628972, 0.2304583159551348),
    (0.2078160475368885, -0.4484927510364469),
    (0.2078160475368885, 0.4484927510364469),
    (0.1781459807619457, -0.6423493394403402),
    (0.1781459807619457, 0.6423493394403402),
    (0.1388735102197872, -0.8015780907333099),
    (0.1388735102197872, 0.8015780907333099),
    (0.0921214998377285, -0.9175983992229779),
    (0.0921214998377285, 0.9175983992229779),
    (0.0404840047653159, -0.9841830547185881),
    (0.0404840047653159, 0.9841830547185881)
]

### EVALUATE GAUSSIAN FUNCTION FROM VALUES OF A, B, T
def f_Gaussian (a, b, t):
    x = (b + a)/2 + ((b - a) * t)/2
    dx = (b - a)/2

    f = (1/sqrt(6 * pi)) * pow(e , pow(x,2)/-6) * dx

    return f

### EVALUATE T-DISTRIBUTION FUNCTION FROM VALUES OF A, B, T
def f_tDist (a, b, t):
    x = (b + a)/2 + ((b - a) * t)/2
    dx = (b - a)/2
    gamma = 0.8862269

    f = (1/(gamma * sqrt(3 * pi))) * pow(1 + pow(x,2)/3 , -2) * dx

    return f

### MAIN FUNCTION
if __name__ == '__main__':
    y_CDF_Gaussian = []
    y_CDF_tDist = []
    xValues = []

    ### FOR x IN [-6, 6] AT 0.1 INTERVALS
    ### x < 0
    for a in range (-60, 0, 1):
        a /= 10.0

        area_Gaussian = 0
        area_tDist = 0

        for i in range(13):
            area_Gaussian += LG_VALUES[i][0] * f_Gaussian( a, 0, LG_VALUES[i][1] )
            area_tDist += LG_VALUES[i][0] * f_tDist( a, 0, LG_VALUES[i][1] )

        xValues.append(a)

        y_CDF_Gaussian.append(0.5 - area_Gaussian)
        y_CDF_tDist.append(0.5 - area_tDist)

    ### x > 0
    for b in range (0, 61, 1):
        b /= 10.0

        area_Gaussian = 0
        area_tDist = 0

        for i in range(13):
            area_Gaussian += LG_VALUES[i][0] * f_Gaussian(0, b, LG_VALUES[i][1])
            area_tDist += LG_VALUES[i][0] * f_tDist(0, b, LG_VALUES[i][1])

        xValues.append(b)

        y_CDF_Gaussian.append(0.5 + area_Gaussian)
        y_CDF_tDist.append(0.5 + area_tDist)
    
    ### GRAPHING
    figure, axes = pyplot.subplots()

    axes.plot (xValues, y_CDF_Gaussian, label = 'Gaussian CDF')
    axes.plot (xValues, y_CDF_tDist, label = 't-Distribution CDF')

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.legend()

    pyplot.savefig('Gaussian vs t Distribution.png')
    
    pyplot.show()












### CACHE

# @title Inverse survivability
# solved as root-finding problem using bisection method
# def surv_inv (alpha, a, b, k, mu):
#     isDiffSign = lambda x1, x2 : np.sign(x1) != np.sign(x2)

#     f = lambda t : cdf(t, k, mu) - alpha

#     tol = 1.0e-9

#     fa = f(a)
#     fb = f(b)

#     if fa == 0.0:
#         return a
    
#     if fb == 0.0:
#         return b
    
#     print("-> CDF(" + str(a) + ") - " + str(alpha) + " = " + str(fa))
#     print("-> CDF(" + str(b) + ") - " + str(alpha) + " = " + str(fb))
    
#     if not isDiffSign(fa, fb):
#         print("---> no root at: ", alpha)
#         return None
    
#     n = int (math.ceil (math.log(abs(b-a)/tol) / math.log(2.0)))

#     print("---> iterations: ", n)

#     for i in range(n):
#         print("-> CDF(" + str(a) + ") - " + str(alpha) + " = " + str(fa))
#         print("-> CDF(" + str(b) + ") - " + str(alpha) + " = " + str(fb))

#         c = 0.5 * (a + b)
#         fc = f(c)

#         if fc == 0.0:
#             print("---> returning c =", c)
#             return c

#         if isDiffSign(fa, fc):
#             b = c
#             fb = fc
        
#         elif isDiffSign(fb, fc):
#             a = c
#             fa = fc

#     return 0.5 * (a+b)
