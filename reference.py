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