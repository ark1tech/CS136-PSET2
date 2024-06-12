import math

start = -5
end = 3
n_points = 9 


for x in range(1, n_points+1):
    print(start+(end-start)*((-math.cos(((x-1)*math.pi)/n_points)+1)/2), x)
    # print((start + end)/2 + ((end-start)/2)*(math.cos(((2*x+1)/(2*n_points))*math.pi)), x)