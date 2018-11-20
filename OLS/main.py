
x = [1, 1, 2, 3, 4, 3, 4, 6, 4]
y = [2, 1, 0.5, 1, 3, 3, 2, 5, 4]
#print(len(y))
def cost (y, x, k = 0, b = 0):
    J = 0
    for i in range(len(y)):
        J += (y[i] - (k*x[i] + b))**2
        #print(J)
    return J/len(y)

print (cost(y, x))