
x = [1, 1, 2, 3, 4, 3, 4, 6, 4]
y = [2, 1, 0.5, 1, 3, 3, 2, 5, 4]
#print(len(y))

def epsilon (y, x, k, b):
    return (y - (k*x + b))**2

def cost (y, x, k=0, b=0):
    J = 0
    for i in range(len(y)):
        J += epsilon(y[i], x[i], k, b)
        #print(J)
    return J/len(y)
print (cost(y, x))

