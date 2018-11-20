
def E(x, k, b, y):
    return (y - (k*x + b))**2

def J(x, k, b, y):
    sum = 0
    for i in range(len(x)):
        sum += E(x[i], k, b, y[i])
    return sum/len(x)

def delta(x, k, b, y):
    return (y - (k*x + b))

def sum_delta(x, k, b, y):
    sum = 0
    for i in range(len(x)):
        sum += delta(x[i], k, b, y[i])
    return sum

def sum_delta_x(x, k, b, y):
    sum = 0
    for i in range(len(x)):
        sum += delta(x[i], k, b, y[i])*x[i]
    return sum


x = [1, 1, 2, 3, 4, 3, 4, 6, 4]
y = [2, 1, 0.5, 1, 3, 3, 2, 5, 4]
k = 0
b = 0
alpha = 0.1
N = 10
for i in range(N):
    k = k - 2*alpha/len(x)*sum_delta_x(x, k, b, y)
    b = b - 2*alpha/len(x)*sum_delta(x, k, b, y)
    print(J(x, k, b, y))

