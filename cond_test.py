import numpy as np


m = 40
n = 40
A = np.random.randn(m, n)
b = np.random.randn(m, 1)

B1 = A.copy()
B1[:, -1] = A[:, 0]
print("cond B1 no noise")
print(np.linalg.cond(B1))

print(np.log(np.linalg.solve(B1, b)))

B2 = A.copy()
B2[:, -1] = A[:, 0] + np.random.randn(m)*(10**(-14))

print("cond B2 with noise")
print(np.linalg.cond(B2))


print(np.log(np.linalg.solve(B2, b)))
