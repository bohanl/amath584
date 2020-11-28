import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag

np.set_printoptions(suppress=True)

m = 10
A = np.random.randn(m, m) + np.random.randn(m, m)*1j
# A = A.T @ A  # make `A` symmetric

eigenvals, eigenvecs = np.linalg.eig(A)

print("========")
print(eigenvals)
print("========")

# Rayleigh quotient iteration
def rayqot(A, v0, maxiter=50, eps=1e-10):
    v0 = v0 / np.linalg.norm(v0)
    sigma = v0.T @ A @ v0
    sigma_iters = np.array([sigma])
    i = 0
    while i < maxiter:
        if np.linalg.norm((A - sigma * np.eye(m)) @ v0) < eps:
            break
        w = np.linalg.solve(A - sigma * np.eye(m), v0)
        v0 = w / np.linalg.norm(w)
        sigma = v0.T.conjugate() @ A @ v0
        sigma_iters = np.append(sigma_iters, sigma)
        i += 1

    return sigma, np.abs(sigma_iters - sigma), i


V = np.random.randn(m, 2 * m) + np.random.randn(m, 2 * m)*1j
for j in range(V.shape[1]):
    V[:, j] /= np.linalg.norm(V[:, j])

for j in range(V.shape[1]):
    v = V[:,j]
    for i in range(j):
        v -= v.dot(V[:,i]) * V[:,i]
    V[:, j] = v / np.linalg.norm(v)
    sigma, errors, niters = rayqot(A, v)
    print(f'{sigma} took {niters}')


"""

ax = plt.subplot()

ax.plot([1,2,3,4,5], [0.71611657, 0.02879863, 0.00001023, 0.,         0.], label='Sample 1')
ax.plot([1,2,3,4,5], [0.98277819, 0.3117679,  0.03959511, 0.00033583, 0.], label='Sample 2')
ax.plot([1,2,3,4,5], [2.10297736, 1.20417417, 0.22875746, 0.0011606,  0.], label='Sample 3')

ax.set(xlabel='Iterations', ylabel='Error',
      title='Rayleigh Quotient Iterations')
ax.grid(True)
ax.legend()
plt.show()
"""
