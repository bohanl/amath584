import numpy as np


def lu(A):
  """
  LU decomposition of a square matrix `A` with pivoting.
  """
  m, n = A.shape
  assert m == n, "requires a square matrix"

  U, L, P = A.copy(), np.eye(m), np.eye(m)
  for k in range(0, m - 1):
    # select the row `i` (i >= k) maximizes `U[k:,k]`
    i = k + np.argmax(np.abs(U[k:,k]))
    # exchange `k`th and `i`th row
    if i != k:
      U[[i,k],k:m] = U[[k,i],k:m]
      if k > 0:
        L[[i,k],:k-1] = L[[k,i],:k-1]
      P[[i,k],:] = P[[k,i],:]

    for j in range(k + 1, m):
      L[j,k] = U[j,k] / U[k,k]
      U[j,k:m] = U[j,k:m] - L[j,k]*U[k,k:m]

  return P, L, U


def main():
  np.set_printoptions(suppress=True, linewidth=1000)
  A = np.random.rand(5,5)

  P, L, U = lu(A)
  print(L)
  print()
  print(U)

  import scipy.linalg
  p, l, u = scipy.linalg.lu(A)
  print(l)
  print()
  print(u)


if __name__ == '__main__':
  main()
