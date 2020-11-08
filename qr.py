import numpy as np
import matplotlib.pyplot as plt

def qr(a):
  """
  QR factorization (Modified GS).
  Returns:
    q - reduced `Q`
    r - `R`
  """
  m, n = a.shape
  q, r = a.copy(), np.zeros((n, n))
  for i in range(n):
    # normalize the current column
    r[i, i] = np.linalg.norm(q[:, i])
    q[:, i] /= r[i, i]
    # for all other columns, substract the projection
    # of each column onto the j-th orthonormal basis.
    for j in range(i + 1, n):
      r[i, j] = np.dot(q[:, i], q[:, j])
      q[:, j] -= r[i, j] * q[:, i]
  return q, r


def qr_class(a):
  """
  QR factorization developed in class (HH).
  (Translated from the MATLAB version).
  Returns:
    q - `Q`
    r - `R`
  """
  m, n = a.shape
  q, a1 = np.eye(m, m), a.copy()
  for k in range(n):
    # Find the HH reflector
    z = a1[k:m, k]
    v = -z
    v[0] += -np.sign(z[0])*np.linalg.norm(z)
    v = v / np.linalg.norm(v)
    # Apply the HH reflection to each column of A and Q
    for j in range(n):
      a1[k:m, j] = a1[k:m, j] - v * (2 * np.dot(v, a1[k:m, j]))
    for j in range(m):
      q[k:m, j] = q[k:m, j] - v * (2 * np.dot(v, q[k:m, j]))
  return q.T, np.triu(a1)

def main():
  m = 10
  A, b = np.random.randn(m, m), np.random.randn(m, 1)
  #A[:,-1] = A[:,0] + (10**-14)*np.random.randn(m)
  print(np.linalg.cond(A))
  Q1, R1 = qr(A)
  Q2, R2 = qr_class(A)
  Q3, R3 = np.linalg.qr(A)
  y1, y2, y3 = Q1.T @ b, Q2.T @ b, Q3.T @ b
  x1, x2, x3 = np.linalg.solve(R1, y1), np.linalg.solve(R2, y2), np.linalg.solve(R3, y3)

  ax = plt.subplot()
  w = 0.75
  dimw = w / 3
  x = np.arange(len(x1))
  ax.bar(x,            x1[:,0], dimw, bottom=0.001, label='Modified-GS')
  ax.bar(x + dimw,     x2[:,0], dimw, bottom=0.001, label='HH')
  ax.bar(x + dimw * 2, x3[:,0], dimw, bottom=0.001, label='Numpy')
  ax.set_xticks(x + dimw / 2)
  ax.set_xticklabels(map(str, x))
  ax.set(ylabel='Solution', title=f'cond(A) = {np.linalg.cond(A)}')
  ax.legend()
  plt.show()

if __name__ == '__main__':
  main()
