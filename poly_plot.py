import numpy as np
import matplotlib.pyplot as plt


def main():
  x = np.arange(1.920, 2.080, 0.001)
  y1 = x**9-18*(x**8)+144*(x**7)-672*(x**6)+2016*(x**5)-4032*(x**4) \
       +5376*(x**3)-4608*(x**2)+2304*x-512
  y2 = (x-2)**9

  ax = plt.subplot()
  ax.plot(x, y1, label='RHS')
  ax.plot(x, y2, label='LHS')
  ax.set(xlabel='x values', ylabel='y values',
       title='Polynomial Plot')
  ax.grid(True)
  ax.legend()
  plt.show()

if __name__ == '__main__':
  main()
