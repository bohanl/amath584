import argparse
import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage import io

parser = argparse.ArgumentParser(description='yale_faces')
parser.add_argument('--cropped', dest='cropped', action='store_true')
parser.add_argument('--uncropped', dest='cropped', action='store_false')
parser.set_defaults(cropped=True)

args = parser.parse_args()

_SUPPORTED_IMAGE_DIM = (243, 320)
_DATA_DIR = 'data/yalefaces_uncropped/yalefaces'
if args.cropped:
  _SUPPORTED_IMAGE_DIM = (192, 168)
  _DATA_DIR = 'data/CroppedYale'


def concat_images(images):
  """
  Flatten all images and combine them into a matrix
  where each column is a flattened image. This method
  only takes gray scaled images.

  Args:
    images - a list of gray scaled images.
  Returns:
    A matrix with flattened images as its columns.
  """
  if len(images) == 0:
    return None

  dim = images[0].shape
  A = np.empty((dim[0] * dim[1], len(images)))
  for i, image in enumerate(images):
    A[:, i] = image.flatten()

  return A


def read_images():
  """
  Read all images under a directory recursively.

  Returns:
    A list of all images found and sorted by their names.
  """
  image_file_ext = '.pgm' if args.cropped else ''
  images = []
  for f in sorted(glob.glob(f'{_DATA_DIR}/**/*{image_file_ext}',
                            recursive=True)):
    image = io.imread(f, as_gray=True)
    assert image.shape == _SUPPORTED_IMAGE_DIM, f'Image must be gray scaled '\
      f'{_SUPPORTED_IMAGE_DIM}, but got {image.shape}'

    images.append(image)

  return images


def compute_svd(image_matrix):
  """
  Center the image matrix and compute its SVD decomposition.

  Returns:
    U - An orthonormal basis for `image_matrix`.
    s - An array of singular values.
    Vh - The projection matrix that projects `image_matrix` onto `U`.
  """
  # Center all images at the "origin"
  avg = np.mean(image_matrix, axis=1)
  for j in range(image_matrix.shape[1]):
    image_matrix[:, j] -= avg

  # Compute the SVD of the centered matrix. Images are stored as column vectors
  # in matrix `A`, therefore `U` provides an orthonormal basis and `V` is the
  # projection matrix that projects `A` onto `U`.
  U, s, Vh = np.linalg.svd(image_matrix, full_matrices=False)
  return U, s, Vh


def randomized_svd(A, k):
    """
    SVD low rank approximation using randomized
    algorithms.
    """
    m, n = A.shape
    assert k > 0 and k < n

    # Stage A - find an orthonormal basis Q
    #   1) Random sampling columns of A
    Omega = np.random.randn(n, k)
    Y = A @ Omega  # Y is m x k
    #   2) Econ QR on Y and use Q as the new
    #      basis for A
    Q, R = np.linalg.qr(Y, mode='reduced')

    # Stage B - SVD on B
    #  1) Project A onto the new basis Q
    B = Q.T @ A  # B is k x n

    return np.linalg.svd(B)


def power_iteration(A):
  m, _ = A.shape
  # Initial guess v0 (normalized).
  v = np.random.randn(m)
  v = v / np.linalg.norm(v)
  # Power iterations
  eigenval = 0
  steps = 0
  while True:
    w = A @ v
    v = w / np.linalg.norm(w)
    # The largest eigenvalue in each iteration.
    new_eigenval = v.T.conjugate() @ A @ v
    if np.abs(new_eigenval - eigenval) < 1e-10:
        break
    eigenval = new_eigenval
    steps += 1

  return eigenval, v, steps


def main():
  """
  # Do SVD analysis for cropped images.
  images = concat_images(read_images())
  print(f'Images size: {images.shape}')
  # True SVD
  U, sigmas, Vh = compute_svd(images)
  # SVD using randomized sampling
  print(f'image rank is {np.linalg.matrix_rank(images)}')

  eigenval, _, steps = power_iteration(images.T @ images)

  print(f'iterations = {steps}, largest eigenvalue = {eigenval}')
  print(f'Leading SVD mode = {sigmas[0]}')

  ax = plt.subplot()

  ks = np.array([10, 20, 30, 40, 50, 100])
  ax.plot(np.arange(ks[-1]), sigmas[:ks[-1]], label='True')
  for k in ks:
      U_rnd, sigmas_rnd, Vh_rnd = randomized_svd(images, k=k)
      ax.plot(np.arange(k), sigmas_rnd, label=f'# Samples = {k}')
  ax.set(xlabel='Modes', ylabel='Values',
       title='Singular Value Decay')
  ax.grid(True)
  ax.legend()
  plt.show()
  """

  m = 10
  A = np.random.randn(m, m)
  A = A.T @ A  # make `A` symmetric

  eigenvals, eigenvecs = np.linalg.eig(A)

  print("========")
  print(eigenvals)
  print("========")

  eigenval, _, steps = power_iteration(A)

  print(f'iterations = {steps}, largest eigenvalue = {eigenval}')
  print(f'True largest eigenvalue = {eigenvals[0]}')


if __name__ == '__main__':
  main()
