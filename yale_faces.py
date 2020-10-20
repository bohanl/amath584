import argparse
import glob
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.function_base import average
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


def main():
  # Do SVD analysis for cropped images.
  images = concat_images(read_images())
  print(f'Images size: {images.shape}')
  U, s, Vh = compute_svd(images)

  # Plot a first few reshaped columns of `U`
  fig = plt.figure(figsize=(8, 8))
  fig.suptitle('Orthonormal Basis for Images')
  for i in range(1, 26):
    img = U[:, i - 1].reshape(_SUPPORTED_IMAGE_DIM)
    fig.add_subplot(5, 5, i)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
  plt.show()

  # Print out a few singular values.
  print(f'Rank: {len(s)}')
  print(s[:100])

  # Plot singular values (normalized) to show the decay.
  plt.title('Singular Value Decay for Images')
  plt.ylabel('Singular Values')
  t = np.arange(0, len(s))
  plt.xticks(np.arange(0, len(s), 500))
  plt.plot(t, s)
  plt.show()


if __name__ == '__main__':
  main()
