import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import glob

SUPPORTED_IMAGE_DIM = (192, 168)


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


def read_images(dir_path, type='pgm'):
  """
  Read all images under a directory recursively.

  Args:
    dir_path - directory to read images recursively.
    type - the file extension that it searches for.
  Returns:
    A list of all images found and sorted by their names.
  """
  images = []
  for f in sorted(glob.glob(f'{dir_path}/**/*.{type}', recursive=True)):
    image = io.imread(f)
    assert image.shape == SUPPORTED_IMAGE_DIM, f'Image must be gray scaled '
    f'{SUPPORTED_IMAGE_DIM}'
    images.append(image)

  return images


def compute_svd(images):
  """
  """
  # Center all images at the "origin"
  avg = np.mean(images, axis=1)
  for j in range(images.shape[1]):
    images[:, j] -= avg

  # Compute the SVD of the centered matrix. Images are stored as column vectors
  # in matrix `A`, therefore `U` provides an orthonormal basis and `V` is the
  # projection matrix that projects `A` onto `U`.
  U, s, Vh = np.linalg.svd(images, full_matrices=False)
  return U, s, Vh


def main():
  # Do SVD analysis for cropped images.
  cropped_images = concat_images(read_images('data/CroppedYale/'))
  print(f'Cropped images size: {cropped_images.shape}')
  cropped_U, cropped_s, cropped_Vh = compute_svd(cropped_images)

  # Plot a first few reshaped columns of `cropped_U`
  fig = plt.figure(figsize=(8, 8))
  fig.suptitle('Orthonormal Basis for Cropped Images')
  for i in range(1, 26):
    img = cropped_U[:, i-1].reshape(SUPPORTED_IMAGE_DIM)
    fig.add_subplot(5, 5, i)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
  plt.show()

  print(f'Rank: {len(cropped_s)}')
  print(cropped_s[:100])

  # Plot singular values to show the decay.
  plt.title('Singular Value Decay for Cropped Images')
  plt.ylabel('Singular Values')
  plt.xticks(np.arange(0, len(cropped_s), 100))
  plt.plot(cropped_s)
  plt.show()


if __name__ == '__main__':
  main()
