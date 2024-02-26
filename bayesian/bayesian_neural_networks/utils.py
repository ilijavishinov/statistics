import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import figure
from matplotlib.backends import backend_agg
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
IMAGE_SHAPE = [28, 28, 1]


class DatasetSequence(tf.keras.utils.Sequence):
    """
    Tensorflow sequence class for training on the MNIST images dataset
    """
    def __init__(self, data, batch_size=128):
        images, labels = data
        self.images, self.labels = self.__preprocessing(images, labels)
        self.batch_size = batch_size
    
    @staticmethod
    def __preprocessing(images, labels):
        images = 2 * (images / 255.) - 1.
        images = images[..., tf.newaxis]
        labels = tf.keras.utils.to_categorical(labels)
        return images, labels
    
    def __len__(self):
        return int(tf.math.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_x = self.images[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    


def plot_heldout_prediction(input_vals, probs, fname, n=3):
  """
  A function for plotting the predictions for specific images
  """
  
  fig = figure.Figure(figsize=(9, n*3))
  canvas = backend_agg.FigureCanvasAgg(fig)
  for i in range(n):
    ax = fig.add_subplot(n, 3, 1+i*3)
    ax.imshow(input_vals[i, :].reshape(IMAGE_SHAPE[:-1]),
              interpolation='None')

    ax = fig.add_subplot(n, 3, 3*i + 2)
    for prob_sample in probs:
      sns.barplot(np.arange(10), prob_sample[i, :], alpha=0.05, ax=ax)
      ax.set_ylim([0, 1])
    ax.set_title('posterior samples')

    ax = fig.add_subplot(n, 3, 3*i + 3)
    sns.barplot(np.arange(10), np.mean(probs[:, i, :], axis=0), ax=ax)
    ax.set_ylim([0, 1])
    ax.set_title('predictive probs')
  
  fig.tight_layout()
  canvas.print_figure(fname, format='png')

