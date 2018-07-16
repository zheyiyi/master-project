#%matplotlib inline
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from glob import glob
import seaborn as sns
import random
from keras.preprocessing import image
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


path_all_data = './train'

n_classes = 0
images_number = []
sample_images = {}
class_dirs = []
for class_dir in os.listdir(path_all_data):
    train_dir = os.path.join(path_all_data, class_dir)
    if train_dir != './train/.DS_Store':
        n_classes += 1
        total_images = os.listdir(train_dir)
        images_number.append(len(total_images))
        print(class_dir, len(total_images))

        images_path = os.path.join(train_dir, '**')
        class_images = glob(images_path)
        #print(class_images)
        images = random.sample(class_images, 10)
        #print(images)

        sample_images[class_dir] = images
        class_dirs.append(class_dir)




plt.figure(figsize=(10, 8))
plt.title("Number of cases per fruit (Training data)")
plt.bar(range(n_classes), list(images_number))
plt.xticks(range(n_classes), os.listdir(path_all_data), rotation=90)
plt.show()

datas = []
for class_id, class_name in enumerate(class_dirs):
    for image_name in os.listdir(os.path.join(path_all_data, class_dir)):
        datas.append([os.path.join(path_all_data, class_dir,image_name),class_id, class_name])

data_pandas = pd.DataFrame(datas, columns = ['file', 'class_id', 'class_name'])
print(data_pandas.head(5))
print(data_pandas.shape)


fig = plt.figure(1, figsize=(n_classes, n_classes))
grid = ImageGrid(fig, 111, nrows_ncols=(n_classes, n_classes), axes_pad=0.05)
i = 0
for class_id, class_name in enumerate(class_dirs):
    for filepath in data_pandas[data_pandas['class_name'] == class_name]['file'].values[:n_classes]:
        ax = grid[i]
        img = image.load_img(filepath, target_size=(224, 224))
        img = image.img_to_array(img)
        ax.imshow(img / 255.)
        ax.axis('off')
        if i % n_classes == n_classes - 1:
            ax.text(250, 112, filepath.split('/')[1], verticalalignment='center')
        i += 1
plt.show()





