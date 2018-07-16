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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


#PCA
images = []
labels = []
path_all_data = './train'

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

for class_dir in os.listdir(path_all_data):
    train_dir = os.path.join(path_all_data, class_dir)
    for image_path in glob(os.path.join(train_dir, "*.png")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (150, 150))
        mask = create_mask_for_plant(image)
        image = cv2.bitwise_and(image, image, mask = mask)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (45, 45))

        image = image.flatten()

        images.append(image)
        labels.append(class_dir)

images = np.array(images)
labels = np.array(labels)


label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
label_ids = np.array([label_to_id_dict[x] for x in labels])
images_scaled = StandardScaler().fit_transform(images)


def visualize_scatter(data_2d, label_ids, figsize=(20, 20)):
    plt.figure(figsize=figsize)
    plt.grid()


    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):

        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])

    plt.legend(loc='best')
    plt.show()



pca = PCA(n_components=180)
pca_result = pca.fit_transform(images_scaled)
print(pca_result.shape)


tsne = TSNE(n_components=2, perplexity=40.0)
tsne_result = tsne.fit_transform(pca_result)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)


visualize_scatter(tsne_result_scaled, label_ids)

