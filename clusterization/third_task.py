import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from  first_task import get_standaridized_data

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color
from skimage import segmentation
from skimage.segmentation import mark_boundaries
from sklearn.cluster import KMeans

def show_clusters_boundaries(image, amount_of_colors):
    image = img_as_float(image)
    image = img_as_float(image)
    image_slic = segmentation.slic(image, n_segments=amount_of_colors)
    plt.imshow(segmentation.mark_boundaries(image, image_slic));
    plt.axis('off')
    plt.show()

def change_image_with_new_amount_of_colours(df,image,amount_of_colors,height,width):
    kmeans = KMeans(init="k-means++", n_clusters=amount_of_colors)
    kmeans.fit(df)

    for i in range(0, height):
        for j in range(0, width):
            label = kmeans.labels_[i * width + j]
            label_value = kmeans.cluster_centers_[label]
            for k in range(0, 3):
                image[i][j][k]=label_value[k]

    # image_slic = segmentation.slic(image, n_segments=155)
    # plt.imshow(color.label2rgb(image_slic, image, kind='avg'));

    plt.imshow(image);
    plt.axis('off')
    plt.show()


def third_task():
   image = io.imread('files/mh2.jpg')
   image = np.array(image)
   image = img_as_float(image)

   height, width,_=image.shape

   pixels=[[]]
   for i in range(0, height):
       for j in range(0, width):
           rgb = []
           for k in range(0, 3):
               if (np.isnan(image[i][j][k])==False):
                   rgb.append(image[i][j][k])
           if ((i == 0) and (j == 0)):
               pixels[0]=rgb
           else:
               pixels.append(rgb)
   df = pd.DataFrame(pixels,columns=['r','g','b'])
   print(df)


   amount=[2,4,8,16,32,64]
   for i in amount:
       change_image_with_new_amount_of_colours(df,image,i,height,width)

