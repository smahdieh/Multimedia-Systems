from unicodedata import normalize
from utils import *
import numpy as np
import math


def grayScaledFilter(img):

    A = [0.2989 , 0.5870 , 0.1140]
    T = np.array([A , A , A])
    return Filter(img , T)

def histogram_CDF_Pic(img):

    hist,bins = np.histogram(img.flatten(),256,[0,256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

def Transformfunction1(file_name):

    img = Image.open(file_name)
    
    pix_img = np.zeros((3 , 256))
    w, h = img.size
    for x in range(w):
        for y in range(h):
            pixel = img.getpixel((x, y))
            pix_img[0][pixel[0]] += 1
            pix_img[1][pixel[1]] += 1
            pix_img[2][pixel[2]] += 1
    color_levels = np.zeros(3)
    for i in range(0 , 3):
        for j in range(0 , 256):
            if(pix_img[i][j] != 0):
                color_levels[i] += 1
    sum = np.zeros((3 , 256))
    for i in range(0 , 3):
        for j in range(0 , 256):
            if(pix_img[i][j] != 0):
                sum[i][j] = sum[i][j - 1] + pix_img[i][j]
    normalized_sum = np.zeros((3 , 256))
    for i in range(0 , 3):
        for j in range(0 , 256):
            if(pix_img[i][j] != 0):
                normalized_sum[i][j] = round(((color_levels[i] - 1) * sum[i][j]) / (w * h))
    return normalized_sum


def contrast(img , T):

    newimg1 = np.zeros_like(img)
    newimg2 = np.zeros_like(img)
    a = img.min()
    b = img.max()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for t in range(img.shape[2]):
                newimg1[i][j][t] = T[t][img[i][j][t]]
                newimg2[i][j][t] = round((img[i][j][t] - a)*(255 / (b - a)))

    return newimg1 , newimg2

if __name__ == "__main__":
    image_matrix = get_input('image.png')
    #remove 4th channel
    image_matrix = image_matrix[:, :, :3]

    #show input image
    showImage(image_matrix, title="Input Image")

    #grey scale of image
    grayScalePic = grayScaledFilter(image_matrix)
    showImage(grayScalePic, title="Gray Scaled")
    
    #histogram & CDF
    histogram_CDF_Pic(image_matrix)

    #contrast Transform function
    T = Transformfunction1('image.png')
    #c = find_C('image.png')

    #contrast image convert
    contrastPic1 , contrastPic2 = contrast(image_matrix , T)
    showImage(contrastPic1, title="contrast pic1")
    showImage(contrastPic2, title="contrast pic2")

    #histogram & CDF of contrast image
    histogram_CDF_Pic(contrastPic1)
    histogram_CDF_Pic(contrastPic2)

    





