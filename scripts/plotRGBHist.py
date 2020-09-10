import numpy as np
import scipy.stats as sstats
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2
import seaborn as sns

com1 = cv2.cvtColor(cv2.imread('../tori1/tori1-1.png',1), cv2.COLOR_BGR2RGB)
com2 = cv2.cvtColor(cv2.imread('../tori1/tori1-2.png',1), cv2.COLOR_BGR2RGB)

def rgb_hist(rgb_img):
    sns.set()
    sns.set_style(style='ticks')
    fig = plt.figure(figsize=[15,4])
    ax1 = fig.add_subplot(1,2,1)
    sns.set_style(style='whitegrid')
    ax2 = fig.add_subplot(1,2,2)
    
    ax1.imshow(rgb_img)

    color=['r','g','b']

    for (i,col) in enumerate(color): # 各チャンネルのhist
        # cv2.calcHist([img], [channel], mask_img, [binsize], ranges)
        hist = cv2.calcHist([rgb_img], [i], None, [255], [0,255])
        # グラフの形が偏りすぎるので √ をとってみる
        hist = np.sqrt(hist)
        ax2.plot(hist,color=col)
        ax2.set_xlim([0,256])

    plt.show()

rgb_hist(com2)
