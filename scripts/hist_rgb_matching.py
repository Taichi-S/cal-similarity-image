#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""hist matching.

Usage:
    hist_matching.py
        --target_file_path=<target_file_path>
        --comparing_dir_path=<comparing_dir_path>
    hist_matching.py -h | --help
Options:
    -h --help show this screen and exit.
"""

import cv2
from docopt import docopt
import glob
import logging
import os
import sys
from statistics import mean

if __name__ == '__main__':
    # logging config
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
    )
    logging.info('%s start.' % (__file__))

    # get parameters
    args = docopt(__doc__)
    target_file_path = args['--target_file_path']
    comparing_dir_path = args['--comparing_dir_path']

    # setting
    img_size = (200, 200)
    channels = (0, 1, 2)
    mask = None
    hist_size = 255
    ranges = (0, 255)
    ret = {}

    # get comparing files
    pattern = '%s/*.png'
    comparing_files = glob.glob(pattern % (comparing_dir_path))
    if len(comparing_files) == 0:
        logging.error('no files.')
        sys.exit(1)

    # read target image
    target_file_name = os.path.basename(target_file_path)
    target_img = cv2.imread(target_file_path)
    target_img = cv2.resize(target_img, img_size)

    for comparing_file in comparing_files:
        comparing_file_name = os.path.basename(comparing_file)
        if comparing_file_name == target_file_name:
            continue

        tmp = []
        for channel in channels:
            # calc hist of target image
            # target_hsv = cv2.cvtColor(target_img,cv2.COLOR_BGR2HSV)
            # target_hist = cv2.calcHist([target_hsv], [0], None, [360], [0, 360])
            target_hist = cv2.calcHist([target_img], [channel], mask, [hist_size], ranges)

            # read comparing image
            comparing_img_path = os.path.join(
                os.path.abspath(os.path.dirname(__file__)) + '/../',
                comparing_file,
            )
            comparing_img = cv2.imread(comparing_img_path)
            comparing_img = cv2.resize(comparing_img, img_size)
            # calc hist of comparing image
            # comparing_hsv = cv2.cvtColor(comparing_img,cv2.COLOR_BGR2HSV)
            # comparing_hist = cv2.calcHist([comparing_hsv], [0], None, [360], [0, 360])
            comparing_hist = cv2.calcHist([comparing_img], [channel], mask, [hist_size], ranges)

            # compare hist
            tmp.append(cv2.compareHist(target_hist, comparing_hist, 0))

        # mean hist
        ret[comparing_file] = mean(tmp)

    # sort
    for k, v in sorted(ret.items(), reverse=True, key=lambda x: x[1]):
        logging.info('%s: %f.' % (k, v))

    logging.info('%s end.' % (__file__))
    sys.exit(0)
