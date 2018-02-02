#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import cv2
import itertools
from collections import defaultdict
from PIL import Image

def generate_file(disp_file_name):
    disp_noc_file = open("disp_noc.txt", "wb")
    disp_occ_file = open("disp_occ.txt", "wb")
    disp_pred_file = open("disp_dispnet.txt", "wb")
    for i in xrange(200):
        disp_noc_file.write('/home/sensetime/Desktop/data_scene_flow/training/disp_noc_0/' + str(i).zfill(6) + "_10.png" + "\n")
        disp_occ_file.write('/home/sensetime/Desktop/data_scene_flow/training/disp_occ_0/'+str(i).zfill(6) + "_10.png"+"\n")
        disp_pred_file.write('/home/sensetime/Desktop/{}/'.format(disp_file_name)+str(i).zfill(6) + "_10.png"+"\n")



def readimg(f):
    # print(f)
    return np.array(Image.open(f))/256.0

def evaluateValid(gt, d, metrics):
    if metrics == 'EPE':
        gt = gt.astype(np.float32)
        d = d.astype(np.float32)

        e = np.abs(gt - d)
        epe = np.sum(e)
        return epe
    elif metrics == '3pixels':
        # 3 pixels or 5% error, used in KITTI
        e = np.abs(gt - d)
        y1 = e < 3.0
        y2 = (e / gt < 0.05)
        y = y1.astype(np.int32) + y2.astype(np.int32)
        return np.sum((y == 0).astype(np.int32))
    else:
        print 'Evaluation Metric {} cannot be used'.format(metrics)
        exit()


def main(part, metric='EPE'):
    dataset = 'kitti'
    # metric = '3pixels'
    result = {}
    log = open('./log', 'w')

    dirs = '/home/sensetime/Desktop/File/'
    disp_gt = os.path.join(dirs, 'disp_{}.txt'.format(part))
    disp_dispnet = os.path.join(dirs, 'disp_dispnet.txt')

    fgt = open(disp_gt, 'r')
    fnet = open(disp_dispnet, 'r')
    fgt_lines = fgt.readlines()
    fnet_lines = fnet.readlines()

    count = defaultdict(int)
    for i in xrange(200):
        gt = readimg(fgt_lines[i].rstrip('\n'))
        net = readimg(fnet_lines[i].rstrip('\n'))

        ind = gt > 0  # get valid diaparity
        # n = np.sum(ind.astype(np.int32))
        n = np.sum(ind)
        y_net = evaluateValid(gt[ind], net[ind], metric)

        count['all'] += n
        count['net'] += y_net

        log_str = '{}_{}: \tNet_{}: {}'.format(dataset, i, metric, float(y_net) / n)
        log.write(log_str + '\n')
        print log_str

    result[dataset] = count
    result_str = '{}_{}: \tNet_{}_mean: {}'.format(dataset, 'all', metric, float(count['net']) / count['all'])
    print result_str
    log.write(result_str + '\n')
    fgt.close()
    fnet.close()

    # output final results
    log.close()
    print '*' * 40
    for dataset in result:
        count = result[dataset]
        print '{}_{}: \tNet_{}_mean: {}'.format(dataset, 'all', metric, float(count['net']) / count['all'])


if __name__ == '__main__':
    part = "noc" # noc or occ
    metric = '3pixels' # EPE or 3pixels, default: EPE
    main(part)
    # disp_file_name = "disp_GR"
    # generate_file(disp_file_name)