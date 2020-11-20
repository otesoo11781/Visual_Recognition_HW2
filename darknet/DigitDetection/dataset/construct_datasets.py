# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 13:28:27 2018
@author: PavitrakumarPC
@Editor: Sheng-Hung Kuo
"""

import cv2
import os
import h5py


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        attrs[key] = values
    return attrs


def img_boundingbox_data_constructor(img_dir, mat_file_name):
    mat_file = os.path.join(img_dir, mat_file_name)
    f = h5py.File(mat_file, 'r')
    print('image bounding box data construction starting...')
    for j in range(f['/digitStruct/bbox'].shape[0]):
        img_name = get_name(j, f)
        row_dict = get_bbox(j, f)
        img = cv2.imread(os.path.join(img_dir, img_name))
        img_h, img_w, _ = img.shape
        with open(os.path.join(img_dir, img_name.split('.')[0] + '.txt'), 'w') as fout:
            #print(os.path.join(img_dir, img_name.split('.')[0] + '.txt'))
            for idx in range(len(row_dict['label'])):
                label = int(row_dict['label'][idx])
                if label == 10:
                    label = 0
                center_x = (row_dict['left'][idx] + 0.5*row_dict['width'][idx]) / img_w
                center_y = (row_dict['top'][idx] + 0.5*row_dict['height'][idx]) / img_h
                w = row_dict['width'][idx] / img_w
                h = row_dict['height'][idx] / img_h
                print(label, center_x, center_y, w, h, file=fout)

    print('finished image bounding box data construction...')


if __name__ == '__main__':
    train_folder = "./train"
    img_boundingbox_data_constructor(train_folder, 'digitStruct.mat')
