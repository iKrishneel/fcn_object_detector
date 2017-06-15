#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2 as cv
from bs4 import BeautifulSoup

class PascalVOT:
    def __init__(self, vot_dir):
    
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat', \
                        'bottle', 'bus', 'car', 'cat', 'chair', \
                        'cow', 'diningtable', 'dog', 'horse', \
                        'motorbike', 'person', 'pottedplant', \
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.voc_root = vot_dir
        self.img_dir_ = os.path.join(vot_dir, 'JPEGImages/')
        self.ann_dir_ = os.path.join(vot_dir, 'Annotations/')
        self.set_dir_ = os.path.join(vot_dir, 'ImageSets', 'Main/')
        
        suffix_train = '_train.txt'
        suffix_val = '_val.txt'
        suffix_trainval = '_trainval.txt'

        self.train_file = self.set_dir_ + self.classes[0] + suffix_train
        self.val_file = self.set_dir_ + self.classes[0] + suffix_val
        
        self.__img_ext = '.jpg'
        self.__ano_ext = '.xml'

        self.create()

    #! just read one
    def create(self):

        out_dir = '/home/krishneel/Desktop/'
        
        type_train = False
        if type_train:
            lines = self.read_textfile(self.train_file)
            outfile_name = out_dir + 'train.txt'
        else:
            lines = self.read_textfile(self.val_file)
            outfile_name = out_dir + 'val.txt'

        label_manifest = out_dir + 'class_label_names.txt'
        with open(outfile_name, 'w') as text_file:
            for line in lines:
                indx = line.split(' ')[0]
                im_fn = self.img_dir_ + indx + self.__img_ext
                an_fn = self.ann_dir_ + indx + self.__ano_ext            
                
                annotation = self.get_bounding_box(an_fn)

                str_labels = im_fn + ','
                for i, anno in enumerate(annotation):
                    name, (x,y,w,h) = anno
                    if self.classes.count(name):
                        label = self.classes.index(name)
                        str_labels += (str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + ' ' + str(label) + ',')
                if str_labels[-1:] == ',':
                    str_labels = str_labels[:-1]
                    str_labels += '\n'
                text_file.write(str_labels)
        
        with open(label_manifest, 'w') as f:
            for i, c in enumerate(self.classes):
                f.write(str(i) + ' ' + c + '\n')

    def get_bounding_box(self, filename):
        annotation = self.load_annotation(filename)
        objects = annotation.findAll('object')

        annot_info = []
        for obj in objects:
            obj_names = obj.findChildren('name')
            for name_tag in obj_names:
                fname = annotation.findChild('filename').contents[0]
                bbox = obj.findChildren('bndbox')[0]
                xmin = int(bbox.findChildren('xmin')[0].contents[0])
                ymin = int(bbox.findChildren('ymin')[0].contents[0])
                xmax = int(bbox.findChildren('xmax')[0].contents[0])
                ymax = int(bbox.findChildren('ymax')[0].contents[0])
        
                rect = [xmin, ymin, xmax - xmin, ymax - ymin]
                annot_info.append((str(name_tag.contents[0]), rect))
                
        return annot_info


    def load_annotation(self, filename):
        xml = ""
        with open(filename) as f:
            xml = f.readlines()
            xml = ''.join([line.strip('\t') for line in xml])
        return BeautifulSoup(xml, 'lxml')

        
    def read_textfile(self, path_to_txt):
        lines = [line.rstrip('\n')
                 for line in open(path_to_txt)
        ]
        return lines


def main(argv):
    if len(argv) < 2:
        raise Exception('Provide path to /VOC<YEAR>/ImageSets/Main/')
        
    vot_dir = str(argv[1])
    if not os.path.isdir(vot_dir):
        raise Exception('VOT Directory does not exist')
        
    vot = PascalVOT(vot_dir)


def test(argv):
    lines = [line.rstrip('\n')
             for line in open(argv[1])
         ]
    im_paths = []
    rects = []
    labels = []
    for line in lines:
        segment = line.split(',')
        im_paths.append(str(segment[0]))

        labs = []
        bbox = []
        for index in xrange(1, len(segment), 1):
            seg = segment[index].split(' ')
            bbox.append(map(int, seg[:-1]))
            labs.append(int(seg[-1]))
        labels.append(labs)
        rects.append(bbox)
        
    print labels

if __name__ == '__main__':
    main(sys.argv)
    #! test(sys.argv)
