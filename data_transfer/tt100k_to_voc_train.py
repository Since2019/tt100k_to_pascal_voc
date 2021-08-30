import json
import os
import sys
import shutil
from PIL import Image
from voc_xml_generator import xml_fill


def find_image_size(filename):
    with Image.open(filename) as img:
        img_width = img.size[0]
        img_height = img.size[1]
        img_mode = img.mode
        if img_mode == "RGB":
            img_depth = 3
        elif img_mode == "RGBA":
            img_depth = 3
        elif img_mode == "L":
            img_depth = 1
        else:
            print("img_mode = %s is neither RGB or L" % img_mode)
            exit(0)
        return img_width, img_height, img_depth


class ConvertTrainLabels:
    def __init__(self, dataset_root_in, dataset_root_out):

        # dataset_root

        # 输入的目录
        self.dataset_root_in = dataset_root_in

        # 输出的目录
        self.dataset_root_out = dataset_root_out

    def load_mask(self, annos, datadir, imgid, filler):
        img = annos["imgs"][imgid]
        path = img['path']

        for obj in img['objects']:
            name = obj['category']
            box = obj['bbox']
            xmin = int(box['xmin'])
            ymin = int(box['ymin'])
            xmax = int(box['xmax'])
            ymax = int(box['ymax'])
            filler.add_obj_box(name, xmin, ymin, xmax, ymax)

        # HACK : 将tt100k_parent_dir 直接设置为浏览到的root folder
        tt100k_parent_dir = f"{self.dataset_root_out}/"

        #
        work_sapce_dir = os.path.join(tt100k_parent_dir, "VOCdevkit/")

        if not os.path.isdir(work_sapce_dir):
            os.mkdir(work_sapce_dir)

        work_sapce_dir = os.path.join(work_sapce_dir, "VOC2007/")

        if not os.path.isdir(work_sapce_dir):
            os.mkdir(work_sapce_dir)

        jpeg_images_path = os.path.join(work_sapce_dir, 'JPEGImages')
        annotations_path = os.path.join(work_sapce_dir, 'Annotations')

        #
        if not os.path.isdir(jpeg_images_path):
            os.mkdir(jpeg_images_path)
        if not os.path.isdir(annotations_path):
            os.mkdir(annotations_path)

        # TT100K 数据集的目录位置
        # TT100K's input path
        datadir = f"{self.dataset_root_in}"
        filedir = datadir + "/annotations.json"

        ids = open(datadir + "/train/ids.txt").read().splitlines()
        annos = json.loads(open(filedir).read())

    def execute(self):
        for i, value in enumerate(ids):
            imgid = value
            filename = datadir + "/train/" + imgid + ".jpg"
            width, height, depth = find_image_size(filename)
            filler = xml_fill(filename, width, height, depth)
            # HACK: 改为了class function
            self.load_mask(annos, datadir, imgid, filler)
            filler.save_xml(annotations_path + '/' + imgid + '.xml')
            print("%s.xml saved\n" % imgid)
