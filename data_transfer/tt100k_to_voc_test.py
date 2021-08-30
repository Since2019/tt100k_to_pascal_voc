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


def load_mask(annos, datadir, imgid, filler):
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


class ConvertTestLabels:

    def __init__(self, input_dataset_dir, output_dataset_dir):
        # TT100K 原始数据集的根目录
        self.input_dataset_dir = input_dataset_dir
        # 转换后输出PASCAL VOC的目录
        self.output_dataset_dir = output_dataset_dir

    def execute(self):
        #
        tt100k_parent_dir = self.input_dataset_dir
        #sys.path.append(tt100k_parent_dir +"data_transfer")

        # work_space_dir 构建完毕后将是输出结果的根目录
        work_sapce_dir = os.path.join(tt100k_parent_dir, "VOCdevkit/")

        if not os.path.isdir(work_sapce_dir):
            os.mkdir(work_sapce_dir)

        work_sapce_dir = os.path.join(work_sapce_dir, "VOC2007/")

        if not os.path.isdir(work_sapce_dir):
            os.mkdir(work_sapce_dir)

        # 将会有一个JPEGImages文件夹专门存放所有的图片(分不同的imagesets : train test val)
        jpeg_images_path = os.path.join(work_sapce_dir, 'JPEGImages')
        # 存放 XML 格式的 Annotation, 包含图片中学习范围的位置和class
        annotations_path = os.path.join(work_sapce_dir, 'Annotations')

        if not os.path.isdir(jpeg_images_path):
            os.mkdir(jpeg_images_path)
        if not os.path.isdir(annotations_path):
            os.mkdir(annotations_path)

        # TT100K 的位置
        # 重复的变量名主要是为了和原来的脚本尽量保持一致
        datadir = self.input_dataset_dir

        filedir = datadir + "/annotations.json"
        ids = open(datadir + "/test/ids.txt").read().splitlines()
        annos = json.loads(open(filedir).read())

        for i, value in enumerate(ids):
            imgid = value
            filename = datadir + "/test/" + imgid + ".jpg"
            width, height, depth = find_image_size(filename)
            filler = xml_fill(filename, width, height, depth)
            load_mask(annos, datadir, imgid, filler)
            filler.save_xml(annotations_path + '/' + imgid + '.xml')
            print("%s.xml saved\n" % imgid)
