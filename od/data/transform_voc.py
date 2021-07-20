import xml.etree.ElementTree as ET
import os
import shutil

global year
year = '2012'


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = max((box[0] + box[1]) / 2.0 - 1, 0)
    y = max((box[2] + box[3]) / 2.0 - 1, 0)
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(data_dir, image_id, train, classes):
    in_file = open(os.path.join(data_dir, 'VOC' + year + '/Annotations/%s.xml' % (image_id)), encoding='utf-8')
    if train:
        out_file = open(os.path.join(data_dir, 'labels/train/%s.txt' % (image_id)), 'w', encoding='utf-8')
    else:
        out_file = open(os.path.join(data_dir, 'labels/val/%s.txt' % (image_id)), 'w', encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes:
          continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def transform_voc(data_dir, classes, c_year=None):
    if c_year is not None:
        year = c_year

    if not os.path.exists(os.path.join(data_dir, 'images')):
        os.makedirs(os.path.join(data_dir, 'images/train'))
        os.makedirs(os.path.join(data_dir, 'images/val'))
    if not os.path.exists(os.path.join(data_dir, 'labels')):
        os.makedirs(os.path.join(data_dir, 'labels/train'))
        os.makedirs(os.path.join(data_dir, 'labels/val'))

    # make train labels
    train_image_ids = open(os.path.join(data_dir, 'VOC' + year + '/ImageSets/Main/train.txt'), encoding='utf-8')
    for image_id in train_image_ids:
        image_id = image_id.strip()
        convert_annotation(data_dir, image_id, True, classes)
        img_path = os.path.join(data_dir, 'VOC' + year + "/JPEGImages", image_id + '.jpg')
        shutil.copy(img_path, os.path.join(data_dir, 'images/train/'))

    # make val labels
    val_image_ids = open(os.path.join(data_dir, 'VOC' + year + '/ImageSets/Main/val.txt'), encoding='utf-8')
    for image_id in val_image_ids:
        image_id = image_id.strip()
        convert_annotation(data_dir, image_id, False, classes)
        img_path = os.path.join(data_dir, 'VOC' + year + "/JPEGImages", image_id + '.jpg')
        shutil.copy(img_path, os.path.join(data_dir, 'images/val/'))

    return os.path.join(data_dir, 'images/train'), os.path.join(data_dir, 'images/val')


if __name__ == '__main__':
    data_dir = ''
    classes = ['person']
    transform_voc(data_dir, classes, year)
