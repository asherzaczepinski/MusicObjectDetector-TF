# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import os
import tensorflow as tf
from lxml import etree
import PIL.Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from tqdm import tqdm

flags = tf.compat.v1.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'training', 'Convert training set, validation set or '
                                      'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                         'difficult instances')
FLAGS = flags.FLAGS

SETS = ['training', 'validation', 'test']

def dict_to_tf_example(data, dataset_directory, set_name, label_map_dict, ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.
    Notice that this function normalizes the bounding box coordinates provided by the raw data.
    Args:
      data: dict holding PASCAL XML fields for a single image (obtained by running dataset_util.recursive_parse_xml_to_dict)
      dataset_directory: Path to root directory holding PASCAL dataset
      set_name: name of the set training, validation or test
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the dataset  (default: False).
    Returns:
      example: The converted tf.Example.
    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    img_path = os.path.join(set_name, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_image = fid.read()
    encoded_image_io = io.BytesIO(encoded_image)
    image = PIL.Image.open(encoded_image_io)
    if image.format != 'JPEG' and image.format != 'PNG':
        raise ValueError('Image format not JPEG or PNG')
    key = hashlib.sha256(encoded_image).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    if 'object' not in data:
        data['object'] = []
    for obj in data['object']:
        if obj['name'] in label_map_dict:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image.format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example

def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    os.makedirs(os.path.dirname(FLAGS.output_path), exist_ok=True)

    writer = tf.io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    # Set data_dir to the 'data' directory
    data_dir = FLAGS.data_dir  # This should be 'data'

    # Adjust examples_path to correctly point to the 'training.txt' file
    examples_path = os.path.join(data_dir, FLAGS.set + '.txt')

    # Set annotations_dir to point to the 'Full_Page_Annotations' directory
    annotations_dir = os.path.join(data_dir, 'Full_Page_Annotations')

    examples_list = dataset_util.read_examples_list(examples_path)

    with tqdm(enumerate(examples_list), desc="Parsing annotations...", total=len(examples_list)) as pbar:
        for idx, example in pbar:
            path = os.path.join(annotations_dir, example + '.xml')  # Forms the correct path to each XML file
            print("Trying to open XML file:", path)  # Debug print
            with tf.io.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, FLAGS.data_dir, FLAGS.set, label_map_dict, FLAGS.ignore_difficult_instances)
            writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    tf.compat.v1.app.run()
