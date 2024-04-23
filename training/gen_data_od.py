import os
import tensorflow as tf

from object_detection.protos import string_int_label_map_pb2
from object_detection.utils import dataset_util
from google.protobuf import text_format

SINGLE_CLASS_ID = 1
SINGLE_CLASS_TEXT = 'default'
# # SINGLE_CLASS_TEXT = 'bolt'
TFRECORD_NAME = 'combined.tfrecord'
LABEL_MAP_NAME = 'combined_label_map.pbtxt'
IMAGE_FEATURE_DESCRIPTION = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
}


def create_cat_tf_example(parsed, class_text, class_num):
    height = parsed['image/height'].numpy()
    width = parsed['image/width'].numpy()
    filename = parsed['image/filename'].numpy()
    source_id = parsed['image/source_id'].numpy()
    encoded_image_data = parsed['image/encoded'].numpy()
    image_format = parsed['image/format'].numpy()
    xmins = [value.numpy() for value in parsed['image/object/bbox/xmin'].values]
    xmaxs = [value.numpy() for value in parsed['image/object/bbox/xmax'].values]
    ymins = [value.numpy() for value in parsed['image/object/bbox/ymin'].values]
    ymaxs = [value.numpy() for value in parsed['image/object/bbox/ymax'].values]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(source_id),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            [class_text.encode('utf8')]),
        'image/object/class/label': dataset_util.int64_list_feature(
            [class_num]),
    }))
    return tf_example


def main():
    # Scan dataset folders of each class under 'input/'
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
    class_folders = [os.path.abspath(f.path) for f in os.scandir(input_dir) if f.is_dir()]
    class_folders.sort()
    if len(class_folders) <= 0:
        return

    # Specify output directory and files
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'combined')
    os.makedirs(out_dir, exist_ok=True)
    out_label_map = os.path.join(out_dir, LABEL_MAP_NAME)
    out_tfrecord = os.path.join(out_dir, TFRECORD_NAME)

    # Write combined label map
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    item = string_int_label_map_pb2.StringIntLabelMapItem()
    # All data should be merged into one single class
    item.id = SINGLE_CLASS_ID
    item.name = SINGLE_CLASS_TEXT
    label_map.item.append(item)

    ########################################
    # item.id = 1
    # item.name = 'bolt'
    # label_map.item.append(item)
    # item = string_int_label_map_pb2.StringIntLabelMapItem()
    # item.id = 2
    # item.name = 'default'
    # label_map.item.append(item)
    ######################################

    with open(out_label_map, 'w') as f:
        f.write(text_format.MessageToString(label_map))

    # Write combined tf record
    writer = tf.io.TFRecordWriter(out_tfrecord)
    # Walk through folders for all classes
    for class_dir in class_folders:

        ######################################
        # if os.path.basename(class_dir) == 'bolt':
        #     CLASS_ID = 1
        #     CLASS_TEXT = 'bolt'
        # else:
        #     CLASS_ID = 2
        #     CLASS_TEXT = 'default'
        ######################################

        record_folders = [os.path.abspath(f.path) for f in os.scandir(class_dir) if f.is_dir()]

        # Walk through different dataset folder within each class
        for record_dir in record_folders:
            print(SINGLE_CLASS_ID, record_dir)

            ######################################
            # print(CLASS_ID, record_dir)
            ######################################

            full_dataset_path = os.path.join(record_dir, 'default.tfrecord')
            dataset = tf.data.TFRecordDataset(full_dataset_path)

            # Iterate through all images within each dataset
            it = iter(dataset)
            for value in it:
                parsed = tf.io.parse_single_example(value, IMAGE_FEATURE_DESCRIPTION)
                num_values = len(parsed['image/object/class/text'].values)
                if num_values == 0:
                    continue
                if num_values != 1:
                    raise Exception

                # All data should be merged into one single class
                tf_example = create_cat_tf_example(parsed, SINGLE_CLASS_TEXT, SINGLE_CLASS_ID)

                ######################################
                # tf_example = create_cat_tf_example(parsed, CLASS_TEXT, CLASS_ID)
                ######################################

                writer.write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    main()
