import os
import tensorflow as tf
from PIL import Image


IMAGE_FEATURE_DESCRIPTION = {
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
}


def write_cropped_image(parsed, image_count):
    class_label = parsed['image/object/class/text'].values.numpy()[0].decode('utf-8')
    tf_image = tf.image.decode_jpeg(parsed['image/encoded'])
    image = Image.fromarray(tf_image.numpy())

    x1 = parsed['image/object/bbox/xmin'].values[0].numpy() * image.width
    x2 = parsed['image/object/bbox/xmax'].values[0].numpy() * image.width
    y1 = parsed['image/object/bbox/ymin'].values[0].numpy() * image.height
    y2 = parsed['image/object/bbox/ymax'].values[0].numpy() * image.height

    cropped = image.crop((x1, y1, x2, y2))
    write_dir_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cropped_images')
    write_dir = os.path.join(write_dir_folder, class_label)
    os.makedirs(write_dir, exist_ok=True)
    cropped.save(os.path.join(write_dir, '{}.jpg'.format(image_count)))


def main():
    image_count = 0
    # Scan dataset folders of each class under 'input/'
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input')
    class_folders = [os.path.abspath(f.path) for f in os.scandir(input_dir) if f.is_dir()]
    class_folders.sort()
    if len(class_folders) <= 0:
        return

    # Walk through folders for all classes
    for class_id, class_dir in enumerate(class_folders):
        record_folders = [os.path.abspath(f.path) for f in os.scandir(class_dir) if f.is_dir()]

        # Walk through different dataset folder within each class
        for record_dir in record_folders:
            print(class_id + 1, record_dir)
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

                # The class label inside default.tfrecord should be correct
                write_cropped_image(parsed, image_count)
                image_count += 1


if __name__ == '__main__':
    main()
