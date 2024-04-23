## Prepare Image Dataset

First, for each step, download TFRecord 1.0 dataset from cvat, put the
.tfrecord file and the .pbtxt file in a folder with the same name as the
class label (i.e. the step number). Then put these folders under the `input/`
directory.

Example usage of the scripts:
```bash
# from outside of the tf-raskog/ folder
python3 tf-raskog/gen_data_od.py
python3 tf-raskog/crop_videos_from_labels.py
python3 tf-raskog/remove_dup.py
python3 tf-raskog/split_dataset.py tf-raskog/cropped_images \
    fast-MPN-COV/data/step0-6/train \
    fast-MPN-COV/data/step0-6/val \
    fast-MPN-COV/data/step0-6/test \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --shuffle
```

## Train Object Detector

[Install](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md#docker-installation)
the Object Detection API using Docker.

[Start](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md#local)
training

## Train Image Classifier

[Clone](https://github.com/rogerthat94/fast-MPN-COV) the repository and follow the instructions in README.

Set the environment variable for model path:
```bash
export R50=path/to/pretrained/model
```

## Test

Test Image Classifier
```bash
python3 tf-raskog/test_model.py
```

Start Server (OD + Classifier)
```bash
python3 tf-raskog/server/fine_grained.py
```
