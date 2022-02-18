# Music Object Detector

This is the repository for the fast and reliable Music Symbol detector with Deep Learning, based on the Tensorflow Object Detection API:

![Animation of inference results on a full page of sheet music](MusicObjectDetector/images/full-page-detection_animated.gif)

If you want to try out the full-page detection on your own images, you can try it online in the [DIVAServices Spotlight](http://divaservices.unifr.ch/spotlight).

The scientific reasoning can be found in [this scientific article](http://www.mdpi.com/2076-3417/8/9/1488). The detailed results for various combinations of object-detector, feature-extractor, etc. can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/14i3Y64YgcdA1eMQ8-SkGvwhCYLKvhhZ4B7DqTjMFpkg).

## Running an Inference Demo

The fastest way to try out this model is by running the docker demo, which can be found in the [demo directory](MusicObjectDetector/demo).
To run it, read the [demo readme](MusicObjectDetector/demo/README.md) file.

## Music object detection and image crops

If you are interested in previous work, presented at the [DAS 2018](https://das2018.cvl.tuwien.ac.at/en/) on cropped images like these, please refer to the [corresponding release](https://github.com/apacha/MusicObjectDetector-TF/releases/tag/crop-images-detection)

| Original Image | Detection results as training progresses |
|:--------------:|:------------------:|
| ![Original image](MusicObjectDetector/images/crop_undetected.png) | ![Image with detected objects](MusicObjectDetector/images/individualImage1-animation.gif) |

The scientific reasoning can be found in [this scientific article](https://alexanderpacha.files.wordpress.com/2018/05/handwritten-music-object-detection.pdf).
The detailed results for various combinations of object-detector, feature-extractor, etc. can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/174-CnLO-rAoVMst0ngVGHguTlD39ebdxLX9ZLE9Pscw).

## Preparing the application

This repository contains several scripts that can be used independently of each other.
Before running them, make sure that you have the necessary requirements installed.

### Install required libraries

- Python 3.6 / 3.7
- Tensorflow 1.13.1 (or optionally tensorflow-gpu 1.13.1)
- pycocotools (more [infos](https://github.com/matterport/Mask_RCNN/issues/6#issuecomment-341503509))
  - On Mac/Linux, run `pip install pycocotools`
  - On Windows, run `pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI`
- Some additional libraries, as specified in [requirements.txt](MusicObjectDetector/requirements.txt)

### Build Protobuf files on Linux

```bash
cd research
protoc object_detection/protos/*.proto --python_out=.
```

### Build Protobuf files on Windows

> Run [`DownloadAndBuildProtocolBuffers.ps1`](MusicObjectDetector/DownloadAndBuildProtocolBuffers.ps1) to automate this step or manually build the protobufs by first installing [protocol buffers](https://developers.google.com/protocol-buffers/docs/downloads) and then run:

```bash
cd research
protoc object_detection/protos/*.proto --python_out=.
```

Note, that you have to use [version 3.4.0](https://github.com/google/protobuf/releases/download/v3.4.0/) because of a [bug in 3.5.0 and 3.5.1](https://github.com/google/protobuf/issues/3957)

## Dataset

Run [`PrepareMuscimaFullPageDatasetForTensorflow.ps1`](MusicObjectDetector/PrepareMuscimaFullPageDatasetForTensorflow.ps1) or [`PrepareMuscimaStavewiseDatasetForTensorflow.ps1`](MusicObjectDetector/PrepareMuscimaStavewiseDatasetForTensorflow.ps1) to automate this step on Windows or manually prepare the datasets by running:

```bash
# cd into MusicObjectDetector folder
python download_muscima_dataset.py

# For full-page training:
python prepare_muscima_stavewise_annotations.py
# For stavewise training:
python prepare_muscima_full_page_annotations.py

# Split image directory according to predefined train/test split
python dataset_splitter.py --source_directory=data/muscima_pp_cropped_images_with_stafflines --destination_directory=data/training_validation_test
```
  
These scripts will download the datasets automatically, prepare the annotations and split the images into three reproducible parts for training, validation and test.

Now you can create the Tensorflow Records that are required for actually running the training.

```bash
# Full-page
python create_muscima_tf_record.py --data_dir=data/training_validation_test --set=training --annotations_dir=Full_Page_Annotations --output_path=data/all_classes_writer_independent_split/training.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test --set=validation --annotations_dir=Full_Page_Annotations --output_path=data/all_classes_writer_independent_split/validation.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test --set=test --annotations_dir=Full_Page_Annotations --output_path=data/all_classes_writer_independent_split/test.record --label_map_path=mapping_all_classes.txt

# Stavewise
python create_muscima_tf_record.py --data_dir=data/training_validation_test --set=training --annotations_dir=Stavewise_Annotations --output_path=data/stave_crop_writer_independent_split/training.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test --set=validation --annotations_dir=Stavewise_Annotations --output_path=data/stave_crop_writer_independent_split/validation.record --label_map_path=mapping_all_classes.txt
python create_muscima_tf_record.py --data_dir=data/training_validation_test --set=test --annotations_dir=Stavewise_Annotations --output_path=data/stave_crop_writer_independent_split/test.record --label_map_path=mapping_all_classes.txt
```

 By providing a different mapping, you can reduce the classes, you want to be able to detect, e.g. `mapping_71_classes.txt`:

## Training the network

### Adding source to Python path

There are two ways of making sure, that the python script discoveres the correct binaries:

### Permanently linking the source code as pip package

To permanently link the source-code of the project, for Python to be able to find it, you can link the two packages by running:

```bash
# From tensorflow/models/research/
pip install -e .
cd slim
# From inside tensorflow/models/research/slim
pip install -e .
```

### Temporarily adding the source code before starting the training

Make sure you have all required folders appended to the [Python path](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath). This can temporarily be done inside a shell, before calling any training scrips by the following commands:

For Linux:

```bash
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

For Windows (Powershell):

```powershell
$pathToGitRoot = "[GIT_ROOT]"
$pathToSourceRoot = "$($pathToGitRoot)/object_detection"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot);$($pathToGitRoot)/slim"
```

### Adjusting paths

For running the training, you need to change the paths, according to your system

- in the configuration, you want to run, e.g. `configurations/faster_rcnn_inception_resnet_v2_atrous_muscima_pretrained_reduced_classes.config`
- if you use them, in the PowerShell scripts in the `training_scripts` folder.

Run the actual training script, by using the pre-defined Powershell scripts in the `training_scripts` folder, or by directly calling

```bash
# Start the training
python [GIT_ROOT]/research/object_detection/train.py --logtostderr --pipeline_config_path="[GIT_ROOT]/MusicObjectDetector/configurations/[SELECTED_CONFIG].config" --train_dir="[GIT_ROOT]/MusicObjectDetector/data/checkpoints-[SELECTED_CONFIG]-train"

# Start the validation
python [GIT_ROOT]/research/object_detection/eval.py --logtostderr --pipeline_config_path="[GIT_ROOT]/MusicObjectDetector/configurations/[SELECTED_CONFIG].config" --checkpoint_dir="[GIT_ROOT]/MusicObjectDetector/data/checkpoints-[SELECTED_CONFIG]-train" --eval_dir="[GIT_ROOT]/MusicObjectDetector/data/checkpoints-[SELECTED_CONFIG]-validate"
```

A few remarks: The two scripts can and should be run at the same time, to get a live evaluation during the training. The values, may be visualized by calling `tensorboard --logdir=[GIT_ROOT]/MusicObjectDetector/data`.

You might have to download pre-trained models from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).

### Restricting GPU memory usage

Notice that usually Tensorflow allocates the entire memory of your graphics card for the training. In order to run both training and validation at the same time, you might have to restrict Tensorflow from doing so, by opening `train.py` and `eval.py` and uncomment the respective (prepared) lines in the main function. E.g.:

```python
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

### Training with pre-trained weights

It is recommended that you use pre-trained weights for known networks to speed up training and improve overall results. To do so, head over to the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), download and unzip the respective trained model, e.g. `faster_rcnn_inception_resnet_v2_atrous_coco` for reproducing the best results, we obtained. The path to the unzipped files, must be specified inside of the configuration in the `train_config`-section, e.g.

```yaml
train-config: {
  fine_tune_checkpoint: "C:/Users/Alex/Repositories/MusicObjectDetector-TF/MusicObjectDetector/data/faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08/model.ckpt"
  from_detection_checkpoint: true
}
```

> Note that inside that folder, there is no actual file, called `model.ckpt`, but multiple files called `model.ckpt.[something]`.

### Dimension clustering

For optimizing the performance of the detector, we adopted the dimensions clustering algorithm, proposed in the [YOLO 9000 paper](https://arxiv.org/abs/1612.08242).
While preparing the dataset, the `muscima_image_cutter.py` script created a file called `Annotations.csv` and a folder called `Annotations`.
Both will contain the same annotations, but in different formats. While the csv-file contains all annotations in a plain list, the Annotations
folder contains one xml-file per image, complying with the format used for the [Pascal VOC project](http://host.robots.ox.ac.uk/pascal/VOC/).

To perform dimension clustering on the cropped images, run the following scripts:

```bash
python generate_muscima_statistics.py
python muscima_dimension_clustering.py
```

The first script will load all annotations and create four csv-files containing the dimensions for each annotation from all images, including their relative sizes, compared to the entire image.
The second script loads those statistics and performs dimension clustering, use a k-means algorithm on the relative dimensions of annotations.

## Inference

### Standalone inference

We recommend to check out the [demo](MusicObjectDetector/demo) folder first, which provides a self-contained docker image for performing object detection and does not depend on this library. It comes with a pre-trained model for convenience and a simple csv output for interoperability with other applications.

### Inference from within this library

If you have trained a model by yourself, [this document](research/object_detection/g3doc/exporting_models.md) describes how to prepare it. Basically, you just run `export_inference_graph.py` with appropriate arguments or `freeze_model.ps1` after setting the paths accordingly. Alternatively, pre-trained models can be download from the [release section](https://github.com/apacha/MusicObjectDetector-TF/releases).

Once you have the frozen model, you can perform inference on a single image by running

```bash
# From [GIT_ROOT]/MusicObjectDetection
python inference_over_image.py \
    --inference_graph ${frozen_inference_graph.pb} \
    --label_map mapping.txt \
    --input_image ${IMAGE_TO_BE_CLASSIFIED} \
    --output_image image_with_detection.jpg
```

or for an entire directory of images by running

```bash
# From [GIT_ROOT]/MusicObjectDetection
python inference_over_directory.py \
    --inference_graph ${frozen_inference_graph.pb} \
    --label_map mapping.txt \
    --input_directory ${DIRECTORY_TO_IMAGES} \
    --output_directory ${OUTPUT_DIRECTORY}
```

## License

Published under MIT License,

Copyright (c) 2019 [Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha) and Kwon-Young Choi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
