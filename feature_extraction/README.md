# Feature Extraction for WTS dataset
This folder contains the code for extracting visual features of video frames in the WTS dataset using the [OpenAI Clip](https://github.com/openai/CLIP).

## 1. List of available configurations for the WTS Dataset

| TYPE 	| Input path 	| Anno path 	| Output path 	| Feature type 	|
|---	|---	|---	|---	|---	|
| train 	| /home/data/wts_dataset_zip/videos/train 	| /home/data/wts_dataset_zip/annotations_new/bbox_annotated/pedestrian/train 	| /home/features/<TYPE>/wts_dataset_zip/train 	| global/sub_global/local 	|
| train 	| /home/data/wts_dataset_zip/videos/train/normal_trimmed 	| /home/data/wts_dataset_zip/annotations_new/bbox_annotated/pedestrian/train/normal_trimmed 	| /home/features/<TYPE>/wts_dataset_zip/train/normal_trimmed 	| global/sub_global/local 	|
| train 	| /home/data/wts_dataset_zip/external/BDD_PC_5K/videos/train 	| /home/data/wts_dataset_zip/external/BDD_PC_5K/annotations_new/bbox_annotated/train 	| /home/features/<TYPE>/external/BDD_PC_5K/train 	| global/sub_global/local 	|
| valid 	| /home/data/wts_dataset_zip/videos/val 	| /home/data/wts_dataset_zip/annotations_new/bbox_annotated/pedestrian/val 	| /home/features/<TYPE>/wts_dataset_zip/val 	| global/sub_global/local 	|
| valid 	| /home/data/wts_dataset_zip/videos/val/normal_trimmed 	| /home/data/wts_dataset_zip/annotations_new/bbox_annotated/pedestrian/val/normal_trimmed 	| /home/features/<TYPE>/wts_dataset_zip/val/normal_trimmed 	| global/sub_global/local 	|
| valid 	| /home/data/wts_dataset_zip/external/BDD_PC_5K/videos/val 	| /home/data/wts_dataset_zip/external/BDD_PC_5K/annotations_new/bbox_annotated/val 	| /home/features/<TYPE>/external/BDD_PC_5K/val 	| global/sub_global/local 	|
| test 	| /home/data/wts_dataset_test/WTS_DATASET_PUBLIC_TEST/videos/test/public 	| /home/data/wts_dataset_test/WTS_DATASET_PUBLIC_TEST_BBOX/annotations/bbox_annotated/pedestrian/test/public 	| /home/features/<TYPE>/wts_dataset_test/new_bbox/test 	| global/sub_global/local 	|
| test 	| /home/data/wts_dataset_test/WTS_DATASET_PUBLIC_TEST/videos/test/public/normal_trimmed 	| /home/data/wts_dataset_test/WTS_DATASET_PUBLIC_TEST_BBOX/annotations/bbox_annotated/pedestrian/test/public/normal_trimmed 	| /home/features/<TYPE>/wts_dataset_test/new_bbox/test/normal_trimmed 	| global/sub_global/local 	|
| test 	| /home/data/wts_dataset_test/WTS_DATASET_PUBLIC_TEST/external/BDD_PC_5K/videos/test/public 	| /home/data/wts_dataset_test/WTS_DATASET_PUBLIC_TEST_BBOX/external/BDD_TC_5K/annotations/bbox_annotated/test/public 	| /home/features/<TYPE>/external/BDD_PC_5K/new_bbox/test 	| global/sub_global/local 	|

## 2. Steps to extract features

> Step 1: Modify the [configuration](config.yaml) file

Destination folders (output paths) are as follows:

```Markdown
    |-- global
    |   |-- external
    |   |   `-- BDD_PC_5K
    |   |       |-- new_bbox/test
    |   |       |-- train
    |   |       `-- val
    |   |-- wts_dataset_test
    |   |   `-- new_bbox/test
    |   |                `-- normal_trimmed
    |   `-- wts_dataset_zip
    |       |-- train
    |       |   `-- normal_trimmed
    |       `-- val
    |           `-- normal_trimmed
    |-- local
    `-- semi_global
```

The content of 'local' and 'sub_global' are the same as 'global'

Here's an example of [config.yaml](config.yaml) with the variables are subjected to change according to the [TABLE](https://github.com/quangminhdinh/TrafficVLM/edit/main/feature_extraction/readme.md#1-list-of-available-configurations-for-the-wts-dataset) above.

```yaml
paths:
  input_path:  "/home/data/wts_dataset_zip/videos/train" 
  anno_path:   "/home/data/wts_dataset_zip/annotations_new/bbox_annotated/pedestrian/train" 
  output_path: "/home/features/global/wts_dataset_zip/train" 
  CLIP_PATH:   "/home/CLIP" 

feature_type:  "local" 
```

> Step 2: Run [wts_extractor.py](wts_extractor.py)

```zsh
python wts_extractor.py --config_path ./config.yaml --is_external False  # specify if the data is not external
```

## General pipeline

- Get the bounding boxes from the annotation file & video dimension
- Loop through each frame of the video to crop out the frame
- Resize the cropped frame to 224x224 in RGB colorspace
- Feed that frame through CLIP-ViT-L/14 and append to the array that represents the encoded features of the whole video

## Example of levels of view

![Local and Sub-global](example.png)
