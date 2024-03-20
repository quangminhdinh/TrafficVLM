import os


ROOT_PATH = "/home"
FEATURE_PATH = os.path.join(ROOT_PATH, "features")
ANNOTATION_PATH = os.path.join(ROOT_PATH, "data")
SELECTED_BBOX = "bbox_annotated"


wts_train_main = {
  "name": "wts_train_main",
  "features": os.path.join(FEATURE_PATH, "wts_dataset_zip/train"),
  "captions": os.path.join(ANNOTATION_PATH, "wts_dataset_zip/annotations/caption/train"),
  "bbox_pedestrian": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/annotations/{SELECTED_BBOX}/pedestrian/train"),
  "bbox_vehicle": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/annotations/{SELECTED_BBOX}/vehicle/train"),
  "local_annotated": os.path.join(FEATURE_PATH, "local/wts_dataset_zip/new_bbox/train")
}

wts_val_main = {
  "name": "wts_val_main",
  "features": os.path.join(FEATURE_PATH, "wts_dataset_zip/val"),
  "captions": os.path.join(ANNOTATION_PATH, "wts_dataset_zip/annotations/caption/val"),
  "bbox_pedestrian": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/annotations/{SELECTED_BBOX}/pedestrian/val"),
  "bbox_vehicle": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/annotations/{SELECTED_BBOX}/vehicle/val"),
  "local_annotated": os.path.join(FEATURE_PATH, "local/wts_dataset_zip/new_bbox/val")
}

wts_train_external = {
  "name": "wts_train_external",
  "features": os.path.join(FEATURE_PATH, "external/BDD_PC_5K/train"),
  "captions": os.path.join(ANNOTATION_PATH, "wts_dataset_zip/external/annotations/caption/train"),
  "bbox_pedestrian": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/external/BDD_PC_5K/annotations/bbox_annotated/train"),
  "bbox_vehicle": None,
  "local_annotated": os.path.join(FEATURE_PATH, "local/external/BDD_PC_5K/train")
}

wts_val_external = {
  "name": "wts_val_external",
  "features": os.path.join(FEATURE_PATH, "external/BDD_PC_5K/val"),
  "captions": os.path.join(ANNOTATION_PATH, "wts_dataset_zip/external/annotations/caption/val"),
  "bbox_pedestrian": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/external/BDD_PC_5K/annotations/bbox_annotated/val"),
  "bbox_vehicle": None,
  "local_annotated": os.path.join(FEATURE_PATH, "local/external/BDD_PC_5K/val")
}

wts_test_main = {
  "name": "wts_test_main",
  "features": os.path.join(FEATURE_PATH, "wts_dataset_test/new_bbox/test"),
  "captions": os.path.join(ANNOTATION_PATH, "wts_dataset_test/annotations/caption/test/public_challenge"),
  "bbox_pedestrian": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/WTS_DATASET_PUBLIC_TEST_BBOX/annotations/{SELECTED_BBOX}/pedestrian/test/public"),
  "bbox_vehicle": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/WTS_DATASET_PUBLIC_TEST_BBOX/annotations/{SELECTED_BBOX}/vehicle/test/public"),
  "local_annotated": os.path.join(FEATURE_PATH, "local/wts_dataset_test/new_bbox/test")
}

wts_test_normal = {
  "name": "wts_test_normal",
  "features": os.path.join(FEATURE_PATH, "wts_dataset_test/new_bbox/test/normal_trimmed"),
  "captions": os.path.join(ANNOTATION_PATH, "wts_dataset_test/annotations/caption/test/public_challenge/normal_trimmed"),
  "bbox_pedestrian": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/WTS_DATASET_PUBLIC_TEST_BBOX/annotations/{SELECTED_BBOX}/pedestrian/test/public"),
  "bbox_vehicle": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/WTS_DATASET_PUBLIC_TEST_BBOX/annotations/{SELECTED_BBOX}/vehicle/test/public"),
  "local_annotated": os.path.join(FEATURE_PATH, "local/wts_dataset_test/new_bbox/test/normal_trimmed")
}

wts_test_external = {
  "name": "wts_test_external",
  "features": os.path.join(FEATURE_PATH, "external/BDD_PC_5K/new_bbox/test"),
  "captions": os.path.join(ANNOTATION_PATH, "wts_dataset_test/external/annotations/caption/test/public_challenge"),
  "bbox_pedestrian": os.path.join(ANNOTATION_PATH, f"wts_dataset_zip/WTS_DATASET_PUBLIC_TEST_BBOX/external/BDD_TC_5K/annotations/{SELECTED_BBOX}/test/public"),
  "bbox_vehicle": None,
  "local_annotated": os.path.join(FEATURE_PATH, "local/external/BDD_PC_5K/new_bbox/test")
}


AVAILABLE_DATASETS = [
  wts_train_main,
  wts_val_main,
  wts_train_external,
  wts_val_external,
  wts_test_main,
  wts_test_normal,
  wts_test_external
]


def get_dataset_paths(*args):
  ds_num = len(args)
  assert ds_num > 0

  ds_configs = [cfg for cfg in AVAILABLE_DATASETS if cfg["name"] in args]
  assert len(ds_configs) == len(args)
  return ds_configs
