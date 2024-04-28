import os

from PIL import Image
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from loguru import logger
import cv2 as cv
import yaml
import numpy as np
import torch
import clip
import fire

from bbox import get_square_box


def load_configs(config_file_path):
    """Load configurations from the config file.
    Args:
        config_file_path: path to the config file.
    Returns:
        configs: configurations loaded from the file
    """
    with open(config_file_path, 'r', encoding='UTF-8') as file:
        configs = yaml.safe_load(file)
    return configs


def setup_model(clip_model_dir):
    """Setup CLIP model.
    Args:
        clip_model_dir: path to the CLIP model directory.
    Returns:
        model: CLIP model.
        preprocess: preprocess function for the model
    """
    model, preprocess = clip.load("ViT-L/14", download_root=clip_model_dir)
    model.eval()
    model.cuda()
    return model, preprocess


def ensure_directory_exists(path):
    """Ensure the directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)


def process_frame(frame, box_coordinates, preprocess, model, output_path):
    """Process each frame and extract features.
    Args:
        frame: input frame.
        box_coordinates: bounding box coordinates.
        preprocess: preprocess function for the model.
        model: CLIP model.
        output_path: path to save the output.
    Returns:
        None
    """
    x1, y1, x2, y2 = box_coordinates
    frame = frame[y1:y2, x1:x2]
    frame = cv.cvtColor(cv.resize(frame, (224, 224)), cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    image = preprocess(image).unsqueeze(0).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image).cpu().numpy()
    
    if output_path is not None:  # Save the output for local feature extraction
        image_features = np.expand_dims(image_features, axis=0)  # Add batch dimension
        np.save(output_path, image_features)
    else:  # Return the output for concatenation
        return image_features

def extract_features(configs, is_external):
    """Extract features for the given configuration.
    Args:
        configs: configurations.
        is_external: whether the dataset is external or internal.
    Returns:
        None
    """
    video_path = configs['paths']['input_path']
    anno_path = configs['paths']['anno_path']
    output_path = configs['paths']['output_path']
    feature_type = configs['feature_type']

    logger.info(f"Video path: {video_path}")
    logger.info(f"Annotation path: {anno_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Feature type: {feature_type}")

    model, preprocess = setup_model(configs['paths']['CLIP_PATH'])
    ensure_directory_exists(output_path)

    file_list = os.listdir(video_path)

    # check video_path exist
    assert os.path.exists(video_path), f"Video path {video_path} does not exist."
    # check anno_path exist
    assert os.path.exists(anno_path), f"Annotation path {anno_path} does not exist." 
    # check feature_type is valid
    assert feature_type in ['global', 'sub_global', 'local'], f"Invalid feature type {feature_type}."

    with logging_redirect_tqdm():
        if is_external:
            for filename in tqdm(file_list, desc='Processing videos in external dataset'):
                logger.info(f"Processing video: {filename}")
                extract_features_external(filename, video_path, anno_path, output_path, feature_type, preprocess, model)
        else:
            # print("Internal dataset")  # Debug statement
            for view_name in tqdm(file_list, desc='Processing views in internal dataset'):
                # print(f"View: {view_name}")  # Debug statement
                logger.info(f"Processing view: {view_name}")
                extract_features_internal(view_name, video_path, anno_path, output_path, feature_type, preprocess, model)


def extract_features_external(filename, video_path, anno_path, output_path, feature_type, preprocess, model):
    """Extract features for external dataset.
    Args:
        filename: name of the video file.
        video_path: path to the video files.
        anno_path: path to the annotation files.
        output_path: path to save the output.
        feature_type: type of feature extraction, 'global', 'sub_global', 'local'.
        preprocess: preprocess function for the model.
        model: CLIP model.
    Returns:
        None
    """
    filename = filename[:filename.rfind('.')]  # e.g., vid.mp4 -> vid
    video_full_path = os.path.join(video_path, filename + '.mp4')
    bbox_full_path = os.path.join(anno_path, filename + '_bbox.json')
    bbox_to_cut = get_square_box(video_full_path, bbox_full_path, feature_type)


    # print("Video full path: ", video_full_path)  # Debug statement
    # print("Bbox full path: ", bbox_full_path)  # Debug statement
    # print("Bbox to cut: ", bbox_to_cut)  # Debug statement

    if feature_type == 'local':
        ensure_directory_exists(os.path.join(output_path, filename))
    else:
        imfeat = []

    frame_count = 0
    cap = cv.VideoCapture(video_full_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if feature_type == 'local':
            # Process each frame and save the output
            if frame_count in bbox_to_cut:
                output_frame_path = os.path.join(output_path, filename, f'frame{frame_count}.npy')
                process_frame(frame, bbox_to_cut[frame_count], preprocess, model, output_frame_path)

            # Skip the frame if not in the list
            else:
                pass

        elif feature_type in ['global', 'sub_global']:
            x1, y1, x2, y2 = bbox_to_cut[-1]
            process_frame(frame, (x1, y1, x2, y2), preprocess, model, imfeat)
        frame_count += 1
    cap.release()

    # Save the output for global and sub_global feature extraction
    if feature_type != 'local':
        imfeat = np.concatenate(imfeat, axis=0)
        np.save(os.path.join(output_path, filename + '.npy'), imfeat)


# Extract features for internal
def extract_features_internal(view_name, video_path, anno_path, output_path, feature_type, preprocess, model):
    """Extract features for internal dataset.
    Args:
        view_name: name of the view.
        video_path: path to the video files.
        anno_path: path to the annotation files.
        output_path: path to save the output.
        feature_type: type of feature extraction, 'global', 'sub_global', 'local'.
        preprocess: preprocess function for the model.
        model: CLIP model.
    Returns:
        None
    """
    view_path = os.path.join(video_path, view_name)
    ensure_directory_exists(os.path.join(output_path, view_name))

    views = ['overhead_view', 'vehicle_view']
    for view in views:
        extract_features_view(view, view_name, view_path, anno_path, output_path, feature_type, preprocess, model)


# Extract features per view
def extract_features_view(view, view_name, view_path, anno_path, output_path, feature_type, preprocess, model):
    """Extract features for each view.
    Args:
        view: name of the view.
        view_name: name of the view.
        view_path: path to the view files.
        anno_path: path to the annotation files.
        output_path: path to save the output.
        feature_type: type of feature extraction, 'global', 'sub_global', 'local'.
        preprocess: preprocess function for the model.
        model: CLIP model.
    Returns:
        None
    """
    view_files = os.listdir(os.path.join(view_path, view))
    for file in view_files:
        file = file[:file.rfind('.')]
        video_full_path = os.path.join(view_path, view, file + '.mp4')
        bbox_full_path = os.path.join(anno_path, view_name, view, file + '_bbox.json')
        bbox_to_cut = get_square_box(video_full_path, bbox_full_path, feature_type)
        # print(bbox_to_cut)  # Debug statement

        if feature_type == 'local':
            ensure_directory_exists(os.path.join(output_path, view_name, view, file))
        else:
            imfeat = []

        frame_count = 0
        cap = cv.VideoCapture(video_full_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if feature_type == 'local':
                # Process each frame and save the output
                if frame_count in bbox_to_cut:
                    output_frame_path = os.path.join(output_path, view_name, view, file, f'frame{frame_count}.npy')
                    process_frame(frame, bbox_to_cut[frame_count], preprocess, model, output_frame_path)
                
                # Skip the frame if not in the list
                else:
                    pass

            elif feature_type in ['global', 'sub_global']:
                x1, y1, x2, y2 = bbox_to_cut[-1]
                process_frame(frame, (x1, y1, x2, y2), preprocess, model, imfeat)
            frame_count += 1
        cap.release()

        if feature_type != 'local':
            imfeat = np.concatenate(imfeat, axis=0)
            np.save(os.path.join(output_path, view_name, view, file + '.npy'), imfeat)


@logger.catch
def main(config_path, is_external):
    """
    CLI interface to extract features from videos using CLIP model.
    Args:
        config_path (str): Path to the configuration YAML file.
        is_external (bool): Whether the dataset is external or not.
    Returns:
        None
    """
    try:
        configs = load_configs(config_path)
        extract_features(configs, is_external)
        print("Feature extraction completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Using Fire to handle the CLI
if __name__ == "__main__":
    fire.Fire(main)
