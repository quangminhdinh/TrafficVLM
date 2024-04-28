"""Bounding box extraction for video frames."""
from loguru import logger
import json
import cv2 as cv

def get_square_box(video, bbox_file, feature_type='global'):
    """Get the square bounding box for the video.
    Args:
        video: path to video file.
        bbox_file: path to annotation file.
        feature_type: type of feature extraction, 'global', 'sub-global', 'local'.
    Returns:
        bbox_to_cut: dictionary containing frame_ids and bboxes.
    """
    # Open video capture
    try:
        cap = cv.VideoCapture(video)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # Returned dictionary for the bbox - (x1,y1),(x2,y2)
        bbox_to_cut = {}
    except FileNotFoundError as e:
        logger.error(f"Error opening video file: {e}")
        raise FileNotFoundError
    except Exception as e:
        logger.error(f"Error opening video file: {e}")
        raise Exception

    try:
        # Load JSON data
        json_data = json.load(open(bbox_file))['annotations']
        
        # Initialize bounding box coordinates
        x1, y1, x2, y2 = 1e6, 1e6, 0, 0

        
        if feature_type in ['global', 'sub_global']:
            # Calculate the bounding box for global or sub_global
            for bbox_data in json_data:
                x, y, w, h = bbox_data['bbox']
                x1 = int(min(x1, x))
                y1 = int(min(y1, y))
                x2 = int(max(x2, x+w))
                y2 = int(max(y2, y+h))
                
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        
        if feature_type == 'global':
            # Calculate square dimension and adjust the bounding box
            square_dim = min(width, height)
            # Define square crop
            # Calculate x1, x2
            if center_x-square_dim//2 < 0:  # If the center is too close to the left edge 
                x1 = 0
                x2 = square_dim
            elif center_x+square_dim//2 > width:  # If the center is too close to the right edge 
                x1 = width - square_dim
                x2 = width
            else:  # If the center is within the frame
                x1, x2 = center_x-square_dim//2, center_x+square_dim//2
            
            if center_y-square_dim//2 < 0:  # If the center is too close to the top edge
                y1 = 0
                y2 = square_dim
            elif center_y+square_dim//2 > height:  # If the center is too close to the bottom edge
                y1 = height - square_dim
                y2 = height
            else:  # If the center is within the frame
                y1, y2 = center_y-square_dim//2, center_y+square_dim//2
            bbox_to_cut[-1] = (x1, y1, x2, y2)
        
        elif feature_type == 'sub_global':
            # Use the maximum dimension of the bounding box
            square_dim = max(x2 - x1, y2 - y1)
            x1, x2 = max(0, center_x - square_dim // 2), min(width, center_x + square_dim // 2)
            y1, y2 = max(0, center_y - square_dim // 2), min(height, center_y + square_dim // 2)
            bbox_to_cut[-1] = (x1, y1, x2, y2)
        
        elif feature_type == 'local':
            # Local feature type, process each bbox individually
            for bbox_data in json_data:
                x, y, w, h = bbox_data['bbox']
                x1 = int(min(x1, x))
                y1 = int(min(y1, y))
                x2 = int(max(x2, x+w))
                y2 = int(max(y2, y+h))
                frame_id = int(bbox_data['image_id'])

                square_dim = max(x2 - x1, y2 - y1)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Define square crop
                final_x1, final_x2 = max(0, center_x - square_dim // 2), min(width, center_x + square_dim // 2)
                final_y1, final_y2 = max(0, center_y - square_dim // 2), min(height, center_y + square_dim // 2)
                bbox_to_cut[frame_id] = (final_x1, final_y1, final_x2, final_y2)
        
        return bbox_to_cut
    
    except Exception as e:
        logger.error(f"Error processing video or JSON data: {e}")
        # In case of error, return default square center crop
        square_dim = min(width, height)
        bbox_to_cut[-1] = (
                            (width - square_dim) // 2,
                            (height - square_dim) // 2,
                            (width + square_dim) // 2,
                            (height + square_dim) // 2
                        )
        return bbox_to_cut