#!/usr/bin/python3
"""
Evaluate validation set result for the AI City Challenge, Track 2, 2024.
"""
import json
from .utils import (
    convert_to_dict,
    compute_metrics_single,
    convert_desc_to_dict,
    convert_dict_to_desc_list
)
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from typing import List, Dict
import os

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

EPS = 1e-10


def calculate_tIoU(interval_pred, interval_gt):
    """Calculate the temporal Intersection over Union (tIoU) between two intervals."""
    start_pred, end_pred = interval_pred
    start_gt, end_gt = interval_gt
    
    intersection_start = max(start_pred, start_gt)
    intersection_end = min(end_pred, end_gt)
    intersection = max(0, intersection_end - intersection_start)
    
    union = (end_pred - start_pred) + (end_gt - start_gt) - intersection
    
    if union == 0:
        return 0
    else:
        return intersection / union

def compute_metrics_scenario(pred_scenario: list, gt_scenario: list, scenario_name: str, tiou_threshold):
    """Compute metrics for one scenario and return a dict"""
    pred_scenario_dict = convert_to_dict(pred_scenario)
    gt_scenario_dict = convert_to_dict(gt_scenario)


    metrics_ped_scenario_total = {
        "bleu":    0,
        "meteor":  0,
        "rouge-l": 0,
        "cider":   0,
    }
    metrics_veh_scenario_total = {
        "bleu":    0,
        "meteor":  0,
        "rouge-l": 0,
        "cider":   0,
    }
    num_segments = 0

    for segment, gt_segment_dict in gt_scenario_dict.items():
        if segment not in pred_scenario_dict:
            print(f"Segment captions missing for scenario {scenario_name}, segment number {segment}")
            # Skip adding score to this segment but still increment segment number since it is in GT
            num_segments += 1
            continue

        pred_segment_dict = pred_scenario_dict[segment]
        interval_pred = (pred_segment_dict['start_time'], pred_segment_dict["end_time"])
        interval_gt = (gt_segment_dict['start_time'], gt_segment_dict["end_time"])
        tiou = calculate_tIoU(interval_pred, interval_gt)

        # compute caption metrics for this segment
        metrics_ped_segment_total = compute_metrics_single(pred_segment_dict["caption_pedestrian"], gt_segment_dict["caption_pedestrian"], tiou, tiou_threshold)
        metrics_veh_segment_total = compute_metrics_single(pred_segment_dict["caption_vehicle"], gt_segment_dict["caption_vehicle"], tiou, tiou_threshold)

        # add segment metrics total to scenario metrics total
        for metric_name, metric_score in metrics_ped_segment_total.items():
            metrics_ped_scenario_total[metric_name] += metric_score
        for metric_name, metric_score in metrics_veh_segment_total.items():
            metrics_veh_scenario_total[metric_name] += metric_score

        # increment segment count
        num_segments += 1

    return metrics_ped_scenario_total, metrics_veh_scenario_total, num_segments


def compute_metrics_overall(pred_all, gt_all, tiou_threshold):
    """Compute metrics for all scenarios and return results for pedestrian, vehicle separately and number of averaged segments."""
    metrics_pedestrian_overall = {
        "bleu":    0,
        "meteor":  0,
        "rouge-l": 0,
        "cider":   0,
    }
    metrics_vehicle_overall = {
        "bleu":    0,
        "meteor":  0,
        "rouge-l": 0,
        "cider":   0,
    }
    num_segments_overall = 0
    with logging_redirect_tqdm(): 
        for scenario_name, gt_scenario in gt_all.items():
            if scenario_name not in pred_all:
                # print(f"Scenario {scenario_name} exists in ground-truth but not in predictions. "
                #     f"Counting zero score for this scenario.")
                continue

            pred_scenario = pred_all[scenario_name]

            # Get total scores for this scenario (for pedestrian and vehicle separately; for each separate metric)
            # and number of segments
            metrics_ped_scenario_total, metrics_veh_scenario_total, num_segments = compute_metrics_scenario(pred_scenario, gt_scenario, scenario_name, tiou_threshold)

            # Accumulate metric and num_segments for this scenario to overall sum
            for metric_name, metric_score in metrics_ped_scenario_total.items():
                metrics_pedestrian_overall[metric_name] += metric_score
            for metric_name, metric_score in metrics_veh_scenario_total.items():
                metrics_vehicle_overall[metric_name] += metric_score
            num_segments_overall += num_segments

    return metrics_pedestrian_overall, metrics_vehicle_overall, num_segments_overall


def compute_mean_metrics(metrics_overall, num_segments_overall):
    """Compute mean metrics from overall metrics and number of segments."""
    metrics_mean = metrics_overall
    for metric_name in metrics_overall.keys():
        metrics_mean[metric_name] /= (num_segments_overall + EPS)

    return metrics_mean


def batch_evaluate_scenario(pred_all_sentences, gt_all_sentences) -> Dict[str, float]:
    metrics_all_category_means = {}
    raw_metrics = [
        compute_metrics_single(
            pred, gt, 1, 0
        ) for pred, gt in zip(pred_all_sentences, gt_all_sentences)
    ]
    
    raw_metrics_dict = {}
    for k in raw_metrics[0].keys():
        raw_metrics_dict[f"{k.upper()}"] = [met[k] for met in raw_metrics]
    
    raw_total = 0
    for k, v in raw_metrics_dict.items():
        met = sum(v) / len(v)
        metrics_all_category_means[k] = met
        
        if k != "cider".upper():
            raw_total += met * 10
        else:
            raw_total += met * 100
            
    metrics_all_category_means["TOTAL"] = raw_total / 4
    return metrics_all_category_means


def compute_metrics_single_wrapper(pair):
    pred, gt = pair
    return compute_metrics_single(
        pred, gt, 1, 0
    )


def batch_evaluate_concurrent(pred_all_sentences, gt_all_sentences) -> Dict[str, float]:
    metrics_all_category_means = {}
        
    with ProcessPoolExecutor(max_workers=1) as exe:
        result = exe.map(compute_metrics_single_wrapper,
                         zip(pred_all_sentences, gt_all_sentences))
        raw_metrics = list(result)
    # pool = Pool(2)
    # raw_metrics = list(pool.map(
    #     compute_metrics_single_wrapper,
    #     zip(pred_all_sentences, gt_all_sentences)
    # ))
    
    raw_metrics_dict = {}
    for k in raw_metrics[0].keys():
        raw_metrics_dict[f"{k.upper()}"] = [met[k] for met in raw_metrics]
    
    raw_total = 0
    for k, v in raw_metrics_dict.items():
        met = sum(v) / len(v)
        metrics_all_category_means[k] = met
        
        if k != "cider".upper():
            raw_total += met * 10
        else:
            raw_total += met * 100
            
    metrics_all_category_means["TOTAL"] = raw_total / 4
    return metrics_all_category_means


def batch_evaluate(pred_all_sentences, gt_all_sentences, tiou_thresholds=[0.0, 0.3, 0.5, 0.7]) -> Dict[str, float]:
    """Evaluate the predictions and ground truth and return the mean score.
    Parameters:
        pred_all_sentences (List[str]): List of sentences containing multiple predicted descriptions for each scenario
        gt_all_sentences (List[str]): List of sentences containing multiple descriptions for each scenario
        tiou_thresholds (List[float]): List of float tIoU thresholds to use for evaluation. Languistic evaluation metrics are only calculated if tiou between pred and gt is larger than tiou_threshold
    Returns:
        metrics_all_category_means (Dict[float, Dict[str, float]]): Dict of mean scores for each category (pedestrian, vehicle) and each metric (bleu, meteor, rouge-l, cider) for 
    """
    pred_all = {f"id: {i}": convert_desc_to_dict(pred_all_sentences[i]) for i in range(len(pred_all_sentences))}
    gt_all = {f"id: {i}": convert_desc_to_dict(gt_all_sentences[i]) for i in range(len(gt_all_sentences))}
    
    # sanity check
    if isinstance(tiou_thresholds, float):
        tiou_thresholds = [tiou_thresholds]

    metrics_all_category_means = {}
    raw_metrics = [
        compute_metrics_single(
            pred, gt, 1, 0
        ) for pred, gt in zip(pred_all_sentences, gt_all_sentences)
    ]
    
    raw_metrics_dict = {}
    for k in raw_metrics[0].keys():
        raw_metrics_dict[f"RAW {k.upper()}"] = [met[k] for met in raw_metrics]
    
    raw_total = 0
    for k, v in raw_metrics_dict.items():
        met = sum(v) / len(v)
        metrics_all_category_means[k] = met
        
        if k != "raw cider".upper():
            raw_total += met * 10
        else:
            raw_total += met * 100
            
    metrics_all_category_means["RAW TOTAL"] = raw_total / 4
    
    for tiou_threshold in tiou_thresholds:
        tiou_repr = f"tIoU {tiou_threshold}"
        metrics_pedestrian_overall, metrics_vehicle_overall, num_segments_overall = compute_metrics_overall(pred_all, gt_all, tiou_threshold)
        if num_segments_overall == 0:
            return metrics_all_category_means
        metrics_pedestrian_mean = compute_mean_metrics(metrics_pedestrian_overall, num_segments_overall)
        metrics_vehicle_mean = compute_mean_metrics(metrics_vehicle_overall, num_segments_overall)

        metrics_all_category_mean = {}
        for metric_name, ped_score in metrics_pedestrian_mean.items():
            veh_score = metrics_vehicle_mean[metric_name]
            metrics_all_category_mean[metric_name] = (ped_score + veh_score) / 2
            metrics_all_category_means[f"{tiou_repr}/{metric_name.upper()}"] = \
                metrics_all_category_mean[metric_name]
            metrics_all_category_means[f"{tiou_repr}/pedestrian/{metric_name.upper()}"] = ped_score
            metrics_all_category_means[f"{tiou_repr}/vehicle/{metric_name.upper()}"] = veh_score

        total = 0
        for metric_name, score in metrics_all_category_mean.items():
            if metric_name in ["bleu", "meteor", "rouge-l"]:
                total += score * 100
            elif metric_name == "cider":
                total += score * 10
                
        metrics_all_category_means[f'{tiou_repr}/AVG'] = total / 4

    return metrics_all_category_means

# # Evaluate either internal or external dataset. If video_list is provided, only evaluate on this subset.
# def evaluate_one_dataset(predictions_file, ground_truth_dir_path, internal):
#     try:
#         # Read pred and gt to pred_all and gt_all, which will both look like:
#         # {
#         #     "<scenario-name-1>": [  # scenario name for multiple view or video file name for single view of BDD_TC_5K
#         #         {
#         #             "labels": [  # segment number, this is known information will be given
#         #                 "0"
#         #             ],
#         #             "caption_pedestrian": "",  # caption regarding pedestrian
#         #             "caption_vehicle": ""      # caption regarding vehicle
#         #         },
#         #         {
#         #             ...
#         #         }
#         #     ]
#         # },
#         # {
#         #     "<scenario-name-2>": [  # scenario name
#         #         {
#         #             ...
#         #         },
#         #     ]
#         # }
#         pred_all = read_pred(predictions_file)
#         gt_all = read_gt(ground_truth_dir_path)

#         # Only evaluate internal or external data at one time
#         pred_all = filter_internal_or_external_data(pred_all, internal)

#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")

#             # Compute overall metrics (summed over all scenarios and segments)
#             metrics_pedestrian_overall, metrics_vehicle_overall, num_segments_overall = compute_metrics_overall(pred_all, gt_all)
#             # Compute average metrics
#             metrics_pedestrian_mean = compute_mean_metrics(metrics_pedestrian_overall, num_segments_overall)
#             metrics_vehicle_mean = compute_mean_metrics(metrics_vehicle_overall, num_segments_overall)

#         # Compute average metrics over pedestrian and vehicle
#         metrics_all_category_mean = {}
#         for metric_name, ped_score in metrics_pedestrian_mean.items():
#             veh_score = metrics_vehicle_mean[metric_name]
#             metrics_all_category_mean[metric_name] = (ped_score + veh_score) / 2

#         total = 0
#         for metric_name, score in metrics_all_category_mean.items():
#             if metric_name in ["bleu", "meteor", "rouge-l"]:
#                 total += score * 100
#             elif metric_name == "cider":
#                 total += score * 10

#         mean_score = total / 4

        
#         print(f"=== Results for {'internal' if internal else 'external'} videos ===")
#         print(f"Pedestrian mean score over all data provided:")
#         print_metrics(metrics_pedestrian_mean)
#         print(f"Vehicle mean score over all data provided:")
#         print_metrics(metrics_vehicle_mean)
#         print(f"mean score (range [0, 100]): {mean_score:.2f}")
#         print("=="*20)

#     except Exception as e:
#         if mr:
#             print('{"error": "%s"}' % repr(e))
#         else:
#             print("Error: %s" % repr(e))
#         traceback.print_exc()
#         exit()

#     return metrics_all_category_mean, mean_score

# Read prediction json file that contains annotations of all scenarios. File format is specified in
# https://github.com/woven-visionai/wts-dataset/blob/main/README.md#evaluation.
# def read_pred(pred_json_path):
#     with open(pred_json_path) as f:
#         data = json.load(f)

#     return data


# # Read ground truth json file for one scenario
# def read_gt_one_scenario(gt_json_path):
#     with open(gt_json_path) as f:
#         data = json.load(f)

#     return data["event_phase"]


# # Read ground truth for all json files under gt_dir_path and return one dict containing all the annotation
# def read_gt(gt_dir_path):
#     gt_annotations = {}

#     # read json files from GT directory and store in a dict
#     for file_path in glob.iglob(gt_dir_path + '/**/**.json', recursive=True):
#         # skip vehicle view annotations since their captions are the same as overhead view
#         if "vehicle_view" in file_path:
#             continue

#         # get scenario name from file path
#         file_name = file_path.split("/")[-1]
#         scenario_name = file_name.strip("_caption.json")

#         # read annotation of this scenario
#         gt_annotation = read_gt_one_scenario(file_path)
#         gt_annotations[scenario_name] = gt_annotation

#     return gt_annotations

# def print_metrics(metrics_dict):
#     for metric_name, metric_val in metrics_dict.items():
#         print(f"- {metric_name}: {metric_val:.3f}")

# # Filter internal or external data.
# # If internal is True, keep internal data.
# # If internal is False, keep external data.
# def filter_internal_or_external_data(data, internal):
#     filtered_data = {}
#     for key, value in data.items():
#         if (internal and key.startswith("2023")) or (not internal and key.startswith("video")):
#             filtered_data[key] = value

#     return filtered_data


# def filter_data_with_video_list(data, video_list):
#     filtered_data = {}
#     for key, value in data.items():
#         if key in video_list:
#             filtered_data[key] = value
#     return filtered_data


# if __name__ == '__main__':
#     gt = json.load(open("/home/_minh/pipeline_v0/benchmark/gt.json"))
#     pred = json.load(open("/home/_minh/pipeline_v0/benchmark/pred.json"))
#     gt_sent = convert_dict_to_desc_list(gt)
#     pred_sent = convert_dict_to_desc_list(pred)

#     open("gt_sent.txt", "w").write("\n".join(gt_sent))
#     open("pred_sent.txt", "w").write("\n".join(pred_sent))

#     print(batch_evaluate.__doc__)
#     print("___________________________________")
#     metric_all_category_means = batch_evaluate(gt_sent, pred_sent)
#     print()
#     print(metric_all_category_means)
    


