import json
import re

import nltk
from nltk import bleu_score # type: ignore
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from .cider.cider import Cider
from nltk.tokenize import TreebankWordTokenizer


id_text_pattern = f"<time=(\d+)> <time=(\d+)> (.+?)(?=<time=|\Z)" # type: ignore


try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpus/wordnet')
except:
    nltk.download('wordnet', quiet=True)


def convert_dict_to_desc(dict):
    desc = ""
    for segment in dict:
        desc += f"<time: {int(float(segment['start_time']))}> <time: {int(float(segment['end_time']))}> pedestrian: {segment['caption_pedestrian']} vehicle: {segment['caption_vehicle']} "

    return desc.strip()


def tokenize_sentence(sentence):
    tokenizer = TreebankWordTokenizer()
    words = tokenizer.tokenize(sentence)
    if len(words) == 0:
        return ""
    return " ".join(words)


# Compute BLEU-4 score on a single sentence
def compute_bleu_single(tokenized_hypothesis, tokenized_reference):
    # convert tokenized sentence (joined by spaces) into list of words
    tokenized_hypothesis = tokenized_hypothesis.split(" ")
    tokenized_reference = tokenized_reference.split(" ")

    return sentence_bleu([tokenized_reference], tokenized_hypothesis,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=bleu_score.SmoothingFunction().method3)


# Compute METEOR score on a single sentence
def compute_meteor_single(tokenized_hypothesis, tokenized_reference):
    # convert tokenized sentence (joined by spaces) into list of words
    tokenized_hypothesis = tokenized_hypothesis.split(" ")
    tokenized_reference = tokenized_reference.split(" ")

    return meteor_score([tokenized_reference], tokenized_hypothesis)


# Compute ROUGE-L score on a single sentence
def compute_rouge_l_single(sentence_hypothesis, sentence_reference):
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = rouge_l_scorer.score(sentence_hypothesis, sentence_reference)
    rouge_l_score = rouge_score['rougeL']
    return rouge_l_score.fmeasure


# Compute CIDEr score on a single sentence
def compute_cider_single(sentence_hypothesis, sentence_reference):
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score([sentence_reference], [sentence_hypothesis])

    return cider_score


# Compute metrics based for a single caption.
def compute_metrics_single(pred, gt, tiou, tiou_threshold):
    if tiou >= tiou_threshold: 
        tokenized_pred = tokenize_sentence(pred)
        tokenized_gt = tokenize_sentence(gt)

        bleu_score = compute_bleu_single(tokenized_pred, tokenized_gt)
        meteor_score = compute_meteor_single(tokenized_pred, tokenized_gt)
        rouge_l_score = compute_rouge_l_single(pred, gt)
        cider_score = compute_cider_single([tokenized_pred], [tokenized_gt])
    else: 
        bleu_score = 0
        meteor_score = 0
        rouge_l_score = 0
        cider_score = 0

    return {
        "bleu": bleu_score,
        "meteor": meteor_score,
        "rouge-l": rouge_l_score,
        "cider": cider_score,
    }


# Convert a list containing annotation of all segments of a scenario to a dict keyed by segment label.
#   - Example input (segment_list):
#         [
#             {
#                 "labels": [
#                     "0"
#                 ],
#                 "caption_pedestrian": "",
#                 "caption_vehicle": ""
#             },
#             {
#                 ...
#             }
#         ]
#   - Example output (segment_dict):
#         {
#             "0": {
#                 "caption_pedestrian": "",
#                 "caption_vehicle": ""
#             },
#             ...
#         }
def convert_to_dict(segment_list):
    segment_dict = {}
    for segment in segment_list:
        segment_number = segment["labels"][0]

        segment_dict[segment_number] = {
            "caption_pedestrian": segment["caption_pedestrian"],
            "caption_vehicle": segment["caption_vehicle"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
        }

    return segment_dict


def single_parse(desc: str):
    segments = re.findall(id_text_pattern, desc)
    segments_list = [segment[2] for segment in segments]
    return segments_list


def batch_parse(desc_list):
    return [single_parse(desc) for desc in desc_list]


def convert_desc_to_dict(desc: str):
    """Convert description to iterable dict
    Example input: <time: 1> <time: 5> pedestrian: meowmeow vehicle: gogoo <time: 10> <time: 12> pedestrian: argarg vehicle: hehe <time: 15> <time: 25> pedestrian: hihi vehicle: sdsdsdsd
    Example output: [
        {
            "labels": [
                "3"
            ],
            "caption_pedestrian": "hihi",
            "caption_vehicle": "sdsdsdsd",
            "start_time: "15",
            "end_time": "25",
        },
        ...
    ]
    """
    # split into segments
    segments_list = []
    # TODO: pedestrian -> <pedestrian>, vehicle -> <vehicle>
    pattern = r"<time: (\d+)> <time: (\d+)> pedestrian: (.+?) vehicle: (.+?)(?=<time:|\Z)"
    segments = re.findall(pattern, desc)
    for i in range(len(segments)):
        segments_list.append({
            "labels": [str(i)],
            "caption_pedestrian": segments[i][2],
            "caption_vehicle": segments[i][3],
            "start_time": float(segments[i][0]),
            "end_time": float(segments[i][1]),
        })
    return segments_list

def convert_dict_to_desc_list(data):
    output = []
    for sub in data.values():
        output.append(convert_dict_to_desc(sub))
    return output

if __name__ == '__main__':
    gt = json.load(open("gt.json"))
    sentences = []
    for sub_gt in gt.values():
        sentences.append(convert_dict_to_desc(sub_gt))
    print(sentences[0])
    print(convert_desc_to_dict(sentences[0]))