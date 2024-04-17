from yacs.config import CfgNode as CN


_C = CN()


_C.GLOB = CN()
_C.GLOB.EXP_PARENT_DIR = "/home/logs"
_C.GLOB.SEED = 42
_C.GLOB.DEVICE = "cuda"


# Dataset configurations
_C.DATA = CN()

# Max 1 from 0-1, others must be -1
_C.DATA.TRAIN_DATASETS = [
  { "name": "wts_train_main", "ratio": -1 },
  { "name": "wts_train_external", "ratio": -1 },
  { "name": "wts_val_external", "ratio": -1 },
]
_C.DATA.VAL_DATASETS = [
  { "name": "wts_val_main", "ratio": -1 },
  # { "name": "wts_val_external", "ratio": -1 },
]
_C.DATA.TEST_DATASETS = [
  { "name": "wts_test_main", "ratio": -1 },
  { "name": "wts_test_normal", "ratio": -1},
  { "name": "wts_test_external", "ratio": -1},
]

_C.DATA.MAX_FEATS = 100
_C.DATA.FEATURES_DIM = 768
_C.DATA.MAX_OUTPUT_TOKENS = 256
_C.DATA.MAX_PAD_TIME = 5 #s
_C.DATA.TRAIN_RANDOM_PAD_TIME = True
_C.DATA.NUM_BINS = 100

_C.DATA.FPS = 1
_C.DATA.EXTRACT_FPS = 30

_C.DATA.OVERHEAD_RATIO = 0.5

_C.DATA.MAIN_FEATURE = "sub_global"
_C.DATA.SUB_FEATURE = None

_C.DATA.AUGMENT = CN()
_C.DATA.AUGMENT.NLP_AUGS = [] # ["backtrans"]
_C.DATA.AUGMENT.DEVICE = "cuda"

_C.DATA.AUGMENT.NLP_PROB = 0.2
_C.DATA.AUGMENT.NLP_PER_PHASE_PROB = 0.5


_C.MODEL = CN()
_C.MODEL.DEC_DROP = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.T5_PATH = "t5-base"
_C.MODEL.EMBED_DIM = 768
_C.MODEL.DEPTH = 12
_C.MODEL.NUM_HEADS = 12
_C.MODEL.MLP_DIM = 2048
_C.MODEL.VIS_DROP = 0.1

_C.MODEL.FEATURE_BRANCHES = ["mix"] # ["vehicle", "overhead", "mix"]
_C.MODEL.VEHICLE_PROJ = False
_C.MODEL.OVERHEAD_PROJ = False
_C.MODEL.FEAT_MLP_DIM = 768
_C.MODEL.TARGET_EMBED_SIZE = 2

_C.MODEL.USE_LOCAL = False
_C.MODEL.MAX_PHASES = 6
_C.MODEL.ENCODE_LOCAL_TEMPORAL = False

_C.MODEL.VID2SEQ_PATH = "/home/pretrained/vid2seq_htmchaptersvitt.pth"
_C.MODEL.LOAD_VID2SEQ_CKPT = True


_C.SOLVER = CN()
_C.SOLVER.LOAD_FROM_EPOCH = -1
_C.SOLVER.LOAD_FROM_PATH = None

_C.SOLVER.FAULT_TOLERANCE = 5
_C.SOLVER.LOG_TO_WANDB = True

_C.SOLVER.TRAIN = CN()
_C.SOLVER.TRAIN.MAX_EPOCH = 100
_C.SOLVER.TRAIN.BATCH_SIZE = 25

_C.SOLVER.TRAIN.LOG_UPDATES = 5
_C.SOLVER.TRAIN.CHECKPOINT_METRICS = ['wts_val_main_vehicle/RAW TOTAL']
_C.SOLVER.TRAIN.SAVE_INTERVAL = 5

_C.SOLVER.TRAIN.CLIP_MAX_NORM = 1.

_C.SOLVER.TRAIN.DENOISING = False
_C.SOLVER.TRAIN.PHASE_NOISE_DENSITY = 0.5

_C.SOLVER.TRAIN.PEDESTRIAN_FACTOR = 1.0

_C.SOLVER.TRAIN.OPTIMIZER = CN()
_C.SOLVER.TRAIN.OPTIMIZER.FRACTION_WARMUP_STEPS = 0.1
_C.SOLVER.TRAIN.OPTIMIZER.SCHEDULE = ""
_C.SOLVER.TRAIN.OPTIMIZER.LR = 3e-4
_C.SOLVER.TRAIN.OPTIMIZER.BETA1 = 0.9
_C.SOLVER.TRAIN.OPTIMIZER.BETA2 = 0.999
_C.SOLVER.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.

_C.SOLVER.VAL = CN()
_C.SOLVER.VAL.VAL_INTERVAL = 5
_C.SOLVER.VAL.BATCH_SIZE = None

_C.SOLVER.VAL.NUM_BEAMS = 2
_C.SOLVER.VAL.TOP_P = 0.9
_C.SOLVER.VAL.REPETITION_PENALTY = 1.
_C.SOLVER.VAL.LENGTH_PENALTY = 1.
_C.SOLVER.VAL.TEMPERATURE = 0.

_C.SOLVER.TEST = CN()
_C.SOLVER.TEST.NUM_BEAMS = 4
_C.SOLVER.TEST.TOP_P = 0.9
_C.SOLVER.TEST.REPETITION_PENALTY = 1.
_C.SOLVER.TEST.LENGTH_PENALTY = 1.
_C.SOLVER.TEST.TEMPERATURE = 1.


_C.ENSEMBLE = CN()
_C.ENSEMBLE.EXPERIMENT_LIST = []
_C.ENSEMBLE.ROOT_EXP = None


def get_cfg_defaults():
  return _C.clone()


def convert_to_dict(cfg_node, key_list=[]):
  if not isinstance(cfg_node, CN):
    return cfg_node
  else:
    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
      cfg_dict[k] = convert_to_dict(v, key_list + [k])
    return cfg_dict
  
def get_sig(cfg_dict):
  import json
  from hashlib import sha1
  
  # Return signature from a jsonable content.
  _repr = json.dumps(cfg_dict, sort_keys=True).encode('utf8')
  return sha1(_repr).hexdigest()[:8]
