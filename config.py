from yacs.config import CfgNode as CN


_C = CN()


_C.GLOB = CN()
_C.GLOB.EXP_PARENT_DIR = "/home/logs"


# Dataset configurations
_C.DATA = CN()

# Max 1 from 0-1, others must be -1
_C.DATA.TRAIN_DATASETS = [
  { "name": "wts_train_main", "ratio": -1 },
  { "name": "wts_train_external", "ratio": -1 },
]
_C.DATA.VAL_DATASETS = [
  { "name": "wts_val_main", "ratio": -1 },
  { "name": "wts_val_external", "ratio": -1 },
]
_C.DATA.TEST_DATASETS = [
  { "name": "wts_test_main", "ratio": -1 },
  { "name": "wts_test_external", "ratio": -1},
]

_C.DATA.MAX_FEATS = 100
_C.DATA.FEATURES_DIM = 768
_C.DATA.MAX_OUTPUT_TOKENS = 128
_C.DATA.MAX_PAD_TIME = 10 #s
_C.DATA.TRAIN_RANDOM_PAD_TIME = True
_C.DATA.NUM_BINS = 100

_C.DATA.FPS = 1
_C.DATA.EXTRACT_FPS = 30


_C.MODEL = CN()
_C.MODEL.DEC_DROP = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.T5_PATH = "t5-base"
_C.MODEL.EMBED_DIM = 768
_C.MODEL.DEPTH = 12
_C.MODEL.NUM_HEADS = 12
_C.MODEL.MLP_DIM = 2048
_C.MODEL.VIS_DROP = 0.1

_C.MODEL.FEATURE_BRANCHES = ["vehicle"] # ["vehicle", "overhead"]
_C.MODEL.VEHICLE_PROJ = False
_C.MODEL.OVERHEAD_PROJ = False
_C.MODEL.FEAT_MLP_DIM = 768

_C.MODEL.VID2SEQ_PATH = None # ADD CHECKPOINT HERE
_C.MODEL.LOAD_VID2SEQ_CKPT = True


_C.SOLVER = CN()
_C.SOLVER.LOAD_FROM_EPOCH = -1

_C.SOLVER.TRAIN = CN()
_C.SOLVER.TRAIN.MAX_EPOCH = 100
_C.SOLVER.TRAIN.LOG_UPDATES = 5
_C.SOLVER.TRAIN.CHECKPOINT_METRICS = [] # ROGUE, ...
_C.SOLVER.TRAIN.SAVE_INTERVAL = 5

_C.SOLVER.TRAIN.OPTIMIZER = CN()
_C.SOLVER.TRAIN.OPTIMIZER.FRACTION_WARMUP_STEPS = 0.1
_C.SOLVER.TRAIN.OPTIMIZER.SCHEDULE = ""
_C.SOLVER.TRAIN.OPTIMIZER.LR = 3e-4

_C.SOLVER.TRAIN.CLIP_MAX_NORM = 1.


def get_cfg_defaults():
  return _C.clone()


def convert_to_dict(cfg_node, key_list):
  if not isinstance(cfg_node, CN):
    return cfg_node
  else:
    cfg_dict = dict(cfg_node)
    for k, v in cfg_dict.items():
      cfg_dict[k] = convert_to_dict(v, key_list + [k])
    return cfg_dict
  
def _get_sig(cfg_dict):
  import json
  from hashlib import sha1
  
  # Return signature from a jsonable content.
  _repr = json.dumps(cfg_dict, sort_keys=True).encode('utf8')
  return sha1(_repr).hexdigest()[:8]
