import argparse


def get_args_parser():
  parser = argparse.ArgumentParser(add_help=False)
  
  # experiment yaml file name
  parser.add_argument(
    "experiment",
    type=str,
    help="experiment name",
  )
  
  return parser


def get_test_args_parser():
  parser = argparse.ArgumentParser(add_help=False)
  
  # experiment yaml file name
  parser.add_argument(
    "experiment",
    type=str,
    help="experiment name",
  )
  
  parser.add_argument(
    "-d", "--device",
    type=str, default="cuda",
    help="device to be used (i.e. cuda:0 for single gpu)",
  )
  
  parser.add_argument(
    "-b", "--batch",
    type=int, default=8, #15,
    help="batch size to be used for testing",
  )
  
  return parser
