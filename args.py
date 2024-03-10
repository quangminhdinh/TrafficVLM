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
