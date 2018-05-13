import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from cityscape import CityscapeConfig
from models import Decision
import mrcnn.model as modellib
from mrcnn import utils

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train decision network")
    parser.add_argument("--flowmodel", type=str, default='flownets',
                        choices=['flownets', 'flownetS'],
                        help="chose flow model")
    return parser.parse_args()

def main():
    args = get_arguments()
    print(args)
    
    MODEL_DIR = os.path.join(ROOT_DIR, "logs/decision")
    data_dir = "./data/"

    config = CityscapeConfig()
    config.FLOW = args.flowmodel
    decision = Decision(config, dropout=0.5)
    decision.train(data_dir, MODEL_DIR, epochs=200)

        
if __name__ == '__main__':
    main()