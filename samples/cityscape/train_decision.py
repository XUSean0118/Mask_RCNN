import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from models import Decision
import mrcnn.model as modellib
from mrcnn import utils


def main():
    MODEL_DIR = os.path.join(ROOT_DIR, "logs/decision")
    data_dir = "./data/"

    decision = Decision(dropout=0.5)
    decision.train(data_dir, MODEL_DIR, epochs=200)

        
if __name__ == '__main__':
    main()