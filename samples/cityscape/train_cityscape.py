import os
import sys
import argparse
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from cityscape import CityscapeConfig, CityscapeDataset
from mrcnn import utils
import mrcnn.model as modellib

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train cityscape")
    parser.add_argument("--flowmodel", type=str, default='flownets',
                        choices=['flownets', 'flownetS'],
                        help="chose flow model")
    parser.add_argument("--resnetmodel", type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help="chose resnet model")
    parser.add_argument("--flow", action="store_true",
                        help="Use flow or not.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="number of gpus.")
    parser.add_argument("--num_images", type=int, default=2,
                        help="number of images per gpu.")
    return parser.parse_args()

def main():
    args = get_arguments()
    print(args)
    
    config = CityscapeConfig()
    config.GPU_COUNT = args.num_gpus
    config.IMAGES_PER_GPU = args.num_images
    config.BATCH_SIZE = args.num_images * args.num_gpus     
    config.BACKBONE = args.resnetmodel
    if args.flow:
        config.IMAGE_SHAPE = [1024, 1024, 6]
        config.FLOW = args.flowmodel

    data_dir = '/data/cityscapes_dataset/cityscape'
    # Training dataset
    dataset_train = CityscapeDataset()
    if args.flow:
        dataset_train.load_cityscape(data_dir, "train", 11)
    else:
        dataset_train.load_cityscape(data_dir, "train", 1)
    dataset_train.prepare()
    # Validation dataset
    dataset_val = CityscapeDataset()
    if args.flow:
        dataset_val.load_cityscape(data_dir, "val", 11)
    else:
        dataset_val.load_cityscape(data_dir, "val", 1)
    dataset_val.prepare()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
   
    if not args.flow:
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=5, 
                    layers='heads')
        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also 
        # pass a regular expression to select which layers to
        # train by name pattern.
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=20, 
                    layers='all')
    else:
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
        # Fine tune flow
        # Passing layers="finetune" trains flow layers. You can also 
        # pass a regular expression to select which layers to
        # train by name pattern.
        model.loadflow_weights("./flow/flownet-s.h5", by_name=True)
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE/10,
                    epochs=30, 
                    layers="finetune")

        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE/100,
                    epochs=40, 
                    layers="finetune")
        
if __name__ == '__main__':
    main()
