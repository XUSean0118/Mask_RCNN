import os
import sys
import time
import argparse
import skimage
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from cityscape import CityscapeConfig, CityscapeDataset
from models import ResNet, FlowNet, MaskRCNN, Warp, Decision
from mrcnn import utils, visualize
import mrcnn.model as modellib

DATA_DIRECTORY = '/data/cityscapes_dataset/cityscape_video/'
DATA_LIST_PATH = '/data/cityscapes_dataset/cityscape_video/list/video_list0.txt'
RESTORE_FROM = ""
DECISION_FROM = ""
SAVE_DIR = './video0/'
NUM_STEPS = 599 # Number of images in the video.
TARGET = 70.0

label_colors = [(0, 0, 142), (0, 0, 69), (219, 19, 60)
                # 0 = car, 1 = truck, 2 = person
                ,(119, 10, 32), (255, 0, 0), (0, 60, 100)
                # 3 = bicycle, 4 = rider, 5 = bus
                ,(0, 0, 230), (0, 79, 100)]
                # 6 = motocycle, 7 = train
class_names = ['BG', 'car', 'truck', 'person', 'bicycle', 'rider', 'bus', 'motorcycle', 'train']

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--flowmodel", type=str, default='flownets',
                        choices=['flownets', 'flownetS'],
                        help="chose flow model")
    parser.add_argument("--resnetmodel", type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help="chose resnet model")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--data_list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--decision_from", type=str, default=DECISION_FROM,
                        help="Where restore decision model parameters from.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of images in the video.")
    parser.add_argument("--target", type=float, default=TARGET,
                        help="confidence score threshold.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save segmented output.")
    parser.add_argument("--is_save", action="store_true",
                        help="whether to save output")
    parser.add_argument("--dynamic", action="store_true",
                        help="Whether to dynamically adjust target")
    return parser.parse_args()

    
def mold_inputs(config, images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matricies [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matricies:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    """
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
        # Resize image
        # TODO: move resizing to mold_image()
        molded_image, window, scale, padding, crop = utils.resize_image(
            image,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
        molded_image = molded_image.astype(np.float32) - config.MEAN_PIXEL
        # Build image_meta
        image_meta = modellib.compose_image_meta(
            0, image.shape, molded_image.shape, window, scale,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))
        # Append
        molded_images.append(molded_image)
        windows.append(window)
        image_metas.append(image_meta)
    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows
    
def main():
    args = get_arguments()
    print(args)
    
    config = CityscapeConfig()
    config.FLOW = args.flowmodel
    config.BACKBONE = args.resnetmodel
    config.IMAGE_SHAPE = [1024, 1024, 6]
    config.POST_NMS_ROIS_INFERENCE = 500
    #config.display()
    
    resnet = ResNet(config=config)
    flownet = FlowNet(config=config)
    maskrcnn = MaskRCNN(config=config)
    warp = Warp(config=config)
    decision = Decision(config=config)

    model_path = args.restore_from
    resnet.load_weights(model_path, by_name=True)
    flownet.load_weights(model_path, by_name=True)
    maskrcnn.load_weights(model_path, by_name=True)
    decision.load_weights(args.decision_from, by_name=True)

    seg_step = 0
    flow_step = 0
    target = args.target
    list_file = open(args.data_list, 'r')
    if args.is_save and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for step in range(args.num_steps):
        f1 = list_file.readline().split('\n')[0]
        f1 = os.path.join(args.data_dir, f1)
        current = np.expand_dims(skimage.io.imread(f1), 0)
        current, image_metas, window = mold_inputs(config, current)
        image = current[0] + config.MEAN_PIXEL
        if step == 0:
            seg_step += 1
            key_P2, key_P3, key_P4, key_P5, key_P6 = resnet.keras_model.predict(current)
            key = current
            P2, P3, P4, P5, P6 = key_P2, key_P3, key_P4, key_P5, key_P6
        else:
            images = np.concatenate([current, key], 3)
            flow, flow_feature = flownet.keras_model.predict(images)
            score = decision.keras_model.predict(flow_feature)[0][0]
            print("step: {:4d} predict score: {:.3f} target: {:.2f}".format(step, score, target))
            
            if score < target:
                if args.dynamic and target < 50:
                    target -= 0.5
                seg_step += 1
                key_P2, key_P3, key_P4, key_P5, key_P6 = resnet.keras_model.predict(current)
                key = current
                P2, P3, P4, P5, P6 = key_P2, key_P3, key_P4, key_P5, key_P6
            else:
                if args.dynamic and target < 95:
                    target += 0.25
                flow_step += 1
                P2, P3, P4, P5, P6 = warp.predict([key_P2, key_P3, key_P4, key_P5, key_P6, flow])

        inputs=[image_metas, P2, P3, P4, P5, P6]
        result = maskrcnn.detect_molded(inputs)
        
        # Save
        if args.is_save:
            save_name = args.save_dir + 'mask' + str(step) + '.png'
            colors = np.array(label_colors)/255.0
            pred_img = visualize.display_instances(image, result['rois'], result['masks'], result['class_ids'], 
                            class_names, result['scores'], colors = colors, save_name=save_name)
    print("segmentation steps:", seg_step, "flow steps:", flow_step)


if __name__ == '__main__':
    main()
    
