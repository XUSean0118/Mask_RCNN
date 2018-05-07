import os
import sys
import time
import argparse
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from cityscape import CityscapeConfig, CityscapeDataset
from models import ResNet, FlowNet, MaskRCNN, Warp, Decision
from mrcnn import utils
import mrcnn.model as modellib

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--num_frames", type=int, default=20,
                        help="Snippets length.")
    parser.add_argument("--target", type=float, default=85,
                        help="confidence score threshold.")
    return parser.parse_args()


def main():
    args = get_arguments()
    print(args)
    
    data_dir = '/data/cityscapes_dataset/cityscape'
    config = CityscapeConfig()
    config.IMAGE_SHAPE = [1024, 1024, 6]
    config.Flow =True
    config.POST_NMS_ROIS_INFERENCE = 500
    #config.display()

    # Validation dataset
    dataset = CityscapeDataset()
    dataset.load_cityscape(data_dir, "val", args.num_frames)
    dataset.prepare()
    print("Image Count: {}".format(len(dataset.image_ids)))
    
    resnet = ResNet(config=config)
    flownet = FlowNet(config=config)
    maskrcnn = MaskRCNN(config=config)
    warp = Warp(config=config)
    decision = Decision()

    model_path = ""
    resnet.load_weights(model_path, by_name=True)
    flownet.load_weights(model_path, by_name=True)
    maskrcnn.load_weights(model_path, by_name=True)
    decision.load_weights("", by_name=True)

    AP50s = []
    APs = []
    seg_step = 0
    flow_step = 0
    score = 0
    for image_id in range(len(dataset.image_ids)):
         # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)

        current = np.expand_dims(image[:,:,3:], 0)
        image_metas = np.expand_dims(image_meta, 0)

        if image_id % args.num_frames != 0:
            images = np.concatenate([current, key], 3)
            flow, flow_feature = flownet.keras_model.predict(images)
            score = decision.keras_model.predict(flow_feature)

        if score < args.target or image_id % args.num_frames == 0:
            seg_step += 1
            key_P2, key_P3, key_P4, key_P5, key_P6 = resnet.keras_model.predict(current)
            key = current
            P2, P3, P4, P5, P6 = key_P2, key_P3, key_P4, key_P5, key_P6
        else:
            flow_step += 1
            P2, P3, P4, P5, P6 = warp.predict([key_P2, key_P3, key_P4, key_P5, key_P6, flow])

        # Compute AP
        if (image_id+1) % args.num_frames == 0:
            inputs=[image_metas, P2, P3, P4, P5, P6]
            result = maskrcnn.detect_molded(inputs)

            if np.sum(result["scores"]) == 0:
                print("{} Fasle".format(image_id))
                continue
                
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            result["rois"], result["class_ids"], result["scores"], result['masks'])
            AP50s.append(AP)
            AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                           result["rois"], result["class_ids"], result["scores"], result['masks'], verbose=0)
            APs.append(AP)
            print("step: {:3d}, AP50: {:.3f}, mAP: {:.3f}".format(image_id, np.mean(AP50s), np.mean(APs)))
        
    print("step: {:3d}, AP50: {:.3f}, mAP: {:.3f}".format(image_id, np.mean(AP50s), np.mean(APs)))
    print("segmentation steps:", seg_step-492, "flow steps:", flow_step)

if __name__ == '__main__':
    main()
