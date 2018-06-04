import os
import sys
import time
import argparse
import numpy as np
from scipy import misc

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from cityscape import CityscapeConfig, CityscapeDataset
from models import ResNet, FlowNet, MaskRCNN, Warp, Decision
from mrcnn import utils, visualize
import mrcnn.model as modellib

DATA_DIRECTORY = '/data/cityscapes_dataset/cityscape/'
RESTORE_FROM = ""
DECISION_FROM = ""
SAVE_DIR = './output/'
TARGET = 80.0

label_colors = [(0, 0, 142), (0, 0, 69), (219, 19, 60)
                # 0 = car, 1 = truck, 2 = person
                ,(119, 10, 32), (255, 0, 0), (0, 60, 100)
                # 3 = bicycle, 4 = rider, 5 = bus
                ,(0, 0, 230), (0, 79, 100)]
                # 6 = motocycle, 7 = train
    
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--flowmodel", type=str, default='flownets',
                        choices=['flownets', 'flownetS'],
                        help="chose flow model")
    parser.add_argument("--resnetmodel", type=str, default='resnet50',
                        choices=['resnet50', 'resnet101'],
                        help="chose resnet model")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the dataset.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--decision_from", type=str, default=DECISION_FROM,
                        help="Where restore decision model parameters from.")
    parser.add_argument("--num_frames", type=int, default=20,
                        help="Snippets length.")
    parser.add_argument("--fix", action="store_true",
                        help="Fix key frame.")
<<<<<<< HEAD
    parser.add_argument("--target", type=float, default=TARGET,
=======
    parser.add_argument("--method", type=int, default=2,
                        choices=[0, 1, 2],
                        help="0 = frame_difference, 1 = flow_magnitude, 2 = confidence_score")
    parser.add_argument("--target", type=float, default=80,
>>>>>>> 493f2ce62c8a55486e82d4c9f70e203bf712b6e3
                        help="confidence score threshold.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save segmented output.")
    parser.add_argument("--is_save", action="store_true",
                        help="whether to save output")
    return parser.parse_args()


def main():
    args = get_arguments()
    print(args)
    
    config = CityscapeConfig()
    config.FLOW = args.flowmodel
    config.BACKBONE = args.resnetmodel
    config.IMAGE_SHAPE = [1024, 1024, 6]
    config.POST_NMS_ROIS_INFERENCE = 500
    #config.display()

    # Validation dataset
    dataset = CityscapeDataset()
    dataset.load_cityscape(args.data_dir, "val", args.num_frames, args.fix)
    dataset.prepare()
    print("Image Count: {}".format(len(dataset.image_ids)))
    
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

    AP50s = []
    APs = []
    seg_step = 0
    flow_step = 0
    score = 0
    if args.method == 2:
        target = -args.target
    else:
        target = args.target
    if args.is_save and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for image_id in range(len(dataset.image_ids)):
         # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)

        current = np.expand_dims(image[:,:,3:], 0)# when args.fix it is key frame
        image_metas = np.expand_dims(image_meta, 0)

        if args.fix:
            images = np.expand_dims(image, 0)
            flow, flow_feature = flownet.keras_model.predict(images)
            if args.num_frames == 1:
                P2, P3, P4, P5, P6 = resnet.keras_model.predict(current)
            else:
                key_P2, key_P3, key_P4, key_P5, key_P6 = resnet.keras_model.predict(current)
                P2, P3, P4, P5, P6 = warp.predict([key_P2, key_P3, key_P4, key_P5, key_P6, flow])
        else:
            if image_id % args.num_frames != 0:
                images = np.concatenate([current, key], 3)
                flow, flow_feature = flownet.keras_model.predict(images)
                if args.method == 0:
                    im1_gray = cv2.cvtColor(np.squeeze(current), cv2.COLOR_RGB2GRAY)
                    im2_gray = cv2.cvtColor(np.squeeze(key), cv2.COLOR_RGB2GRAY)
                    score = np.mean(np.abs(im1_gray - im2_gray))
                    print("step: {:4d} frame difference: {:.3f}".format(image_id, score))
                else:
                    if args.method == 1:
                        flow_mag = flow*20
                        score = np.mean(np.sqrt(np.square(flow_mag[:,:,:,0]) + np.square(flow_mag[:,:,:,1])))
                        print("step: {:4d} flow magnitude: {:.3f}".format(image_id, score))
                    elif args.method == 2:
                        score = -decision.keras_model.predict(flow_feature)[0][0]
                        print("step: {:4d} predict score: {:.3f}".format(image_id, -score))
                       
            if score > target or image_id % args.num_frames == 0:
                seg_step += 1
                key_P2, key_P3, key_P4, key_P5, key_P6 = resnet.keras_model.predict(current)
                key = current
                P2, P3, P4, P5, P6 = key_P2, key_P3, key_P4, key_P5, key_P6
            else:
                flow_step += 1
                P2, P3, P4, P5, P6 = warp.predict([key_P2, key_P3, key_P4, key_P5, key_P6, flow])

        inputs=[image_metas, P2, P3, P4, P5, P6]
        result = maskrcnn.detect_molded(inputs)
        
        # Compute AP
        if (image_id+1) % args.num_frames == 0 or args.fix:
            if np.sum(result["scores"]) == 0:
                print("{} Fasle".format(image_id))
                continue
                
            AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            result["rois"], result["class_ids"], result["scores"], result['masks'])
            AP50s.append(AP)
            AP = utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                           result["rois"], result["class_ids"], result["scores"], result['masks'], verbose=0)
            APs.append(AP)
            print("step: {:4d}, AP50: {:.3f}, mAP: {:.3f}".format(image_id, np.mean(AP50s), np.mean(APs)))
        
        # Save
        if args.is_save:
            save_name = args.save_dir + 'mask' + str(image_id) + '.png'
            colors = np.array(label_colors)/255.0
            pred_img = visualize.display_instances(image[:,:,3:], result['rois'], result['masks'], result['class_ids'], 
                            dataset.class_names, result['scores'], colors = colors, save_name=save_name)
    print("step: {:4d}, AP50: {:.3f}, mAP: {:.3f}".format(image_id, np.mean(AP50s), np.mean(APs)))
    print("segmentation steps:", seg_step, "flow steps:", flow_step)

if __name__ == '__main__':
    main()