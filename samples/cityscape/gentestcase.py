import os
import sys
import argparse
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

from cityscape import CityscapeConfig, CityscapeDataset
from models import ResNet101, FlowNet, MaskRCNN, Warp
from mrcnn import utils
import mrcnn.model as modellib

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate testcases")
    parser.add_argument("--subset", type=str, default="train",
                        help="testcase subset.")
    parser.add_argument("--offset", type=int, default=13,
                        help="testcase offset.")
    parser.add_argument("--clip", type=float, default=0.5,
                        help="trim extreme confidence score")
    return parser.parse_args()
                        
def overlap_pred(class_ids, scores, masks):
    overlap_mask = np.zeros(masks.shape[:2], dtype=np.uint8)
    overlap_score = np.zeros(masks.shape[:2])
    for i in range(len(class_ids)):
        tmp_score = masks[:,:,i]*scores[i]
        overlap_mask = np.where(tmp_score > overlap_score, class_ids[i], overlap_mask)
        overlap_score = np.where(tmp_score > overlap_score, tmp_score, overlap_score)
    return overlap_mask

def overlap_gt(class_ids, masks):
    overlap_mask = masks*class_ids
    overlap_mask = np.amax(overlap_mask,axis=2).astype(np.uint8)
    return overlap_mask

def confidence(gt, pred):
    index = np.greater(gt+pred,0)
    score = np.sum(np.equal(gt[index],pred[index]))/np.sum(index)
    return score

def main():
    args = get_arguments()
    print(args)
    
    data_dir = '/data/cityscapes_dataset/cityscape'
    config = CityscapeConfig()
    config.IMAGE_SHAPE = [1024, 1024, 6]
    config.Flow =True
    #config.display()

    # Validation dataset
    dataset = CityscapeDataset()
    dataset.load_cityscape(data_dir, args.subset, args.offset)
    dataset.prepare()
    print("Image Count: {}".format(len(dataset.image_ids)))

    resnet = ResNet101(config=config)
    flownet = FlowNet(config=config)
    maskrcnn = MaskRCNN(config=config)
    warp = Warp(config=config)

    model_path = ""

    resnet.load_weights(model_path, by_name=True)
    flownet.load_weights(model_path, by_name=True)
    maskrcnn.load_weights(model_path, by_name=True)

    score_list = []
    ft_list = []
    for image_id in range(len(dataset.image_ids)):
         # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)

        images = np.expand_dims(image, 0)
        image_metas = np.expand_dims(image_meta, 0)
        
        if image_id % args.offset == 0:
            true_P2, true_P3, true_P4, true_P5, true_P6 = resnet.keras_model.predict(images[:,:,:,:3])
            true_inputs = [image_metas, true_P2, true_P3, true_P4, true_P5, true_P6]
            result1 = maskrcnn.detect_molded(true_inputs)
            true = overlap_pred(result1["class_ids"],result1["scores"],result1["masks"])
            #print(confidence(overlap_gt(gt_class_id,gt_mask), true))
        
        if np.sum(result1["scores"]) == 0:
            print("{} True Fasle".format(image_id))
            continue
            
        key_P2, key_P3, key_P4, key_P5, key_P6 = resnet.keras_model.predict(images[:,:,:,3:])
        flow, flow_feature = flownet.keras_model.predict(images)
        P2, P3, P4, P5, P6 = warp.predict([key_P2, key_P3, key_P4, key_P5, key_P6, flow])
        inputs=[image_metas, P2, P3, P4, P5, P6]
        result2 = maskrcnn.detect_molded(inputs)
        
        if np.sum(result2["scores"]) == 0:
            print("{} Pred Fasle".format(image_id))
            continue
        
        pred = overlap_pred(result2["class_ids"],result2["scores"],result2["masks"])
        score = confidence(true, pred)
        #print(score)
        if score > args.clip:
            ft_list.append(np.squeeze(flow_feature))
            score_list.append(score)

        if image_id % 100 == 0:
            print("step: {:3d}".format(image_id))

    # save confidence score and feature
    np.save(args.subset+"X", ft_list) 
    np.save(args.subset+"Y", score_list)


if __name__ == '__main__':
    main()
