import os
import sys
import numpy as np
import skimage

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class CityscapeConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscapes"
    
    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + 8 class

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    #BACKBONE_STRIDES = [8, 16, 32, 64]
    
    # Length of square anchor side in pixels
    #RPN_ANCHOR_SCALES = (64, 128, 256, 512)
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([122.67891434,116.66876762,104.00698793])
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (128, 128)  # (height, width) of the mini-mask
    
    Flow = None
    
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
class CityscapeDataset(utils.Dataset):
    def load_cityscape(self, data_dir, subset, offset=10, fix=False, class_ids=None):
        """Load a subset of the cityscapes dataset.
           dataset_dir: The root directory of the cityscapes dataset.
           subset: What to load (train, val)
        """
        cityscape = COCO("{}/annotations/instancesonly_filtered_gtFine_{}.json".format(data_dir, subset))
        image_dir = "{}/leftImg8bit_sequence/{}".format(data_dir, subset)
        
        if not class_ids:
            # All classes
            class_ids = sorted(cityscape.getCatIds())
            
        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(cityscape.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(cityscape.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("cityscape", i, cityscape.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            filename = cityscape.imgs[i]['file_name'].split('_')[0] + '/' + cityscape.imgs[i]['file_name']
            index = int(filename[-22:-16])
            if fix:
                imagename = filename[:-22]+str(index-offset+1).zfill(6)+filename[-16:]
                self.add_image(
                    "cityscape", image_id=i,
                    path=[os.path.join(image_dir, filename),os.path.join(image_dir, imagename)],
                    width=cityscape.imgs[i]["width"],
                    height=cityscape.imgs[i]["height"],
                    annotations=cityscape.loadAnns(cityscape.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))
            else:
                for j in range(index-offset+1,index+1):
                    imagename = filename[:-22]+str(j).zfill(6)+filename[-16:]
                    self.add_image(
                        "cityscape", image_id=i,
                        path=[os.path.join(image_dir, filename),os.path.join(image_dir, imagename)],
                        width=cityscape.imgs[i]["width"],
                        height=cityscape.imgs[i]["height"],
                        annotations=cityscape.loadAnns(cityscape.getAnnIds(
                            imgIds=[i], catIds=class_ids, iscrowd=None)))
        
    def image_reference(self, image_id):
        """Return the cityscapes information of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cityscape":
            return info["source"]
        else:
            super(self.__class__).image_reference(self, image_id)
            
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        current_image = skimage.io.imread(self.image_info[image_id]['path'][0])
        key_image = skimage.io.imread(self.image_info[image_id]['path'][1])
        images = np.concatenate([current_image, key_image], axis=2)
        return images
    
    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cityscape":
            return super(self.__class__, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "cityscape.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__, self).load_mask(image_id)

    # The following two functions are from pycocotools with a few changes.
            
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
