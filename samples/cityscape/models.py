import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

def find_last(config, model_dir):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Returns:
        log_dir: The directory where events and weights are saved
        checkpoint_path: the path to the last checkpoint file
    """
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(model_dir))[1]
    key = config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        return None, None
    # Pick last directory
    dir_name = os.path.join(model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint

############################################################
# Warping function
############################################################
def get_index(b,h,w,num):
    index_list = []
    for k in range(b):
        for i in range(h):
            for j in range(w):
                for index in range(num):
                    index_list.append([index])
    return np.array(index_list, dtype=np.float32).reshape((b,h,w,num))

def get_hindex(b,h,w,num):
    index_list = []
    for k in range(b):
        for i in range(h):
            for j in range(w):
                for index in range(num):
                    index_list.append([i])
    return np.array(index_list, dtype=np.float32).reshape((b,h,w,num))

def get_windex(b,h,w,num):
    index_list = []
    for k in range(b):
        for i in range(h):
            for j in range(w):
                for index in range(num):
                    index_list.append([j])
    return np.array(index_list, dtype=np.float32).reshape((b,h,w,num))

class WarpingLayer(KE.Layer):
    def __init__(self, backbone_shape, stride, depth=256, config=None, **kwargs):
        super(WarpingLayer, self).__init__(**kwargs)
        self.batch = config.IMAGES_PER_GPU
        self.height = backbone_shape[0]
        self.width = backbone_shape[1]
        self.depth = depth
        self.scale = 20.0/stride
        
    def call(self, inputs):
        key_feature, flow = inputs
        batch_size = self.batch
        height = self.height
        width = self.width
        key_feature = tf.image.resize_bilinear(key_feature, [height, width])
        flow = tf.image.resize_bilinear(flow, [height, width])*self.scale
        h_a = tf.Variable(get_index(batch_size,height,width,height))
        h_p = tf.Variable(get_hindex(batch_size,height,width,height))
        w_a = tf.Variable(get_index(batch_size,height,width,width))
        w_p = tf.Variable(get_windex(batch_size,height,width,width))

        flow_x = tf.reshape(flow[:,:,:,0],[batch_size,height,width,1])
        flow_y = tf.reshape(flow[:,:,:,1],[batch_size,height,width,1])

        h_b = tf.maximum(tf.minimum(h_p + flow_y, height-1.0), 0.0)
        w_b = tf.maximum(tf.minimum(w_p + flow_x, width-1.0), 0.0)

        h_kernal = tf.expand_dims(tf.maximum(1.0-tf.abs(h_a-h_b), 0.0), axis=4)
        w_kernal = tf.expand_dims(tf.maximum(1.0-tf.abs(w_a-w_b), 0.0), axis=3)

        g_kernal = tf.reshape(tf.matmul(h_kernal,w_kernal),[self.batch, height*width, height*width])
        key_kernal = tf.reshape(key_feature,[batch_size, height*width, -1])

        warp_mat = tf.matmul(g_kernal,key_kernal)
        warp_pred = tf.reshape(warp_mat,[batch_size, height, width, self.depth])
        return warp_pred

    def compute_output_shape(self, input_shape):
        return (None, self.height, self.width, self.depth)
    
def get_xyindex(h,w):
    index_list = []
    for i in range(h):
        for j in range(w):
            index_list.append([j,i])
    return np.array(index_list)

def get_batchindex(b,h,w):
    index_list = []
    for k in range(b):
        for i in range(h):
            for j in range(w):
                index_list.append([k])
    return np.array(index_list)

class WarpingForwardLayer(KE.Layer):
    def __init__(self, backbone_shape, stride, depth=256, config=None, **kwargs):
        super(WarpingForwardLayer, self).__init__(**kwargs)
        self.batch = config.IMAGES_PER_GPU
        self.height = backbone_shape[0]
        self.width = backbone_shape[1]
        self.depth = depth
        self.scale = 20.0/stride
        
    def call(self, inputs):
        key_feature, flow = inputs
        batch_size = self.batch
        height = self.height
        width = self.width
        key_feature = tf.image.resize_bilinear(key_feature, [height, width])
        flow = tf.image.resize_bilinear(flow, [height, width])*self.scale
        flow_index = flow + tf.constant(get_xyindex(height, width),shape=[height, width, 2],dtype=tf.float32)
        flow_index = tf.minimum(flow_index, [width-1.0,height-1.0])
        flow_index = tf.maximum(flow_index, [0.0,0.0])
        batch_index = tf.constant(get_batchindex(batch_size, height, width),shape=[batch_size, height, width, 1],dtype=tf.float32)
        x_index = tf.reshape(flow_index[:,:,:,0], [batch_size, height, width, 1])
        y_index = tf.reshape(flow_index[:,:,:,1], [batch_size, height, width, 1])
        x_floor = tf.floor(x_index)
        x_ceil = tf.ceil(x_index)
        y_floor = tf.floor(y_index)
        y_ceil = tf.ceil(y_index)
        flow_index_ff = tf.cast(tf.concat([batch_index,y_floor,x_floor], 3), tf.int32)
        flow_index_cf = tf.cast(tf.concat([batch_index,y_ceil,x_floor], 3), tf.int32)
        flow_index_fc = tf.cast(tf.concat([batch_index,y_floor,x_ceil], 3), tf.int32)
        flow_index_cc = tf.cast(tf.concat([batch_index,y_ceil,x_ceil], 3), tf.int32)
        thetax = x_index - x_floor
        _thetax = 1.0 - thetax
        thetay =  y_index - y_floor
        _thetay = 1.0 - thetay
        coeff_ff = _thetax * _thetay
        coeff_cf = _thetax * thetay
        coeff_fc = thetax * _thetay
        coeff_cc = thetax * thetay
        ff = tf.gather_nd(key_feature, flow_index_ff) * coeff_ff
        cf = tf.gather_nd(key_feature, flow_index_cf) * coeff_cf
        fc = tf.gather_nd(key_feature, flow_index_fc) * coeff_fc
        cc = tf.gather_nd(key_feature, flow_index_cc) * coeff_cc
        warp_image = tf.add_n([ff,cf,fc,cc])
        return warp_image
    
    def compute_output_shape(self, input_shape):
        return (None, self.height, self.width, self.depth)
def Warp(config):
    P2 = KL.Input(shape=[None,None,256], name="P2")
    P3 = KL.Input(shape=[None,None,256], name="P3")
    P4 = KL.Input(shape=[None,None,256], name="P4")
    P5 = KL.Input(shape=[None,None,256], name="P5")
    P6 = KL.Input(shape=[None,None,256], name="P6")
    flow = KL.Input(shape=[None,None,2], name="flow")
    
    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    flow_p6 = WarpingForwardLayer(backbone_shapes[4], config.BACKBONE_STRIDES[4], config=config, name="flow_p6")([P6, flow])
    flow_p5 = WarpingForwardLayer(backbone_shapes[3], config.BACKBONE_STRIDES[3], config=config, name="flow_p5")([P5, flow])
    flow_p4 = WarpingForwardLayer(backbone_shapes[2], config.BACKBONE_STRIDES[2], config=config, name="flow_p4")([P4, flow])
    flow_p3 = WarpingForwardLayer(backbone_shapes[1], config.BACKBONE_STRIDES[1], config=config, name="flow_p3")([P3, flow])
    flow_p2 = WarpingForwardLayer(backbone_shapes[1], config.BACKBONE_STRIDES[1], config=config, name="flow_p2")([P2, flow])

    return KM.Model([P2, P3, P4, P5, P6, flow], [flow_p2, flow_p3, flow_p4, flow_p5, flow_p6], name="warp_model")
                
class Models(object):
    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')

        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
            
        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()
            
class ResNet101(Models):
    def __init__(self, config):
        """
        config: A Sub-class of the Config class
        """
        self.config = config
        self.keras_model = self.build(config=config)
        
    def build(self, config):
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=[None, None, 3], name="input_image")
        
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        _, C2, C3, C4, C5 = modellib.resnet_graph(input_image, "resnet101", stage5=True, train_bn=config.TRAIN_BN)
        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
        P4 = KL.Add(name="fpn_p4add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
            KL.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)])
        P3 = KL.Add(name="fpn_p3add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
            KL.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)])
        P2 = KL.Add(name="fpn_p2add")([
            KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
            KL.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
        P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
        P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
        P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = KL.MaxPooling2D(pool_size=(2, 2), strides=2, name="fpn_p6")(P5)
        model = KM.Model(input_image, [P2, P3, P4, P5, P6], name='resnet101')
        return model
            
class FlowNet(Models):
    def __init__(self, config):
        """
        config: A Sub-class of the Config class
        """
        self.config = config
        self.keras_model = self.build(config=config)
        
    def build(self, config):
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=[None, None, 6], name="input_image")
        
        flow, flow_feature = modellib.flow_graph(input_image)
        model = KM.Model(input_image, [flow, flow_feature], name='flownet')
        return model
    
class MaskRCNN(Models):
    def __init__(self, config):
        """
        config: A Sub-class of the Config class
        """
        self.config = config
        self.keras_model = self.build(config=config)
        
    def build(self, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        P2 = KL.Input(shape=[None,None,256], name="P2")
        P3 = KL.Input(shape=[None,None,256], name="P3")
        P4 = KL.Input(shape=[None,None,256], name="P4")
        P5 = KL.Input(shape=[None,None,256], name="P5")
        P6 = KL.Input(shape=[None,None,256], name="P6")
        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
        
        P2_resize = KL.Lambda(lambda x: tf.image.resize_bilinear(x,[256,256]), name="P2_resize")(P2)
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2_resize, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2_resize, P3, P4, P5]
        
        # Generate Anchors
        anchors = input_anchors

        # RPN Model
        rpn = modellib.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), 256)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count =  config.POST_NMS_ROIS_INFERENCE
        rpn_rois = modellib.ProposalLayer(proposal_count=proposal_count,
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI",
                                 config=config)([rpn_class, rpn_bbox, anchors])
        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
            modellib.fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                         config.POOL_SIZE, config.NUM_CLASSES,
                                         train_bn=config.TRAIN_BN)

        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in 
        # normalized coordinates
        detections = modellib.DetectionLayer(config, name="mrcnn_detection")(
            [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

        # Create masks for detections
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = modellib.build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                          input_image_meta,
                                          config.MASK_POOL_SIZE,
                                          config.NUM_CLASSES,
                                          train_bn=config.TRAIN_BN)

        model = KM.Model([input_image_meta, P2, P3, P4, P5, P6, anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')
        return model
    
    def detect_molded(self, inputs, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also retruned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = inputs[0][0, 4:7].astype(int)

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        inputs.append(anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict(inputs, verbose=0)
        
        # Process detections
        window = [0, 0, image_shape[0], image_shape[1]]
        final_rois, final_class_ids, final_scores, final_masks =\
            self.unmold_detections(detections[0], mrcnn_mask[0],
                                   image_shape, image_shape,
                                   window)
        results={
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        }
        return results
    
    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(masks.shape[1:3] + (0,))

        return boxes, class_ids, scores, full_masks
    
    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = modellib.compute_backbone_shapes(self.config, image_shape)
        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]
    
class Decision(Models):
    def __init__(self, dropout=0.0):
        """
        config: A Sub-class of the Config class
        """
        self.keras_model = self.build(dropout=dropout)
        
    def build(self, dropout):
        # Inputs
        input_feature = KL.Input(shape=[16, 16, 384], name="input_feature")
        
        conv1 = KL.Conv2D(384, (3, 3), strides=2, padding='SAME', name='decision_conv1')(input_feature)
        conv2 = KL.Conv2D(96, (1, 1), strides=1, padding='SAME', name='decision_conv2')(conv1)
        conv = KL.Flatten()(conv2)
        fc1 = KL.Dense(1024, kernel_initializer='he_normal', bias_initializer=KI.Constant(value=0.1), name='decision_fc1')(conv)
        fc1 = KL.LeakyReLU(alpha=0.1)(fc1)
        fc1 = KL.Dropout(dropout)(fc1)
        fc2 = KL.Dense(1024, kernel_initializer='he_normal', bias_initializer=KI.Constant(value=0.1), name='decision_fc2')(fc1)
        fc2 = KL.LeakyReLU(alpha=0.1)(fc2)
        fc2 = KL.Dropout(dropout)(fc2)
        fc3 = KL.Dense(10  , kernel_initializer='he_normal', bias_initializer=KI.Constant(value=0.1), name='decision_fc3')(fc2)
        fc3 = KL.LeakyReLU(alpha=0.1)(fc3)
        fc3 = KL.Dropout(dropout)(fc3)
        fc4 = KL.Dense(1   , kernel_initializer='he_normal', bias_initializer=KI.Constant(value=0.1), name='decision_fc4')(fc3)
        fc4 = KL.LeakyReLU(alpha=0.1)(fc4)
        
        model = KM.Model(input_feature, fc4, name='decision')
        return model
    
    def train(self, data_dir, log_dir, learning_rate=0.002, decay=1e-6, batch_size=32, epochs=100, clipnorm=5):
        # Data 
        trX = np.load(data_dir+'trainX.npy')
        trY = np.expand_dims(np.load(data_dir+'trainY.npy'),1)*100
        vaX = np.load(data_dir+'valX.npy')
        vaY = np.expand_dims(np.load(data_dir+'valY.npy'),1)*100
        
        # Callbacks
        checkpoint_path = os.path.join(log_dir, "deciosion_{epoch:04d}_{val_mean_absolute_error:.2f}.h5")
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_mean_absolute_error', mode='min',
                                            verbose=0, save_weights_only=True),
        ]
        
        # Train     
        optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay, clipnorm=clipnorm)
        self.keras_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        self.keras_model.fit(
            x=trX,
            y=trY,
            validation_data = (vaX, vaY),
            batch_size = batch_size,
            epochs=epochs,
            #steps_per_epoch=trY.shape[0],
            #validation_steps=vaY.shape[0],
            callbacks=callbacks,
        )
               
        
