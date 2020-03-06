# Limit gpu usage to only gpu:0 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tfserve import TFServeApp
import tensorflow as tf
import sys
import tempfile

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.utils as utils
import numpy as np
import math
import cv2
from PIL import Image
from io import BytesIO
import json
import tdmlabelbox as tdmlb

# Build validation dataset
import tdmcoco_serve as tdmcoco

# File path to Frozen Graph
FROZEN_GRAPH = "./frozen_graphs/graphene_tdm20190827T1836_mask_rcnn_tdm_0120.pb" # Graphene 2019/08/27

# Load configurations
config = tdmcoco.CocoConfig()
#COCO_DIR = "/workspace/raid/2dmaterials/graphene"  # TODO: enter value here
COCO_DIR ="/workspace/raid/2dmaterials/graphene006"

if config.NAME == 'shapes':
    dataset = shapes.ShapesDataset()
    dataset.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
elif config.NAME == "tdm":
    dataset = tdmcoco.CocoDataset()
    dataset.load_coco(COCO_DIR, "val")

# Must call before using the dataset
dataset.prepare()

#from tdmcoco import CocoConfig
#config = CocoConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 3
#    DETECTION_MIN_CONFIDENCE = 0.7   
    DETECTION_MIN_CONFIDENCE = 0.5

config = InferenceConfig()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple segmentation_datasets
        where not all classes are present in all segmentation_datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +  # size=3
        list(window) +  # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +  # size=1
        list(active_class_ids)  # size=num_classes
    )
    return meta

def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride))]
            for stride in config.BACKBONE_STRIDES])

def mold_inputs(images):
    """Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matrices [height,width,depth]. Images can have
    different sizes.

    Returns 3 Numpy matrices:
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
        molded_image = mold_image(molded_image, config)
        # Build image_meta
        image_meta = compose_image_meta(
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

def get_anchors(image_shape):
    """Returns anchor pyramid for the given image size."""
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    # Cache anchors and reuse if image shape is the same
    try:
        _anchor_cache
    except NameError:
        _anchor_cache = {}
        
    if not tuple(image_shape) in _anchor_cache:
        # Generate Anchors
        a = utils.generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)
        # Keep a copy of the latest anchors in pixel coordinates because
        # it's used in inspect_model notebooks.
        # TODO: Remove this after the notebook are refactored to not use it
        anchors = a
        # Normalize coordinates
        _anchor_cache[tuple(image_shape)] = utils.norm_boxes(a, image_shape[:2])
    return _anchor_cache[tuple(image_shape)]

def load_graph(self):
    with tf.gfile.GFile(self.model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with self.graph.as_default():
        tf.import_graph_def(graph_def)

def build_outputs(self):
    outputs = {}
        
        
    for name in self.outputs:
        tensor_name = self.outputs[name]
        tensor_name = tensor_name + ':0' if ':0' != tensor_name[-2:] else tensor_name
        tensor_name = tensor_name
        tensor = self.graph.get_tensor_by_name(tensor_name)
        outputs[name] = tensor
    self.outputs = outputs

def unmold_detections(detections, mrcnn_mask, original_image_shape,
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
        if full_masks else np.empty(original_image_shape[:2] + (0,))

    return boxes, class_ids, scores, full_masks

def __call__(self, images):
    molded_images, image_metas, windows = self.mold_inputs(images)

    # Validate image sizes
    # All images in a batch MUST be of the same size
    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

    # Anchors
    anchors = self.get_anchors(image_shape)
    anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
    # Run object detection
    feed_dict = self._build_feed({
        'input_image': molded_images,
        'input_image_meta': image_metas,
        'input_anchors': anchors
    })
    result_dict = self.sess.run(self.outputs, feed_dict=feed_dict)

    # Process detections
    results = []
    for i, image in enumerate(images):
        final_rois, final_class_ids, final_scores, final_masks = \
            self.unmold_detections(result_dict['detection'][i], result_dict['mask'][i],
                                    image.shape, molded_images[i].shape,
                                    windows[i])
        results.append({
            "rois": final_rois,
            "class": final_class_ids,
            "scores": final_scores,
            "mask": final_masks,
        })
        
    return results

def _build_feed(network_inputs):
    feed_dict = {}
    for name in network_inputs:
        p = 'import/' + name + ':0'
        feed_dict[p] = network_inputs[name]
    return feed_dict

def inspect_tensors(self, network_inputs, tensor_names):
    feed_dict = self._build_feed(network_inputs)
    output_tensors = {x: self.graph.get_tensor_by_name('import/'+   x) for x in tensor_names}
    return self.sess.run(output_tensors, feed_dict=feed_dict)


def encode(request_data):
    """
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".jpg") as f:
        #f.write(request_data)    
        #img = cv2.imread(f.name)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    """
    img = np.array(Image.open(BytesIO(request_data)))
    images = [img]
    molded_images, image_metas, windows = mold_inputs(images)

    # Validate image sizes
    # All images in a batch MUST be of the same size
    image_shape = molded_images[0].shape
    for g in molded_images[1:]:
        assert g.shape == image_shape, \
            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

    # Anchors
    anchors = get_anchors(image_shape)
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
    # Run object detection
    feed_dict = {
        'import/input_image:0': molded_images[0],
        'import/input_image_meta:0': image_metas[0],
        'import/input_anchors:0': anchors[0]
    }
    return feed_dict
    

def decode(outputs):
    img = np.zeros((1086,2040,3))
    molded_images, image_metas, windows = mold_inputs([img])

    final_rois, final_class_ids, final_scores, final_masks = \
        unmold_detections(outputs['import/mrcnn_detection/Reshape_1:0'],
                          outputs['import/mrcnn_mask/Reshape_1:0'],
                          img.shape, molded_images[0].shape,
                          windows[0])
    
    final_labels = []
    for i in range(final_rois.shape[0]):
        segmentation_map = np.zeros(final_masks[:,:,i].shape)
        segmentation_map[final_masks[:,:,i]] = 1
        segmentation_map = segmentation_map.astype(np.int16)
        label = tdmlb.vectorize_to_v4_label(segmentation_map, {1: str(final_class_ids[i])})
        final_labels.append(label)
        
    results = {
        "rois": json.dumps(final_rois.tolist()),
        "class": json.dumps(final_class_ids.tolist()),
        "scores": json.dumps(final_scores.tolist()),
        "masks": json.dumps(final_labels)
    }
    return results

app = TFServeApp(FROZEN_GRAPH,
                ["import/input_image:0",
                 "import/input_image_meta:0",
                 "import/input_anchors:0"],
                ["import/mrcnn_detection/Reshape_1:0", #detection
                 "import/mrcnn_class/Reshape_1:0",     #class
                 "import/mrcnn_mask/Reshape_1:0"       #mask
                ], encode, decode)

app.run('0.0.0.0', 5000)
