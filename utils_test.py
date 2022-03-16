import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

# 3RScan dataset path
data_path = '../data'

# Original image sizes from the 3RScan dataset's images in the sequence folder
orig_width = 960
orig_height = 540

bbox_data_aligned_vertically = False

resize_image_shape = (224, 224) # (256, 256), None

# Transformations to apply to the images
rotate_vertical_transform = transforms.Compose([
    transforms.Pad(padding=(0, 210), fill=0),  # padding needed for rotation to keep all information
    transforms.Lambda(lambda x: TF.rotate(x, -90)),
])

if resize_image_shape is not None:
    transform = transforms.Compose([
      rotate_vertical_transform,
      transforms.Resize(resize_image_shape),
      transforms.ToTensor()
    ])
else:
    transform = transforms.Compose([
      rotate_vertical_transform,
      transforms.ToTensor()
    ])

def rotate_bbox_minus_90(bbox):
    """
    This is a temporary fix for wrongly calculated bboxes.
    They are assumed to be calculated w.r.t. an already rotated image (image is already vertical).
    Instead, we need them to be calculated w.r.t. the camera images in their original orientation (horizontally).
    Thus, we rotate them back 90 degree counter-clockwise.

    TODO This is only a temporary fix because we should not have to expect the bboxes to be in this wrong format.
    TODO As soon as this format issue is fixed, we can completely remove this method.

    :param bbox:
    :return:
    """

    x1 = bbox[1]
    x2 = bbox[3]

    y1 = orig_height - bbox[2]
    y2 = orig_height - bbox[0]

    return x1, y1, x2, y2

def apply_bbox_sanity_check(bbox, parsed_instance):
    """
    Check if a bbox still is large enough and if not make it larger.
    A bbox must at least have a width and height of 8.

    :param bbox: bbox as loaded from a row of 2DInstances.txt
    :param parsed_instance: instance for this bbox, used for logging purposes

    :return: bbox but with ensured minimum width and height
    """

    if bbox[0] == -1 and bbox[1] == -1 and bbox[2] == -1 and bbox[3] == -1:
        # See this as error, we do not know how to resolve this
        raise ValueError("Needed to fix -1 bbox case for ", parsed_instance["scan"], parsed_instance["frame_nr"],
                         parsed_instance["label"], bbox, parsed_instance["bbox"])

    if bbox[2] - bbox[0] < 8:
        # This can be no error because we just "enlarge" the bbox a little bit
        # print("Needed to fix x bbox case for ", parsed_instance["scan"], parsed_instance["frame_nr"],
        #      parsed_instance["label"], bbox, parsed_instance["bbox"])
        x1 = bbox[0] - 8
        x1 = x1 if x1 >= 0 else 0
        x2 = bbox[2] + 8
        x2 = x2 if x2 <= 959 else 959
        bbox = (x1, bbox[1], x2, bbox[3])

    if bbox[3] - bbox[1] < 8:
        # This can be no error because we just "enlarge" the bbox a little bit
        # print("Needed to fix y bbox case for ", parsed_instance["scan"], parsed_instance["frame_nr"],
        #      parsed_instance["label"], bbox, parsed_instance["bbox"])
        y1 = bbox[1] - 8
        y1 = y1 if y1 >= 0 else 0
        y2 = bbox[3] + 8
        y2 = y2 if y2 <= 959 else 959
        bbox = (bbox[0], y1, bbox[2], y2)

    return bbox

def transform_bbox(bbox, bbox_data_aligned_vertically, transform):
    """
    Resizes/rescales the bbox by applying the specified transformation.

    :param bbox: bbox as loaded from a row of 2DInstances.txt
    :return: transformed bbox in format (x1, y1, x2, y2) or (-1, -1, -1, -1) if bbox is empty after 
    transformation
    """
    
    if bbox_data_aligned_vertically:
        x1, y1, x2, y2 = rotate_bbox_minus_90(bbox)
    else:
        x1, y1, x2, y2 = bbox

    if transform is None:
        return x1, y1, x2, y2

    # convert to boolean mask image
    bbox_mask = np.zeros((orig_height, orig_width),
                         dtype=np.uint8)
    bbox_mask[y1:y2 + 1, x1:x2 + 1] = 255

    # apply torch transforms, convert back to numpy
    pil = Image.fromarray(bbox_mask).convert("RGB")

    if isinstance(transform.transforms[-1], torchvision.transforms.ToTensor):
        transform = torchvision.transforms.Compose([
            *transform.transforms[:-1]
        ])
        pil = transform(pil).convert("L")
    else:
        pil = transform(pil).convert("L")

    bbox_mask = np.asarray(pil)

    # extract new bbox
    mask_indices = np.argwhere(bbox_mask > 0)  # where is mask now == 1 after transforming it?
    if mask_indices.shape[0] != 0:
        # maybe after downscaling a very small mask area, it vanished completely: in such cases do not add the bbox
        min_mask = np.amin(mask_indices, axis=0)  # minimum (y,x) index: upper-left corner of mask
        max_mask = np.amax(mask_indices, axis=0)  # maximum (y,x) index: bottom-right corner of mask
        x1 = min_mask[1]  # min x value
        y1 = min_mask[0]  # min y value
        x2 = max_mask[1]  # max x value
        y2 = max_mask[0]  # max y value
        transformed_bbox = x1, y1, x2, y2
    else:
        transformed_bbox = -1, -1, -1, -1

    return transformed_bbox

def parse_instance(instance):
        """
        Parse a row of 2DInstances.txt into its columns as defined in 2DInstances.txt

        :param instance: a row of 2DInstances.txt
        :return: dict with format
        {
            "reference": ref_scan_id,
            "scan": scan_id,
            "frame_nr": frame_id,
            "instance_id": instance_id,
            "label": label,
            "bbox": [x1, y1, x2, y2],
            "visibility": [truncation_number_pixels_original_image, truncation_number_pixels_larger_fov_image, truncation, occlusion_number_pixels_original_image, occlusion_number_pixels_only_with_that_instance, occlusion],
            "number_other_instances": number_other_instances,
            "other_instance_ids": other_instance_ids,
            "other_bboxes": other_bboxes (list of [x1, y1, x2, y2] where i-th bbox belongs to other_instance_id[i])
        }
        """
        
        ref_scan_id = instance[0]
        scan_id = instance[1]
        frame_id = int(instance[2])
        instance_id = int(instance[3])
        label = instance[4]
        bbox = [int(x) for x in instance[5:9]]
        visibility = [float(x) for x in instance[9:15]]
        number_other_instances = int(instance[15])
        other_instance_ids = []
        other_bboxes = []
        if number_other_instances > 0:
            for i in range(number_other_instances):
                start = 16 + i * 5  # we have 5 values in the file for each other instance: <instance_id> <bbox>
                other_instance_ids.append(int(instance[start]))  # every 5-th value is a new instance id
                other_bboxes.append([int(x) for x in instance[start + 1:start + 5]])  # next 4 values are the bbox

        result =  {
            "reference": ref_scan_id,
            "scan": scan_id,
            "frame_nr": frame_id,
            "instance_id": instance_id,
            "label": label,
            "bbox": bbox,
            "visibility": visibility,
            "number_other_instances": number_other_instances,
            "other_instance_ids": other_instance_ids,
            "other_bboxes": other_bboxes
        }

        return result
    
def load_instance(instance, transform=None):
        """
        Loads a row of 2DInstances.txt by parsing its columns, loading the image from disk, loading 
        the bbox and applying transformations.

        :param instance: a row of 2DInstances.txt
        :return: loaded instance
        """

        # parse instance into dict
        parsed_instance = parse_instance(instance)

        # load image
        frame_name = "frame-{:06d}.color.jpg".format(parsed_instance["frame_nr"])
        image_path = os.path.join(data_path, parsed_instance["scan"], "sequence", frame_name)
        image = Image.open(image_path)
        bbox = parsed_instance["bbox"]
        
        if transform is not None:
            image = transform(image)
            bbox = transform_bbox(parsed_instance["bbox"], bbox_data_aligned_vertically, transform)
            
        # make sure that bbox is valid in all cases
        bbox = apply_bbox_sanity_check(bbox, parsed_instance)

        # convert bbox into dict
        bbox_dict = {
            "x": bbox[0],
            "y": bbox[1],
            "w": bbox[2] - bbox[0],
            "h": bbox[3] - bbox[1]
        }

        # construct loaded dict
        loaded_instance = {
            "image": image,
            "bbox": bbox_dict,
            "label": parsed_instance["label"],
            "instance_id": parsed_instance["instance_id"],
            "reference": parsed_instance["reference"],
            "scan": parsed_instance["scan"],
            "frame_nr": parsed_instance["frame_nr"],
        }

        return loaded_instance