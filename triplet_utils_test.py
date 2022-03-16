import json
import os
import numpy as np
from utils_test import *

sample_treshold = 20
scans = {}
metadata = json.load(open(os.path.join(data_path, "3RScan.json")))
pos_minimum_visibility = [0.6, 0.6]
pos_maximum_visibility = [1.0, 1.0]
neg_background_overlap_minimum = 256*256
positive_sample_probabilities = [0.25, 0.25, 0.25, 0.25]
negative_sample_probabilities = [0, 0, 0, 0.5, 0.5]
number_negative_samples = 1

def add_all_instances_of_same_scan(instance, idx, instances_list):

    # ref scan id of the instance for which adding should be performed
    ref_scan_id = parse_instance(instance)["reference"]

    # search backwards from instance
    for i in range(idx-1, -1, -1):
        scan_id = parse_instance(instances[i])["reference"]
        if scan_id == ref_scan_id:
            instances_list.append(instances[i])
        else:
            break # since 2DInstances.txt is ordered alphabetically w.r.t. scan id's we know that now no other scan will have the same scan_id again

    # search forwards from instance
    for i in range(idx+1, len(instances), 1):
        scan_id = parse_instance(instances[i])["reference"]
        if scan_id == ref_scan_id:
            instances_list.append(instances[i])
        else:
            break # since 2DInstances.txt is ordered alphabetically w.r.t. scan id's we know that now no other scan will have the same scan_id again

def sample_treshold_reached(list_of_lists):

    return all(len(l) >= sample_treshold for l in list_of_lists)

pos_minimum_visibility = [0.6, 0.6]
pos_maximum_visibility = [1.0, 1.0]

def visibility_in_range(visibility):
    """
    Check if a given visibility satisfies the range of [self.pos_minimum_visibility, self.pos_maximum_visibility].
    :param visibility: in the format as loaded from parse_instance
    :return: True or False
    """
    truncation = visibility[2]
    occlusion = visibility[5]

    return truncation >= pos_minimum_visibility[0] \
        and truncation <= pos_maximum_visibility[0] \
        and occlusion >= pos_minimum_visibility[1] \
        and occlusion <= pos_maximum_visibility[1]

scans = {}
metadata = json.load(open(os.path.join(data_path, "3RScan.json")))

def get_scan(scan):
    """
    Return the json object for the scan as defined in 3RScan.json.
    If it is a reference scan, it will return the whole reference json object with all rescans as sub-object.
    If it is a rescan scan, it will also return the whole reference json object with this rescan being a sub-object.
    :param scan: scan_id of the scan to load
    :return: json object or ValueError if scan cannot be found in 3RScan.json
    """

    # try to find in scans cache if accessed before
    result = scans.get(scan, None)
    if result is not None:
        return result

    # search for scan in 3RScan.json
    for reference in metadata:

        # search in reference
        if scan == reference["reference"]:
            result = reference
            break # early stopping in reference

        # search in rescans
        found = False # early stopping in rescans
        for rescan in reference["scans"]:
            if scan == rescan["reference"]:
                result = reference
                found = True
                break # early stopping in rescans
        if found:
            break # early stopping in rescans

    # store in cache or throw error
    if result is None:
        raise ValueError("Scan not found: ", scan)
    else:
        scans[scan] = result

    return result

def is_same_room(first_scan_id, second_scan_id):
    """
    Return if the two scan_ids reference the same room.
    :param first_scan_id:
    :param second_scan_id:
    :return: true or false
    """

    ref1_id = get_scan(first_scan_id)["reference"]
    ref2_id = get_scan(second_scan_id)["reference"]
    return ref1_id == ref2_id

def has_ambiguity(first_scan, first_instance, second_scan, second_instance):
    """
    Check if the two instances share an ambiguity that is defined in 3RScan.json
    :param first: a row of 2DInstances.txt (an entry in self.instances)
    :param second: a row of 2DInstances.txt (an entry in self.instances)
    :return: true or false
    """

    # if not same room: immediately return false: is a sanity check
    if not is_same_room(first_scan, second_scan):
        return False

    # find reference
    ref = get_scan(first_scan)

    # check if ambiguity present for this reference id and if instance numbers are defined in it
    for ambiguity in ref["ambiguity"]:
        for entry in ambiguity:
            source = int(entry["instance_source"])
            target = int(entry["instance_target"])
            if (first_instance == source and second_instance == target) or (
                    second_instance == source and first_instance == target):
                return True
    return False

def has_rigid_movement(first_scan_id, second_scan_id, instance_id):
    """
    Check if the two scan_ids have a different rigid movement w.r.t the specified instance.
    Either one of them has no rigid movement and the other one does (i.e. one is a reference scan),
    or both have rigid movements (i.e. both are rescans).
    :param first_scan_id:
    :param second_scan_id:
    :param instance_id: for which instance to check the rigid movement defined
    :return: true or false
    """

    # if not same room: immediately return false: is a sanity check
    if not is_same_room(first_scan_id, second_scan_id):
        return False

    # there can be no movement when it is the same scan: is a sanity check
    if first_scan_id == second_scan_id:
        return False

    # find reference
    ref = get_scan(first_scan_id)

    # retrieve rigid movement for instance for each scan
    rigid_first = None
    rigid_second = None
    for rescan in ref["scans"]:
        if rescan["reference"] == first_scan_id:
            for rigid in rescan["rigid"]:
                if rigid["instance_reference"] == instance_id or rigid["instance_rescan"] == instance_id:
                    rigid_first = rigid
        if rescan["reference"] == second_scan_id:
            for rigid in rescan["rigid"]:
                if rigid["instance_reference"] == instance_id or rigid["instance_rescan"] == instance_id:
                    rigid_second = rigid

    # evaluate rigid movements to judge whether there is rigid movement for this instance
    if rigid_first is None and rigid_second is None:
        return False

    if rigid_first is not None:
        if rigid_second is None:
            return True
        else:
            return rigid_first["transform"] != rigid_second["transform"]

    if rigid_second is not None:
        if rigid_first is None:
            return True
        else:
            return rigid_first["transform"] != rigid_second["transform"]

def add_instance(instance, list, instances_counter):
    if not sample_treshold_reached([list]):
        list.append(instance)
        instances_counter += 1

    return instances_counter

def find_positive_pair(anchor, instances):
    """
    Sample 4 types of positives:
        -   (SSD): Same scan, same object, different view
        -   (OSD): Other scan, same object, different view
        -   (COA): Same or other scan, same class, other object, ambiguity between anchor and other
        -   (OSR): Other scan, same object, rigid movement happened
    Then choose with a probability between the 4 types and sample from chosen type at random.
    :param anchor: a row of 2DInstances.txt (an entry in self.instances)
    :return: a row of 2DInstances.txt (an entry in self.instances) which is a positive match for anchor
    """
    # create lists as defined in documentation
    ssd = []
    osd = []
    coa = []
    osr = []

    # retrieve anchor attributes
    a = parse_instance(anchor)

    # save overall number of found instances
    instances_counter = 0

    # go through all instances and select the ones that match any of the 4 types of positives
    for instance in instances:

        # have we collected enough samples to randomly sample from?
        if sample_treshold_reached([ssd, osd, coa, osr]):
            break

        # retrieve instance attributes
        i = parse_instance(instance)

        # if this instance is not that much visible, we do not want to have it as a positive sample at all (no matter the case).
        if not visibility_in_range(i["visibility"]):
            continue

        # do some common checks helpful for all type checks
        same_room = is_same_room(a["scan"], i["scan"])
        same_label = i["label"] == a["label"]
        same_scan = i["scan"] == a["scan"]
        same_instance = i["instance_id"] == a["instance_id"]
        same_frame = i["frame_nr"] == a["frame_nr"]
        ambiguous = has_ambiguity(a["scan"], a["instance_id"], i["scan"], i["instance_id"])
        rigid_movement = has_rigid_movement(a["scan"], i["scan"], i["instance_id"])

        # do type checks
        if same_scan:
            # check SSD
            if same_instance and not same_frame:
                # matches SSD
                instances_counter = add_instance(instance, ssd, instances_counter)

            # check COA
            if same_label and not same_instance and ambiguous:
                # matches COA
                instances_counter = add_instance(instance, coa, instances_counter)

        elif same_room:
            # check OSD & OSR
            if same_instance:
                # matches OSD
                instances_counter = add_instance(instance, osd, instances_counter)
                if rigid_movement:
                    # matches OSR
                    instances_counter = add_instance(instance, osr, instances_counter)

            # check COA
            if same_label and not same_instance and ambiguous:
                # matches COA
                instances_counter = add_instance(instance, coa, instances_counter)
        # else: matches no criteria, so ignore it

    # create positives list and sample probabilities
    positives = [ssd, osd, coa, osr]

    return positives, instances_counter

def bboxes_overlapping(first_bbox, second_bbox):
    """
    Return if two axis-aligned-bboxes in format [x1, y1, x2, y2] are overlapping where
    (x1, y1): upper left corner
    (x2, y2): lower right corner
    (0, 0): origin at upper left
    :param first_bbox: first bbox to check
    :param second_bbox: second bbox to check
    :return: (True, area_of_overlap) if overlapping, (False, 0) otherwise.
    """
    first_min_x = first_bbox[0]
    first_min_y = first_bbox[1]
    first_max_x = first_bbox[2]
    first_max_y = first_bbox[3]

    second_min_x = second_bbox[0]
    second_min_y = second_bbox[1]
    second_max_x = second_bbox[2]
    second_max_y = second_bbox[3]

    dx = min(first_max_x, second_max_x) - max(first_min_x, second_min_x)
    dy = min(first_max_y, second_max_y) - max(first_min_y, second_min_y)

    if (dx >= 0) and (dy >= 0):
        return True, dx * dy
    else:
        return False, 0

def contains_instance(instance, query_id):
    """
    Return if an object with given instance_id from the "other_instances" in a frame represented by a parsed instance
    (see self.parse_instance) is overlapping with the parsed instance (w.r.t. their bboxes).
    :param instance: instance in the format defined in self.parse_instance
    :param query_id: instance_id for which to compare to the bbox. If this instance_id is not even part of the "other_instances" of this parsed instance, return False as well.
    :return: (True, area_of_overlap) if overlapping, (False, 0) otherwise.
    """
    idx_in_other_instances = [idx for idx, id in enumerate(instance["other_instance_ids"]) if id == query_id]
    # if that instance id is contained in the frame represented by this parsed instance
    if len(idx_in_other_instances) > 0:
        other_bbox = instance["other_bboxes"][idx_in_other_instances[0]]
        return bboxes_overlapping(other_bbox, instance["bbox"])
    return False, 0

neg_background_overlap_minimum = 256*256

def find_negative_pairs(anchor, instances):
    """
    Sample 5 types of negatives:
        -   (OAC): Other room, any other class
        -   (SAC): Same room, any other class
        -   (OSC): Other room, same class
        -   (SCA): Same room, other scan, same class, other instance, no ambiguity to anchor instance, rigid movement happened (due to bad ambiguity-annotation-rate in dataset for non-moved objects)
        -   (AVB): Same room, same scan, anchor no longer visible, but something in the background of anchor still is, that something is not ambiguous to anchor
    Then choose with a probability between the 5 types and sample from chosen type at random.
    :param anchor: a row of 2DInstances.txt (an entry in self.instances)
    :return: list of rows of 2DInstances.txt (an entry in self.instances) that are negative matches for anchor.
    """
    # create lists as defined in documentation
    oac = []
    sac = []
    osc = []
    sca = []
    avb = []

    # retrieve anchor attributes
    a = parse_instance(anchor)

    # save overall number of found instances
    instances_counter = 0

    # go through all instances and select the ones that match any of the 5 types of negatives
    for instance in instances:

        # have we collected enough samples to randomly sample from?
        if sample_treshold_reached([oac, sac, osc, sca, avb]):
            break

        # retrieve instance attributes
        i = parse_instance(instance)

        # do some common checks helpful for all type checks
        same_room = is_same_room(a["scan"], i["scan"])
        same_label = i["label"] == a["label"]
        same_scan = i["scan"] == a["scan"]
        same_instance = i["instance_id"] == a["instance_id"]
        anchor_in_instance, _ = contains_instance(i, a["instance_id"])
        instance_in_anchor, instance_in_anchor_overlap_area = contains_instance(a, i["instance_id"])
        instances_overlapping = anchor_in_instance or instance_in_anchor
        ambiguous = has_ambiguity(a["scan"], a["instance_id"], i["scan"], i["instance_id"])
        rigid_movement = has_rigid_movement(a["scan"], i["scan"], i["instance_id"])

        # do type checks
        if same_room:
            # check SAC
            if not same_label and not instances_overlapping:
                # matches SAC
                instances_counter = add_instance(instance, sac, instances_counter)

            # check SCA
            if not same_scan and same_label and not same_instance and not instances_overlapping and not ambiguous and rigid_movement:
                # matches SCA
                instances_counter = add_instance(instance, sca, instances_counter)

            # check AVB
            if same_scan and not anchor_in_instance and instance_in_anchor and instance_in_anchor_overlap_area > neg_background_overlap_minimum and not ambiguous:
                # matches AVB
                instances_counter = add_instance(instance, avb, instances_counter)
        else:
            # check OAC & OSC
            if not same_label:
                # matches OAC
                instances_counter = add_instance(instance, oac, instances_counter)
            else:
                # matches OSC
                instances_counter = add_instance(instance, osc, instances_counter)
        # else: matches no criteria, so ignore it

    # create negatives list
    negatives = [oac, sac, osc, sca, avb]

    return negatives, instances_counter

def find_triplet(anchor_instance, instances):
    """
    Sample positive and negative for this anchor instance.
    If no positive or negative could be found, we signal this via the first object in tuple, which is then false.
    :param anchor_instance: a row of 2DInstances.txt (an entry in self.instances)
    :return: (valid, pos, neg) tuple
    """

    all_possible_positives, pos_counter = find_positive_pair(anchor_instance, instances)

    all_possible_negatives, neg_counter = find_negative_pairs(anchor_instance, instances)

    return all_possible_positives, all_possible_negatives, pos_counter > 0 and neg_counter > 0

def calculate_probabilities(lists, probs):
    """
    For a given list of lists and probability to sample from each list, we redistribute the probability such that:
    - Only probabilities that are >0 are still >0
    - Only probabilities for a non-empty list are >0
        Will set probability to 0 for empty lists and redistribute among all non-empty lists with a probability >0
    - probabilities still sum up to the same number as before
    :param lists: list of lists
    :param probs: probabilities to sample from each list
    :return: altered probabilities to sample from each list satisfying the above criteria
    """

    # how much probability needs to be redistributed among remaining probabilities to sum to 1?
    zero_sum = np.sum([p if len(lists[i]) == 0 else 0 for i, p in enumerate(probs)])

    # set probability to 0 if no list entry present
    probs = [p if len(lists[i]) > 0 else 0 for i, p in enumerate(probs)]

    # how many probabilities are non-zero and can get an increment from the others
    non_zero_probs = np.count_nonzero(probs)

    if non_zero_probs > 0:
        # calculate the increment for each remaining probability
        increment = zero_sum / non_zero_probs

        # increment each probability that is not zero
        probs = [p + increment if p > 0 else 0 for p in probs]

    return probs

def sample_from_types(types, probs, n):
    """
    Samples n items from types where types contains k lists and each list contains a different number of items.
    A type k is selected with probability probs[k] and the resulting item is sampled uniformly from types[k].
    :param types: list of lists where types contains k lists and each list contains a different number of items
    :param probs: list of floats with len(probs) == len(types) and probs[k] is sample probability for types[k]
    :param n: how many samples to take with above procedure
    :return: (True, samples): if at least one probability is greater than zero we can return all samples.
                If n=1 then samples is just one sample, if n>1 then samples is a list of samples.
                (False, None): if all sample probabilities are zero, then we return no samples.
    """
    try:
        # select from which type to sample from for each sample
        types_per_sample = np.random.choice(len(types), n, p=probs)

        # count how many samples to select from each type
        unique, counts = np.unique(types_per_sample, return_counts=True)
        sample_count_per_type = dict(zip(unique, counts))

        # sample indices per type at random without replacement
        sample_indices_per_type = {k: np.random.choice(len(types[k]), v, replace=False) for k, v in
                                       sample_count_per_type.items()}

        # create list of samples from indices
        samples = [types[k][index] for k, v in sample_indices_per_type.items() for index in v]

        return True, samples
    except:
        return False, None

positive_sample_probabilities = [0.25, 0.25, 0.25, 0.25]
negative_sample_probabilities = [0, 0, 0, 0.5, 0.5]

def sample_positives(positives):
    probs = calculate_probabilities(positives, positive_sample_probabilities)

    valid, samples = sample_from_types(positives, probs, 1)

    if valid:
        samples = samples[0]  # we do not need a list when only ever having one positive sample at a time

    return valid, samples

number_negative_samples = 1

def sample_negatives(negatives):
    probs = calculate_probabilities(negatives, negative_sample_probabilities)

    valid, samples = sample_from_types(negatives, probs, number_negative_samples)

    return valid, samples

def sample_triplets(all_possible_positives, all_possible_negatives):
    # find positive instance: only one positive item
    valid_pos, positive_instance = sample_positives(all_possible_positives)

    # find negative instance(s): list of length self.number_negative_samples
    valid_neg, negative_instances = sample_negatives(all_possible_negatives)

    return valid_pos and valid_neg, positive_instance, negative_instances

def triplets_as_batches(batch, number_negative_samples):
        """
        Take in one batch of this dataset and concatenate all anchor, pos and neg attributes into one large image tensor.
        This is done for faster processing on the GPU instead of processing each anchor, pos and neg image tensor independently.
        We also concatenate the bboxes and labels into one large tensor / list.
        The rest of the batch is discarded. (TODO: we might change this later by creating a list for each of the other attributes if needed)
        :param batch: batched samples of this dataset
        :return: dict with format
        {
            "image": images stacked in a tensor with size <batch_size>*(2+self.number_negative_samples).
                     The first <batch_size> images correspond to the anchor images, the second to the positive images.
                     The remaining <batch_size>*self.number_negative_samples correspond to the negative images.
            "bbox": bboxes stacked in a tensor with size <batch_size>*(2+self.number_negative_samples).
                     Same ordering as for images.
            "label": labels stacked in a list with size <batch_size>*(2+self.number_negative_samples)
                     Same ordering as for images.
        }
        """

        x = batch

        # load anchor, pos, negs images and cat them into one tensor.
        # final batch_size = batch_size of incoming batch * (2+self.number_negative_samples)
        anchor_images = x["anchor"]["image"]
        pos_images = x["pos"]["image"]
        if number_negative_samples > 0:
            neg_images = tuple([x["neg"][i]["image"] for i in range(number_negative_samples)])
        else:
            neg_images = ()

        images = (anchor_images, pos_images)
        images += neg_images

        images = torch.cat(images, dim=0)

        # load anchor, pos, negs bboxes and cat each x,y,w,h into one tensor
        # final batch_size = batch_size of incoming x,y,w,h * (2+self.number_negative_samples)
        bboxes = {}
        for k in x["anchor"]["bbox"].keys():
            anchor_bbox = x["anchor"]["bbox"][k]
            pos_bbox = x["pos"]["bbox"][k]
            if number_negative_samples > 0:
                neg_bbox = tuple([x["neg"][i]["bbox"][k] for i in range(number_negative_samples)])
            else:
                neg_bbox = ()

            bboxes[k] = (anchor_bbox, pos_bbox)
            bboxes[k] += neg_bbox

            bboxes[k] = torch.cat(bboxes[k], dim=0)

        # load anchor, pos, negs labels and extend them into one list
        labels = []

        labels.extend(x["anchor"]["label"])
        labels.extend(x["pos"]["label"])
        if number_negative_samples > 0:
            for neg in x["neg"]:
                labels.extend(neg["label"])

        return {
            "image": images,
            "bbox": bboxes,
            "label": labels,
        }
    
def outputs_as_triplets(output, number_negative_samples):
        """
        Reverses the process of self.triplets_as_batches by splitting the tensors into anchor, pos and neg again.
        The incoming 'output' vector is assumed to contain the image encodings of each image only.
        Either the encodings are stored in a list with the n-th entry being the n-th encoding for one image or we only
        have one encoding per image.
        :param output: image encodings
        :return: dictionary with format
        {
            "anchor": all anchor encodings in a stacked tensor of size <batch_size>,
            "pos": all positive encodings in a stacked tensor of size <batch_size>,
            "neg": all negative encodings in a stacked tensor of size <batch_size>*self.number_negative_samples
        }
        """

        if (isinstance(output, list)):
            out = []
            for i in range(len(output)):
                out_i = output[i]
                bs = int(out_i.shape[0] / (number_negative_samples + 2))

                if number_negative_samples > 0:
                    x = torch.split(out_i, [bs, bs, number_negative_samples * bs], dim=0)
                else:
                    x = torch.split(out_i, [bs, bs], dim=0)

                out_i = {}
                out_i["anchor"] = x[0]
                out_i["pos"] = x[1]
                if number_negative_samples > 0:
                    out_i["neg"] = [x[2][i * bs:(i + 1) * bs] for i in range(number_negative_samples)]

                out.append(out_i)

            return out
        else:
            bs = int(output.shape[0] / (number_negative_samples + 2))

            if number_negative_samples > 0:
                x = torch.split(output, [bs, bs, number_negative_samples * bs], dim=0)
            else:
                x = torch.split(output, [bs, bs], dim=0)

            out = {}
            out["anchor"] = x[0]
            out["pos"] = x[1]
            if number_negative_samples > 0:
                out["neg"] = [x[2][i * bs:(i + 1) * bs] for i in range(number_negative_samples)]

            return [out]