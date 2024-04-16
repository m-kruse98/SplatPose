
import numpy as np
from bisect import bisect
from scipy.ndimage.measurements import label

# Code is taken from the official reference of the MVTec evaluation code 
# found at https://www.mvtec.com/company/research/datasets/mvtec-ad 

def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by
    x- and corresponding y-values. In contrast to, e.g., 'numpy.trapz()',
    this function allows to define an upper bound to the integration range by
    setting a value x_max.

    Points that do not have a finite x or y value will be ignored with a
    warning.

    Args:
        x: Samples from the domain of the function to integrate
          Need to be sorted in ascending order. May contain the same value
          multiple times. In that case, the order of the corresponding
          y values will affect the integration with the trapezoidal rule.
        y: Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be
          determined by interpolating between its neighbors. Must not lie
          outside of the range of x.

    Returns:
        Area under the curve.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print("WARNING: Not all x and y values passed to trapezoid(...)"
              " are finite. Will continue with only the finite values.")
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after
            # np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the
            # insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between
            # the last x[ins-1] and x_max. Since we do not know the exact value
            # of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) *
                                     (x_max - x[ins - 1]) /
                                     (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction

def compute_pro(anomaly_maps, ground_truth_maps):
    """Compute the PRO curve for a set of anomaly maps with corresponding ground
    truth maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a
          real-valued anomaly score at each pixel.

        ground_truth_maps: List of ground truth maps (2D numpy arrays) that
          contain binary-valued ground truth labels for each pixel.
          0 indicates that a pixel is anomaly-free.
          1 indicates that a pixel contains an anomaly.

    Returns:
        fprs: numpy array of false positive rates.
        pros: numpy array of corresponding PRO values.
    """

    # print("Compute PRO curve...")

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    num_ok_pixels = 0
    num_gt_regions = 0

    shape = (len(anomaly_maps),
             anomaly_maps[0].shape[0],
             anomaly_maps[0].shape[1])
    fp_changes = np.zeros(shape, dtype=np.uint32)
    assert shape[0] * shape[1] * shape[2] < np.iinfo(fp_changes.dtype).max, \
        'Potential overflow when using np.cumsum(), consider using np.uint64.'

    pro_changes = np.zeros(shape, dtype=np.float64)

    for gt_ind, gt_map in enumerate(ground_truth_maps):

        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)
        num_gt_regions += n_components

        # Compute the mask that gives us all ok pixels.
        ok_mask = labeled == 0
        num_ok_pixels_in_map = np.sum(ok_mask)
        num_ok_pixels += num_ok_pixels_in_map

        # Compute by how much the FPR changes when each anomaly score is
        # added to the set of positives.
        # fp_change needs to be normalized later when we know the final value
        # of num_ok_pixels -> right now it is only the change in the number of
        # false positives
        fp_change = np.zeros_like(gt_map, dtype=fp_changes.dtype)
        fp_change[ok_mask] = 1

        # Compute by how much the PRO changes when each anomaly score is
        # added to the set of positives.
        # pro_change needs to be normalized later when we know the final value
        # of num_gt_regions.
        pro_change = np.zeros_like(gt_map, dtype=np.float64)
        for k in range(n_components):
            region_mask = labeled == (k + 1)
            region_size = np.sum(region_mask)
            pro_change[region_mask] = 1. / region_size

        fp_changes[gt_ind, :, :] = fp_change
        pro_changes[gt_ind, :, :] = pro_change

    # Flatten the numpy arrays before sorting.
    anomaly_scores_flat = np.array(anomaly_maps).ravel()
    fp_changes_flat = fp_changes.ravel()
    pro_changes_flat = pro_changes.ravel()

    # Sort all anomaly scores.
    # print(f"Sort {len(anomaly_scores_flat)} anomaly scores...")
    sort_idxs = np.argsort(anomaly_scores_flat).astype(np.uint32)[::-1]

    # Info: np.take(a, ind, out=a) followed by b=a instead of
    # b=a[ind] showed to be more memory efficient.
    np.take(anomaly_scores_flat, sort_idxs, out=anomaly_scores_flat)
    anomaly_scores_sorted = anomaly_scores_flat
    np.take(fp_changes_flat, sort_idxs, out=fp_changes_flat)
    fp_changes_sorted = fp_changes_flat
    np.take(pro_changes_flat, sort_idxs, out=pro_changes_flat)
    pro_changes_sorted = pro_changes_flat

    del sort_idxs

    # Get the (FPR, PRO) curve values.
    np.cumsum(fp_changes_sorted, out=fp_changes_sorted)
    fp_changes_sorted = fp_changes_sorted.astype(np.float32, copy=False)
    np.divide(fp_changes_sorted, num_ok_pixels, out=fp_changes_sorted)
    fprs = fp_changes_sorted

    np.cumsum(pro_changes_sorted, out=pro_changes_sorted)
    np.divide(pro_changes_sorted, num_gt_regions, out=pro_changes_sorted)
    pros = pro_changes_sorted

    # Merge (FPR, PRO) points that occur together at the same threshold.
    # For those points, only the final (FPR, PRO) point should be kept.
    # That is because that point is the one that takes all changes
    # to the FPR and the PRO at the respective threshold into account.
    # -> keep_mask is True if the subsequent score is different from the
    # score at the respective position.
    # anomaly_scores_sorted = [7, 4, 4, 4, 3, 1, 1]
    # ->          keep_mask = [T, F, F, T, T, F]
    keep_mask = np.append(np.diff(anomaly_scores_sorted) != 0, np.True_)
    del anomaly_scores_sorted

    fprs = fprs[keep_mask]
    pros = pros[keep_mask]
    del keep_mask

    # To mitigate the adding up of numerical errors during the np.cumsum calls,
    # make sure that the curve ends at (1, 1) and does not contain values > 1.
    np.clip(fprs, a_min=None, a_max=1., out=fprs)
    np.clip(pros, a_min=None, a_max=1., out=pros)

    # Make the fprs and pros start at 0 and end at 1.
    zero = np.array([0.])
    one = np.array([1.])

    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))

def compute_classification_roc(
        anomaly_maps,
        scoring_function,
        ground_truth_labels):
    """Compute the ROC curve for anomaly classification on the image level.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain
          a real-valued anomaly score at each pixel.
        scoring_function: Function that turns anomaly maps into a single
          real valued anomaly score.

        ground_truth_labels: List of integers that indicate the ground truth
          class for each input image. 0 corresponds to an anomaly-free sample
          while a value != 0 indicates an anomalous sample.

    Returns:
        fprs: List of false positive rates.
        tprs: List of correspoding true positive rates.squeue 
    """
    assert len(anomaly_maps) == len(ground_truth_labels)

    # Compute the anomaly score for each anomaly map.
    anomaly_scores = map(scoring_function, anomaly_maps)
    num_scores = len(anomaly_maps)

    # Sort samples by anomaly score. Keep track of ground truth label.
    sorted_samples = \
        sorted(zip(anomaly_scores, ground_truth_labels), key=lambda x: x[0])

    # Compute the number of OK and NOK samples from the ground truth.
    ground_truth_labels_np = np.array(ground_truth_labels)
    num_nok = ground_truth_labels_np[ground_truth_labels_np != 0].size
    num_ok = ground_truth_labels_np[ground_truth_labels_np == 0].size

    # Initially, every NOK sample is correctly classified as anomalous
    # (tpr = 1.0), and every OK sample is incorrectly classified as anomalous
    # (fpr = 1.0).
    fprs = [1.0]
    tprs = [1.0]

    # Keep track of the current number of false and true positive predictions.
    num_fp = num_ok
    num_tp = num_nok

    # Compute new true and false positive rates when successively increasing
    # the threshold.
    next_score = None

    for i, (current_score, label) in enumerate(sorted_samples):

        if label == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        if i < num_scores - 1:
            next_score = sorted_samples[i + 1][0]
        else:
            next_score = None  # end of list

        if (next_score != current_score) or (next_score is None):
            fprs.append(num_fp / num_ok)
            tprs.append(num_tp / num_nok)

    # Return (FPR, TPR) pairs in increasing order.
    fprs = fprs[::-1]
    tprs = tprs[::-1]

    return fprs, tprs



def calculate_au_pro_au_roc(gt_filenames,
                            prediction_filenames,
                            integration_limit=0.3):
    """Compute the area under the PRO curve for a set of ground truth images
    and corresponding anomaly images.

    In addition, the function computes the area under the ROC curve for image
    level classification.

    Args:
        gt_filenames: List of filenames that contain the ground truth images
          for a single dataset object.
        prediction_filenames: List of filenames that contain the corresponding
          anomaly images for each ground truth image.
        integration_limit: Integration limit to use when computing the area
          under the PRO curve. in (0,1]

    Returns:
        au_pro: Area under the PRO curve computed up to the given integration
          limit.
        au_roc: Area under the ROC curve.
        pro_curve: PRO curve values for localization (fpr,pro).
        roc_curve: ROC curve values for image level classifiction (fpr,tpr).
    """
    # Read all ground truth and anomaly images.
    # ground_truth = []
    # predictions = []

    # print("Read ground truth files and corresponding predictions...")
    # for (gt_name, pred_name) in tqdm(zip(gt_filenames, prediction_filenames),
    #                                  total=len(gt_filenames)):
    #     prediction = util.read_tiff(pred_name)
    #     predictions.append(prediction)
    #
    #     if gt_name is not None:
    #         ground_truth.append(np.asarray(Image.open(gt_name)))
    #     else:
    #         ground_truth.append(np.zeros(prediction.shape))


    # NOTE: the way i use this function, the predictions and ground truth are already provided in proper
    #       numpy array form
    predictions = prediction_filenames
    ground_truth = gt_filenames

    # Compute the PRO curve.
    # MK: here we need the list of 2D anomaly map predictions, as well as the 2D GROUND TRUTH
    pro_curve = compute_pro(
        anomaly_maps=predictions,
        ground_truth_maps=ground_truth)

    # Compute the area under the PRO curve.
    au_pro = trapezoid(
        pro_curve[0], pro_curve[1], x_max=integration_limit)
    au_pro /= integration_limit

    # Derive binary labels for each input image:
    # (0 = anomaly free, 1 = anomalous).
    binary_labels = [int(np.any(x > 0)) for x in ground_truth]
    del ground_truth

    # Compute the classification ROC curve.
    roc_curve = compute_classification_roc(
        anomaly_maps=predictions,
        scoring_function=np.max,
        ground_truth_labels=binary_labels)

    # Compute the area under the classification ROC curve.
    au_roc = trapezoid(roc_curve[0], roc_curve[1])
    # print(f"Image-level classification AU-ROC: {au_roc}")

    # Return the evaluation metrics.
    return au_pro, au_roc, pro_curve, roc_curve