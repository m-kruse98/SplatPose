 
import os
import json
import numpy as np
from sklearn.metrics import pairwise_distances
from PIL import Image
import cv2

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def generate_samples(n_samples, reference_translations, distance_factor=0.8, deviation_factor=4):
    """
    Generates new points by:
        - Calculating the middle of all reference translation positions
        - Estimates the average distance from the middle to all other points
        - According to estimates, samples n new distances, which are placed closer to the middle than the reference
        - Samples n direction vectors in all possible directions
        - Each sample marches from the middle in the sampled direction until the sampled distance is reached
    """
    middle = np.mean(reference_translations, axis=0)
    # distances from middle to train points
    distances = np.linalg.norm(reference_translations - middle, axis=1)
    mu, sigma = np.mean(distances), np.std(distances)
    print(f"Sampling with mu: {mu:1.3f} * {distance_factor} = {(mu * distance_factor):1.3f} " \
            f"and sigma: {sigma:1.3f} * {deviation_factor} = {(sigma * deviation_factor):1.3f}")
    # sampled distances of our new points to the middle
    sample_dists = np.random.normal(mu * distance_factor,
                                    sigma * deviation_factor, size=n_samples)    
    # random directional vectors in R^3  
    direction = (np.random.rand(n_samples, 3) * 2) - 1
    dist_to_dir = np.linalg.norm((middle - (middle + direction)), axis=1)
    direction_multiplier = sample_dists / dist_to_dir
    # march from middle_point in the sampled direction until distance is reached
    new_points = middle + np.broadcast_to(direction_multiplier[...,None], (n_samples, 3)) * direction
    return new_points


def fix_mad_filenames(path_to_mad, classnames):
    # Fixes the filenames to include trailing zeros. This avoids ambiguous data loading in later stages 
    def regular_loop(cur_dir):
        for entry in os.listdir(cur_dir):
            cur_index = int(entry.split(".")[0])
            os.rename(os.path.join(cur_dir, entry), os.path.join(cur_dir, f"{cur_index:03d}.png"))
    def masks_loop(cur_dir):
        for entry in os.listdir(cur_dir):
            cur_index = int(entry.split("_")[0])
            os.rename(os.path.join(cur_dir, entry), os.path.join(cur_dir, f"{cur_index:03d}_mask.png"))
             
    for cl in classnames:
        cur_class_path = os.path.join(path_to_mad, cl)
        
        train_dir = os.path.join(cur_class_path, "train", "good")
        test_dir_good = os.path.join(cur_class_path, "test", "good")
        test_dir_burr = os.path.join(cur_class_path, "test", "Burrs")
        test_dir_miss = os.path.join(cur_class_path, "test", "Missing")
        test_dir_stain = os.path.join(cur_class_path, "test", "Stains")
        mask_dir_burr = os.path.join(cur_class_path, "ground_truth", "Burrs")
        mask_dir_miss = os.path.join(cur_class_path, "ground_truth", "Missing")
        mask_dir_stain = os.path.join(cur_class_path, "ground_truth", "Stains")
        test_dirs = [test_dir_good, test_dir_burr, test_dir_miss, test_dir_stain]
        mask_dirs = [mask_dir_stain, mask_dir_burr, mask_dir_miss]
        
        regular_loop(train_dir)
        for d in test_dirs:
            regular_loop(d)
        for d in mask_dirs:
            masks_loop(d)
        

# number of extra generated test samples
k_augments = 5
# change this path to the directory you want to fill with the "new" data set
result_base_path = "MAD-Sim_3dgs"
# path to the MAD-Sim data set
mad_base_path = "MAD-Sim/"
classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]
fix_mad_filenames(mad_base_path, classnames)

prepare_pose_dataset = False
split_to_train = 1.0

os.makedirs(result_base_path, exist_ok=True)

for cl in classnames:
    
    print(f"Processing MAD class {cl}")
    class_dir = os.path.join(result_base_path, cl)
    os.makedirs(class_dir, exist_ok=True)

    # copy over training_data
    orig_train_dir = os.path.join(mad_base_path, cl, "train", "good")
    new_train_dir = os.path.join(class_dir, "train")
    os.makedirs(new_train_dir, exist_ok=True)
    
    # create test dir
    new_test_dir = os.path.join(class_dir, "test")
    os.makedirs(new_test_dir, exist_ok=True)
    
    n_train_samples = len(os.listdir(orig_train_dir))
    
    # sample indices in case we want to split the train set
    chosen_train_idx = np.random.choice(n_train_samples, int(n_train_samples * split_to_train), replace=False)
    chosen_test_idx = np.array([a for a in range(n_train_samples) if a not in chosen_train_idx])
        
    train_samples = sorted(os.listdir(orig_train_dir), key=lambda x : int(x.split("_")[-1].split(".")[0]))
    
    for sample_idx, train_sample in enumerate(train_samples):

        # TODO: determine whether I really need to work with transparent Images
        #       white background and the "-w" flag should probably work just as well
        img = cv2.imread(os.path.join(orig_train_dir, train_sample))
        mask = np.abs((cv2.threshold(img[:,:,1], 254, 1, cv2.THRESH_BINARY)[1]) - 1).astype(np.uint8)
        result = np.dstack((img, mask))

        # if we create pose dataset. split the training set, else write everything to the new one
        if prepare_pose_dataset:
            path_to_write = os.path.join(new_train_dir, f"train_{train_sample}") if sample_idx in chosen_train_idx else os.path.join(new_test_dir, f"test_{train_sample}")
        else:
            path_to_write = os.path.join(new_train_dir, f"train_{train_sample}")
        res = cv2.imwrite(path_to_write, result)
        if not res:
            raise RuntimeError(f"Could not save transparent image to {path_to_write}")

    # load training poses
    with open(os.path.join(mad_base_path, cl, "transforms.json"), "r") as f:
        train_transforms = json.load(f)
    camera_angle_x = train_transforms["camera_angle_x"]
    
    if prepare_pose_dataset:
        test_transforms = {
        "camera_angle_x" : camera_angle_x,
        "frames" : []
        }
        new_train_transforms = {
            "camera_angle_x" : camera_angle_x,
            "frames" : []
        }
        
        for sample_idx in range(n_train_samples):
            cur_num = int(train_transforms["frames"][sample_idx]["file_path"].split("/")[-1].split(".")[0])
            is_train = sample_idx in chosen_train_idx
            frame_entry = {
                "file_path" : f"./train/train_{cur_num:03d}" if is_train else f"./test/test_{cur_num:03d}",
                "transform_matrix" : train_transforms["frames"][sample_idx]["transform_matrix"]
            }
            (new_train_transforms if is_train else test_transforms)["frames"].append(frame_entry)
        # dump training poses back to the data set
        with open(os.path.join(class_dir, "transforms_train.json"), "w") as f:
            json.dump(new_train_transforms, f, indent=2)
        # dump the json
        with open(os.path.join(class_dir, "transforms_test.json"), "w") as f:
            json.dump(test_transforms, f, indent=2)
        print(f"Done with preparing pose data set at split {split_to_train} and {len(new_train_transforms['frames'])} & {len(test_transforms['frames'])}")
        continue
                
    
    training_poses = list()
    # refactor training filepaths
    for frame in train_transforms["frames"]:
        cur_num = int(frame["file_path"].split("/")[-1].split(".")[0])
        frame["file_path"] = f"./train/train_{cur_num:03d}"
        training_poses.append(np.array(frame["transform_matrix"])[None,...])
    
    # dump training poses back to the data set
    with open(os.path.join(class_dir, "transforms_train.json"), "w") as f:
        json.dump(train_transforms, f, indent=2)

    # gather training pose information
    training_poses = np.concatenate(training_poses, axis=0)
    all_translations = training_poses[:,:3,3]
    all_rotations = training_poses[:,:3,:3]
    mean_point = np.mean(all_translations, axis=0)

    # generate test poses (translation vectors for now)
    test_translations = generate_samples(k_augments, all_translations)
    test_translations_2 = generate_samples(k_augments, all_translations, 1.2)
    test_translations = np.concatenate((test_translations, test_translations_2), axis=0)
    # for each test sample grab the closest rotations and translations in the trainset
    distances = pairwise_distances(X=test_translations, Y=all_translations).argmin(axis=1)
    closest_translations = all_translations[distances]
    closest_rotations = all_rotations[distances]

    test_transforms = {
        "camera_angle_x" : camera_angle_x,
        "frames" : []
    }
    test_poses = np.zeros((test_translations.shape[0], 4, 4))

    empty_image = Image.fromarray(np.ones((800,800), dtype=np.uint8) * 255)

    for idx in range(test_translations.shape[0]):

        # calculate rotation from given translation
        cur_vec = test_translations[idx]
        base_vec = closest_translations[idx] - mean_point
        # NOTE: original procedure where we first rotate /w nearest available rotation and then to our point from
        #       there to our desired point
        rot = closest_rotations[idx]
        rot = rotation_matrix_from_vectors(base_vec, cur_vec)
        rot = rot @ closest_rotations[idx]
        
        test_poses[idx, 3,3] = 1
        test_poses[idx, :3,3] = cur_vec
        test_poses[idx, :3,:3] = rot
        # append to transforms json
        test_transforms["frames"].append(
            {
                "file_path" : f"./test/test_{idx:03d}",
                "transform_matrix" : test_poses[idx].tolist()
            }
        )
        empty_image.save(os.path.join(class_dir, "test", f"test_{idx:03d}.png"))

    # dump the json
    with open(os.path.join(class_dir, "transforms_test.json"), "w") as f:
        json.dump(test_transforms, f, indent=2)

    print(f"Done!\n")
