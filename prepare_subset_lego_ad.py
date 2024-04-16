 
import os
import json
import numpy as np
import cv2


# number of extra generated test samples
k_augments = 300
# change this path to the directory you want to fill with the "new" data set
result_base_path = "MAD-Sim_Subsets"
# path to the MAD-Sim data set
mad_base_path = "MAD-Sim/"
classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

prepare_pose_dataset = True
splits_to_train = [0.8]
n_runs = 1

os.makedirs(result_base_path, exist_ok=True)

for cl in classnames:
    print(f"Processing MAD class {cl}")
    for split_to_train in splits_to_train:
        print(f"Cur subset: {split_to_train}")
            
        for run_idx in range(n_runs):
            print("run idx: ", run_idx)
            pose_class_dir = os.path.join(result_base_path, f"{split_to_train}_{run_idx}", "pose", cl)
            ano_class_dir = os.path.join(result_base_path, f"{split_to_train}_{run_idx}", "ano", cl)
            os.makedirs(pose_class_dir, exist_ok=True)
            os.makedirs(ano_class_dir, exist_ok=True)

            # copy over training_data
            orig_train_dir = os.path.join(mad_base_path, cl, "train", "good")
            new_train_dir = os.path.join(pose_class_dir, "train")
            os.makedirs(new_train_dir, exist_ok=True)
            
            # create test dir
            new_test_dir = os.path.join(pose_class_dir, "test")
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
  
                path_to_write = os.path.join(new_train_dir, f"train_{train_sample}") if sample_idx in chosen_train_idx else os.path.join(new_test_dir, f"test_{train_sample}")

                res = cv2.imwrite(path_to_write, result)
                if not res:
                    raise RuntimeError(f"Could not save transparent image to {path_to_write}")

            # load training poses
            with open(os.path.join(mad_base_path, cl, "transforms.json"), "r") as f:
                train_transforms = json.load(f)
            camera_angle_x = train_transforms["camera_angle_x"]
            
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
            with open(os.path.join(pose_class_dir, "transforms_train.json"), "w") as f:
                json.dump(new_train_transforms, f, indent=2)
            # dump the json
            with open(os.path.join(pose_class_dir, "transforms_test.json"), "w") as f:
                json.dump(test_transforms, f, indent=2)
            print(f"Done with preparing pose data set at split {split_to_train} and {len(new_train_transforms['frames'])} & {len(test_transforms['frames'])}")
            
            ######
            # Copy train split from constructed pose subset data set, while testset is just symlink to old testset
            # copy over training_data
            os.makedirs(os.path.join(ano_class_dir, "train"), exist_ok=True)
            new_train_dir = os.symlink(new_train_dir, os.path.join(ano_class_dir, "train", "good"))
            
            # create test dir
            orig_test_dir = os.path.join(mad_base_path, cl, "test")
            new_test_dir = os.symlink(orig_test_dir, os.path.join(ano_class_dir, "test"))
            
            gt = os.symlink(os.path.join(mad_base_path, cl, "ground_truth"), os.path.join(ano_class_dir, "ground_truth"))
            
            transforms = os.symlink(os.path.join(os.path.join(pose_class_dir, "transforms_train.json")), os.path.join(ano_class_dir, "transforms.json"))
                        
            print(new_train_dir, new_test_dir, gt, transforms)

