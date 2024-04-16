
from gaussian_splatting.train import *
import torch
import numpy as np
import wandb

# This code can be used to run quantitative experiments for the pose estmation
# NOTE: You need to first construct a pose data set using prepare_subset_lego_ad.py


classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]


pre_parser = ArgumentParser(description="Parameters of the LEGO training run")
pre_parser.add_argument("-k", metavar="K", type=int, help="number of pose estimation steps", default=175)
pre_parser.add_argument("-c", "-classname", metavar="c", type=str, help="current class to run experiments on",
                        default="01Gorilla")
pre_parser.add_argument("-wandb_config", metavar="WC", type=str, help="the wandb config to use", default="None")
pre_parser.add_argument("-p", "-prefix", metavar="pf", type=str, help="prefix for the wandb run name", default="to_delete")
pre_parser.add_argument("-seed", type=int, help="seed for random behavior", default=0)
pre_parser.add_argument("-gauss_iters", type=int, help="number of training iterations for 3DGS", default=30000)
pre_parser.add_argument("-wandb", type=int, help="whether we track with wandb", default=0)
pre_parser.add_argument("-train", type=int, help="whether we train or look for a saved model", default=1)
pre_parser.add_argument("-data_path", type=str, help="path pointing towards the usable data set", default="MAD-Sim_Subsets/0.8_0/pose")                        


lego_args = pre_parser.parse_args()
data_base_dir = lego_args.data_path
config = {
    "k" : lego_args.k,
    "classname" : lego_args.c,
    "seed" : lego_args.seed,
    "3dgs_iters" : lego_args.gauss_iters,
    "prefix" : lego_args.p,
    "wandb" : lego_args.wandb,
    "train" : lego_args.train,
    "datadir" : data_base_dir
}

projectname = config["prefix"]
if config["wandb"] != 0:
    run = wandb.init(project=projectname, config=config, name=f"{config['prefix']}_{config['classname']}")

data_path = os.path.join(data_base_dir, f"{config['classname']}")
result_dir = os.path.join(data_base_dir, f"results_{config['prefix']}_{config['seed']}", f"{config['classname']}")

print("saving model to: ", result_dir)
os.makedirs(result_dir, exist_ok=True)

if config["train"] != 0:
    # Set up command line argument parser
    training_args = ["-w", "--eval", "-s", data_path, "-m", result_dir, "--iterations", str(config["3dgs_iters"]), "--sh_degree", "0"]
    parser = ArgumentParser(description="3DGS Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, config["3dgs_iters"]])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[config["3dgs_iters"]])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(training_args)
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet, config["seed"])
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
else:
    print("skipping training!")
    
from pose_estimation import evaluate_pose_estimation

return_dict = evaluate_pose_estimation(cur_class=config["classname"], model_dir_location=result_dir,
                                       k=int(config["k"]), verbose=True)

translation_errors = return_dict["translation_error"]
rotation_errors = return_dict["rotation_error"]

rot_error_other1 = return_dict["rotation_error_other1"]
rot_error_other2 = return_dict["rotation_error_other2"]

coarse_trans_error = return_dict["coarse_t_error"]
coarse_rot_error = return_dict["coarse_rot_error"]

result_dict = {
        "coarse_translation" : np.mean(coarse_trans_error),
        "coarse_rotation" : np.mean(coarse_rot_error),
        "translation_mean" : np.mean(translation_errors),
        "translation_median" : np.median(translation_errors),
        "translation_std" : np.std(translation_errors),
        "rotation_mean" : np.mean(rotation_errors),
        "rotation_median" : np.median(rotation_errors),
        "rotation_std" : np.std(rotation_errors),
        "rotation_other1" : np.mean(rot_error_other1),
        "rotation_other2" : np.mean(rot_error_other2),
    }

for k in result_dict.keys():
    print(f"{k}: {result_dict[k]}")

if config["wandb"] != 0:
    wandb.log(result_dict)



