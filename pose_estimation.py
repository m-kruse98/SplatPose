import os

from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene import DiffGaussianModel
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.render import *
from gaussian_splatting.scene.cameras import Camera

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils_pose_est import DefectDataset, pose_retrieval_loftr, camera_transf

classnames = ["01Gorilla", "02Unicorn", "03Mallard", "04Turtle", "05Whale", "06Bird", "07Owl", "08Sabertooth",
              "09Swan", "10Sheep", "11Pig", "12Zalika", "13Pheonix", "14Elephant", "15Parrot", "16Cat", "17Scorpion",
              "18Obesobeso", "19Bear", "20Puppy"]

def main_pose_estimation(cur_class, model_dir_location, k=150, verbose=False, data_dir=None):
    
    model_dir = model_dir_location
    data_dir = "MAD-Sim/" if data_dir is None else data_dir
    trainset = DefectDataset(data_dir, cur_class, "train", True, True)

    train_imgs = torch.cat([a[0][None,...] for a in trainset], dim=0)
    train_poses = np.concatenate([np.array(a["transform_matrix"])[None,...] for a in trainset.camera_transforms["frames"]])
    train_imgs = torch.movedim(torch.nn.functional.interpolate(train_imgs, (400,400)), 1, 3).numpy()
    
    testset = DefectDataset(data_dir, cur_class, "test", True, True)
    camera_angle_x = trainset.camera_angle

    # Set up command line argument parser
    eval_args = ["-w", "--eval", "-m", model_dir]
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser, my_cmdline=eval_args)


    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if verbose:
        save_to = os.path.join(model_dir_location, "3dgs_imgs")
        os.makedirs(save_to, exist_ok=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    normal_images = list()
    reference_images = list()
    all_labels = list()
    gt_masks = list()
    times = list()
    
    print("STARTING POSE ESTIMATION")
    
    for i in tqdm(range(len(testset))):
        cur_path = testset.images[i].split("/")
        filename = f"{cur_path[-2]}_{cur_path[-1]}.png"

        set_entry = testset[i]
        
        all_labels.append(set_entry[1])
        
        gt_masks.append(set_entry[2].cpu().numpy())
        obs_img = torch.movedim(torch.nn.functional.interpolate(set_entry[0][None,...], (400, 400)).squeeze(), 0, 2)
        c2w_init = pose_retrieval_loftr(train_imgs * 255, obs_img.numpy() * 255, train_poses)

        start.record()
        
        c2w_init[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w_init)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        c2w_init[:3,:3] = R
        c2w_init[:3,3] = T

        c2w_init = torch.from_numpy(c2w_init).type(torch.float).to("cuda")

        cam_transf = camera_transf().to("cuda")
        cam_transf.train()
        optimizer = torch.optim.Adam(cam_transf.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        gaussians = DiffGaussianModel(dataset.sh_degree, c2w_init, cam_transf)
        gaussians.load_ply(os.path.join(model_dir,
                                        "point_cloud",
                                        "iteration_" + str(30000),
                                        "point_cloud.ply"))
        init_image = None

        for iters in range(k):
            optimizer.zero_grad()
            cur_view = Camera(colmap_id=123, R=c2w_init[:3,:3].cpu().numpy(), T=c2w_init[:3,3].cpu().numpy(),
                            FoVx=camera_angle_x, FoVy=camera_angle_x,
                            image=set_entry[0], gt_alpha_mask=None, image_name="aha", uid=123)
            rendering = render(cur_view, gaussians, pipeline, background)["render"]

            if init_image is None:
                init_image = torch.clone(rendering).cpu().detach()

            gt_image = set_entry[0].to("cuda")
            loss = 0.8 * l1_loss(rendering, gt_image) + 0.2 * (1 - ssim(rendering, gt_image))
            loss.backward()
          
            optimizer.step()
            
            new_lrate = 0.01 * (0.8 ** ((iters + 1) / 100))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            if iters == k - 1:
                if verbose:
                    
                    cur_save = os.path.join(save_to, filename.split(".")[0])
                    os.makedirs(cur_save, exist_ok=True)
                    torchvision.utils.save_image(set_entry[0].cpu().detach(), os.path.join(cur_save, "gt.png"))
                    torchvision.utils.save_image(init_image, os.path.join(cur_save, "first_pose.png"))
                    torchvision.utils.save_image(rendering.cpu().detach(), os.path.join(cur_save, "result.png"))
                    
                    fig, axs = plt.subplots(2,2, figsize=(10, 6.4))
                    axs[0, 0].set_title("original image"), axs[0, 1].set_title("first pose")
                    axs[1,0].set_title(f"iteration {iters}"), axs[1,1].set_title(f"diff")
                    axs[0,0].imshow(torch.movedim(set_entry[0], 0, 2).cpu().detach())
                    axs[0,1].imshow(torch.movedim(init_image, 0, 2))
                    axs[1,0].imshow(torch.movedim(rendering, 0, 2).cpu().detach())
                    axs[1,1].imshow(torch.movedim(torch.abs(rendering - set_entry[0].to("cuda")), 0, 2).sum(dim=2).cpu().detach())
                    fig.savefig(os.path.join(save_to, filename))
                    plt.close(fig)

                normal_images.append(set_entry[0].cpu().detach())
                reference_images.append(rendering.cpu().detach())

        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        
    print("all labels:", all_labels, len(all_labels))
    for i in range(5):
        print(f"norm: {normal_images[i].shape}. ref: {reference_images[i].shape}. gt: {gt_masks[i].shape}")
    
    assert len(normal_images) == len(reference_images) == len(testset), f"Wrongly sized sets!" \
                                                                         f"{len(normal_images)}. {len(reference_images)}. {len(testset)}"
    assert len(normal_images) == len(gt_masks), f"Wrongly sized sets! {len(normal_images)}. {len(gt_masks)}"
    return normal_images, reference_images, all_labels, gt_masks, times
 
 
def evaluate_pose_estimation(cur_class, model_dir_location, k=150, verbose=False):
    
    model_dir = model_dir_location
    
    from utils_pose_est import DatasetPose, pose_retrieval_loftr, camera_transf, matrix_to_quaternion
    import numpy as np
    import wandb

    
    if cur_class in classnames:
        data_dir = "MAD-Sim_Subsets/0.8_0/"
    else:
        data_dir = "nerf_synthetic/"
    
    trainset = DatasetPose(data_dir, cur_class, "train", True)
    train_imgs = torch.cat([a[0][None,...] for a in trainset], dim=0)
    train_poses = np.concatenate([np.array(a["transform_matrix"])[None,...] for a in trainset.camera_transforms["frames"]])
    train_imgs = torch.movedim(torch.nn.functional.interpolate(train_imgs, (400,400)), 1, 3).numpy()
    
    testset = DatasetPose(data_dir, cur_class, "test", True)    
    
    camera_angle_x = trainset.camera_angle

    all_labels = list()

    # Set up command line argument parser
    eval_args = ["-w", "--eval", "-m", model_dir]
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser, my_cmdline=eval_args)

    dataset = model.extract(args)
    pipeline = pipeline.extract(args)
    bg_color = [1,1,1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    if verbose:
        save_to = os.path.join(model_dir, "img_results")
        os.makedirs(save_to, exist_ok=True)

    translation_errors = list()
    quat_differences = list()
    quat_other1 = list()
    quat_other2 = list()
    coarse_t_error = list()
    coarse_rot_error = list()
    
    print("STARTING POSE ESTIMATION")
    
    for i in tqdm(range(len(testset))):
        
        cur_path = testset.images[i].split("/")
        filename = f"{cur_path[-2]}_{cur_path[-1]}"
        set_entry = testset[i]
                        
        gt_pose = set_entry[2]
        gt_trans = gt_pose[4:]
        gt_quat = torch.nn.functional.normalize(gt_pose[:4], dim=0)          
        all_labels.append(set_entry[1])
        
        # grab initial pose
        obs_img = torch.movedim(torch.nn.functional.interpolate(set_entry[0][None,...], (400, 400)).squeeze(), 0, 2)
        c2w_init = pose_retrieval_loftr(train_imgs * 255, obs_img.numpy() * 255, train_poses) # my_init_poses[i] # 
        orig_init = torch.from_numpy(np.copy(c2w_init))
        
        # prepare pose for input into gaussian splatting
        c2w_init[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w_init)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        c2w_init[:3,:3] = R
        c2w_init[:3,3] = T
        c2w_init = torch.from_numpy(c2w_init).type(torch.float).to("cuda")
        
        # set up learnable SE(3) transform and splatting model
        cam_transf = camera_transf().to("cuda")
        cam_transf.train()
        gaussians = DiffGaussianModel(dataset.sh_degree, c2w_init, cam_transf)
        gaussians.load_ply(os.path.join(model_dir,
                                        "point_cloud",
                                        "iteration_" + str(30000),
                                        "point_cloud.ply"))
        optimizer = torch.optim.Adam(cam_transf.parameters(), lr=0.001, betas=(0.9, 0.999))
        init_image = None
        
        # calculate errors of coarse pose estimation
        coarse_t_error.append(torch.sqrt(torch.sum((gt_trans - orig_init[:3,3]) ** 2)).item())
        init_quat = torch.nn.functional.normalize(matrix_to_quaternion(orig_init[:3,:3]), dim=0).type(torch.float)
        coarse_rot_error.append(torch.arccos(torch.abs(torch.dot(gt_quat, init_quat))).item())        
           
        for iters in range(k):
            optimizer.zero_grad()
            cur_view = Camera(colmap_id=123, R=c2w_init[:3,:3].cpu().numpy(), T=c2w_init[:3,3].cpu().numpy(),
                            FoVx=camera_angle_x, FoVy=camera_angle_x,
                            image=set_entry[0], gt_alpha_mask=None, image_name="aha", uid=123)
            rendering = render(cur_view, gaussians, pipeline, background)["render"]


            if init_image is None:
                init_image = torch.clone(rendering).cpu().detach()

            gt_image = set_entry[0].to("cuda")
            
            loss = 0.8 * l1_loss(rendering, gt_image) + 0.2 * (1 - ssim(rendering, gt_image))
            loss.backward()
            optimizer.step()
            
            new_lrate = 0.01 * (0.8 ** ((iters + 1) / 100))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
                
                
            if iters == k - 1:
                # construct estimated SE(3) transform
                estimation = torch.zeros((4,4), dtype=torch.float32)
                estimation[:3,:3] = gaussians.R
                estimation[:3,3] = gaussians.T
                estimation[3,3] = 1.0
                                
                # apply my estimated SE(3) transform to the original initial pose 
                est_t = estimation.detach() @ orig_init.type(torch.float32)
                trans_error = torch.sqrt(torch.sum((gt_trans - est_t[:3,3]) ** 2))
                
                est_quat = torch.nn.functional.normalize(matrix_to_quaternion(est_t[:3,:3]), dim=0)
                
                # measured as ph_1 from "Metrics for 3D Rotations: Comparison and Analysis"
                # which ranges from [0,sqrt(2)]
                norm_of_diff = min(torch.linalg.vector_norm(gt_quat - est_quat, ord=2),
                                   torch.linalg.vector_norm(gt_quat + est_quat, ord=2))
                # phi_2, this one is used for the SplatPose paper                
                quat_diff = torch.arccos(min(torch.abs(torch.dot(gt_quat, est_quat)), torch.tensor(1.0)))
                # phi_3
                quat_diff_2 = 1 - torch.abs(torch.dot(gt_quat, est_quat))
                
                translation_errors.append(trans_error.item())
                quat_differences.append(norm_of_diff)
                quat_other1.append(quat_diff)
                quat_other2.append(quat_diff_2)
                                
                
                if verbose:
                    cur_save = os.path.join(save_to, filename.split(".")[0])
                    os.makedirs(cur_save, exist_ok=True)
                    torchvision.utils.save_image(set_entry[0].cpu().detach(), os.path.join(cur_save, "gt.png"))
                    torchvision.utils.save_image(init_image, os.path.join(cur_save, "first_pose.png"))
                    torchvision.utils.save_image(rendering.cpu().detach(), os.path.join(cur_save, "result.png"))
                    continue
                    
    ret_dict = {
        "translation_error" : translation_errors,
        "rotation_error" : quat_differences,
        "rotation_error_other1" : quat_other1,
        "rotation_error_other2" : quat_other2,
        "coarse_t_error" : coarse_t_error,
        "coarse_rot_error" : coarse_rot_error
    }
    
    my_data = [[testset.images[i], coarse_t_error[i], coarse_rot_error[i], translation_errors[i], quat_differences[i], quat_other1[i], quat_other2[i]] for i in range(len(testset))]
    columns = ["path", "coarse_t", "coarse_rot", "trans", "rot_1", "rot_2", "rot_3"]
    cur_table = wandb.Table(data=my_data, columns=columns)
    # wandb.log({"pose_est": cur_table})
    
    assert all([len(a) == len(testset) for a in ret_dict.values()]), "Wrongly sized lists of errors!"
    return ret_dict
    