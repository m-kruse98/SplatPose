import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from scipy.ndimage.morphology import binary_dilation, binary_opening
from copy import deepcopy
from PAD_utils.loftr import LoFTR, default_cfg
import cv2

dilate_size = 8
img_size = 800
img_len = img_size
map_len = 24
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

import torch.nn as nn
import copy
import importlib


def qvec2rotmat(qvec):
    return torch.tensor(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def quaternion_invert(q: torch.Tensor) -> torch.Tensor:
    scaling = torch.as_tensor([1, -1, -1, -1], device=q.device)
    return q * scaling


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        """
        Returns torch.sqrt(torch.max(0, x))
        but with a zero subgradient where x is 0.
        """
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret
    def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert a unit quaternion to a standard form: one in which the real
        part is non negative.

        Args:
            quaternions: Quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Standardized quaternions as tensor of shape (..., 4).
        """
        return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def vec2ss_matrix(vector):  # vector to skewsym. matrix

    ss_matrix = torch.zeros((3,3)).to("cuda")
    ss_matrix[0, 1] = -vector[2]
    ss_matrix[0, 2] = vector[1]
    ss_matrix[1, 0] = vector[2]
    ss_matrix[1, 2] = -vector[0]
    ss_matrix[2, 0] = -vector[1]
    ss_matrix[2, 1] = vector[0]

    return ss_matrix

class camera_transf(nn.Module):
    def __init__(self):
        super(camera_transf, self).__init__()
        self.w = nn.Parameter(torch.normal(0, 1e-6, size=(3,)), requires_grad=True)
        self.v = nn.Parameter(torch.normal(0, 1e-6, size=(3,)), requires_grad=True)
        self.theta = nn.Parameter(torch.normal(0, 1e-6, size=()), requires_grad=True)
        self.eye = torch.eye(3, requires_grad=False).to("cuda")
        # self.ones = torch.ones(3, requires_grad=False).to("cuda")

    def forward(self):
        w_skewsym = vec2ss_matrix(self.w)
        # Proposition 3.11 in Modern Robotics (p. 84)
        R = self.eye + torch.sin(self.theta) * w_skewsym + (1 - torch.cos(self.theta)) * torch.matmul(w_skewsym, w_skewsym)
        # Proposition 3.25 in Modern Robotics (p.105)
        T = torch.matmul(self.eye * self.theta + (1 - torch.cos(self.theta)) * w_skewsym + (self.theta - torch.sin(self.theta)) * torch.matmul(w_skewsym, w_skewsym), self.v)

        r_quat = torch.nn.functional.normalize(matrix_to_quaternion(R[None,...]), dim=1) 
        return R, T, r_quat
    


backbone_info = {
    "resnet18": {
        "layers": [1, 2, 3, 4],
        "planes": [64, 128, 256, 512],
        "strides": [4, 8, 16, 32],
    },
    "resnet34": {
        "layers": [1, 2, 3, 4],
        "planes": [64, 128, 256, 512],
        "strides": [4, 8, 16, 32],
    },
    "resnet50": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
    "resnet101": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
    "wide_resnet50_2": {
        "layers": [1, 2, 3, 4],
        "planes": [256, 512, 1024, 2048],
        "strides": [4, 8, 16, 32],
    },
    "efficientnet_b0": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [0, 2, 4, 10, 15],
        "planes": [16, 24, 40, 112, 320],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b1": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [1, 4, 7, 15, 22],
        "planes": [16, 24, 40, 112, 320],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b2": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [1, 4, 7, 15, 22],
        "planes": [16, 24, 48, 120, 352],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b3": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [1, 4, 7, 17, 25],
        "planes": [24, 32, 48, 136, 384],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b4": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [1, 5, 9, 21, 31],
        "planes": [24, 32, 56, 160, 448],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b5": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [2, 7, 12, 26, 38],
        "planes": [24, 40, 64, 176, 512],
        "strides": [2, 4, 8, 16, 32],
    },
    "efficientnet_b6": {
        "layers": [1, 2, 3, 4, 5],
        "blocks": [2, 8, 14, 30, 44],
        "planes": [32, 40, 72, 200, 576],
        "strides": [2, 4, 8, 16, 32],
    },
}

class ModelHelper(torch.nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            print(mtype, kwargs)
            module = self.build(mtype, kwargs)
            self.add_module(mname, module)
            break

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        input = copy.copy(input)
        if input.device != self.device:
            # input = to_device(input, device=self.device)
            input=input.cuda()
        for submodule in self.children():
            output = submodule(input)
            # input.update(output)
        feat=[]
        size=(224,224)
        return output['features']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

def update_config(config):
    # update feature size
    _, reconstruction_type = config.net[2].type.rsplit(".", 1)
    if reconstruction_type == "UniAD":
        input_size = config.dataset.input_size
        outstride = config.net[1].kwargs.outstrides[0]
        assert (
            input_size[0] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        assert (
            input_size[1] % outstride == 0
        ), "input_size must could be divided by outstrides exactly!"
        feature_size = [s // outstride for s in input_size]
        config.net[2].kwargs.feature_size = feature_size

    # update planes & strides
    backbone_path, backbone_type = config.net[0].type.rsplit(".", 1)
    # TODO: currently commented out by me and instead the backbone_info is copied into this file
    # module = importlib.import_module(backbone_path)
    # backbone_info = getattr(module, "backbone_info")
    backbone = backbone_info[backbone_type]
    outblocks = None
    if "efficientnet" in backbone_type:
        outblocks = []
    outstrides = []
    outplanes = []
    for layer in config.net[0].kwargs.outlayers:
        if layer not in backbone["layers"]:
            raise ValueError(
                "only layer {} for backbone {} is allowed, but get {}!".format(
                    backbone["layers"], backbone_type, layer
                )
            )
        idx = backbone["layers"].index(layer)
        if "efficientnet" in backbone_type:
            outblocks.append(backbone["blocks"][idx])
        outstrides.append(backbone["strides"][idx])
        outplanes.append(backbone["planes"][idx])
    if "efficientnet" in backbone_type:
        config.net[0].kwargs.pop("outlayers")
        config.net[0].kwargs.outblocks = outblocks
    config.net[0].kwargs.outstrides = outstrides
    config.net[1].kwargs.outplanes = [sum(outplanes)]

    return config


def pose_retrieval_loftr(imgs,obs_img,poses):
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("PAD_utils/model/indoor_ds_new.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()
    if obs_img.shape[-1] == 3:
        query_img = cv2.cvtColor(obs_img, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(query_img)[None][None].cuda() / 255.
    max_match=-1
    max_index=-1
    for i in range(len(imgs)):
        if imgs[i].shape[-1] == 3:
            gallery_img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2GRAY)
        img1 = torch.from_numpy(gallery_img)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}

        # Inference with LoFTR and get prediction
        with torch.no_grad():
            matcher(batch)
            mkpts0 = batch['mkpts0_f'].cpu().numpy()
            mkpts1 = batch['mkpts1_f'].cpu().numpy()
            mconf = batch['mconf'].cpu().numpy()
        match_num=len(mconf)
        if match_num>max_match:
            max_match=match_num
            max_index=i
    return np.copy(poses[max_index])
        

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def dilation(map, size):
    # Mathis' version
    map = t2np(map)
    kernel = np.ones([size, size])
    map = binary_dilation(map, kernel)
    map = torch.FloatTensor(map)

    return map

def opening(map, size):
    map = t2np(map)
    kernel = np.ones([size, size])
    for i in range(len(map)):
        map[i, 0] = binary_opening(map[i, 0], kernel)
    map = torch.FloatTensor(map).to("cuda")

    return map

def make_dataloaders(trainset, testset, batch_size, shuffle_train=True, drop_last=True):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=batch_size, shuffle=shuffle_train,
                                              drop_last=drop_last)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=batch_size * 2, shuffle=False,
                                             drop_last=False)
    return trainloader, testloader


def downsampling(x, size, to_tensor=False, bin=True):
    if to_tensor:
        x = torch.FloatTensor(x).to("cuda")
    down = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    if bin:
        down[down > 0] = 1
    return down


class DefectDataset(Dataset):
    def __init__(self, dataset_dir, class_name, set='train', get_mask=True, get_features=True,
                 train_subset=None):
        super(DefectDataset, self).__init__()
        self.set = set
        self.labels = list()
        self.masks = list()
        self.images = list()
        self.class_names = ['good']
        self.get_mask = get_mask
        self.get_features = get_features
        self.image_transforms = transforms.Compose([transforms.Resize(img_size),
                                                    transforms.ToTensor(),
                                                   ])
        root = os.path.join(dataset_dir, class_name)
        set_dir = os.path.join(root, set)
        subclass = os.listdir(set_dir)
        subclass.sort()

        self.pretrained_params = None

        with open(os.path.join(root, "transforms.json"), "r") as f:
            self.camera_transforms = json.load(f)

        self.camera_angle = self.camera_transforms["camera_angle_x"] if set == "train" else None

        aug_transform_path = os.path.join(dataset_dir, class_name, "augmented_transforms.json")
        if os.path.isfile(aug_transform_path):
            with open(aug_transform_path, "r") as f:
                aug_transforms = json.load(f)
            tmp_len = len(self.camera_transforms["frames"])
            self.camera_transforms["frames"].extend(aug_transforms["frames"])
            print(f"Adding {len(aug_transforms['frames'])} to the dict with {tmp_len} for total of {len(self.camera_transforms['frames'])}")
        else:
            pass

        self.width, self.height = None, None
        class_counter = 1
        for sc in subclass:
            if sc == 'good' or sc == "nerf":
                label = 0
            else:
                label = class_counter
                self.class_names.append(sc)
                class_counter += 1
            sub_dir = os.path.join(set_dir, sc)
            img_dir = sub_dir
            img_paths = os.listdir(img_dir)
            img_paths.sort()
            for p in img_paths:
                i_path = os.path.join(img_dir, p)
                if not i_path.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                    continue
                self.images.append(i_path)
                self.labels.append(label)
                if self.set == 'test' and self.get_mask:
                    extension = '_mask' if sc != 'good' else ''
                    mask_path = os.path.join(root, 'ground_truth', sc, p[:-4] + extension + p[-4:])
                    self.masks.append(mask_path)
                elif self.get_mask:
                    self.masks.append(0)

            if self.width is None:
                with Image.open(os.path.join(img_dir, img_paths[0])) as img:
                    self.width, self.height = img.size


        self.img_mean = torch.FloatTensor(norm_mean)[:, None, None]
        self.img_std = torch.FloatTensor(norm_std)[:, None, None]

    def __len__(self):
        return len(self.images)

    def grab_mask_from_file(self, pth, index):
        if self.labels[index] == 0:
            return torch.zeros(self.height, self.width)
        with open(pth, 'rb') as f:
            mask = Image.open(f)
            mask = np.array(mask).copy()
            mask = torch.FloatTensor(mask)
            mask[mask > 0] = 1
        return mask

    def __getitem__(self, index):
        
        with open(self.images[index], 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.image_transforms(img)
        label = self.labels[index]

        ret = [img, label]
        if self.set == 'test' and self.get_mask:
            true_mask = self.grab_mask_from_file(self.masks[index], index)
            ret.append(true_mask)
        elif self.get_mask:
            true_mask = torch.zeros(self.width, self.height)
            ret.append(true_mask)

        return ret

    def insert_pretrained_params(self, params : torch.Tensor):
        # assert self.set == "test", "Inserting custom camera params into the training set is nonsensical"
        assert self.pretrained_params is None, "Overwriting existing pretrained camera params. Stopping!"
        assert self.__len__() == params.shape[0], "Number of poses doesn't match data set size"
        print("Inserting estimated poses of shape: ", params.shape)
        self.pretrained_params = params

    def remove_pretrained_params(self):
        assert self.pretrained_params is not None, "Removing non existant camera params. Stoppin!"
        self.pretrained_params = None
        

class DatasetPose(Dataset):
    def __init__(self, dataset_dir, class_name, set='train', get_features=True):
        super(DatasetPose, self).__init__()
        self.set = set
        self.labels = list()
        self.masks = list()
        self.images = list()
        self.class_names = ['good']
        self.get_features = get_features
        self.image_transforms = transforms.Compose([transforms.ToPILImage(),
                                                    transforms.Resize(img_size),
                                                    transforms.ToTensor(),
                                                   ])
        root = os.path.join(dataset_dir, class_name)
        img_dir = os.path.join(root, set)
        img_paths = os.listdir(img_dir)

        self.correct_alpha = True
        self.pretrained_params = None

        with open(os.path.join(root, f"transforms_{set}.json"), "r") as f:
            self.camera_transforms = json.load(f)

        self.camera_angle = self.camera_transforms["camera_angle_x"]
        self.width, self.height = None, None
        
        for p in img_paths:
            i_path = os.path.join(img_dir, p)
            if not i_path.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                continue
            if "depth" in i_path or "normal" in i_path:
                continue
            self.images.append(i_path)
            self.labels.append(0)
        self.images.sort(key=lambda x:int(x.split("_")[-1].split(".")[0]))
        if self.width is None:
            with Image.open(os.path.join(self.images[0])) as img:
                self.width, self.height = img.size


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        with open(self.images[index], 'rb') as f:
            img = np.array(Image.open(f))
        if self.correct_alpha:
            img[img[:,:,3] == 0.0,:3] = 255.0
            img = img[:,:,:3]
            
        img = self.image_transforms(img)

        trans_entry = self.camera_transforms["frames"][index]
        transform_matrix = torch.tensor(trans_entry["transform_matrix"])
        qu =  matrix_to_quaternion(transform_matrix[:3,:3])
        translation_params = transform_matrix[:3,3]
        transform_params = torch.cat((qu, translation_params))

        label = self.labels[index]
        ret = [img, label, transform_params]

        return ret
