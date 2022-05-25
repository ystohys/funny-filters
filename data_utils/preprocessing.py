import cv2
import numpy as np
import pandas as pd
from PIL import Image

import copy

import torch
from torch.utils.data import Dataset


###########################################
# Image scaling + normalization functions #
###########################################


def per_pixel_mean_std(train_df):
    """
    Obtains mean and std at each pixel based on entire training image dataset.
    :param train_df: Training dataset with last column being string of pixel values
    :return: the per-pixel mean and standard deviation at each pixel
    """
    img_col = train_df.iloc[:, -1]
    img_list = img_col.tolist()
    pre_np = [0]*len(train_df)
    for idx, each_img in enumerate(img_list):
        not_minmax = np.fromstring(each_img, sep=' ')
        pre_np[idx] = minmax_scaling(not_minmax)
    post_np = np.array(pre_np)
    ppm_flat = np.apply_along_axis(np.mean, axis=0, arr=post_np)
    pps_flat = np.apply_along_axis(np.std, axis=0, arr=post_np)
    ppm = ppm_flat.reshape(96, 96)
    pps = pps_flat.reshape(96, 96)
    return ppm, pps


def minmax_scaling(img, return_arr=True):
    tmp_arr = img
    if isinstance(img, Image.Image):
        tmp_arr = np.asarray(img)

    norm_arr = (tmp_arr - tmp_arr.min()) / (tmp_arr.max() - tmp_arr.min())
    norm_img = Image.fromarray(norm_arr)

    if return_arr:
        return norm_arr
    else:
        return norm_img


#############################
# Data formatting functions #
#############################


def parse_row(df, idx):
    img = df.iloc[idx, -1]
    kp = df.iloc[idx, :-1]
    return img, kp


def str_to_img_arr(s):
    img = np.fromstring(s, sep=' ')
    img = img.reshape(96, 96)
    return img


def get_kp_as_arr(kp, one_d=False):
    """
    Takes in the x and y coordinates of the facial keypoints and returns them in
    numpy array form
    :param kp: keypoint coordinates as a Pandas Series
    :param one_d: whether to return one dimensional numpy array > [x, y, x, y]
    :return: returns two dimensional array by default (see below)
    [[y1, y2, y3, ...]  --> because y-coordinate is the 'row index'
     [x1, x2, x3, ...]]  --> because x_coordinate is the 'column index'
    """
    one_d_arr = np.array(kp)
    two_d_arr = one_d_arr.reshape(2, 15, order='F')
    two_d_arr[[0, 1]] = two_d_arr[[1, 0]]
    new_one_d_arr = two_d_arr.flatten(order='F')
    if one_d:
        return new_one_d_arr
    else:
        return two_d_arr


def downsize_frame(img, return_arr=True):
    """
    :param img: takes in a PIL Image object
    :param return_arr: whether to return Image object or numpy array
    :return: downsized image with height and width of 96 (or numpy array of image)
    """
    tmp = copy.copy(img)
    down_img = tmp.resize((96, 96))
    down_img_arr = np.asarray(down_img)

    if return_arr:
        return down_img_arr
    else:
        return down_img


def upsize_frame(img, height, width=None, return_arr=True):
    """
    :param img: takes in PIL image
    :param height: height of new upsized image
    :param width: width of new upsized image (ideally should be same as height)
    :param return_arr: whether to return Image object or numpy array
    :return:
    """
    if not width:
        width = height
    tmp = copy.copy(img)
    up_img = tmp.resize((height, width), Image.BICUBIC)
    up_img_arr = np.asarray(up_img)

    if return_arr:
        return up_img_arr
    else:
        return up_img


def tensorify_img(img):
    """
    Converts an image from its numpy array form to a torch.Tensor in the correct shape
    :param img:
    :return:
    """
    tmp = torch.from_numpy(img)
    tmp = tmp.unsqueeze(0)
    tmp = tmp.unsqueeze(0).to(torch.device('cpu'), dtype=torch.float)
    return tmp


def upsize_kp(tensor_kp, x_start, y_start, width, height):
    """
    Converts coordinates of keypoints from a (0,96) coordinate system to the
    'real-world' coordinate system of the entire image captured.
    :param tensor_kp: keypoints outputted by model, in torch.Tensor form
    :param x_start: x-coordinate of face bounding box top left
    :param y_start: y-coordinate of face bounding box top left
    :param width: width of face bounding box
    :param height: height of face bounding box
    :return: numpy array of keypoints where kp[0] = [x0, y0]
    """
    kp = tensor_kp.detach().cpu().numpy()
    kp *= 96.0
    kp = kp.squeeze(0)
    kp[[1, 0]] = kp[[0, 1]]
    kp = np.array([x_start + (kp[0]/96)*width,
                   y_start + (kp[1]/96)*height])
    kp = np.transpose(kp)
    kp = kp.astype(int)
    return kp


def conv_to_uint(img):
    tmp = copy.copy(img)
    new_tmp = minmax_scaling(tmp, return_arr=True)
    new_tmp *= 255.0
    res = np.round(new_tmp).astype(np.uint8)
    res_img = Image.fromarray(res)

    return res_img


#####################################
# Transformations and Augmentations #
#####################################

class PerPixelNorm:

    def __init__(self,
                 ppm_path='data_utils/per_pixel_mean.npy',
                 pps_path='data_utils/per_pixel_std.npy'):
        self.ppm = np.load(ppm_path)
        self.pps = np.load(pps_path)

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        standard_img = (image - self.ppm) / self.pps
        return {'image': standard_img, 'keypoints': keypoints}


class OutputScale:

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        scaled_kp = keypoints / 96.0
        return {'image': image, 'keypoints': scaled_kp}


class HorizontalFlip:

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        rng = np.random.default_rng()
        flagger = rng.integers(low=1, high=10)
        if flagger % 2 != 0:
            return {'image': image, 'keypoints': keypoints}
        else:
            new_img = np.fliplr(image)
            new_kp = hori_flip_kp(keypoints)
            return {'image': new_img, 'keypoints': new_kp}


class Rotate:
    """
    Class wrapper for rotate_img_and_kp function below.
    (Purely to assimilate with the PyTorch workflow of applying functions)
    """

    def __init__(self, deg_range):
        assert isinstance(deg_range, (int, tuple)), "Needs to be int or 2-tuple!"
        if isinstance(deg_range, int):
            self.deg_range = (-abs(deg_range), abs(deg_range))
        else:
            self.deg_range = deg_range
        rng = np.random.default_rng()
        self.deg = (rng.random() * (self.deg_range[1]-self.deg_range[0])) + self.deg_range[0]

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        rot_img, rot_kp = rotate_img_and_kp(image, keypoints, self.deg)

        return {'image': rot_img, 'keypoints': rot_kp}


class CustomToTensor:

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        tensor_img = torch.from_numpy(image)
        tensor_kp = torch.from_numpy(keypoints)

        tensor_w_chnl = torch.unsqueeze(tensor_img, 0)

        return {'image': tensor_w_chnl, 'keypoints': tensor_kp}


def rotate_kp(kp, deg, height=96, width=96, one_d=False):
    """
    Rotates the facial key points counter-clockwise by specified degrees
    :param kp: Facial keypoints in np.array or pd.Series format
    :param deg: Number of degrees to rotate by (set as negative to rotate clockwise)
    :param height: Height of image (default = 96)
    :param width: Width of image (default = 96) <- SHOULD BE SAME AS HEIGHT
    :param one_d: Return a one-dimensional array [x1,y1,x2,y2,...]
    :return: Array containing new x and y coordinate of keypoints after rotation
    """
    kp_arr = kp
    if isinstance(kp, pd.Series):
        kp_arr = get_kp_as_arr(kp)  # 2nd row are all the x-coordinates

    origin_h = height / 2
    origin_w = width / 2

    kp_vec_arr = kp_arr - np.array([[origin_w], [origin_h]])

    rot_mat = np.array([[np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                        [np.sin(np.radians(deg)), np.cos(np.radians(deg))]])

    new_kp_vec_arr = np.matmul(rot_mat, kp_vec_arr)
    new_kp_arr = new_kp_vec_arr + np.array([[origin_w], [origin_h]])

    if one_d:
        new_kp_arr.flatten(order='F')
    else:
        return new_kp_arr


def rotate_img_and_kp(img, kp, deg):
    if isinstance(img, str):
        img = str_to_img_arr(img)

    img_rot_mat = cv2.getRotationMatrix2D(center=(img.shape[1]/2, img.shape[0]/2),
                                          angle=deg,
                                          scale=1)

    rot_img = cv2.warpAffine(img, img_rot_mat, (img.shape[1], img.shape[0]))
    rot_kp = rotate_kp(kp, deg)

    return rot_img, rot_kp


def hori_flip_kp(kp, height=96, width=96, one_d=False):
    kp_arr = kp
    if isinstance(kp, pd.Series):
        kp_arr = get_kp_as_arr(kp)  # 2nd row are all the x-coordinates

    origin_h = height / 2
    origin_w = width / 2

    kp_vec_arr = kp_arr - np.array([[origin_w], [origin_h]])
    new_kp_vec_arr = np.multiply(kp_vec_arr, np.array([np.ones(15), -np.ones(15)]))
    new_kp_arr = new_kp_vec_arr + np.array([[origin_w], [origin_h]])

    if one_d:
        new_kp_arr.flatten(order='F')
    else:
        return new_kp_arr


# Dataset Class

class FaceKeyDataset(Dataset):

    def __init__(self, csv_file,
                 transform=None,
                 target_transform=None):
        self.full_df = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.full_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_str, pre_kp = parse_row(self.full_df, idx)
        img = str_to_img_arr(img_str)
        mm_img = minmax_scaling(img)
        kp = get_kp_as_arr(pre_kp).astype('float')
        presamp = {'image': mm_img, 'keypoints': kp}
        sample = presamp

        if self.transform:
            sample = self.transform(presamp)
        if self.target_transform:
            kp = self.target_transform(kp)
            sample['keypoints'] = kp

        return sample
