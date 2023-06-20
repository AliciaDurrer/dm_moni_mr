"""
Generate 2D samples, stack them to a 3D volume.
"""

import argparse
import os
import nibabel as nib
import torch
from visdom import Visdom
viz = Visdom(port=8875) # adjust if required
import sys
import random
sys.path.append(".")
import numpy as np
import cv2
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.mrloader import MRDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    test_id = '50' # adapt ID depending on sample
    output_folder = 'results_sampling/results_sampling_' # adapt folder depending on samples

    ds = MRDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=None)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    vol = []
    
    while len(all_images) * args.batch_size < args.num_samples:
        # load data slice-wise
        b, path = next(data)  # should return an image from the dataloader "data"
        folder_name = path[0].split("/", -1)[3]
        slice_name = path[0].split("/", -1)[4]
        # ground truth loading
        slice_name_gt = slice_name[3:-9]
        GT_path = 'data/test/' + str(test_id) + '_test_gt/' + str(folder_name) + '/BL_' + str(slice_name_gt) + 'BL.nii.gz' # adapt it (here BL or FU), depending on direction of translation (also adapt mrloader.py!)
        gt_img = nib.load(GT_path)
        gt_img_arr = np.asarray(gt_img.dataobj)
        gt_img_crop = gt_img_arr[..., 16:-16, 16:-16] 
        gt_img_vis = visualize(gt_img_crop)
        viz.image(visualize(gt_img_vis), opts=dict(caption="groundtruth to input " + str(slice_name)))
        # original input image loading for visualization
        org_img = nib.load(path[0])
        org_img_arr = np.asarray(org_img.dataobj)
        org_img_crop = org_img_arr[..., 16:-16, 16:-16] 
        org_img_vis = visualize(org_img_crop)
        viz.image(visualize(org_img_vis), opts=dict(caption="original input img " + str(slice_name)))
        # prepare input image
        c = th.randn_like(b[:, :1, ...])
        img = th.cat((b, c), dim=1)     #add a noise channel$
        slice_ID=path[0].split("/", -1)[4]
        viz.image(visualize(img[0,0,...]), opts=dict(caption="img input 0"))

        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)

        if not os.path.exists(str(output_folder) + str(test_id) + '/' + str(folder_name) + '/'):
            os.makedirs((str(output_folder) + str(test_id) + '/' + str(folder_name) + '/'))

        model_kwargs = {}
        start.record()
        # sampling
        sample_fn = (
            diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        )
        sample, x_noisy, org = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size), img,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        end.record()
        th.cuda.synchronize()
        print('time for 1 sample slice', start.elapsed_time(end))

        sample_vis = visualize(sample)

        # save sample
        sam = sample_vis[0,0,:,:].clone().cpu().detach().numpy()
        nib.save(nib.Nifti1Image(sam, np.eye(4)),(str(output_folder) + str(test_id) + '/' + str(folder_name) + '/sample_' + str(i) + '_from_input_' + str(slice_name)))
        # visualize sample, calculate MSE sample & GT
        viz.image(visualize(sample[0,0,:,:]), opts=dict(caption="generated sample slice " + str(slice_ID)))
        MSE_sample_gt = np.square(np.subtract(np.asarray(gt_img), np.asarray(sample.cpu()))).mean()

        print('MSE sample & GT: ', format(MSE_sample_gt, ".8f"))

        # mask background of generated sample
        mask_input = th.clone(th.as_tensor(org_img))
        mask_input_sq = mask_input.squeeze()
        mask_input_arr = np.asarray(mask_input_sq)
        mask_input_re = mask_input_arr * 255
        thr, im_thr = cv2.threshold(mask_input_re, 50, 255, cv2.THRESH_BINARY)
        im_floodfill = im_thr.copy()
        h, w = im_thr.shape[:2]
        im_floodfill_u = im_floodfill.astype(np.uint8)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill_u, mask, (0, 0), 255)
        im_floodfill_inv = im_floodfill_u.copy()
        im_floodfill_inv[im_floodfill_inv == 0] = 127
        im_floodfill_inv[im_floodfill_inv == 255] = 0
        im_floodfill_inv[im_floodfill_inv == 127] = 255

        im_thr_int = im_thr.astype(int)
        im_floodfill_int = im_floodfill_u.astype(int)
        im_floodfill_inv_int = im_floodfill_inv.astype(int)

        mask_input_int = im_thr_int | im_floodfill_inv_int
        mask_input_div = mask_input_int / 255
        mask_input_th = th.as_tensor(mask_input_div)
        mask_input_unsq = mask_input.unsqueeze(0).unsqueeze(1)
        mask_input_float = mask_input_unsq.type(th.FloatTensor)
        mask_input_float_sq = mask_input_float.squeeze()
        viz.image(visualize(mask_input_float_sq), opts=dict(caption="mask " + str(slice_ID)))

        sample_masked = th.as_tensor(sam) * mask_input_float_sq

        viz.image(visualize(sample_masked), opts=dict(caption="sample masked ID " + str(test_id) + ' slice ' + str(slice_ID)))

        MSE_masked_sample_gt = np.square(np.subtract(gt_img_vis, sample_masked)).mean()
        print('MSE masked sample & GT: ', format(MSE_masked_sample_gt, ".8f"))

        nib.save(nib.Nifti1Image(np.asarray(sample_masked), np.eye(4)), (str(output_folder) + str(test_id) + '/' + str(folder_name) + '/sample_masked_from_input_' + str(slice_name)))
        MSE_gt_input = np.square(np.subtract(np.asarray(gt_img_vis), np.asarray(org_img_vis))).mean()
        MSE_gt_sample = np.square(np.subtract(np.asarray(gt_img_vis), np.asarray(sample_masked[:,:].cpu()))).mean()
        MSE_input_sample = np.square(np.subtract(np.asarray(org_img_vis), np.asarray(sample_masked[:,:].cpu()))).mean()

        print('MSE GT & input: ', format(MSE_gt_input, ".8f"))
        print('MSE GT & masked sample: ', format(MSE_gt_sample, ".8f"))
        print('MSE input & masked sample: ', format(MSE_input_sample, ".8f"))

        # calculate the differences between input image, ground truth and generated sample
        diff_gt_input = gt_img_vis - org_img_vis
        diff_gt_sample = gt_img_vis - sample_masked
        diff_input_sample = org_img_vis - sample_masked

        viz.image(visualize(diff_gt_input), opts=dict(caption="difference input & gt, input " + str(slice_name)))
        viz.heatmap(np.flipud(diff_gt_input), opts=dict(title=("difference input & gt, heatmap, input " + str(slice_name))))
        viz.image(visualize(diff_gt_sample), opts=dict(caption="difference masked sample & gt, input " + str(slice_name)))
        viz.heatmap(np.flipud(diff_gt_sample), opts=dict(title=("difference masked sample & gt, heatmap, input " + str(slice_name))))
        viz.image(visualize(diff_input_sample), opts=dict(caption="difference masked sample & input, input " + str(slice_name)))
        viz.heatmap(np.flipud(diff_input_sample), opts=dict(title=("difference masked sample & input, heatmap, input " + str(slice_name))))

        nib.save(nib.Nifti1Image(diff_gt_input, np.eye(4)), (str(output_folder) + str(test_id) + '/' + str(folder_name) + '/diff_gt_input_from_input_' + str(slice_name)))
        nib.save(nib.Nifti1Image(diff_gt_sample, np.eye(4)), (str(output_folder) + str(test_id) + '/' + str(folder_name) + '/diff_sample_gt_from_input_' + str(slice_name)))
        nib.save(nib.Nifti1Image(diff_input_sample, np.eye(4)), (str(output_folder) + str(test_id) + '/' + str(folder_name) + '/diff_sample_input_from_input_' + str(slice_name)))

        slice_temp = nib.load(str(output_folder) + str(test_id) + '/' + str(folder_name) + '/sample_masked_from_input_' + str(slice_name)).dataobj
        slice_temp_arr = np.asarray(slice_temp)
        vol.append(th.as_tensor(slice_temp_arr).unsqueeze(0))

        # create the 3D volume slice-by-slice
        vol_stack = th.cat(vol, dim=0)
        vol_stack_arr = np.asarray(vol_stack)
        nib.save(nib.Nifti1Image(vol_stack_arr, np.eye(4)), (str(output_folder) + str(test_id) + '/volume_samples_from_input_ID_' + str(test_id) + '.nii.gz'))

def create_argparser():
    defaults = dict(
        data_dir="./dataset/test",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False, # FALSE --> show different outputs, varying!!
        model_path="",
        num_ensemble=5 # number of samples in the ensemble, not used for this implementation
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
