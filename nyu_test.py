import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from mono.utils.do_test import get_prediction, transform_test_data_scalecano, align_scale_shift
from mono.utils.running import load_ckpt
from mmcv.utils import Config
from mono.utils.avg_meter import MetricAverageMeter
from mono.model.monodepth_model import get_configured_monodepth_model
import os.path as osp
import os

def postprocess_per_image_test(i, pred_depth, gt_depth, intrinsic, rgb_origin, normal_out, pad, an, dam, dam_median, dam_global, is_distributed, save_imgs_dir, save_pcd_dir, normalize_scale, scale_info):

    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad[0] : pred_depth.shape[0] - pad[1], pad[2] : pred_depth.shape[1] - pad[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], [rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear').squeeze() # to original size
    pred_depth = pred_depth * normalize_scale / scale_info

    pred_depth = (pred_depth > 0) * (pred_depth < 300) * pred_depth
    if gt_depth is not None:
        pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], (gt_depth.shape[0], gt_depth.shape[1]), mode='bilinear').squeeze() # to original size
        gt_depth = torch.from_numpy(gt_depth).cuda()
        pred_depth_median = pred_depth * gt_depth[gt_depth != 0].median() / pred_depth[gt_depth != 0].median()
        pred_global, _ = align_scale_shift(pred_depth, gt_depth)
        mask = (gt_depth > 1e-8)
        dam.update_metrics_gpu(pred_depth, gt_depth, mask, is_distributed)
        dam_median.update_metrics_gpu(pred_depth_median, gt_depth, mask, is_distributed)
        dam_global.update_metrics_gpu(pred_global, gt_depth, mask, is_distributed)
        print(gt_depth[gt_depth != 0].median() / pred_depth[gt_depth != 0].median(), )
    
    os.makedirs(osp.join(save_imgs_dir), exist_ok=True)
    rgb_torch = torch.from_numpy(rgb_origin).to(pred_depth.device).permute(2, 0, 1)
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None].to(rgb_torch.device)
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None].to(rgb_torch.device)
    rgb_torch = torch.div((rgb_torch - mean), std)
    pred_depth = pred_depth.detach().cpu().numpy()
 
    if normal_out is not None:
        pred_normal = normal_out[:3, :, :] # (3, H, W)
        H, W = pred_normal.shape[1:]
        pred_normal = pred_normal[ :, pad[0]:H-pad[1], pad[2]:W-pad[3]]
        pred_normal = torch.nn.functional.interpolate(pred_normal[None, :], size=[rgb_origin.shape[0], rgb_origin.shape[1]], mode='bilinear', align_corners=True).squeeze()
        
    return pred_depth


def visualize_depth_map(depth_map, cmap="plasma", title="Depth Map"): #For visualization
    """
    Visualizes a depth map from a NumPy array.

    Parameters:
        depth_map (np.ndarray): 2D NumPy array representing depth values.
        cmap (str): Colormap for visualization (default is "plasma").
        title (str): Title of the plot.
    """
    if depth_map is None or depth_map.size == 0:
        print("Error: Depth map is empty or None.")
        return

    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap=cmap, interpolation="nearest")
    plt.colorbar(label="Depth")
    plt.title(title)
    plt.axis("off")  # Hide axes for better visualization
    plt.show()


def compute_rmse(pred_depth, gt_depth):
    """Compute RMSE between predicted and ground truth depth."""
    valid_mask = gt_depth > 0  # Avoid invalid depth values
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]
    rmse = np.sqrt(mean_squared_error(gt_depth, pred_depth))
    return rmse




def compute_depth_metrics(pred_depth, gt_depth):
    """Compute depth estimation error metrics: RMSE, log RMSE, AbsRel, SqRel, and SI Log Error."""
    valid_mask = gt_depth > 0  # Avoid invalid depth values
    pred_depth = pred_depth[valid_mask]
    gt_depth = gt_depth[valid_mask]
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(gt_depth, pred_depth))
    
    # Log RMSE
    log_rmse = np.sqrt(mean_squared_error(np.log(gt_depth), np.log(pred_depth)))
    
    # Absolute Relative Difference
    absrel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
    
    # Squared Relative Difference
    sqrel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)
    
    # Scale Invariant Log Error
    log_diff = np.log(pred_depth) - np.log(gt_depth)
    silog = np.sqrt(np.mean(log_diff ** 2) - (np.mean(log_diff) ** 2))
    
    return {
        "RMSE": rmse,
        "Log RMSE": log_rmse,
        "AbsRel": absrel,
        "SqRel": sqrel,
        "SI Log Error": silog
    }

def process_dataset(dataset):
    total_metrics = {"RMSE": 0, "Log RMSE": 0, "AbsRel": 0, "SqRel": 0, "SI Log Error": 0}
    num_samples = 0
    
    for idx, sample in enumerate(list(dataset)):
        print(f"Processing Sample {idx}")
        rgb_image = sample['image']  # RGB image
        gt_depth = np.array(sample['depth_map'])  # Ground truth depth
        inferred_depth = infer_depth(rgb_image)
        
        metrics = compute_depth_metrics(inferred_depth, gt_depth)
        
        for key in total_metrics:
            total_metrics[key] += metrics[key]
        
        num_samples += 1
        print(f"Metrics: {metrics}")
    
    avg_metrics = {key: total_metrics[key] / num_samples for key in total_metrics}
    print(f"Average Metrics: {avg_metrics}")
    return avg_metrics




def infer_depth(img):
    img=np.array(img)
    intrinsic = None
    if intrinsic is None:
        intrinsic = [500.0, 500.0, img.shape[1]/2, img.shape[0]/2]
      # Ensure input has batch dimension (B, C, H, W)
    
    rgb_input, _, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic, cfg.data_basic)
    if len(rgb_input.shape) == 3:  # (C, H, W)
        rgb_input = rgb_input.unsqueeze(0)  # Add batch dimension → (1, C, H, W)
    #check if preprocessing is correct
    pred_depth, output = get_prediction(
                model=model,
                input=rgb_input,  # Stack inputs for batch processing
                cam_model=None,
                pad_info=pad,
                scale_info=None,
                gt_depth=None,
                normalize_scale=None,
            )

    #add postprocessing function with relevant parameters#######################
    depth = postprocess_per_image_test(
                1, #iteration number ig
                pred_depth,
                None, #No ground truth given, no cheating here!
                intrinsic,
                img,
                None, #normal_out
                pad,
                [img],
                dam,
                dam_median,
                dam_global,
                False, #not distributed, single gpu
                "./output_imgs/", #save imgs dir
                "./output_pcds/",
                normalize_scale,
                label_scale_factor,
            )

    return depth

home_dir = os.environ["HOME"] # to save the nyu_cache
print(home_dir)

dataset = load_dataset("sayakpaul/nyu_depth_v2", split="validation[:654]", cache_dir=home_dir+"/nyu_cache")
#dataset = dataset.select(range(0, 654, 6))  # Sample every 6th data in dataset

cfg = Config.fromfile('./mono/configs/HourglassDecoder/vit.raft5.large.py')
cfg.load_from = "weight/metric_depth_vit_large_800k.pth"
model = get_configured_monodepth_model(cfg, )
model = torch.nn.DataParallel(model).cuda()
model, _,  _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
model.eval()
# Define your depth estimation model (Replace with your actual model)
dam = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
dam_median = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
dam_global = MetricAverageMeter(['abs_rel', 'rmse', 'silog', 'delta1', 'delta2', 'delta3'])
normalize_scale = cfg.data_basic.depth_range[1]

# Evaluate model
process_dataset(dataset)
