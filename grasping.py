import os
import sys
import rowan
import torch
import numpy as np
import open3d as o3d
from graspnetAPI import GraspGroup

GRASPNET_DIR = "../graspnet-baseline"
sys.path.append(os.path.join(GRASPNET_DIR, 'models'))
sys.path.append(os.path.join(GRASPNET_DIR, 'dataset'))
sys.path.append(os.path.join(GRASPNET_DIR, 'utils'))

from graspnet import GraspNet, pred_decode

sys.path.append("../contact_graspnet_pytorch/contact_graspnet_pytorch")
sys.path.append("../contact_graspnet_pytorch")

from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils

from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image
from contact_graspnet_pytorch.checkpoints import CheckpointIO 
from data import load_available_input_data


def get_graspnet(model_path: str="./checkpoints/graspnet/checkpoint-rs.tar") -> GraspNet:

    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    return net

def process_point_cloud(point_cloud: np.ndarray,
                        rgb: np.ndarray=np.array([])) -> tuple[dict, o3d.geometry.PointCloud]:
    
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
    if len(rgb):
        print(rgb)
        cloud.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))

    end_points = dict()
    point_cloud = torch.from_numpy(point_cloud[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    point_cloud = point_cloud.to(device)
    end_points['point_clouds'] = point_cloud
    end_points['cloud_colors'] = rgb

    return end_points, cloud

def get_grasps(net: GraspNet, end_points: dict) -> GraspGroup:
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def vis_grasps(gg: GraspGroup, cloud: o3d.geometry.PointCloud, show_top: int=10):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:show_top]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def sample_best_grasp(
        point_cloud: np.ndarray,
        graspnet: GraspNet,
        rgb: np.ndarray=np.array([]),
        verbose: int=0) -> np.ndarray:
    
    #TODO: should maybe be a method in Graspnet
    
    end_points, cloud = process_point_cloud(point_cloud, rgb)
    grasps = get_grasps(graspnet, end_points)
    grasps.sort_by_score()
    best_grasp = grasps[0]
    
    if verbose:
        vis_grasps(grasps, cloud, show_top=1)
    
    pos = best_grasp.translation
    quat = rowan.from_matrix(best_grasp.rotation_matrix)
    
    q_x = rowan.from_axis_angle([1, 0, 0], -np.pi / 2)
    q_z = rowan.from_axis_angle([0, 0, 1],  np.pi / 2)

    quat = rowan.multiply(q_x, quat)
    quat = rowan.multiply(q_z, quat)
    
    pose = np.concatenate((pos, quat))
    
    return pose


def contact_graspnet_inference(point_cloud: np.ndarray,
                                rgb: np.ndarray,
                                ckpt_dir: str="../contact_graspnet_pytorch/checkpoints/contact_graspnet",
                                local_regions: bool=False,
                                filter_grasps: bool=False,
                                forward_passes: int=1,
                                from_top: int=5,
                                verbose: int=0) -> np.ndarray:

    global_config = config_utils.load_config(ckpt_dir, batch_size=forward_passes, arg_configs=[])

    # Build the model
    grasp_estimator = GraspEstimator(global_config)

    # Load the weights
    model_checkpoint_dir = os.path.join(ckpt_dir, 'checkpoints')
    checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=grasp_estimator.model)
    checkpoint_io.load('model.pt')
    
    pc_segments = {}
    # TODO: Look into this
    # if segmap is None and (local_regions or filter_grasps):
    #     raise ValueError('Need segmentation map to extract local regions or filter grasps')
    # if pc_full is None:
    #     pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
    #                                                                             skip_border_objects=skip_border_objects, 
    #                                                                             z_range=z_range)
    
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(point_cloud, 
                                                                                    pc_segments=pc_segments, 
                                                                                    local_regions=local_regions, 
                                                                                    filter_grasps=filter_grasps, 
                                                                                    forward_passes=forward_passes)
    
    idxs = []
    for i, pg in enumerate(pred_grasps_cam[-1]):
        if np.linalg.norm(pg[:3, 3].flatten()) < .7:
            idxs.append(i)
    pred_grasps_cam[-1] = pred_grasps_cam[-1][idxs]
    scores[-1] = scores[-1][idxs]

    if len(scores[-1]) > from_top:
        sorted_idx = np.argsort(scores[-1])[-from_top:]
        best_grasp_idx = np.random.choice(sorted_idx)
    else:
        idxs = np.array(list(range(len(scores[-1]))))
        best_grasp_idx = np.random.choice(idxs)

    best_grasp = pred_grasps_cam[-1][best_grasp_idx]

    best_grasp_pos = best_grasp[:3, 3] + best_grasp[:3, 2:3].flatten() * 0.1034  # Offset to gripper fingers

    best_grasp_rot = rowan.from_matrix(best_grasp[:3, :3])
    q_x = rowan.from_axis_angle([1, 0, 0], np.pi)
    best_grasp_rot = rowan.multiply(q_x, best_grasp_rot)  # 180 degrees around the x axis

    best_pose = np.concatenate((best_grasp_pos, best_grasp_rot))
    
    if verbose:
        visualize_grasps(point_cloud, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=rgb*255)
        pred_grasps_cam[-1] = [pred_grasps_cam[-1][best_grasp_idx]]
        scores[-1] = [scores[-1][best_grasp_idx]]
        visualize_grasps(point_cloud, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=rgb*255)

    return best_pose
    
