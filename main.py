from robot_api import RobotAPI
from vision import get_point_clouds
from scenes import get_config_table
from motions import move_to_look, grasp_motion
from grasping import contact_graspnet_inference

ON_REAL = True
camera_frames = ["l_cameraWrist", "r_cameraWrist"]
robot_api = RobotAPI(verbose=2, use_foundation_stereo=False, address="tcp://130.149.82.15:1234", on_real=ON_REAL) if ON_REAL else None

C = get_config_table(verbose=1)

obj_frame_name = C.getFrameNames()[-1]
path = move_to_look(C, obj_frame_name, distance=.5, verbose=1)

if ON_REAL:
    robot_api.move(path, [3.])
C.setJointState(path[-1])

pcs, rgbs = get_point_clouds(C, camera_frames, robot_api, on_real=ON_REAL, verbose=0)

import torch
torch.cuda.empty_cache()
grasp_pose = contact_graspnet_inference(pcs[0], rgbs[0], local_regions=False, filter_grasps=False, forward_passes=2, verbose=1, from_top=10)

try:
    approach, retract = grasp_motion(C, grasp_pose, verbose=1, arm_prefix="l_")
except:
    approach, retract = grasp_motion(C, grasp_pose, verbose=1, arm_prefix="r_")

if ON_REAL:
    robot_api.move(approach, [10.])
    C.view(True)
    robot_api.gripper_close()
    robot_api.move(retract, [10.])

    # robot_api.close()
