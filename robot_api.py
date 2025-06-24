import zmq
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt

from vision import foundation_stereo_call


class RobotAPI:

    def __init__(self,
                 address: str="tcp://localhost:1234",
                 on_real: bool=True,
                 use_foundation_stereo: bool=False,
                 verbose: int=0):
        
        self.on_real = on_real
        self.use_foundation_stereo = use_foundation_stereo
        self.verbose = verbose
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(address)
        self.home()
        self.gripper_open(left_gripper=True)
        self.gripper_open(left_gripper=False)

    def send_message(self, message: dict) -> dict:
        if self.verbose:
            print(f"Sending mesage: {message}...")
        
        to_send = pickle.dumps(message)
        self.socket.send(to_send)

        reply = self.socket.recv()
        reply = pickle.loads(reply)
        
        if self.verbose:
            print(f"Received reply: {reply}")
        
        return reply

    def move(self, path: np.ndarray, times: list[float]) -> bool:
        
        message = {}
        message["command"] = "move"
        message["path"] = path
        message["times"] = times

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def moveAutoTimed(self, path: np.ndarray, time_cost: float) -> bool:
        
        message = {}
        message["command"] = "moveAutoTimed"
        message["path"] = path
        message["time_cost"] = time_cost

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def home(self) -> bool:
        
        message = {}
        message["command"] = "home"

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def gripper_open(self, left_gripper: bool=True) -> bool:
        
        message = {}
        message["command"] = "gripper_open"
        message["gripper_id"] = "left" if left_gripper else "right"

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def gripper_close(self, left_gripper: bool=True) -> bool:
        
        message = {}
        message["command"] = "gripper_close"
        message["gripper_id"] = "left" if left_gripper else "right"

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def close(self) -> bool:
        
        message = {}
        message["command"] = "close"

        reply = self.send_message(message)
        
        success = reply["success"]
        return success
    
    def getRGBPointCloud(self, sensor_name: str) -> tuple[np.ndarray, np.ndarray]:

        message = {}
        message["sensor_name"] = sensor_name

        if self.use_foundation_stereo:
            message["command"] = "getRealSenseStereo"

            reply = self.send_message(message)
            
            success = reply["success"]
            if success:
                
                if self.verbose > 1:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                    axes[0].imshow(reply["color_image"])
                    axes[0].axis('off')
                    axes[0].set_title('RGB')

                    axes[1].imshow(reply["ir_image"])
                    axes[1].axis('off')
                    axes[1].set_title('IR')

                    plt.tight_layout()
                    plt.show()

                # TODO: Shift point cloud to camera center. Currently the center is at the rgb camera of the realsense.
                pc, rgb = foundation_stereo_call(reply["color_image"], reply["ir_image"], scale=.5, verbose=self.verbose)
                return rgb, pc

        else:
            message["command"] = "getImageDepthPcl"

            reply = self.send_message(message)
            
            success = reply["success"]
            if success:
                
                if self.verbose > 1:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                    axes[0].imshow(reply["rgb"])
                    axes[0].axis('off')
                    axes[0].set_title('RGB')

                    axes[1].imshow(reply["depth"])
                    axes[1].axis('off')
                    axes[1].set_title('Depth')

                    plt.tight_layout()
                    plt.show()

                return reply["rgb"], reply["point_cloud"]
        
        return np.array([]), np.array([])


if __name__ == "__main__":

    robot_api = RobotAPI(verbose=2)
    # robot_api = RobotAPI(address="tcp://130.149.82.15:1234", verbose=1)
    
    robot_api.home()
    
    rgb, depth = robot_api.getImageDepthPcl("camera")
    
    robot_api.close()
