import cv2
import zmq
import pickle
import numpy as np
import robotic as ry
import pyrealsense2 as rs


class RobotServer:

    def __init__(self, address: str="tcp://*:1234", on_real: bool=False, verbose: int=0):
        
        self.C = ry.Config()
        self.C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandasTable.g"))
        self.C.view(False)
        self.bot = ry.BotOp(self.C, on_real)
        self.bot.home(self.C)
        self.verbose = verbose

        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.address = address
        self.socket.bind(address)

        # Realsense
        serial_left = "108322073334"
        serial_right = "102422071099"

        self.pipeline_left = rs.pipeline()
        self.pipeline_right = rs.pipeline()

        config_left = rs.config()
        config_right = rs.config()

        config_left.enable_device(serial_left)
        config_right.enable_device(serial_right)

        config_left.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config_left.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        config_right.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config_right.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        self.pipeline_left.start(config_left)
        self.pipeline_right.start(config_right)

    
    def __del__(self):
        self.pipeline_left.stop()
        self.pipeline_right.stop()

    def send_error_message(self, text):
        message = {}
        message["success"] = False
        message["text"] = text
        to_send = pickle.dumps(message)
        self.socket.send(to_send)

    def execute_command(self, message: dict) -> dict:

        feedback = {}
        command = message["command"]
        if command == "move":
        
            self.bot.move(message["path"], message["times"])
            while self.bot.getTimeToEnd() > 0:
                self.bot.sync(self.C)

        elif command == "moveAutoTimed":
        
            self.bot.moveAutoTimed(message["path"], message["time_cost"])
            while self.bot.getTimeToEnd() > 0:
                self.bot.sync(self.C)

        elif command == "home":
            self.bot.home(self.C)

        elif "gripper" in command:
            which = ry._right if message["gripper_id"] == "right" else ry._left
            
            if "close" in command:
                self.bot.gripperClose(which)
            elif "open" in command:
                self.bot.gripperMove(which)
            else:
                raise Exception(f"Gripper command {message['command']} not implemented.")
            
            while not self.bot.gripperDone(which):
                self.bot.sync(self.C)

        elif command == "getImageDepthPcl":
            rgb, depth, point_cloud = self.bot.getImageDepthPcl(message["sensor_name"])
            feedback["rgb"] = rgb
            feedback["depth"] = depth
            feedback["point_cloud"] = point_cloud

        elif command == "getRealSenseStereo":
            if "r_" in command["sensor_name"]:
                frames = self.pipeline_right.wait_for_frames()
            else:
                frames = self.pipeline_left.wait_for_frames()

            color_frame = frames.get_color_frame()
            ir_frame = frames.get_infrared_frame(2)

            color_image = np.asanyarray(color_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())
            ir_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
            
            feedback["color_image"] = color_image
            feedback["ir_image"] = ir_image
        
        else:
            raise Exception(f"Command {message['command']} not implemented.")
        
        return feedback
        
    def run(self):

        if self.verbose:
            print("Started server at ", self.address)

        running = True
        while running:
            client_input = self.socket.recv()
            try:
                client_input = pickle.loads(client_input)
            except Exception as e:
                self.send_error_message(f"Error while loading message: {e}")
                
            if self.verbose:
                print(f"Received request: {client_input}")

            try:
                if client_input["command"] == "close":
                    running = False
                else:
                    feedback = self.execute_command(client_input)

            except Exception as e:
                self.send_error_message(f"Error while executing command: {e}")
            
            message = {}
            message["success"] = True
            message["command"] = client_input["command"]

            # Feedback
            for k, v in feedback.items():
                message[k] = v

            to_send = pickle.dumps(message)
            self.socket.send(to_send)

            if self.verbose:
                print("Sent a response.")


if __name__ == "__main__":
    robot = RobotServer(verbose=1)
    robot.run()
