import robotic as ry

C = ry.Config()
C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandaSingle.g"))
C.delFrame('panda_collCameraWrist')
C.addFile("./banana.g")
C.view(True)
