import numpy as np
import robotic as ry


def solve_komo(komo: ry.KOMO, verbose: int=0) -> np.ndarray:
    
    sol = ry.NLP_Solver()
    sol.setProblem(komo.nlp())
    sol.setOptions(damping=1e-1, verbose=0, stopTolerance=1e-3, lambdaMax=100., stopInners=20, stopEvals=200)
    ret = sol.solve()
    
    if not ret.feasible:
        print("KOMO not possible :(")
        komo.report(plotOverTime=True)
        komo.view(True)
        raise Exception("KOMO not possible :(")

    if verbose:
        komo.view(True)
        komo.view_play(True, delay=.3)

    path = komo.getPath()
    return path


def move_to_look(C: ry.Config, obj_frame_name: str, distance: float=.3, verbose: int=0) -> np.ndarray:
    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(1, 1, 1, 0)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])
    komo.addObjective([1.], ry.FS.positionRel, [obj_frame_name, "l_cameraWrist"], ry.OT.eq, [1e1], [0., 0., distance])
    # komo.addObjective([1.], ry.FS.positionRel, [obj_frame_name, "r_cameraWrist"], ry.OT.eq, [1e1], [0., 0., distance])
    komo.addObjective([1.], ry.FS.negDistance, ["l_cameraWrist", "table"], ry.OT.ineq, [-1e1], [-.5])
    komo.addObjective([1.], ry.FS.scalarProductYZ, ["l_cameraWrist", "table"], ry.OT.ineq, [1e1], [-.7])
    # komo.addObjective([1.], ry.FS.negDistance, ["r_cameraWrist", "table"], ry.OT.ineq, [-1e1], [-.5])
    
    path = solve_komo(komo)

    if verbose:
        komo.view(True)

    return path


def grasp_motion(C: ry.Config, grasp_pose: np.ndarray,
                 arm_prefix: str="l_", verbose: int=0) -> tuple[np.ndarray, np.ndarray]:
    
    qHome = C.getJointState()
    gripper = f"{arm_prefix}gripper"
    pre_grasp_pose = f"pre_grasp_pose"
    camera_frame_name = f"{arm_prefix}cameraWrist"
    grasp_pose_frame_name = "target_grasp"

    camera_frame = C.getFrame(camera_frame_name)
    C.addFrame("camera_copy") \
        .setPose(camera_frame.getPose())
    C.addFrame(grasp_pose_frame_name, "camera_copy") \
        .setShape(ry.ST.marker, [.1]) \
        .setColor([1., 1., 0.]) \
        .setRelativePose(grasp_pose)

    delta = np.array([0., 0., -1.])
    C.addFrame(pre_grasp_pose, grasp_pose_frame_name) \
        .setRelativePosition(-.1*delta) \
        .setShape(ry.ST.marker, [.05]) \
        .setColor([1., 0., 0.])
    
    if verbose:
        C.view(True)
    
    slices = 32
    komo = ry.KOMO()
    komo.setConfig(C, True)
    komo.setTiming(4, slices, 1, 2)
    komo.addControlObjective([], 0, 1e-1)
    komo.addControlObjective([], 1, 1e-1)
    komo.addControlObjective([], 2, 1e-1)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq, [1e0])

    # Pre-grasp
    komo.addObjective([1.], ry.FS.poseDiff, [gripper, pre_grasp_pose], ry.OT.eq, [1e0])
    # komo.addObjective([1., 3.], ry.FS.angularVel, [gripper], ry.OT.eq, [1e0])
    mat = np.eye(3) - np.outer(delta, delta)
    # komo.addObjective([1., 2.], ry.FS.positionDiff, [gripper, pre_grasp_pose], ry.OT.eq, mat)

    # Grasp
    komo.addObjective([2.], ry.FS.poseDiff, [gripper, grasp_pose_frame_name], ry.OT.eq, [1e0])

    # Return to Pre-grasp
    komo.addObjective([3.], ry.FS.poseDiff, [gripper, pre_grasp_pose], ry.OT.eq, [1e0])

    # Return Home
    komo.addObjective([4.], ry.FS.jointState, [], ry.OT.eq, [1e0], qHome)
    
    path = solve_komo(komo, verbose)

    half_point = slices*2
    return path[:half_point], path[half_point:]
