import robotic as ry


def get_config_floating(verbose: int=0) -> ry.Config:
    C = ry.Config()
    C.addFile(ry.raiPath("../rai-robotModels/scenarios/dual_floating.g"))
    C.addFile("./banana.g")
    if verbose:
        C.view(True)
    return C


def get_config_table(verbose: int=0) -> ry.Config:
    C = ry.Config()
    C.addFile(ry.raiPath("../rai-robotModels/scenarios/pandasTable.g"))
    C.addFile("./banana.g")

    S = ry.Simulation(C, ry.SimulationEngine.physx, verbose=0)
    tau = 3e-4
    for _ in range(2000):
        S.step([], tau,  ry.ControlMode.none)

    if verbose:
        C.view(True)

    return C
