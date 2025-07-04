import numpy as np

from pyrobopath.process import AgentModel
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.process.utilities import create_dependency_graph_by_z
from pyrobopath.toolpath.preprocessing import LayerRangeStep
from pyrobopath.toolpath_scheduling import (
    PlanningOptions,
    DepthBasedParallelPlanner,
    animate_multi_agent_toolpath_full,
)

from utilities import toolpath_from_gcode, print_schedule_info

# ========================== constants =================================
R, Rh = 400, 300
VELOCITY = 50
TRAVEL_VELOCITY = 100
DIMS = (300.0, 50.0, 300.0)

options = PlanningOptions(
    retract_height=10.0,
    collision_offset=3.0,
    collision_gap_threshold=5.0,
)


# ========================== helper functions ==========================
def get_agent_models(n, r, rh, caps, velocity, travel_velocity, col_dims):
    circle_point = lambda r, angle: np.array(
        [r * np.cos(angle), r * np.sin(angle), 0.0]
    )

    bfs = [circle_point(r, 2 * np.pi * i / n) for i in range(n)]
    homes = [circle_point(rh, 2 * np.pi * i / n) for i in range(n)]
    agent_models = {
        f"robot{i + 1}": AgentModel(
            base_frame_position=bfs[i],
            home_position=homes[i],
            capabilities=caps[i],
            velocity=velocity,
            travel_velocity=travel_velocity,
            collision_model=FCLRobotBBCollisionModel(col_dims, bfs[i]),
        )
        for i in range(n)
    }
    return agent_models


def get_toolpath():
    filepath = "../test/test_gcode/multi_tool_demo.gcode"
    toolpath = toolpath_from_gcode(filepath)
    LayerRangeStep(0, 2).apply(toolpath)
    return toolpath


# ======================= multi-robot examples ======================
def multi_robot_example():
    toolpath = get_toolpath()
    dg = create_dependency_graph_by_z(toolpath)

    capabilities = [
        [[0], [1]],
        [[0], [1], [0]],
        [[0], [1], [0], [1]],
        [[0], [1], [0], [1], [0]],
    ]

    for i in range(len(capabilities)):
        caps = capabilities[i]
        agent_models = get_agent_models(
            len(caps), R, Rh, caps, VELOCITY, TRAVEL_VELOCITY, DIMS
        )
        planner = DepthBasedParallelPlanner(agent_models)
        msg = f"Scheduling {len(caps)}-Robot Demo"
        print(f"{(80 * '#')}\n{msg}\n{(80 * '#')}\n")

        sched = planner.plan(toolpath, dg, options)
        dg.reset()

        print(f"Found toolpath plan!\n")
        print_schedule_info(sched)

        animate_multi_agent_toolpath_full(
            toolpath, sched, agent_models, limits=((-600, 600), (-600, 600))
        )


if __name__ == "__main__":
    multi_robot_example()
