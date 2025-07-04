import numpy as np

from pyrobopath.process import AgentModel, create_dependency_graph_by_z
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathPlanner,
    PlanningOptions,
    animate_multi_agent_toolpath_full,
)

from utilities import Materials, print_schedule_info, create_example_toolpath


def two_robot_agent_models():
    bf1 = np.array([-350.0, 0.0, 0.0])
    bf2 = np.array([350.0, 0.0, 0.0])

    # create agent collision models
    agent1 = AgentModel(
        base_frame_position=bf1,
        home_position=np.array([-250.0, 0.0, 0.0]),
        capabilities=[Materials.MATERIAL_A],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300.0), bf1),
    )
    agent2 = AgentModel(
        base_frame_position=bf2,
        home_position=np.array([250.0, 0.0, 0.0]),
        capabilities=[Materials.MATERIAL_B],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300.0), bf2),
    )
    agent_models = {"robot1": agent1, "robot2": agent2}
    return agent_models


def two_robot_simple_example():
    toolpath = create_example_toolpath()
    dg = create_dependency_graph_by_z(toolpath)

    # create planner
    agent_models = two_robot_agent_models()
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=5.0,
        collision_offset=1.0,
        collision_gap_threshold=1.0,
    )

    print(f"{(80 * '#')}\nScheduling Simple Two Materal Toolpath:\n{(80 * '#')}\n")
    sched = planner.plan(toolpath, dg, options)
    print(f"\n{(80 * '#')}\nFound Toolpath Plan!\n{(80 * '#')}\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-250, 250))
    )


if __name__ == "__main__":
    two_robot_simple_example()
