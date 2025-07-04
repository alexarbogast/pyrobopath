import time
import numpy as np

from pyrobopath.process import AgentModel, create_dependency_graph_by_z
from pyrobopath.toolpath.preprocessing import LayerRangeStep, ShuffleStep
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathPlanner,
    DepthBasedSequentialPlanner,
    DepthBasedParallelPlanner,
    BatchedSequentialPlanner,
    BatchedParallelPlanner,
    PlanningOptions,
    animate_multi_agent_toolpath_full,
)

from utilities import print_schedule_info, toolpath_from_gcode


options = PlanningOptions(
    retract_height=10.0,
    collision_offset=3.0,
    collision_gap_threshold=5.0,
)


# ========================== helper functions ==========================
def two_robot_agent_models():
    bf1 = np.array([-350.0, 0.0, 0.0])
    bf2 = np.array([350.0, 0.0, 0.0])

    # create agent collision models
    agent1 = AgentModel(
        base_frame_position=bf1,
        home_position=np.array([-250.0, 0.0, 0.0]),
        capabilities=[0],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300.0), bf1),
    )
    agent2 = AgentModel(
        base_frame_position=bf2,
        home_position=np.array([250.0, 0.0, 0.0]),
        capabilities=[1],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300.0), bf2),
    )
    agent_models = {"robot1": agent1, "robot2": agent2}
    return agent_models


def get_toolpath():
    filepath = "../test/test_gcode/multi_tool_square.gcode"
    toolpath = toolpath_from_gcode(filepath)
    ShuffleStep().apply(toolpath)
    LayerRangeStep(0, 2).apply(toolpath)
    return toolpath


# ========================== planner demos ==========================
def base_planner_example():
    toolpath = get_toolpath()
    dg = create_dependency_graph_by_z(toolpath)

    # create planner
    agent_models = two_robot_agent_models()
    planner = MultiAgentToolpathPlanner(agent_models)

    msg = "MultiAgentToolpathPlanner: Scheduling Multi-tool Square"
    print(f"{(80 * '#')}\n{msg}\n{(80 * '#')}\n")

    start = time.perf_counter()
    sched = planner.plan(toolpath, dg, options)
    end = time.perf_counter()

    print(f"Found plan in {end - start:.6f} seconds!\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-300, 300))
    )


def depth_based_sequential_planner_example():
    toolpath = get_toolpath()
    dg = create_dependency_graph_by_z(toolpath)

    # create planner
    agent_models = two_robot_agent_models()
    planner = DepthBasedSequentialPlanner(agent_models)

    msg = "DepthBasedSequentialPlanner: Scheduling Multi-tool Square"
    print(f"{(80 * '#')}\n{msg}\n{(80 * '#')}\n")

    start = time.perf_counter()
    sched = planner.plan(toolpath, dg, options)
    end = time.perf_counter()

    print(f"Found plan in {end - start:.6f} seconds!\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-300, 300))
    )


def depth_based_parallel_planner_example():
    toolpath = get_toolpath()
    dg = create_dependency_graph_by_z(toolpath)

    # create planner
    agent_models = two_robot_agent_models()
    planner = DepthBasedParallelPlanner(agent_models)

    msg = "DepthBasedParallelPlanner: Scheduling Multi-tool Square"
    print(f"{(80 * '#')}\n{msg}\n{(80 * '#')}\n")

    start = time.perf_counter()
    sched = planner.plan(toolpath, dg, options)
    end = time.perf_counter()

    print(f"Found plan in {end - start:.6f} seconds!\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-300, 300))
    )


def batched_sequential_planner_example():
    toolpath = get_toolpath()
    dg = create_dependency_graph_by_z(toolpath)

    # create planner
    agent_models = two_robot_agent_models()
    planner = BatchedSequentialPlanner(agent_models, 5)

    msg = "BatchedSequentialPlanner: Scheduling Multi-tool Square"
    print(f"{(80 * '#')}\n{msg}\n{(80 * '#')}\n")

    start = time.perf_counter()
    sched = planner.plan(toolpath, dg, options)
    end = time.perf_counter()

    print(f"Found plan in {end - start:.6f} seconds!\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-300, 300))
    )


def batched_parallel_planner_example():
    toolpath = get_toolpath()
    dg = create_dependency_graph_by_z(toolpath)

    # create planner
    agent_models = two_robot_agent_models()
    planner = BatchedParallelPlanner(agent_models, 5)

    msg = "BatchedParallelPlanner: Scheduling Multi-tool Square"
    print(f"{(80 * '#')}\n{msg}\n{(80 * '#')}\n")

    start = time.perf_counter()
    sched = planner.plan(toolpath, dg, options)
    end = time.perf_counter()

    print(f"Found plan in {end - start:.6f} seconds!\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-300, 300))
    )


if __name__ == "__main__":
    base_planner_example()
    depth_based_sequential_planner_example()
    depth_based_parallel_planner_example()
    batched_sequential_planner_example()
    batched_parallel_planner_example()
