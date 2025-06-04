import numpy as np
from example_toolpath import create_example_toolpath, Materials

from gcodeparser import GcodeParser
from pyrobopath.toolpath import *
from pyrobopath.collision_detection import *
from pyrobopath.toolpath_scheduling import *


# ========================== helper functions ==========================
def toolpath_from_gcode(filepath):
    """Parse gcode file to internal toolpath representation."""
    with open(filepath, "r") as f:
        gcode = f.read()
    parsed_gcode = GcodeParser(gcode)

    toolpath = Toolpath.from_gcode(parsed_gcode.lines)
    return toolpath


def print_schedule_info(schedule: MultiAgentToolpathSchedule):
    print(f"Schedule duration: {schedule.duration()}")
    print(f"Total Events: {schedule.n_events()}")


# ========================== example demos ==========================
def simple_toolpath_example_two():
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

    # create toolpath
    toolpath = create_example_toolpath()
    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
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


def simple_toolpath_example_three():
    # create agent collision models
    r = 400
    rh = 300

    bf1 = np.array([r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), 0.0])
    bf2 = np.array([0.0, r, 0.0])
    bf3 = np.array([-r * np.cos(np.pi / 6), -r * np.sin(np.pi / 6), 0.0])

    # fmt: off
    agent1 = AgentModel(
        base_frame_position = bf1,
        home_position = np.array([rh * np.cos(np.pi/6), -rh * np.sin(np.pi/6), 0.]),
        capabilities = [Materials.MATERIAL_B],
        velocity = 50.0,
        travel_velocity = 50.0,
        collision_model = FCLRobotBBCollisionModel((200., 50., 300.), bf1),
    )
    agent2 = AgentModel(
        base_frame_position = bf2,
        home_position = np.array([0., rh, 0.]),
        capabilities = [Materials.MATERIAL_A],
        velocity = 50.0,
        travel_velocity = 50.0,
        collision_model = FCLRobotBBCollisionModel((200., 50., 300.), bf2),
    )
    agent3 = AgentModel(
        base_frame_position = bf3,
        home_position = np.array([-rh * np.cos(np.pi/6), -rh * np.sin(np.pi/6), 0.]),
        capabilities = [Materials.MATERIAL_A],
        velocity = 50.0,
        travel_velocity = 50.0,
        collision_model = FCLRobotBBCollisionModel((200., 50., 300.), bf3),
    )
    agent_models = {"robot1": agent1, "robot2": agent2, "robot3": agent3}

    # create toolpath
    toolpath = create_example_toolpath()
    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=5.0,
        collision_offset=1.0,
        collision_gap_threshold=1.0,
    )

    print(f"{(80 * '#')}\nScheduling Simple Three Material Toolpath:\n{(80 * '#')}\n")
    sched = planner.plan(toolpath, dg, options)
    print(f"\n{(80 * '#')}\nFound Toolpath Plan!\n{(80 * '#')}\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-300, 550))
    )


def multi_material_squares():
    bf1 = np.array([-500.0, 0.0, 0.0])
    bf2 = np.array([500.0, 0.0, 0.0])

    # create agent_models
    agent1 = AgentModel(
        base_frame_position=bf1,
        home_position=np.array([-300, 0.0, 0.0]),
        capabilities=[0],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300.0), bf1),
    )
    agent2 = AgentModel(
        base_frame_position=bf2,
        home_position=np.array([300.0, 0.0, 0.0]),
        capabilities=[1],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300.0), bf2),
    )
    agent_models = {"robot1": agent1, "robot2": agent2}

    filepath = "../test/test_gcode/multi_tool_square.gcode"
    toolpath = toolpath_from_gcode(filepath)
    toolpath.contours = toolpath.contours[0:48]

    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=5.0,
        collision_offset=3.0,
        collision_gap_threshold=5.0,
    )

    print(f"{(80 * '#')}\nScheduling Multi-material Squares Toolpath:\n{(80 * '#')}\n")
    sched = planner.plan(toolpath, dg, options)
    print(f"\n{(80 * '#')}\nFound Toolpath Plan!\n{(80 * '#')}\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-300, 300))
    )


def complex_two_material_two_robots():
    bf1 = np.array([-500.0, 0.0, 0.0])
    bf2 = np.array([500.0, 0.0, 0.0])

    # create agent_models
    agent1 = AgentModel(
        base_frame_position=bf1,
        home_position=np.array([-300, 0.0, 0.0]),
        capabilities=[0],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300.0), bf1),
    )
    agent2 = AgentModel(
        base_frame_position=bf2,
        home_position=np.array([300.0, 0.0, 0.0]),
        capabilities=[1],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300.0), bf2),
    )
    agent_models = {"robot1": agent1, "robot2": agent2}

    filepath = "../test/test_gcode/multi_tool_demo.gcode"
    toolpath = toolpath_from_gcode(filepath)
    toolpath.contours = toolpath.contours[0:130]
    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=5.0,
        collision_offset=3.0,
        collision_gap_threshold=5.0,
    )

    print(f"{(80 * '#')}\nScheduling Complex Multi-material Toolpath:\n{(80 * '#')}\n")
    sched = planner.plan(toolpath, dg, options)
    print(f"\n{(80 * '#')}\nFound Toolpath Plan!\n{(80 * '#')}\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-550, 550), (-300, 300))
    )


def complex_two_material_four_robots():
    # create agent_models
    r = 400
    rh = 300

    bf1 = np.array([r * np.cos(np.pi / 4), -r * np.sin(np.pi / 4), 0.0])
    bf2 = np.array([r * np.cos(np.pi / 4), r * np.sin(np.pi / 4), 0.0])
    bf3 = np.array([-r * np.cos(np.pi / 4), r * np.sin(np.pi / 4), 0.0])
    bf4 = np.array([-r * np.cos(np.pi / 4), -r * np.sin(np.pi / 4), 0.0])

    # fmt: off
    agent1 = AgentModel(
        base_frame_position = bf1,
        home_position = np.array([rh * np.cos(np.pi/4), -rh * np.sin(np.pi/4), 0.]),
        capabilities = [0],
        velocity = 50.0,
        travel_velocity = 50.0,
        collision_model = FCLRobotBBCollisionModel((500., 50., 300.), bf1),
    )
    agent2 = AgentModel(
        base_frame_position = bf2,
        home_position = np.array([rh * np.cos(np.pi/4), rh * np.sin(np.pi/4), 0.]),
        capabilities = [1],
        velocity = 50.0,
        travel_velocity = 50.0,
        collision_model = FCLRobotBBCollisionModel((500., 50., 300.), bf2),
    )
    agent3 = AgentModel(
        base_frame_position = bf3,
        home_position = np.array([-rh * np.cos(np.pi/4), rh * np.sin(np.pi/4), 0.]),
        capabilities = [1],
        velocity = 50.0,
        travel_velocity = 50.0,
        collision_model = FCLRobotBBCollisionModel((500., 50., 300.), bf3),
    )
    agent4 = AgentModel(
        base_frame_position = bf4,
        home_position = np.array([-rh * np.cos(np.pi/4), -rh * np.sin(np.pi/4), 0.]),
        capabilities = [0],
        velocity = 50.0,
        travel_velocity = 50.0,
        collision_model = FCLRobotBBCollisionModel((500., 50., 300.), bf4),
    )
    # fmt: on

    agent_models = {
        "robot1": agent1,
        "robot2": agent2,
        "robot3": agent3,
        "robot4": agent4,
    }

    filepath = "../test/test_gcode/multi_tool_demo.gcode"
    toolpath = toolpath_from_gcode(filepath)
    toolpath.contours = toolpath.contours[0:130]
    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=5.0,
        collision_offset=3.0,
        collision_gap_threshold=5.0,
    )

    print(f"{(80 * '#')}\nScheduling Complex Multi-material Toolpath:\n{(80 * '#')}\n")
    sched = planner.plan(toolpath, dg, options)
    print(f"\n{(80 * '#')}\nFound Toolpath Plan!\n{(80 * '#')}\n")
    print_schedule_info(sched)

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models, limits=((-400, 400), (-400, 400))
    )


if __name__ == "__main__":
    simple_toolpath_example_two()
    simple_toolpath_example_three()
    multi_material_squares()
    complex_two_material_two_robots()
    complex_two_material_four_robots()
