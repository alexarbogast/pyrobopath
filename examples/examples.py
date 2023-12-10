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
    # create agent collision models
    agent1 = AgentModel()
    agent1.base_frame_position = np.array([-350.0, 0.0, 0.0])
    agent1.home_position = np.array([-250.0, 0.0, 0.0])
    agent1.capabilities = [Materials.MATERIAL_A]
    agent1.velocity = 50.0
    agent1.travel_velocity = 50.0
    agent1.collision_model = FCLRobotBBCollisionModel(
        200.0, 50.0, 300.0, agent1.base_frame_position
    )
    agent2 = AgentModel()
    agent2.base_frame_position = np.array([350.0, 0.0, 0.0])
    agent2.home_position = np.array([250.0, 0.0, 0.0])
    agent2.capabilities = [Materials.MATERIAL_B]
    agent2.velocity = 50.0
    agent2.travel_velocity = 50.0
    agent2.collision_model = FCLRobotBBCollisionModel(
        200.0, 50.0, 300.0, agent2.base_frame_position
    )
    agent_models = {"robot1": agent1, "robot2": agent2}

    # create toolpath
    toolpath = create_example_toolpath()
    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=0.1,
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

    # fmt: off
    agent1 = AgentModel()
    agent1.base_frame_position = np.array([r * np.cos(np.pi/6), -r * np.sin(np.pi/6), 0.])
    agent1.home_position = np.array([rh * np.cos(np.pi/6), -rh * np.sin(np.pi/6), 0.])
    agent1.capabilities = [Materials.MATERIAL_B]
    agent1.velocity = 50.0
    agent1.travel_velocity = 50.0
    agent1.collision_model = FCLRobotBBCollisionModel(
        200., 50., 300., agent1.base_frame_position
    )
    agent2 = AgentModel()
    agent2.base_frame_position = np.array([0., r, 0.])
    agent2.home_position = np.array([0., rh, 0.])
    agent2.capabilities = [Materials.MATERIAL_A]
    agent2.velocity = 50.0
    agent2.travel_velocity = 50.0
    agent2.collision_model = FCLRobotBBCollisionModel(
        200., 50., 300., agent2.base_frame_position
    )
    agent3 = AgentModel()
    agent3.base_frame_position = np.array([-r * np.cos(np.pi/6), -r * np.sin(np.pi/6), 0.])
    agent3.home_position = np.array([-rh * np.cos(np.pi/6), -rh * np.sin(np.pi/6), 0.])
    agent3.capabilities = [Materials.MATERIAL_A]
    agent3.velocity = 50.0
    agent3.travel_velocity = 50.0
    agent3.collision_model = FCLRobotBBCollisionModel(
        200., 50., 300., agent3.base_frame_position
    )
    # fmt: on
    agent_models = {"robot1": agent1, "robot2": agent2, "robot3": agent3}

    # create toolpath
    toolpath = create_example_toolpath()
    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=0.1,
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
    # create agent_models
    agent1 = AgentModel()
    agent1.base_frame_position = np.array([-500.0, 0.0, 0.0])
    agent1.home_position = np.array([-300, 0.0, 0.0])
    agent1.capabilities = [0]
    agent1.velocity = 50.0
    agent1.travel_velocity = 50.0
    agent1.collision_model = FCLRobotBBCollisionModel(
        200.0, 50.0, 300, agent1.base_frame_position
    )
    agent2 = AgentModel()
    agent2.base_frame_position = np.array([500.0, 0.0, 0.0])
    agent2.home_position = np.array([300.0, 0.0, 0.0])
    agent2.capabilities = [1]
    agent2.velocity = 50.0
    agent2.travel_velocity = 50.0
    agent2.collision_model = FCLRobotBBCollisionModel(
        200.0, 50.0, 300, agent2.base_frame_position
    )
    agent_models = {"robot1": agent1, "robot2": agent2}

    filepath = "../test/test_gcode/multi_tool_square.gcode"
    toolpath = toolpath_from_gcode(filepath)
    toolpath.contours = toolpath.contours[0:48]

    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=0.1,
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
    # create agent_models
    agent1 = AgentModel()
    agent1.base_frame_position = np.array([-500.0, 0.0, 0.0])
    agent1.home_position = np.array([-300, 0.0, 0.0])
    agent1.capabilities = [0]
    agent1.velocity = 50.0
    agent1.travel_velocity = 50.0
    agent1.collision_model = FCLRobotBBCollisionModel(
        200.0, 50.0, 300, agent1.base_frame_position
    )
    agent2 = AgentModel()
    agent2.base_frame_position = np.array([500.0, 0.0, 0.0])
    agent2.home_position = np.array([300.0, 0.0, 0.0])
    agent2.capabilities = [1]
    agent2.velocity = 50.0
    agent2.travel_velocity = 50.0
    agent2.collision_model = FCLRobotBBCollisionModel(
        200.0, 50.0, 300, agent2.base_frame_position
    )
    agent_models = {"robot1": agent1, "robot2": agent2}

    filepath = "../test/test_gcode/multi_tool_demo.gcode"
    toolpath = toolpath_from_gcode(filepath)
    toolpath.contours = toolpath.contours[0:130]
    dg = create_dependency_graph_by_layers(toolpath)

    # create planner
    planner = MultiAgentToolpathPlanner(agent_models)
    options = PlanningOptions(
        retract_height=0.1,
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

    # fmt: off
    agent1 = AgentModel()
    agent1.base_frame_position = np.array([r * np.cos(np.pi/4), -r * np.sin(np.pi/4), 0.])
    agent1.home_position = np.array([rh * np.cos(np.pi/4), -rh * np.sin(np.pi/4), 0.])
    agent1.capabilities = [0]
    agent1.velocity = 50.0
    agent1.travel_velocity = 50.0
    agent1.collision_model = FCLRobotBBCollisionModel(
        500., 50., 300., agent1.base_frame_position
    )
    agent2 = AgentModel()
    agent2.base_frame_position = np.array([r * np.cos(np.pi/4), r * np.sin(np.pi/4), 0.])
    agent2.home_position = np.array([rh * np.cos(np.pi/4), rh * np.sin(np.pi/4), 0.])
    agent2.capabilities = [1]
    agent2.velocity = 50.0
    agent2.travel_velocity = 50.0
    agent2.collision_model = FCLRobotBBCollisionModel(
        500., 50., 300., agent2.base_frame_position
    )
    agent3 = AgentModel()
    agent3.base_frame_position = np.array([-r * np.cos(np.pi/4), r * np.sin(np.pi/4), 0.])
    agent3.home_position = np.array([-rh * np.cos(np.pi/4), rh * np.sin(np.pi/4), 0.])
    agent3.capabilities = [1]
    agent3.velocity = 50.0
    agent3.travel_velocity = 50.0
    agent3.collision_model = FCLRobotBBCollisionModel(
        500., 50., 300., agent3.base_frame_position
    )
    agent4 = AgentModel()
    agent4.base_frame_position = np.array([-r * np.cos(np.pi/4), -r * np.sin(np.pi/4), 0.])
    agent4.home_position = np.array([-rh * np.cos(np.pi/4), -rh * np.sin(np.pi/4), 0.])
    agent4.capabilities = [0]
    agent4.velocity = 50.0
    agent4.travel_velocity = 50.0
    agent4.collision_model = FCLRobotBBCollisionModel(
        500., 50., 300., agent4.base_frame_position
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
        retract_height=0.1,
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
