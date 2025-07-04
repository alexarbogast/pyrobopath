from pyrobopath.toolpath import visualize_toolpath, visualize_toolpath_projection
from pyrobopath.toolpath.path.transform import Rotation
from pyrobopath.toolpath.preprocessing import *

from utilities import Materials, toolpath_from_gcode


# =========================== visualization ============================
def toolpath_visualization_example():
    filepath = "../test/test_gcode/multi_tool_square.gcode"
    toolpath = toolpath_from_gcode(filepath)
    visualize_toolpath(toolpath, backend="matplotlib", color_method="tool")
    visualize_toolpath(toolpath, backend="pyqtgraph", color_method="cycle")


def toolpath_layer_visualization():
    filepath = "../test/test_gcode/multi_tool_square.gcode"
    toolpath = toolpath_from_gcode(filepath)
    visualize_toolpath_projection(toolpath)


# =========================== preprocessing ============================
def toolpath_preprocessing_example():
    filepath = "../test/test_gcode/multi_tool_square.gcode"
    toolpath = toolpath_from_gcode(filepath)

    # visualize_toolpath before modifications
    visualize_toolpath(toolpath, backend="matplotlib", color_method="cycle", show=False)

    # preprocessing (see other preprocessing steps in pyrobopath documentation)
    preprocessor = ToolpathPreprocessor()
    preprocessor.add_step(TranslateStep([100, 0, 0]))
    preprocessor.add_step(RotateStep(Rotation.Rz(np.pi / 4)))
    preprocessor.add_step(ScalingStep(0.001))
    preprocessor.add_step(MaxContourLengthStep(1.0))
    preprocessor.add_step(LayerRangeStep(5, 10))
    preprocessor.add_step(
        SubstituteToolStep({0: Materials.MATERIAL_A, 1: Materials.MATERIAL_B})
    )

    preprocessor.process(toolpath)

    visualize_toolpath(toolpath, color_method="cycle", backend="matplotlib")


if __name__ == "__main__":
    toolpath_visualization_example()
    toolpath_layer_visualization()
    toolpath_preprocessing_example()
