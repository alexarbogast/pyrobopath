import os
import unittest
from gcodeparser import GcodeParser

from pyrobopath.toolpath import Toolpath

TEST_GCODE1 = os.path.join(os.path.dirname(__file__), 'test_gcode', 'hollow_square.gcode')
TEST_GCODE2 = os.path.join(os.path.dirname(__file__), 'test_gcode', 'multi_tool_square.gcode')


class TestToolpath(unittest.TestCase):
    def test_parse_gcode(self):
        """Test gcode parsing capabilitity"""
        with open(TEST_GCODE1, "r") as f:
            gcode = f.read()
        parsed_gcode = GcodeParser(gcode)

        toolpath = Toolpath.from_gcode(parsed_gcode.lines)
        self.assertEqual(len(toolpath.contours), 128, "Number gcode contours != 128")
        
        with open(TEST_GCODE2, "r") as f:
            gcode = f.read()
        parsed_gcode = GcodeParser(gcode)

        toolpath = Toolpath.from_gcode(parsed_gcode.lines)
        self.assertEqual(len(toolpath.contours), 252, "Number gcode contours != 252")


if __name__ == "__main__":
    unittest.main()
