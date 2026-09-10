"""Entry point for the full autonomous task flow.

This version wires the complete TaskController, including Task 1, Task 2 with
bridge bear recovery, and Task 3 door operation.
"""

import urwid
import os
import threading
import rclpy
import time
import io
import sys
from pros_car_py.joint_config import JOINT_UPDATES_POSITIVE, JOINT_UPDATES_NEGATIVE
from pros_car_py.car_controller import CarController
from pros_car_py.arm_controller_2D import ArmController
from pros_car_py.data_processor import DataProcessor
from pros_car_py.nav_processing import Nav2Processing
from pros_car_py.ros_communicator import RosCommunicator
from pros_car_py.crane_controller import CraneController
from pros_car_py.custom_control import CustomControl
from pros_car_py.ik_solver import PybulletRobotController
from pros_car_py.mode_app import ModeApp
from pros_car_py.task_controller import TaskController
from pros_car_py.frontier_explorer import FrontierExplorer


def init_ros_node():
    """Start the shared ROS communicator in a background spin thread."""
    rclpy.init()
    node = RosCommunicator()
    thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread.start()
    return node, thread


def main():
    ros_communicator, ros_thread = init_ros_node()

    # DataProcessor keeps the latest sensor/perception state; Nav2Processing
    # turns pose, map, and target information into navigation decisions.
    data_processor = DataProcessor(ros_communicator)
    nav2_processing = Nav2Processing(ros_communicator, data_processor)

    # Controllers are intentionally created once and shared by the mode UI and
    # TaskController so manual control and autonomous tasks use the same outputs.
    ik_solver = PybulletRobotController(end_eff_index=5)
    car_controller = CarController(ros_communicator, nav2_processing)
    arm_controller = ArmController(ros_communicator, data_processor)
    crane_controller = CraneController(
        ros_communicator, data_processor, ik_solver, num_joints=7
    )
    custom_control = CustomControl(car_controller, arm_controller)
    frontier_explorer = FrontierExplorer(
        ros_communicator, data_processor, nav2_processing
    )

    # Full TaskController: includes bridge ascent, bridge bear search/grab,
    # descent, return home, and door operation states.
    task_controller = TaskController(
        car_controller,
        arm_controller,
        nav2_processing,
        ros_communicator,
        frontier_explorer,
    )
    app = ModeApp(
        car_controller,
        arm_controller,
        custom_control,
        crane_controller,
        task_controller,
    )

    try:
        # ModeApp owns the terminal menu and repeatedly calls controller logic.
        app.main()
    finally:
        # Always stop ROS cleanly so serial devices and ROS resources are freed.
        rclpy.shutdown()
        ros_thread.join()


if __name__ == "__main__":
    main()
