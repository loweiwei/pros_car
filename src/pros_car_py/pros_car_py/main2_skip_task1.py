import threading

import rclpy

from pros_car_py.arm_controller_2D import ArmController
from pros_car_py.car_controller import CarController
from pros_car_py.crane_controller import CraneController
from pros_car_py.custom_control import CustomControl
from pros_car_py.data_processor import DataProcessor
from pros_car_py.frontier_explorer import FrontierExplorer
from pros_car_py.ik_solver import PybulletRobotController
from pros_car_py.mode_app import ModeApp
from pros_car_py.nav_processing import Nav2Processing
from pros_car_py.ros_communicator import RosCommunicator
from pros_car_py.task_controller import TaskState, TaskType
from pros_car_py.task_controller_skip_task1 import TaskController as SkipTask1Controller


class TaskController(SkipTask1Controller):
    def _set_initial_target(self):
        if self.task_type == TaskType.BEAR_RECOVERY and self.SKIP_TASK1_TO_TASK2_STAGING:
            self._holding_target = False
            self._clear_stale_target_state()
            self.nav_processing.reset_nav_process()
            self._reset_active_path()
            self._task2_staging_position_aligned_count = 0
            self._task2_staging_yaw_aligned_count = 0
            self._bridge_bear_grabbed = False
            self._bridge_exit_stable_count = 0
            self._bridge_bear_close_precision_mode = False
            self._bridge_bear_close_jump_count = 0
            self._bridge_bear_close_precision_entered_at = None
            self._task2_start_bear_search_forward = False

            self._log_event("skip_task1_direct_to_task2_staging enabled=True")
            self._log_event(
                "task2_staging_direct_goal "
                f"position=[{self.TASK2_STAGING_POSITION[0]:.6f},{self.TASK2_STAGING_POSITION[1]:.6f}] "
                f"yaw_deg={self.TASK2_STAGING_YAW_DEG:.1f}"
            )
            self._transition(
                TaskState.NAV_TO_TASK2_STAGING,
                "Task1 skipped. Driving directly to task2 staging pose.",
            )
            return "STOP"

        return super()._set_initial_target()


def init_ros_node():
    rclpy.init()
    node = RosCommunicator()
    thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread.start()
    return node, thread


def main():
    ros_communicator, ros_thread = init_ros_node()
    data_processor = DataProcessor(ros_communicator)
    nav2_processing = Nav2Processing(ros_communicator, data_processor)
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
        app.main()
    finally:
        rclpy.shutdown()
        ros_thread.join()


if __name__ == "__main__":
    main()
