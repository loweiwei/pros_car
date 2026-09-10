from pros_car_py.task_controller import TaskController as BaseTaskController
from pros_car_py.task_controller import TaskState, TaskType


class TaskController(BaseTaskController):
    """TaskController variant that starts the full run from Task2 staging."""

    SKIP_TASK1_TO_TASK2_STAGING = True

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

            self._log_event("skip_task1_to_task2_staging enabled=True")
            self._log_event(
                "task2_staging_goal "
                f"position=[{self.TASK2_STAGING_POSITION[0]:.6f},{self.TASK2_STAGING_POSITION[1]:.6f}] "
                f"yaw_deg={self.TASK2_STAGING_YAW_DEG:.1f}"
            )
            self._transition(
                TaskState.PLAN_TO_TASK2_STAGING,
                "Task1 skipped. Planning path from start pose to task2 staging pose.",
            )
            return "STOP"

        return super()._set_initial_target()
