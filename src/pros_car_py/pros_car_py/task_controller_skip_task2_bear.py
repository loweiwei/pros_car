import time

from pros_car_py.task_controller import TaskController as BaseTaskController
from pros_car_py.task_controller import TaskState, TaskType


class TaskController(BaseTaskController):
    """TaskController variant that skips grabbing the bear after bridge ascent."""

    SKIP_TASK2_BRIDGE_BEAR_GRAB = True
    SKIP_TASK2_BEAR_DESCEND_DISTANCE_M = 0.38
    SKIP_TASK2_BRIDGE_STRAIGHT_YAW_DEG = 90.0
    SKIP_TASK2_BRIDGE_STRAIGHT_YAW_TOLERANCE_DEG = 0.5
    SKIP_TASK2_BRIDGE_STRAIGHT_ACTION = "FORWARD_BRIDGE"
    SKIP_TASK2_BRIDGE_STRAIGHT_LEFT_ACTION = "FORWARD_BRIDGE_LEFT"
    SKIP_TASK2_BRIDGE_STRAIGHT_RIGHT_ACTION = "FORWARD_BRIDGE_RIGHT"
    DOOR_STAGING_FORWARD_ACTION = "FORWARD"
    TASK3_FROM_BRIDGE_ASTAR_RETRY_SECONDS = 1.5
    TASK3_FROM_BRIDGE_TURN_RIGHT_DEG = 90.0
    TASK3_FROM_BRIDGE_TURN_TOLERANCE_DEG = 8.0
    TASK3_FROM_BRIDGE_INTERMEDIATE_POSITION = [1.777592533703811, 2.846921790442113]
    TASK3_FROM_BRIDGE_INTERMEDIATE_REACHED_DISTANCE_M = 0.20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task3_from_bridge_exit = False
        self._task3_from_bridge_astar_retry_until = 0.0
        self._task3_from_bridge_exit_phase = None
        self._task3_from_bridge_turn_target_yaw = None

    def start(self, task_type):
        self._task3_from_bridge_exit = False
        self._task3_from_bridge_astar_retry_until = 0.0
        self._task3_from_bridge_exit_phase = None
        self._task3_from_bridge_turn_target_yaw = None
        return super().start(task_type)

    def _bridge_straight_yaw_action(self, context):
        pose = self._get_current_pose(require_map_frame=True)
        if pose is None:
            self._log_event(
                f"bridge_straight_yaw_no_pose context={context} "
                f"action={self.SKIP_TASK2_BRIDGE_STRAIGHT_ACTION}"
            )
            return self.SKIP_TASK2_BRIDGE_STRAIGHT_ACTION

        _, orientation = pose
        current_yaw = self._yaw_from_quaternion(orientation)
        yaw_error = self._normalize_yaw_deg(
            float(self.SKIP_TASK2_BRIDGE_STRAIGHT_YAW_DEG) - float(current_yaw)
        )

        if abs(yaw_error) <= float(self.SKIP_TASK2_BRIDGE_STRAIGHT_YAW_TOLERANCE_DEG):
            action = self.SKIP_TASK2_BRIDGE_STRAIGHT_ACTION
        elif yaw_error > 0.0:
            action = self.SKIP_TASK2_BRIDGE_STRAIGHT_LEFT_ACTION
        else:
            action = self.SKIP_TASK2_BRIDGE_STRAIGHT_RIGHT_ACTION

        self._log_event(
            f"bridge_straight_yaw context={context} "
            f"current_yaw={self._format_float(current_yaw, 2)} "
            f"target_yaw={self.SKIP_TASK2_BRIDGE_STRAIGHT_YAW_DEG:.2f} "
            f"tolerance={self.SKIP_TASK2_BRIDGE_STRAIGHT_YAW_TOLERANCE_DEG:.2f} "
            f"error={self._format_float(yaw_error, 2)} "
            f"action={action}"
        )
        return action

    def _ascend_bridge(self):
        if self._elapsed() < self.ASCEND_SECONDS:
            action = self._bridge_straight_yaw_action("ascend_bridge")
            self._log_event(f"bridge_ascend_action action={action}")
            return self._apply_obstacle_guard(action, None, "ascend_bridge")

        self._log_event("ascend_bridge_done")
        if self.SKIP_TASK2_BRIDGE_BEAR_GRAB:
            self._bridge_exit_stable_count = 0
            self._return_started_at = None
            self._set_vision(self.DETECTION_MODE, self.BEAR_LABEL)
            self._log_event("skip_task2_bridge_bear_grab enabled=True align_until_close=True")
            self._log_event("bridge_bear_detection_enabled mode=detection label=bear")
            self._transition(
                TaskState.SEARCH_BEAR_ON_BRIDGE,
                "Ascent complete. Aligning to bridge bear until close, then descending.",
            )
            return "STOP"

        return super()._ascend_bridge()

    def _bridge_bear_close_enough_to_descend(self):
        if not (
            self.SKIP_TASK2_BRIDGE_BEAR_GRAB
            and self.task_type == TaskType.BRIDGE_RECOVERY
            and self._target_label == self.BEAR_LABEL
        ):
            return False

        info = self._get_yolo_target_info()
        if not info or info[0] != 1:
            return False

        distance = self._safe_float(info[1])
        delta_x = self._safe_float(info[2])
        if distance is None or distance > self.SKIP_TASK2_BEAR_DESCEND_DISTANCE_M:
            return False

        self._log_event(
            f"skip_task2_bear_close_enough_descend "
            f"distance={distance:.3f} "
            f"threshold={self.SKIP_TASK2_BEAR_DESCEND_DISTANCE_M:.3f} "
            f"delta_x={self._format_float(delta_x, 1)}"
        )
        self._observe_started_at = None
        self._bridge_bear_observe_bad_distance_count = 0
        self._bridge_exit_stable_count = 0
        self._clear_stale_target_state()
        self.nav_processing.reset_nav_process()
        self._transition(
            TaskState.DESCEND_BRIDGE,
            "Bridge bear is close enough. Ignoring bear and descending straight.",
        )
        return True

    def _approach_target(self):
        if self._bridge_bear_close_enough_to_descend():
            return "STOP"

        return super()._approach_target()

    def _observe_target(self):
        if self._bridge_bear_close_enough_to_descend():
            return "STOP"

        return super()._observe_target()

    def _start_task3_from_bridge_exit(self):
        self._reset_active_path()
        self.nav_processing.reset_nav_process()
        self._clear_stale_target_state()

        self.task_type = TaskType.DOOR_UNLOCK
        self._holding_target = False
        self._knob_map_positions = []
        self._stable_knob_map_position = None
        self._door_pre_press_published = False
        self._door_pre_press_started_at = None
        self._door_press_started = False
        self._door_press_done = False
        self._door_press_started_at = None
        self._door_push_started_at = None
        self._door_push_start_position = None
        self._door_staging_position_aligned_count = 0
        self._door_staging_yaw_aligned_count = 0
        self._task3_knob_yolo_align_count = 0
        self._task3_knob_yolo_aligned_info = None
        self._task3_knob_yolo_enabled = False
        self._task3_knob_yolo_disabled = False
        self._drive_to_knob_contact_started_at = None
        self._drive_to_knob_contact_start_position = None
        self._task3_from_bridge_exit = True
        self._task3_from_bridge_astar_retry_until = 0.0
        self._task3_from_bridge_exit_phase = None
        self._task3_from_bridge_turn_target_yaw = None

        self._log_event("skip_task2_bear_start_task3_from_bridge_intermediate")
        self._log_event("task3_knob_yolo_flow_enabled arm_kept_in_vision_pose")
        self._move_arm_to_vision_pose("skip_task2_bear_before_task3_knob_yolo")
        self._transition(
            TaskState.PLAN_TO_DOOR_STAGING,
            "Reached bridge-exit intermediate point. Starting Task3 from there.",
        )

    def _handle_task3_from_bridge_exit_phase(self):
        if self._task3_from_bridge_exit_phase == "turn_right":
            if self._task3_from_bridge_turn_target_yaw is None:
                pose = self._get_current_pose(require_map_frame=True)
                if pose is None:
                    self._log_event("task3_from_bridge_turn_right_no_pose")
                    return "STOP"

                _, orientation = pose
                current_yaw = self._yaw_from_quaternion(orientation)
                self._task3_from_bridge_turn_target_yaw = self._normalize_yaw_deg(
                    current_yaw - float(self.TASK3_FROM_BRIDGE_TURN_RIGHT_DEG)
                )
                self._log_event(
                    f"task3_from_bridge_turn_right_start "
                    f"current_yaw={self._format_float(current_yaw, 2)} "
                    f"target_yaw={self._format_float(self._task3_from_bridge_turn_target_yaw, 2)}"
                )

            action, yaw_error, current_yaw = self._align_to_map_yaw_action(
                self._task3_from_bridge_turn_target_yaw,
                self.TASK3_FROM_BRIDGE_TURN_TOLERANCE_DEG,
                context="task3_from_bridge_turn_right",
            )
            self._log_event(
                f"task3_from_bridge_turn_right "
                f"current_yaw={self._format_float(current_yaw, 2)} "
                f"target_yaw={self._format_float(self._task3_from_bridge_turn_target_yaw, 2)} "
                f"error={self._format_float(yaw_error, 2)} "
                f"action={action}"
            )

            if action == "STOP":
                self._task3_from_bridge_exit_phase = "intermediate"
                self.nav_processing.reset_nav_process()
                self._log_event("task3_from_bridge_turn_right_done_start_intermediate")
                return "STOP"

            return action

        if self._task3_from_bridge_exit_phase == "intermediate":
            current_position = self._get_current_position(require_map_frame=True)
            distance = self._distance(
                current_position,
                self.TASK3_FROM_BRIDGE_INTERMEDIATE_POSITION,
            ) if current_position else None
            self._log_event(
                f"task3_from_bridge_intermediate_nav "
                f"current={current_position} "
                f"target={self.TASK3_FROM_BRIDGE_INTERMEDIATE_POSITION} "
                f"distance={self._format_float(distance, 3)} "
                f"threshold={self.TASK3_FROM_BRIDGE_INTERMEDIATE_REACHED_DISTANCE_M:.3f}"
            )

            if distance is not None and distance <= self.TASK3_FROM_BRIDGE_INTERMEDIATE_REACHED_DISTANCE_M:
                self._log_event("task3_from_bridge_intermediate_reached")
                self._start_task3_from_bridge_exit()
                return "STOP"

            return self._navigate_to_waypoint(
                self.TASK3_FROM_BRIDGE_INTERMEDIATE_POSITION,
                reached_distance=self.TASK3_FROM_BRIDGE_INTERMEDIATE_REACHED_DISTANCE_M,
                context="task3_from_bridge_intermediate",
            )

        return None

    def _descend_bridge(self):
        if not self.SKIP_TASK2_BRIDGE_BEAR_GRAB:
            return super()._descend_bridge()

        phase_action = self._handle_task3_from_bridge_exit_phase()
        if phase_action is not None:
            return phase_action

        current_position = self._get_current_position(require_map_frame=True)
        current_y = None
        if current_position is not None and len(current_position) >= 2:
            current_y = current_position[1]

        self._log_event(
            f"descend_bridge_skip_bear_check current={current_position} "
            f"current_y={self._format_float(current_y, 3)} "
            f"exit_y_threshold={self.BRIDGE_EXIT_Y_THRESHOLD:.3f} "
            f"stable_count={self._bridge_exit_stable_count}/{self.BRIDGE_EXIT_STABLE_COUNT}"
        )

        if current_y is not None and current_y >= self.BRIDGE_EXIT_Y_THRESHOLD:
            self._bridge_exit_stable_count += 1
        else:
            self._bridge_exit_stable_count = 0

        if self._bridge_exit_stable_count >= self.BRIDGE_EXIT_STABLE_COUNT:
            self._log_event(
                f"bridge_exit_reached_skip_bear "
                f"current_y={current_y:.3f} "
                f"threshold={self.BRIDGE_EXIT_Y_THRESHOLD:.3f}"
            )
            self._bridge_exit_stable_count = 0
            self._reset_active_path()
            self.nav_processing.reset_nav_process()
            self._clear_stale_target_state()
            self._task3_from_bridge_exit_phase = "turn_right"
            self._task3_from_bridge_turn_target_yaw = None
            self.status_message = "Exited bridge. Turning right before intermediate point."
            self._log_event("bridge_exit_start_turn_right_before_task3_intermediate")
            return "STOP"

        action = self._bridge_straight_yaw_action("descend_bridge_skip_bear")
        self._log_event(f"descend_bridge_skip_bear_straight_yaw action={action}")
        return action

    def _plan_to_door_staging(self):
        if not self._task3_from_bridge_exit:
            return super()._plan_to_door_staging()

        if self.DOOR_STAGING_POSITION is None:
            self._log_event("door_staging_not_configured")
            self._transition(TaskState.FAILED, "Door staging is not configured for direct door opening.")
            return "STOP"

        now = time.monotonic()
        if now < self._task3_from_bridge_astar_retry_until:
            self._log_event(
                f"task3_from_bridge_door_staging_plan_retry_wait "
                f"until={self._task3_from_bridge_astar_retry_until:.2f}"
            )
            self._transition(
                TaskState.NAV_TO_DOOR_STAGING,
                "Waiting to retry door staging A*. Driving directly meanwhile.",
            )
            return "STOP"

        self._reset_active_path()
        self._door_staging_position_aligned_count = 0
        goal = list(self.DOOR_STAGING_POSITION)
        self._log_event(f"task3_from_bridge_door_staging_plan_start goal={goal}")
        if self._plan_astar_path(goal, "door_staging"):
            path_len = len(self._active_path) if self._active_path else 0
            self._task3_from_bridge_exit = False
            self._task3_from_bridge_astar_retry_until = 0.0
            self._log_event(f"task3_from_bridge_door_staging_plan_success path_len={path_len}")
            self._transition(TaskState.NAV_TO_DOOR_STAGING, "Following path to direct door opening pose from bridge exit.")
            return "STOP"

        self._task3_from_bridge_astar_retry_until = now + float(self.TASK3_FROM_BRIDGE_ASTAR_RETRY_SECONDS)
        self._log_event(
            f"task3_from_bridge_door_staging_plan_failed_fallback_direct_retry_later "
            f"retry_seconds={self.TASK3_FROM_BRIDGE_ASTAR_RETRY_SECONDS:.2f}"
        )
        self._transition(
            TaskState.NAV_TO_DOOR_STAGING,
            "Door staging A* failed. Driving directly while retrying later.",
        )
        return "STOP"

    def _nav_to_door_staging(self):
        if self._task3_from_bridge_exit:
            now = time.monotonic()
            if now >= self._task3_from_bridge_astar_retry_until:
                goal = list(self.DOOR_STAGING_POSITION) if self.DOOR_STAGING_POSITION is not None else None
                if goal is not None:
                    self._log_event(f"task3_from_bridge_door_staging_retry_plan goal={goal}")
                    if self._plan_astar_path(goal, "door_staging"):
                        path_len = len(self._active_path) if self._active_path else 0
                        self._task3_from_bridge_exit = False
                        self._task3_from_bridge_astar_retry_until = 0.0
                        self._log_event(
                            f"task3_from_bridge_door_staging_retry_plan_success path_len={path_len}"
                        )
                    else:
                        self._task3_from_bridge_astar_retry_until = now + float(
                            self.TASK3_FROM_BRIDGE_ASTAR_RETRY_SECONDS
                        )
                        self._log_event(
                            f"task3_from_bridge_door_staging_retry_plan_failed_keep_direct "
                            f"retry_seconds={self.TASK3_FROM_BRIDGE_ASTAR_RETRY_SECONDS:.2f}"
                        )

        return super()._nav_to_door_staging()
