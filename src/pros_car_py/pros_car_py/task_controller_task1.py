import time
import os
import copy
import math
from enum import Enum


class TaskType(Enum):
    BEAR_RECOVERY = "task_1_bear_recovery"
    BRIDGE_RECOVERY = "task_2_bridge_recovery"
    DOOR_UNLOCK = "task_3_door_unlock"


class TaskState(Enum):
    IDLE = "idle"
    RECORD_HOME = "record_home"
    SET_VISION_TARGET = "set_vision_target"
    SEARCH_TARGET = "search_target"
    APPROACH_TARGET = "approach_target"
    OBSERVE = "observe"
    GRAB_TARGET = "grab_target"
    RETURN_HOME = "return_home"
    EXPLORE_MAP = "explore_map"
    SEARCH_BRIDGE = "search_bridge"
    ASCEND_BRIDGE = "ascend_bridge"
    DESCEND_BRIDGE = "descend_bridge"
    UNLOCK_DOOR = "unlock_door"
    PUSH_DOOR = "push_door"
    DONE = "done"
    FAILED = "failed"


class TaskController:
    """Runs high-level final-project tasks as a non-blocking state machine."""

    TARGET_DISTANCE_M = 0.7

    # Bear 會在接近時被手臂/車體遮住，0.9m 太晚。
    # 提早在 1.05m 停下觀察，避免 YOLO/depth 開始跳掉。
    APPROACH_STOP_DISTANCE = 1.05

    # 抓取前必須把目標放到畫面正中間附近。
    # 注意：nav_processing.camera_nav() 的 centered_px 可能比較寬，
    # 但手臂抓取需要更嚴格的置中門檻。
    GRAB_ALIGN_DELTA_X = 50.0
    GRAB_CAMERA_DISTANCE_THRESHOLD = 0.39
    # 只有在目標已經置中時才允許前進靠近。
    # 這避免車子斜著靠近，導致手臂最後偏抓。
    APPROACH_ALIGN_DELTA_X = 50.0

    # Outlier 判斷參數
    OUTLIER_LAST_DISTANCE_MAX = 1.5
    OUTLIER_CURRENT_DISTANCE_MIN = 2.0
    OUTLIER_DELTA_X_JUMP = 120.0
    OUTLIER_MAX_COUNT = 10
    OUTLIER_CLOSE_ENOUGH_DISTANCE = 1.05

    # 近距離鎖定：
    # YOLO 在 bear 靠近後可能因為太近、遮擋、畫面邊緣而丟失近熊，
    # 然後突然切到遠方另一隻 bear。當上一筆穩定距離已經小於此值時，
    # 若下一筆突然跳遠，視為「目標切換/近熊短暫丟失」，不要 search。
    CLOSE_RANGE_LOCK_DISTANCE = 1.35

    # 短暫 lost target 的容錯
    LOST_TARGET_MAX_COUNT = 8
    LOST_CLOSE_ENOUGH_DISTANCE = 1.05

    MIN_APPROACH_CAMERA_DISTANCE = 0.35
    OBSERVE_SECONDS = 5.0
    SEARCH_TIMEOUT_SECONDS = 45.0
    RETURN_TIMEOUT_SECONDS = 90.0
    GRAB_SECONDS = 18.0
    UNLOCK_SECONDS = 5.0
    PUSH_SECONDS = 3.0
    ASCEND_SECONDS = 5.0
    DESCEND_SECONDS = 5.0

    OBSTACLE_GUARD_ENABLED = True
    OBSTACLE_FRONT_STOP_DISTANCE = 0.45
    OBSTACLE_EMERGENCY_STOP_DISTANCE = 0.20
    OBSTACLE_SIDE_CLEAR_DISTANCE = 0.55
    OBSTACLE_TARGET_DEPTH_TOLERANCE = 0.18
    OBSTACLE_TARGET_ALIGN_DELTA_X = 80.0

    # LiDAR /scan safety settings.
    OBSTACLE_USE_LIDAR_SCAN = True
    OBSTACLE_SCAN_FRONT_DEG = 28.0
    OBSTACLE_SCAN_SIDE_FRONT_DEG = 80.0
    OBSTACLE_SCAN_SIDE_DEG_MIN = 65.0
    OBSTACLE_SCAN_SIDE_DEG_MAX = 115.0

    # Filter near-field LiDAR self-hit/noise.
    # Logs showed false/near-field scan_front around 0.18~0.26m while the
    # camera/YOLO target was around 0.9~2.6m. Treat very near scan hits as
    # self-hit/noise unless they are below the hard emergency threshold.
    OBSTACLE_LIDAR_SELF_FILTER_DISTANCE = 0.24
    OBSTACLE_SCAN_MIN_VALID_POINTS = 3
    OBSTACLE_SCAN_PERCENTILE = 0.10

    # Side-wall / rotation protection.
    # Keep true side-wall protection, but do not let a single front-right
    # corner ray at ~0.28m freeze target alignment forever.
    OBSTACLE_SIDE_FRONT_STOP_DISTANCE = 0.24
    OBSTACLE_SIDE_STOP_DISTANCE = 0.25
    OBSTACLE_ROTATION_SIDE_STOP_DISTANCE = 0.30
    OBSTACLE_ROTATION_FRONT_CORNER_STOP_DISTANCE = 0.24
    OBSTACLE_CORRIDOR_BALANCE_DISTANCE = 0.42
    OBSTACLE_CORRIDOR_BALANCE_DELTA = 0.16

    # During centered target approach, camera/YOLO is more reliable than a
    # noisy near-field scan hit for deciding whether to creep forward.
    OBSTACLE_TARGET_APPROACH_CAMERA_CLEAR_DISTANCE = 0.70
    OBSTACLE_TARGET_APPROACH_DELTA_X = 80.0

    # When visually approaching a bear, LiDAR can see the car body / nearby wall
    # at ~0.24~0.27m while the camera target is still clear. Use lower hard
    # thresholds in that special context, otherwise the car keeps steering away
    # from the target and loses it.
    OBSTACLE_APPROACH_SIDE_STOP_DISTANCE = 0.20
    OBSTACLE_APPROACH_ROTATION_SIDE_STOP_DISTANCE = 0.20
    OBSTACLE_APPROACH_ROTATION_FRONT_CORNER_STOP_DISTANCE = 0.20

    # During visual approach, /odom from scan_matcher can barely change even
    # though YOLO distance is decreasing. Do NOT blindly disable stuck forever;
    # instead, use YOLO distance progress to distinguish real movement from a
    # true stall.
    OBSTACLE_DISABLE_STUCK_DURING_TARGET_APPROACH = False
    OBSTACLE_TARGET_PROGRESS_EPS = 0.015
    OBSTACLE_TARGET_PROGRESS_GRACE_SECONDS = 0.9

    # Far target approach was too slow with FORWARD_VERY_SLOW. Use FORWARD_SLOW
    # until the target is close, then creep carefully.
    APPROACH_FAST_UNTIL_DISTANCE = 0.80

    # Stuck detection using pose from /odom fallback.
    OBSTACLE_STUCK_CHECK_SECONDS = 1.0
    OBSTACLE_STUCK_SECONDS = 0.8
    OBSTACLE_STUCK_MIN_MOVE = 0.02
    OBSTACLE_STUCK_RECOVERY_SECONDS = 0.7

    BEAR_LABEL = "bear"
    KNOB_LABEL = "knob"
    BRIDGE_LABEL = "bridge"

    DETECTION_MODE = "detection"
    SEGMENTATION_MODE = "segmentation"

    def __init__(
        self,
        car_controller,
        arm_controller,
        nav_processing,
        ros_communicator,
        frontier_explorer=None,
    ):
        self.car_controller = car_controller
        self.arm_controller = arm_controller
        self.nav_processing = nav_processing
        self.ros_communicator = ros_communicator
        self.frontier_explorer = frontier_explorer

        self.task_type = None
        self.state = TaskState.IDLE
        self.home_position = None
        self.state_started_at = None
        self.status_message = "Task controller idle."

        self._observe_started_at = None
        self._return_started_at = None
        self._target_label = None
        self._vision_mode = None

        self._last_action = "STOP"
        self._last_valid_target_info = None

        # 近距離 target lock：
        # 用來避免多隻 bear 時，近熊短暫消失後 target_info 立刻跳到遠熊。
        self._target_lock_active = False
        self._locked_target_info = None
        self._locked_yolo_marker = None
        self._locked_detection_position = None

        # 避免 outlier / lost target 讓 state machine 卡死或亂跳。
        self._outlier_count = 0
        self._lost_target_count = 0

        self._grab_started = False

        # True 表示已經夾住 bear/目標，車子還沒回到起始點。
        # 這段期間禁止任何 vision pose / reset pose 打開夾爪。
        self._holding_target = False

        self._last_debug_at = 0.0

        # Runtime state for stuck recovery.
        self._last_stuck_pose = None
        self._last_stuck_time = None
        self._stuck_started_at = None
        self._recovery_until = 0.0
        self._recovery_action = "STOP"

        # Used by visual approach stuck detection: if YOLO distance is still
        # decreasing, do not call it stuck even when /odom is noisy/static.
        self._target_progress_last_distance = None
        self._target_progress_last_time = None

        self.log_path = os.environ.get("PROS_TASK_LOG_PATH", "/tmp/pros_task_debug.log")
        self._log_event("TaskController initialized")

    def start(self, task_type):
        if isinstance(task_type, str):
            task_type = TaskType(task_type)

        self.task_type = task_type
        self.home_position = None
        self._observe_started_at = None
        self._return_started_at = None
        self._target_label = None
        self._vision_mode = None

        self._last_action = "STOP"
        self._last_valid_target_info = None
        self._target_lock_active = False
        self._locked_target_info = None
        self._locked_yolo_marker = None
        self._locked_detection_position = None
        self._outlier_count = 0
        self._lost_target_count = 0

        self._grab_started = False
        self._holding_target = False

        self._last_stuck_pose = None
        self._last_stuck_time = None
        self._stuck_started_at = None
        self._recovery_until = 0.0
        self._recovery_action = "STOP"
        self._target_progress_last_distance = None
        self._target_progress_last_time = None

        self._log_event(f"start task={task_type.value}")

        # 關鍵修改：
        # 任務一開始先把手臂抬到不擋相機的位置，避免靠近 bear 時手臂遮住相機。
        self._move_arm_to_vision_pose("task_start")

        self.nav_processing.reset_nav_process()

        if self.frontier_explorer:
            self.frontier_explorer.reset()

        self._transition(TaskState.RECORD_HOME, "Recording start pose as home.")

    def stop(self):
        self.car_controller.update_action("STOP")
        self._log_event("stop requested")

        # 如果手上已經夾著 bear，停止任務時也不能打開夾爪。
        if getattr(self, "_holding_target", False):
            self._move_arm_to_carry_pose("task_stop_holding_target")
        else:
            # 沒有夾東西時，才可以用一般 vision pose。
            self._move_arm_to_vision_pose("task_stop")

        self.nav_processing.reset_nav_process()

        if self.frontier_explorer:
            self.frontier_explorer.reset()

        self._transition(TaskState.IDLE, "Task stopped.")

    def is_running(self):
        return self.state not in (TaskState.IDLE, TaskState.DONE, TaskState.FAILED)

    def tick(self):
        if self.state == TaskState.IDLE:
            return "STOP"

        try:
            if self.state == TaskState.RECORD_HOME:
                return self._record_home()

            if self.state == TaskState.SET_VISION_TARGET:
                return self._set_initial_target()

            if self.state == TaskState.SEARCH_TARGET:
                return self._search_target()

            if self.state == TaskState.APPROACH_TARGET:
                return self._approach_target()

            if self.state == TaskState.OBSERVE:
                return self._observe_target()

            if self.state == TaskState.GRAB_TARGET:
                return self._grab_target()

            if self.state == TaskState.RETURN_HOME:
                return self._return_home()

            if self.state == TaskState.EXPLORE_MAP:
                return self._explore_map()

            if self.state == TaskState.SEARCH_BRIDGE:
                return self._search_bridge()

            if self.state == TaskState.ASCEND_BRIDGE:
                return self._timed_drive(
                    TaskState.DESCEND_BRIDGE,
                    self.ASCEND_SECONDS,
                    "FORWARD_SLOW",
                    "Ascent complete. Driving down from bridge.",
                )

            if self.state == TaskState.DESCEND_BRIDGE:
                return self._timed_drive(
                    TaskState.SET_VISION_TARGET,
                    self.DESCEND_SECONDS,
                    "FORWARD_SLOW",
                    "Descent complete. Searching bear on/near bridge.",
                )

            if self.state == TaskState.UNLOCK_DOOR:
                return self._unlock_door()

            if self.state == TaskState.PUSH_DOOR:
                return self._timed_drive(
                    TaskState.DONE,
                    self.PUSH_SECONDS,
                    "FORWARD_SLOW",
                    "Door cleared. Task 3 done.",
                )

            return "STOP"

        except Exception as exc:
            self._transition(TaskState.FAILED, f"Task failed: {exc}")
            return "STOP"

    def _transition(self, state, message):
        self.state = state
        self.state_started_at = time.monotonic()
        self.status_message = message

        print(f"[TaskController] {state.value}: {message}")
        self._log_event(f"transition state={state.value} message={message}")

    def _debug(self, message, interval=1.0):
        now = time.monotonic()
        if now - self._last_debug_at >= interval:
            print(f"[TaskController][debug] state={self.state.value} {message}")
            self._log_event(f"debug state={self.state.value} {message}")
            self._last_debug_at = now

    def _log_event(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{timestamp} [TaskController] {message}\n")
        except Exception:
            pass

    def _record_home(self):
        pose = self._get_current_position()

        if pose is None:
            self.status_message = "Waiting for robot pose from /amcl_pose, /pose, or /odom before starting task."
            self._debug("no robot pose yet; waiting for /amcl_pose, /pose, or /odom", 2.0)
            return "STOP"

        self.home_position = pose[:2]
        self._transition(TaskState.SET_VISION_TARGET, f"Home recorded at {self.home_position}.")
        return "STOP"

    def _set_initial_target(self):
        if self.task_type == TaskType.BEAR_RECOVERY:
            self._set_vision(self.DETECTION_MODE, self.BEAR_LABEL)
            self._transition(TaskState.EXPLORE_MAP, "Exploring map while searching for any bear.")

        elif self.task_type == TaskType.BRIDGE_RECOVERY:
            if self.state_started_at and self._target_label == self.BEAR_LABEL:
                self._transition(TaskState.EXPLORE_MAP, "Exploring map while searching bear after bridge traversal.")
            else:
                self._set_vision(self.SEGMENTATION_MODE, self.BRIDGE_LABEL)
                self._transition(TaskState.EXPLORE_MAP, "Exploring map while searching bridge with segmentation model.")

        elif self.task_type == TaskType.DOOR_UNLOCK:
            self._set_vision(self.DETECTION_MODE, self.KNOB_LABEL)
            self._transition(TaskState.EXPLORE_MAP, "Exploring map while searching for door knob.")

        return "STOP"

    def _search_target(self):
        if self._target_seen():
            self._outlier_count = 0
            self._lost_target_count = 0
            self._transition(TaskState.APPROACH_TARGET, f"Target {self._target_label} found. Approaching.")
            return "STOP"

        if self._elapsed() > self.SEARCH_TIMEOUT_SECONDS:
            self.status_message = f"Still searching for {self._target_label}; continuing rotation."
            self.state_started_at = time.monotonic()

        return self._apply_obstacle_guard("CLOCKWISE_ROTATION_SLOW", None, "search_target")

    def _approach_target(self):
        info = self._get_yolo_target_info()

        # 1. 看不到目標：短暫容錯，但不要使用舊資料直接 observe / grab。
        # 之前 lost_close_range 會直接鎖舊 marker 進 observe，容易造成亂抓。
        if not info or info[0] != 1:
            self._lost_target_count += 1

            self._debug(
                f"lost target count={self._lost_target_count} "
                f"label={self._target_label}; yolo_target_info={info}",
                1.0,
            )
            self._log_event(
                f"lost_target count={self._lost_target_count} "
                f"label={self._target_label}; yolo_target_info={info}"
            )

            if self._lost_target_count < self.LOST_TARGET_MAX_COUNT:
                # 先停一下，等待 YOLO 恢復；不要用舊 target 直接抓。
                return "STOP"

            self._clear_stale_target_state()
            self.nav_processing.reset_nav_process()
            self._transition(TaskState.SEARCH_TARGET, f"Lost {self._target_label}. Searching again.")
            return "STOP"

        # 2. 有看到目標。
        self._lost_target_count = 0

        # 3. outlier：忽略當前跳動資料，但不要直接用舊資料 observe / grab。
        if self._is_target_info_outlier(info):
            self._outlier_count += 1

            current_distance = self._safe_float(info[1])
            current_delta_x = self._safe_float(info[2])

            self._log_event(
                f"ignore_outlier count={self._outlier_count} "
                f"label={self._target_label} found={info[0]} "
                f"distance={self._format_float(current_distance, 3)} "
                f"delta_x={self._format_float(current_delta_x, 1)}"
            )
            self._debug(
                f"ignore outlier count={self._outlier_count} "
                f"label={self._target_label} "
                f"distance={self._format_float(current_distance, 3)} "
                f"delta_x={self._format_float(current_delta_x, 1)}",
                1.0,
            )

            if self._outlier_count >= self.OUTLIER_MAX_COUNT:
                self._clear_stale_target_state()
                self.nav_processing.reset_nav_process()
                self._move_arm_to_vision_pose("too_many_outliers")
                self._transition(
                    TaskState.SEARCH_TARGET,
                    f"Too many target outliers for {self._target_label}. Clearing stale target and searching again.",
                )
                return "STOP"

            return "STOP"

        # 4. 正常資料：解析 distance / delta_x。
        self._outlier_count = 0

        distance = self._safe_float(info[1])
        delta_x = self._safe_float(info[2])

        if distance is None or delta_x is None:
            self._debug(f"invalid target_info label={self._target_label}; yolo_target_info={info}", 1.0)
            return "STOP"

        self._last_valid_target_info = list(info)

        self._log_event(
            f"target_info label={self._target_label} found={info[0]} "
            f"distance={distance:.3f} delta_x={delta_x:.1f}"
        )
        self._debug(
            f"approach label={self._target_label} distance={distance:.3f} delta_x={delta_x:.1f}",
            1.0,
        )

        # 5. 第一優先：目標必須在畫面正中間附近。
        # 不管距離多近，只要還沒置中，就不 lock、不 observe、不 grab。
        if not self._delta_x_aligned(delta_x, self.GRAB_ALIGN_DELTA_X):
            action = self._alignment_action_from_delta_x(delta_x, self.GRAB_ALIGN_DELTA_X)

            self.status_message = (
                f"Aligning {self._target_label}: "
                f"distance={distance:.2f}m delta_x={delta_x:.1f}"
            )
            self._last_action = action

            self._log_event(
                f"aligning_target distance={distance:.3f} delta_x={delta_x:.1f} "
                f"threshold={self.GRAB_ALIGN_DELTA_X:.1f} action={action}"
            )
            self._debug(
                f"aligning target distance={distance:.3f} delta_x={delta_x:.1f} action={action}",
                0.5,
            )
            return self._apply_obstacle_guard(action, info, "approach_align_target")

        # 6. 第二優先：目標已置中，才允許往前靠近。
        # Bear / Bridge 要等 ArmController 判定 target 在手臂可達範圍內，才准抓。
        if self.task_type in (TaskType.BEAR_RECOVERY, TaskType.BRIDGE_RECOVERY):
            if not self._arm_target_reachable():
                self.status_message = (
                    f"{self._target_label} centered but outside arm reach. Moving closer."
                )

                action = self._target_approach_forward_action(distance)
                self._log_event(
                    f"centered_but_out_of_reach distance={distance:.3f} "
                    f"delta_x={delta_x:.1f} action={action}"
                )
                self._debug(
                    f"centered but outside arm reach; distance={distance:.3f} delta_x={delta_x:.1f} action={action}",
                    0.5,
                )

                # 目標置中但還不可達：慢慢直走靠近，安全判斷交給 obstacle guard。
                self._last_action = action
                return self._apply_obstacle_guard(
                    action,
                    info,
                    "approach_centered_out_of_reach",
                )

            # 7. 第三優先：已置中 + 手臂可達，才鎖定 marker 並進 observe。
            # 這是唯一允許 capture lock 的正常路徑。
            if not self._delta_x_aligned(delta_x, self.GRAB_ALIGN_DELTA_X):
                action = self._alignment_action_from_delta_x(delta_x, self.GRAB_ALIGN_DELTA_X)
                self._log_event(
                    f"grab_align_rejected distance={distance:.3f} delta_x={delta_x:.1f} "
                    f"threshold={self.GRAB_ALIGN_DELTA_X:.1f} action={action}"
                )
                self._last_action = action
                return self._apply_obstacle_guard(action, info, "approach_grab_align_rejected")

                        # 7. 第三優先：已置中 + 手臂可達，但 camera distance 還太遠時，不准抓。
            # 之前 distance=0.47 就開始抓，實測夾不到，所以要繼續慢慢靠近。
            if distance > self.GRAB_CAMERA_DISTANCE_THRESHOLD:
                action = self._target_approach_forward_action(distance)

                self.status_message = (
                    f"{self._target_label} centered and arm-reachable, "
                    f"but camera distance={distance:.2f}m is still too far. Moving closer."
                )

                self._log_event(
                    f"centered_reachable_but_camera_too_far "
                    f"distance={distance:.3f} "
                    f"threshold={self.GRAB_CAMERA_DISTANCE_THRESHOLD:.3f} "
                    f"delta_x={delta_x:.1f} "
                    f"action={action}"
                )
                self._debug(
                    f"centered reachable but camera too far; "
                    f"distance={distance:.3f} "
                    f"threshold={self.GRAB_CAMERA_DISTANCE_THRESHOLD:.3f} "
                    f"delta_x={delta_x:.1f} "
                    f"action={action}",
                    0.5,
                )

                self._last_action = action
                return self._apply_obstacle_guard(
                    action,
                    info,
                    "approach_centered_reachable_but_camera_too_far",
                )

            # 8. 已置中 + 手臂可達 + camera distance 夠近，才真正鎖定並進 observe。
            self._log_event(
                f"target_ready_centered_reachable_and_close "
                f"distance={distance:.3f} "
                f"threshold={self.GRAB_CAMERA_DISTANCE_THRESHOLD:.3f} "
                f"delta_x={delta_x:.1f}"
            )
            self._debug(
                f"target ready centered reachable and close "
                f"distance={distance:.3f} "
                f"threshold={self.GRAB_CAMERA_DISTANCE_THRESHOLD:.3f} "
                f"delta_x={delta_x:.1f}",
                0.5,
            )

            self._capture_locked_target(info, "centered_reachable_and_close")
            self._observe_started_at = time.monotonic()
            self._transition(
                TaskState.OBSERVE,
                f"{self._target_label} centered, reachable, and close enough. "
                f"Observing for {self.OBSERVE_SECONDS:.0f} seconds.",
            )
            return "STOP"

        # Door task 保留較簡單距離邏輯。
        if 0 < distance <= self.APPROACH_STOP_DISTANCE:
            self._observe_started_at = time.monotonic()
            self._transition(
                TaskState.OBSERVE,
                f"Within {self.APPROACH_STOP_DISTANCE} m. Observing for {self.OBSERVE_SECONDS:.0f} seconds.",
            )
            return "STOP"

        # 其他任務：已置中但還太遠，慢慢前進。
        self._last_action = "FORWARD_VERY_SLOW"
        self._log_event("camera_nav action=FORWARD_VERY_SLOW reason=centered_not_close")
        return self._apply_obstacle_guard(
            "FORWARD_VERY_SLOW",
            info,
            "approach_camera_nav",
        )


    def _observe_target(self):
        if self._observe_started_at is None:
            self._observe_started_at = time.monotonic()

        elapsed = time.monotonic() - self._observe_started_at
        remaining = max(0.0, self.OBSERVE_SECONDS - elapsed)
        self.status_message = f"Observing {self._target_label}: {remaining:.1f}s remaining."

        if elapsed < self.OBSERVE_SECONDS:
            return "STOP"

        if self.task_type in (TaskType.BEAR_RECOVERY, TaskType.BRIDGE_RECOVERY):
            self._transition(TaskState.GRAB_TARGET, f"Observation complete. Grabbing {self._target_label}.")

        elif self.task_type == TaskType.DOOR_UNLOCK:
            self._transition(TaskState.UNLOCK_DOOR, "Observation complete. Unlocking door.")

        return "STOP"

    def _grab_target(self):
        if not self._grab_started:
            self._grab_started = True

            # 若剛剛是因為 close-range lost / outlier 進入抓取，
            # 先把 ROS communicator 的 latest marker 還原成鎖定的近距離目標，
            # 避免 ArmController 在抓取時拿到已切換成遠熊的 marker。
            self._restore_locked_target_for_grab()

            marker = getattr(self.ros_communicator, "latest_yolo_marker", None)
            point = self.ros_communicator.get_latest_yolo_detection_position()

            marker_desc = self._describe_marker(marker)
            point_desc = self._describe_point(point)

            self._log_event(
                f"grab_start label={self._target_label} marker={marker_desc} detection_position={point_desc}"
            )

            started = self.arm_controller.auto_control(key="g", mode="auto_arm_human")

            if not started:
                self._grab_started = False
                self._observe_started_at = None

                # auto_control 回傳 False 代表 ArmController 判定目前 target 不可達。
                # 這時不能保留 _last_valid_target_info，否則下一個 tick 會用同一筆舊資料再次 observe/grab。
                self._clear_stale_target_state()

                self.nav_processing.reset_nav_process()
                self._move_arm_to_vision_pose("grab_rejected")
                self._transition(
                    TaskState.SEARCH_TARGET,
                    "Arm target is not reachable. Clearing stale target and searching again.",
                )
                return "STOP"

            self.status_message = "Grab sequence started. Waiting for arm motion."
            return "STOP"

        if self._elapsed() < self.GRAB_SECONDS:
            return "STOP"

        self._return_started_at = time.monotonic()
        self.nav_processing.reset_nav_process()

        # 抓取流程結束後，改成「抱著目標回家姿態」。
        # 重點：不可以呼叫 _move_arm_to_vision_pose，因為 vision pose 會把 gripper 設成 init/max，導致放掉 bear。
        self._holding_target = True
        self._move_arm_to_carry_pose("grab_wait_complete_hold_target")

        self._log_event("grab_wait_complete returning_home holding_target=True")
        self._transition(TaskState.RETURN_HOME, "Returning home while holding target.")

        return "STOP"

    def _return_home(self):
        if not self.home_position:
            self._transition(TaskState.FAILED, "Cannot return home: no home pose recorded.")
            return "STOP"

        if self._return_started_at and time.monotonic() - self._return_started_at > self.RETURN_TIMEOUT_SECONDS:
            self._transition(TaskState.FAILED, "Return home timed out.")
            return "STOP"

        if self.frontier_explorer:
            action = self.frontier_explorer.get_action_to_goal(self.home_position, 0.5)
        else:
            action = self.nav_processing.get_action_from_nav2_plan_no_dynamic_p_2_p(
                goal_coordinates=self.home_position
            )

        current_position = self._get_current_position()

        if current_position and self._distance(current_position, self.home_position) < 0.5:
            # 只有真的回到起始位置，才允許打開夾爪放下 bear。
            if getattr(self, "_holding_target", False):
                self._release_arm_gripper_at_home("return_home_arrived")
                self._holding_target = False
                self._transition(TaskState.DONE, "Returned home and released target. Task done.")
            else:
                self._transition(TaskState.DONE, "Returned home. Task done.")
            return "STOP"

        return self._apply_obstacle_guard(action, None, "return_home")

    def _explore_map(self):
        if self._target_seen():
            if self.task_type == TaskType.BRIDGE_RECOVERY and self._target_label == self.BRIDGE_LABEL:
                if self.frontier_explorer:
                    self.frontier_explorer.reset()

                self.nav_processing.reset_nav_process()
                self._transition(TaskState.ASCEND_BRIDGE, "Bridge found during exploration. Driving up bridge slowly.")
                return "STOP"

            if self.frontier_explorer:
                self.frontier_explorer.reset()

            self.nav_processing.reset_nav_process()
            self._transition(
                TaskState.APPROACH_TARGET,
                f"Target {self._target_label} found during exploration. Approaching.",
            )
            return "STOP"

        if self.frontier_explorer is None:
            self.status_message = "No frontier explorer configured; rotating to search target."
            return self._apply_obstacle_guard("CLOCKWISE_ROTATION_SLOW", None, "explore_no_frontier")

        action = self.frontier_explorer.tick()
        self.status_message = self.frontier_explorer.status_message
        self._debug(f"explore action={action}; {self.status_message}", 1.0)

        return self._apply_obstacle_guard(action, None, "explore_map")

    def _search_bridge(self):
        if self._target_seen():
            self._transition(TaskState.ASCEND_BRIDGE, "Bridge found. Driving up bridge slowly.")
            return "STOP"

        return self._apply_obstacle_guard("CLOCKWISE_ROTATION_SLOW", None, "search_bridge")

    def _unlock_door(self):
        if not self._grab_started:
            self._grab_started = True
            self.arm_controller.auto_control(key="g", mode="auto_arm_human")
            self.status_message = "Unlock sequence started with knob target."
            return "STOP"

        if self._elapsed() < self.UNLOCK_SECONDS:
            return "STOP"

        self._transition(TaskState.PUSH_DOOR, "Unlock done. Pushing door open.")
        return "STOP"

    def _timed_drive(self, next_state, duration, action, message):
        if self._elapsed() < duration:
            return self._apply_obstacle_guard(action, None, "timed_drive")

        self._transition(next_state, message)

        if next_state == TaskState.SET_VISION_TARGET and self.task_type == TaskType.BRIDGE_RECOVERY:
            self._set_vision(self.DETECTION_MODE, self.BEAR_LABEL)

            if self.frontier_explorer:
                self.frontier_explorer.reset()

        return "STOP"

    def _is_forward_action(self, action):
        # Nav2Processing may return plain "FORWARD"; guard must catch it too.
        return action in ("FORWARD", "FORWARD_SLOW", "FORWARD_VERY_SLOW")

    def _is_rotation_action(self, action):
        return action in (
            "CLOCKWISE_ROTATION_FINE",
            "COUNTERCLOCKWISE_ROTATION_FINE",
            "CLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
        )

    def _is_movement_action(self, action):
        return self._is_forward_action(action) or self._is_rotation_action(action)

    def _is_left_turn_action(self, action):
        return action in ("COUNTERCLOCKWISE_ROTATION_FINE", "COUNTERCLOCKWISE_ROTATION_SLOW")

    def _is_right_turn_action(self, action):
        return action in ("CLOCKWISE_ROTATION_FINE", "CLOCKWISE_ROTATION_SLOW")

    def _get_latest_obstacle_depths(self):
        try:
            if hasattr(self.ros_communicator, "get_latest_camera_x_multi_depth"):
                msg = self.ros_communicator.get_latest_camera_x_multi_depth()
                if msg is not None and hasattr(msg, "data"):
                    return list(msg.data)
        except Exception:
            pass

        try:
            data_processor = getattr(self.nav_processing, "data_processor", None)
            if data_processor and hasattr(data_processor, "get_camera_x_multi_depth"):
                depths = data_processor.get_camera_x_multi_depth()
                if depths is not None:
                    return list(depths)
        except Exception:
            pass

        return None

    def _get_latest_scan_msg(self):
        candidates = (
            (self.ros_communicator, "get_latest_lidar"),
            (self.ros_communicator, "get_latest_scan"),
            (self.ros_communicator, "get_scan"),
            (self.ros_communicator, "get_latest_laser_scan"),
            (getattr(self.nav_processing, "data_processor", None), "get_latest_lidar"),
            (getattr(self.nav_processing, "data_processor", None), "get_latest_scan"),
            (getattr(self.nav_processing, "data_processor", None), "get_scan"),
            (getattr(self.nav_processing, "data_processor", None), "get_laser_scan"),
        )

        for obj, method_name in candidates:
            try:
                if obj is not None and hasattr(obj, method_name):
                    msg = getattr(obj, method_name)()
                    if msg is not None and hasattr(msg, "ranges"):
                        return msg
            except Exception:
                pass

        attrs = (
            (self.ros_communicator, "latest_lidar"),
            (self.ros_communicator, "latest_scan"),
            (self.ros_communicator, "latest_laser_scan"),
            (getattr(self.nav_processing, "data_processor", None), "latest_lidar"),
            (getattr(self.nav_processing, "data_processor", None), "latest_scan"),
            (getattr(self.nav_processing, "data_processor", None), "latest_laser_scan"),
        )

        for obj, attr_name in attrs:
            try:
                if obj is not None and hasattr(obj, attr_name):
                    msg = getattr(obj, attr_name)
                    if msg is not None and hasattr(msg, "ranges"):
                        return msg
            except Exception:
                pass

        return None

    def _valid_depth_values(self, values):
        valid = []
        for value in values or []:
            try:
                depth = float(value)
            except Exception:
                continue

            if depth <= 0 or not math.isfinite(depth):
                continue

            valid.append(depth)

        return valid

    def _valid_scan_values(self, values):
        valid = []
        for value in values or []:
            try:
                depth = float(value)
            except Exception:
                continue

            if not math.isfinite(depth):
                continue

            # Fix false emergency_stop: ignore near-field /scan self-hit.
            # Example from your log: front=0.185 while bear is 2.667m away.
            if depth <= self.OBSTACLE_LIDAR_SELF_FILTER_DISTANCE:
                continue

            valid.append(depth)

        return valid

    def _robust_near_depth(self, values):
        valid = self._valid_depth_values(values)
        if not valid:
            return None

        valid.sort()
        if len(valid) < 3:
            return valid[0]

        return sum(valid[:3]) / 3.0

    def _robust_scan_near_depth(self, values):
        valid = self._valid_scan_values(values)
        if len(valid) < self.OBSTACLE_SCAN_MIN_VALID_POINTS:
            return None

        valid.sort()
        index = int(len(valid) * self.OBSTACLE_SCAN_PERCENTILE)
        index = min(max(index, 0), len(valid) - 1)
        return valid[index]

    def _split_depth_regions(self, depths):
        if not depths:
            return None, None, None

        n = len(depths)
        left_end = max(1, (n // 3) + 1)
        right_start = min(n - 1, ((2 * n) // 3) + 1)

        left = depths[:left_end]
        center = depths[left_end:right_start]
        right = depths[right_start:]

        return (
            self._robust_near_depth(left),
            self._robust_near_depth(center),
            self._robust_near_depth(right),
        )

    def _scan_sector_depth(self, scan_msg, min_deg, max_deg):
        if scan_msg is None or not hasattr(scan_msg, "ranges"):
            return None

        try:
            angle_min = float(scan_msg.angle_min)
            angle_increment = float(scan_msg.angle_increment)
            range_min = float(getattr(scan_msg, "range_min", 0.0))
            range_max = float(getattr(scan_msg, "range_max", float("inf")))
        except Exception:
            return None

        values = []
        for index, raw in enumerate(scan_msg.ranges):
            try:
                value = float(raw)
            except Exception:
                continue

            if not math.isfinite(value):
                continue

            if value <= max(0.0, range_min):
                continue

            if math.isfinite(range_max) and value > range_max:
                continue

            angle = angle_min + index * angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            deg = math.degrees(angle)

            if min_deg <= deg <= max_deg:
                values.append(value)

        return self._robust_scan_near_depth(values)

    def _get_camera_obstacle_snapshot(self):
        depths = self._get_latest_obstacle_depths()
        if not depths:
            return None

        left_depth, center_depth, right_depth = self._split_depth_regions(depths)

        if center_depth is None and left_depth is None and right_depth is None:
            return None

        return {
            "source": "camera_depth",
            "front": center_depth,
            "front_left": left_depth,
            "front_right": right_depth,
            "left": None,
            "right": None,
            "camera_front": center_depth,
            "scan_front": None,
        }

    def _get_lidar_obstacle_snapshot(self):
        if not self.OBSTACLE_USE_LIDAR_SCAN:
            return None

        scan_msg = self._get_latest_scan_msg()
        if scan_msg is None:
            return None

        front_deg = float(self.OBSTACLE_SCAN_FRONT_DEG)
        side_front_deg = float(self.OBSTACLE_SCAN_SIDE_FRONT_DEG)
        side_min = float(self.OBSTACLE_SCAN_SIDE_DEG_MIN)
        side_max = float(self.OBSTACLE_SCAN_SIDE_DEG_MAX)

        snapshot = {
            "source": "scan",
            "front": self._scan_sector_depth(scan_msg, -front_deg, front_deg),
            "front_left": self._scan_sector_depth(scan_msg, front_deg, side_front_deg),
            "front_right": self._scan_sector_depth(scan_msg, -side_front_deg, -front_deg),
            "left": self._scan_sector_depth(scan_msg, side_min, side_max),
            "right": self._scan_sector_depth(scan_msg, -side_max, -side_min),
        }

        snapshot["scan_front"] = snapshot.get("front")
        snapshot["camera_front"] = None

        if all(value is None for key, value in snapshot.items() if key != "source"):
            return None

        return snapshot

    def _min_ignore_none(self, *values):
        valid = [
            float(v)
            for v in values
            if v is not None and math.isfinite(float(v)) and float(v) > 0
        ]
        if not valid:
            return None
        return min(valid)

    def _get_obstacle_snapshot(self):
        scan_snapshot = self._get_lidar_obstacle_snapshot()
        camera_snapshot = self._get_camera_obstacle_snapshot()

        if scan_snapshot is None and camera_snapshot is None:
            return None

        if scan_snapshot is None:
            return camera_snapshot

        if camera_snapshot is None:
            return scan_snapshot

        return {
            "source": "scan+camera_depth",
            "front": self._min_ignore_none(scan_snapshot.get("front"), camera_snapshot.get("front")),
            "front_left": self._min_ignore_none(scan_snapshot.get("front_left"), camera_snapshot.get("front_left")),
            "front_right": self._min_ignore_none(scan_snapshot.get("front_right"), camera_snapshot.get("front_right")),
            "left": scan_snapshot.get("left"),
            "right": scan_snapshot.get("right"),
            "scan_front": scan_snapshot.get("front"),
            "camera_front": camera_snapshot.get("front"),
        }

    def _is_center_depth_likely_target(self, center_depth, target_info):
        if center_depth is None or target_info is None or len(target_info) < 3:
            return False

        try:
            found = float(target_info[0]) == 1.0
            distance = float(target_info[1])
            delta_x = float(target_info[2])
        except Exception:
            return False

        return (
            found
            and distance > 0
            and math.isfinite(distance)
            and abs(delta_x) <= self.OBSTACLE_TARGET_ALIGN_DELTA_X
            and abs(center_depth - distance) <= self.OBSTACLE_TARGET_DEPTH_TOLERANCE
        )

    def _is_target_centered_for_obstacle_guard(self, target_info, threshold=None):
        """Return True when YOLO says the target is near the image center.

        This is intentionally independent from depth matching. In the simulator,
        camera x-depth and LiDAR can disagree with YOLO target distance, especially
        near the robot body or when the bear is close.
        """
        if target_info is None or len(target_info) < 3:
            return False

        try:
            found = float(target_info[0]) == 1.0
            distance = float(target_info[1])
            delta_x = float(target_info[2])
        except Exception:
            return False

        if threshold is None:
            threshold = self.OBSTACLE_TARGET_APPROACH_DELTA_X

        return (
            found
            and distance > 0
            and math.isfinite(distance)
            and abs(delta_x) <= float(threshold)
        )

    def _choose_open_turn_action(self, left_depth, right_depth):
        left = left_depth if left_depth is not None else -1.0
        right = right_depth if right_depth is not None else -1.0

        if left >= self.OBSTACLE_SIDE_CLEAR_DISTANCE and left > right:
            return "COUNTERCLOCKWISE_ROTATION_FINE"

        if right >= self.OBSTACLE_SIDE_CLEAR_DISTANCE and right >= left:
            return "CLOCKWISE_ROTATION_FINE"

        if left > right and left > self.OBSTACLE_ROTATION_SIDE_STOP_DISTANCE:
            return "COUNTERCLOCKWISE_ROTATION_FINE"

        if right > self.OBSTACLE_ROTATION_SIDE_STOP_DISTANCE:
            return "CLOCKWISE_ROTATION_FINE"

        return "STOP"

    def _choose_recovery_action(self, snapshot):
        if snapshot:
            left = self._min_ignore_none(snapshot.get("left"), snapshot.get("front_left"))
            right = self._min_ignore_none(snapshot.get("right"), snapshot.get("front_right"))
            return self._choose_open_turn_action(left, right)

        return "STOP"

    def _apply_stuck_guard(self, action, snapshot=None, context="", target_info=None):
        now = time.monotonic()

        target_approach_context = "approach" in str(context)
        if (
            target_approach_context
            and target_info is not None
            and self._is_target_centered_for_obstacle_guard(target_info)
            and self._target_distance_is_progressing(target_info, context)
        ):
            return action

        if now < self._recovery_until:
            self._debug(
                f"stuck_recovery active context={context} action={self._recovery_action}",
                0.3,
            )
            return self._recovery_action

        if not self._is_forward_action(action):
            self._stuck_started_at = None
            return action

        current_position = self._get_current_position()
        if current_position is None or len(current_position) < 2:
            return action

        try:
            current_xy = (float(current_position[0]), float(current_position[1]))
        except Exception:
            return action

        if self._last_stuck_pose is None or self._last_stuck_time is None:
            self._last_stuck_pose = current_xy
            self._last_stuck_time = now
            self._stuck_started_at = None
            return action

        moved = self._distance(current_xy, self._last_stuck_pose)
        elapsed = now - self._last_stuck_time

        if moved >= self.OBSTACLE_STUCK_MIN_MOVE:
            self._last_stuck_pose = current_xy
            self._last_stuck_time = now
            self._stuck_started_at = None
            return action

        if elapsed < self.OBSTACLE_STUCK_CHECK_SECONDS:
            return action

        if self._stuck_started_at is None:
            self._stuck_started_at = now
            self._log_event(
                f"stuck_watch_start context={context} action={action} "
                f"moved={moved:.3f} elapsed={elapsed:.2f}"
            )
            return action

        if now - self._stuck_started_at < self.OBSTACLE_STUCK_SECONDS:
            return action

        recovery_action = self._choose_recovery_action(snapshot)
        self._recovery_action = recovery_action
        self._recovery_until = now + self.OBSTACLE_STUCK_RECOVERY_SECONDS
        self._last_stuck_pose = current_xy
        self._last_stuck_time = now
        self._stuck_started_at = None

        self._log_event(
            f"stuck_detected context={context} raw_action={action} "
            f"moved={moved:.3f} recovery_action={recovery_action}"
        )
        self._debug(
            f"stuck_detected context={context} raw_action={action} recovery_action={recovery_action}",
            0.3,
        )

        return recovery_action

    def _apply_obstacle_guard(self, action, target_info=None, context=""):
        if not self.OBSTACLE_GUARD_ENABLED:
            return action

        if not self._is_movement_action(action):
            return action

        snapshot = self._get_obstacle_snapshot()
        if not snapshot:
            self._log_event(
                f"obstacle_guard no_obstacle_snapshot context={context} raw_action={action}"
            )
            self._debug(
                f"obstacle_guard no_obstacle_snapshot context={context} raw_action={action}",
                1.0,
            )
            return self._apply_stuck_guard(action, None, context)

        source = snapshot.get("source", "unknown")
        front_depth = snapshot.get("front")
        front_left_depth = snapshot.get("front_left")
        front_right_depth = snapshot.get("front_right")
        left_depth = snapshot.get("left")
        right_depth = snapshot.get("right")
        scan_front = snapshot.get("scan_front")
        camera_front = snapshot.get("camera_front")

        likely_target = self._is_center_depth_likely_target(front_depth, target_info)
        target_centered_for_guard = self._is_target_centered_for_obstacle_guard(target_info)
        target_approach_context = str(context).startswith("approach_")
        camera_front_clear_for_target = (
            camera_front is not None
            and camera_front >= self.OBSTACLE_TARGET_APPROACH_CAMERA_CLEAR_DISTANCE
        )
        # When approaching a centered bear, ignore suspicious near-field LiDAR
        # front/corner hits if camera front is clear. This prevents the robot from
        # spinning forever because scan_front/front_right reports ~0.25m.
        target_creep_allowed = (
            target_approach_context
            and target_centered_for_guard
            and camera_front_clear_for_target
        )

        contact_allowed = self.state in (
            TaskState.ASCEND_BRIDGE,
            TaskState.DESCEND_BRIDGE,
            TaskState.PUSH_DOOR,
        )

        if target_approach_context:
            side_stop_distance = self.OBSTACLE_APPROACH_SIDE_STOP_DISTANCE
            rotation_side_stop_distance = self.OBSTACLE_APPROACH_ROTATION_SIDE_STOP_DISTANCE
            rotation_front_corner_stop_distance = (
                self.OBSTACLE_APPROACH_ROTATION_FRONT_CORNER_STOP_DISTANCE
            )
        else:
            side_stop_distance = self.OBSTACLE_SIDE_STOP_DISTANCE
            rotation_side_stop_distance = self.OBSTACLE_ROTATION_SIDE_STOP_DISTANCE
            rotation_front_corner_stop_distance = (
                self.OBSTACLE_ROTATION_FRONT_CORNER_STOP_DISTANCE
            )

        if front_depth is not None and front_depth <= self.OBSTACLE_EMERGENCY_STOP_DISTANCE:
            self._log_event(
                f"obstacle_guard emergency_stop source={source} context={context} "
                f"front={front_depth:.3f} "
                f"scan_front={self._format_float(scan_front, 3)} "
                f"camera_front={self._format_float(camera_front, 3)} "
                f"raw_action={action}"
            )
            self._debug(
                f"obstacle_guard emergency_stop source={source} context={context} "
                f"front={front_depth:.3f} "
                f"scan_front={self._format_float(scan_front, 3)} "
                f"camera_front={self._format_float(camera_front, 3)}",
                0.3,
            )
            return "STOP"

        if self._is_rotation_action(action):
            if (
                self._is_left_turn_action(action)
                and (
                    (left_depth is not None and left_depth <= rotation_side_stop_distance)
                    or (
                        front_left_depth is not None
                        and front_left_depth <= rotation_front_corner_stop_distance
                    )
                )
            ):
                safe_action = (
                    "CLOCKWISE_ROTATION_FINE"
                    if right_depth is not None and right_depth >= rotation_side_stop_distance
                    else "STOP"
                )
                self._log_event(
                    f"obstacle_guard block_left_turn source={source} context={context} "
                    f"left={self._format_float(left_depth, 3)} "
                    f"front_left={self._format_float(front_left_depth, 3)} "
                    f"raw_action={action} safe_action={safe_action}"
                )
                return safe_action

            if (
                self._is_right_turn_action(action)
                and (
                    (right_depth is not None and right_depth <= rotation_side_stop_distance)
                    or (
                        front_right_depth is not None
                        and front_right_depth <= rotation_front_corner_stop_distance
                    )
                )
            ):
                safe_action = (
                    "COUNTERCLOCKWISE_ROTATION_FINE"
                    if left_depth is not None and left_depth >= rotation_side_stop_distance
                    else "STOP"
                )
                self._log_event(
                    f"obstacle_guard block_right_turn source={source} context={context} "
                    f"right={self._format_float(right_depth, 3)} "
                    f"front_right={self._format_float(front_right_depth, 3)} "
                    f"raw_action={action} safe_action={safe_action}"
                )
                return safe_action

            self._log_event(
                f"obstacle_guard allow_rotation source={source} context={context} "
                f"left={self._format_float(left_depth, 3)} "
                f"right={self._format_float(right_depth, 3)} "
                f"front_left={self._format_float(front_left_depth, 3)} "
                f"front_right={self._format_float(front_right_depth, 3)} "
                f"raw_action={action}"
            )
            return action

        safe_action = action

        # True side-wall protection should happen before front-corner steering.
        # In the latest log, left was sometimes ~0.22m while front_right was ~0.28m;
        # the old logic turned left because of front_right, which can scrape the
        # already-close left wall.
        if left_depth is not None and left_depth <= side_stop_distance:
            safe_action = "CLOCKWISE_ROTATION_FINE"

        elif right_depth is not None and right_depth <= side_stop_distance:
            safe_action = "COUNTERCLOCKWISE_ROTATION_FINE"

        elif (
            front_depth is not None
            and front_depth <= self.OBSTACLE_FRONT_STOP_DISTANCE
            and not likely_target
            and not contact_allowed
            and not target_creep_allowed
        ):
            open_left = self._min_ignore_none(front_left_depth, left_depth)
            open_right = self._min_ignore_none(front_right_depth, right_depth)
            safe_action = self._choose_open_turn_action(open_left, open_right)

        elif (
            front_left_depth is not None
            and front_left_depth <= self.OBSTACLE_SIDE_FRONT_STOP_DISTANCE
            and not likely_target
            and not target_creep_allowed
        ):
            safe_action = "CLOCKWISE_ROTATION_FINE"

        elif (
            front_right_depth is not None
            and front_right_depth <= self.OBSTACLE_SIDE_FRONT_STOP_DISTANCE
            and not likely_target
            and not target_creep_allowed
        ):
            safe_action = "COUNTERCLOCKWISE_ROTATION_FINE"

        elif (
            left_depth is not None
            and right_depth is not None
            and left_depth <= self.OBSTACLE_CORRIDOR_BALANCE_DISTANCE
            and right_depth <= self.OBSTACLE_CORRIDOR_BALANCE_DISTANCE
            and abs(left_depth - right_depth) >= self.OBSTACLE_CORRIDOR_BALANCE_DELTA
        ):
            if left_depth < right_depth:
                safe_action = "CLOCKWISE_ROTATION_FINE"
            else:
                safe_action = "COUNTERCLOCKWISE_ROTATION_FINE"

        if safe_action != action:
            self._log_event(
                f"obstacle_guard override_forward source={source} context={context} "
                f"front={self._format_float(front_depth, 3)} "
                f"front_left={self._format_float(front_left_depth, 3)} "
                f"front_right={self._format_float(front_right_depth, 3)} "
                f"left={self._format_float(left_depth, 3)} "
                f"right={self._format_float(right_depth, 3)} "
                f"scan_front={self._format_float(scan_front, 3)} "
                f"camera_front={self._format_float(camera_front, 3)} "
                f"likely_target={likely_target} target_creep_allowed={target_creep_allowed} "
                f"contact_allowed={contact_allowed} raw_action={action} safe_action={safe_action}"
            )
            self._debug(
                f"obstacle_guard override_forward source={source} context={context} "
                f"raw={action} safe={safe_action} "
                f"front={self._format_float(front_depth, 3)} "
                f"scan_front={self._format_float(scan_front, 3)} "
                f"camera_front={self._format_float(camera_front, 3)} "
                f"left={self._format_float(left_depth, 3)} "
                f"right={self._format_float(right_depth, 3)}",
                0.5,
            )
            return safe_action

        self._log_event(
            f"obstacle_guard allow_forward source={source} context={context} "
            f"front={self._format_float(front_depth, 3)} "
            f"front_left={self._format_float(front_left_depth, 3)} "
            f"front_right={self._format_float(front_right_depth, 3)} "
            f"left={self._format_float(left_depth, 3)} "
            f"right={self._format_float(right_depth, 3)} "
            f"scan_front={self._format_float(scan_front, 3)} "
            f"camera_front={self._format_float(camera_front, 3)} "
            f"likely_target={likely_target} target_creep_allowed={target_creep_allowed} "
            f"contact_allowed={contact_allowed} raw_action={action}"
        )

        return self._apply_stuck_guard(action, snapshot, context, target_info)

    def _set_vision(self, mode, label):
        self._vision_mode = mode
        self._target_label = label

        self._last_valid_target_info = None
        self._target_lock_active = False
        self._locked_target_info = None
        self._locked_yolo_marker = None
        self._locked_detection_position = None
        self._outlier_count = 0
        self._lost_target_count = 0
        self._target_progress_last_distance = None
        self._target_progress_last_time = None

        # 關鍵修改：
        # 每次切 YOLO 模式 / label 前都把手臂抬起來，避免手臂擋住相機。
        self._move_arm_to_vision_pose(f"set_vision_{mode}_{label}")

        self._debug(f"set vision mode={mode} label={label}", 0.0)

        if hasattr(self.ros_communicator, "publish_yolo_model_mode"):
            self.ros_communicator.publish_yolo_model_mode(mode)

        self.ros_communicator.publish_target_label(label)

    def _target_seen(self):
        info = self._get_yolo_target_info()
        return bool(info and len(info) >= 1 and info[0] == 1)

    def _get_yolo_target_info(self):
        info = self.nav_processing.data_processor.get_yolo_target_info()

        if info is None or len(info) < 3:
            return None

        return info

    def _is_target_info_outlier(self, info):
        last = self._last_valid_target_info

        if last is None or len(last) < 3:
            return False

        try:
            last_distance = float(last[1])
            current_distance = float(info[1])
            last_delta_x = float(last[2])
            current_delta_x = float(info[2])
        except Exception:
            return False

        return (
            0 < last_distance < self.OUTLIER_LAST_DISTANCE_MAX
            and current_distance > self.OUTLIER_CURRENT_DISTANCE_MIN
            and abs(current_delta_x - last_delta_x) > self.OUTLIER_DELTA_X_JUMP
        )

    def _target_approach_forward_action(self, distance):
        try:
            distance = float(distance)
        except Exception:
            return "FORWARD_VERY_SLOW"

        if distance > self.APPROACH_FAST_UNTIL_DISTANCE:
            return "FORWARD_SLOW"

        return "FORWARD_VERY_SLOW"

    def _target_distance_is_progressing(self, target_info, context=""):
        if target_info is None or len(target_info) < 2:
            return False

        try:
            distance = float(target_info[1])
        except Exception:
            return False

        if distance <= 0 or not math.isfinite(distance):
            return False

        now = time.monotonic()

        if self._target_progress_last_distance is None or self._target_progress_last_time is None:
            self._target_progress_last_distance = distance
            self._target_progress_last_time = now
            return True

        last_distance = self._target_progress_last_distance
        elapsed = now - self._target_progress_last_time

        # Good: YOLO target distance is decreasing, so the robot is making visual progress.
        if distance <= last_distance - self.OBSTACLE_TARGET_PROGRESS_EPS:
            self._target_progress_last_distance = distance
            self._target_progress_last_time = now
            self._last_stuck_pose = None
            self._last_stuck_time = None
            self._stuck_started_at = None
            self._log_event(
                f"target_progress context={context} "
                f"last_distance={last_distance:.3f} current_distance={distance:.3f}"
            )
            return True

        # Give the visual distance a short grace window. YOLO distance is noisy and may
        # repeat the same value for a few frames.
        if elapsed < self.OBSTACLE_TARGET_PROGRESS_GRACE_SECONDS:
            return True

        # No visual progress for long enough; allow normal /odom stuck detection to run.
        return False

    def _arm_target_reachable(self):
        if not hasattr(self.arm_controller, "is_latest_target_reachable"):
            return True

        return self.arm_controller.is_latest_target_reachable()

    def _delta_x_aligned(self, delta_x, threshold=None):
        """Return True if target is close enough to image center."""
        try:
            delta_x = float(delta_x)
        except Exception:
            return False

        if threshold is None:
            threshold = self.GRAB_ALIGN_DELTA_X

        return abs(delta_x) <= float(threshold)

    def _alignment_action_from_delta_x(self, delta_x, threshold=None):
        """Rotate in place using the same direction convention as camera_nav()."""
        try:
            delta_x = float(delta_x)
        except Exception:
            return "STOP"

        if threshold is None:
            threshold = self.GRAB_ALIGN_DELTA_X

        if delta_x > threshold:
            return "CLOCKWISE_ROTATION_FINE"

        if delta_x < -threshold:
            return "COUNTERCLOCKWISE_ROTATION_FINE"

        return "STOP"

    def _format_float(self, value, digits=3):
        try:
            if value is None:
                return "None"
            return f"{float(value):.{digits}f}"
        except Exception:
            return "None"


    def _capture_locked_target(self, info, reason=""):
        """Snapshot the current close-range target.

        YOLO may switch from the near bear to a farther bear when the near bear is
        partially occluded or too close. This lock keeps the last trusted near
        target for the final observe/grab step.
        """
        try:
            if info is None or len(info) < 3:
                return False

            distance = self._safe_float(info[1])
            if distance is None or distance <= 0:
                return False

            self._target_lock_active = True
            self._locked_target_info = list(info)

            marker = getattr(self.ros_communicator, "latest_yolo_marker", None)
            self._locked_yolo_marker = copy.deepcopy(marker) if marker is not None else None

            point = None
            if hasattr(self.ros_communicator, "get_latest_yolo_detection_position"):
                try:
                    point = self.ros_communicator.get_latest_yolo_detection_position()
                except Exception:
                    point = None
            self._locked_detection_position = copy.deepcopy(point) if point is not None else None

            self._log_event(
                f"target_lock_capture reason={reason} "
                f"distance={distance:.3f} delta_x={self._safe_float(info[2]):.1f} "
                f"has_marker={self._locked_yolo_marker is not None} "
                f"has_point={self._locked_detection_position is not None}"
            )
            return True

        except Exception as exc:
            self._log_event(f"target_lock_capture failed reason={reason}: {exc}")
            return False

    def _restore_locked_target_for_grab(self):
        """Restore locked marker before ArmController grabs.

        ArmController reads ros_communicator.latest_yolo_marker directly.
        If YOLO has already switched to a farther bear, restore the locked
        close-range marker so the grab uses the intended near target.
        """
        try:
            if not self._target_lock_active:
                return False

            if self._locked_yolo_marker is not None:
                self.ros_communicator.latest_yolo_marker = copy.deepcopy(self._locked_yolo_marker)
                self._log_event("target_lock_restore marker=True")
                return True

            self._log_event("target_lock_restore marker=False")
            return False

        except Exception as exc:
            self._log_event(f"target_lock_restore failed: {exc}")
            return False


    def _move_arm_to_carry_pose(self, reason=""):
        """Move arm to a camera-clear carry pose while keeping gripper closed.

        跟 vision pose 的差異：
        - vision pose 會使用 joint init，因此 gripper 會變成 init/max_angle，等於開爪。
        - carry pose 只把 shoulder/elbow 收回不擋相機，但 gripper 強制保持 min_angle，也就是夾緊。
        """
        try:
            arm = self.arm_controller

            if arm is None:
                self._log_event(f"move_arm_to_carry_pose skipped reason={reason}: no arm_controller")
                return False

            if hasattr(arm, "move_to_carry_pose"):
                ok = arm.move_to_carry_pose()
                self._log_event(f"move_arm_to_carry_pose method reason={reason} ok={ok}")
                return bool(ok)

            if (
                hasattr(arm, "joint_angles")
                and hasattr(arm, "joint_limits")
                and hasattr(arm, "_clamp_and_publish")
            ):
                carry_pose_degrees = [
                    float(arm.joint_limits[0].get("init", 0.0)),
                    float(arm.joint_limits[1].get("init", 0.0)),
                    float(arm.joint_limits[2].get("min_angle", 0.0)),  # close gripper
                ]

                if hasattr(arm, "_smooth_move_to"):
                    # 先確認夾爪是關的，再收手臂，避免中途放掉。
                    arm._smooth_move_to([None, None, carry_pose_degrees[2]], step=5.0, delay=0.1)
                    arm._smooth_move_to([None, carry_pose_degrees[1], None], step=5.0, delay=0.1)
                    arm._smooth_move_to([carry_pose_degrees[0], None, None], step=5.0, delay=0.1)
                else:
                    arm.joint_angles = list(carry_pose_degrees)
                    arm._clamp_and_publish()

                if hasattr(arm, "_visualize_arm_lines"):
                    try:
                        arm._visualize_arm_lines()
                    except Exception as exc:
                        self._log_event(
                            f"move_arm_to_carry_pose visualize failed reason={reason}: {exc}"
                        )

                self._log_event(
                    f"move_arm_to_carry_pose direct_joint_angles reason={reason} "
                    f"degrees={[round(v, 2) for v in carry_pose_degrees]}"
                )
                return True

            self._log_event(
                f"move_arm_to_carry_pose skipped reason={reason}: no compatible arm API"
            )
            return False

        except Exception as exc:
            self._log_event(f"move_arm_to_carry_pose failed reason={reason}: {exc}")
            return False

    def _release_arm_gripper_at_home(self, reason=""):
        """Release target only after robot has returned to recorded home position."""
        try:
            arm = self.arm_controller

            if arm is None:
                self._log_event(f"release_arm_gripper_at_home skipped reason={reason}: no arm_controller")
                return False

            if hasattr(arm, "release_gripper_at_home"):
                ok = arm.release_gripper_at_home()
                self._log_event(f"release_arm_gripper_at_home method reason={reason} ok={ok}")
                return bool(ok)

            if (
                hasattr(arm, "joint_limits")
                and hasattr(arm, "_clamp_and_publish")
            ):
                release_angle = float(arm.joint_limits[2].get("max_angle", 90.0))

                if hasattr(arm, "_smooth_move_to"):
                    arm._smooth_move_to([None, None, release_angle], step=5.0, delay=0.1)
                elif hasattr(arm, "joint_angles"):
                    arm.joint_angles[2] = release_angle
                    arm._clamp_and_publish()

                if hasattr(arm, "_visualize_arm_lines"):
                    try:
                        arm._visualize_arm_lines()
                    except Exception as exc:
                        self._log_event(
                            f"release_arm_gripper_at_home visualize failed reason={reason}: {exc}"
                        )

                self._log_event(
                    f"release_arm_gripper_at_home direct reason={reason} gripper={release_angle:.1f}"
                )
                return True

            self._log_event(
                f"release_arm_gripper_at_home skipped reason={reason}: no compatible arm API"
            )
            return False

        except Exception as exc:
            self._log_event(f"release_arm_gripper_at_home failed reason={reason}: {exc}")
            return False


    def _move_arm_to_vision_pose(self, reason=""):
        """Move arm to a camera-clear raised pose.

        只改 TaskController，不改 ArmController。
        這版會支援目前新版 ArmController：
        - joint_angles
        - joint_limits
        - _clamp_and_publish()
        - _visualize_arm_lines()
        """
        try:
            arm = self.arm_controller

            if arm is None:
                self._log_event(f"move_arm_to_vision_pose skipped reason={reason}: no arm_controller")
                return False

            # 先嘗試停止/清掉可能存在的 auto target。
            # 注意：目前 ArmController.auto_control(key='q') 可能需要已有 marker 才會進到 q，
            # 所以這裡只是盡量呼叫，失敗不影響後面直接 publish 手臂角度。
            if hasattr(arm, "auto_control"):
                try:
                    arm.auto_control(key="q")
                    self._log_event(f"move_arm_to_vision_pose stop_auto reason={reason}")
                except Exception as exc:
                    self._log_event(f"move_arm_to_vision_pose stop_auto failed reason={reason}: {exc}")

            # 如果以後 ArmController 有正式 move_to_vision_pose，就優先用。
            if hasattr(arm, "move_to_vision_pose"):
                ok = arm.move_to_vision_pose()
                self._log_event(f"move_arm_to_vision_pose method reason={reason} ok={ok}")
                return bool(ok)

            # 新版 ArmController：直接設定 joint_angles，然後呼叫 _clamp_and_publish()
            if (
                hasattr(arm, "joint_angles")
                and hasattr(arm, "joint_limits")
                and hasattr(arm, "_clamp_and_publish")
            ):
                # 先用 ArmController 裡每個 joint 的 init 當作 vision pose。
                # 目前你的 ArmController init 大約是 [-180, 0, 90]。
                vision_pose_degrees = [
                    float(joint.get("init", 0.0))
                    for joint in arm.joint_limits
                ]

                arm.joint_angles = list(vision_pose_degrees)
                arm._clamp_and_publish()

                if hasattr(arm, "_visualize_arm_lines"):
                    try:
                        arm._visualize_arm_lines()
                    except Exception as exc:
                        self._log_event(
                            f"move_arm_to_vision_pose visualize failed reason={reason}: {exc}"
                        )

                self._log_event(
                    f"move_arm_to_vision_pose direct_joint_angles reason={reason} "
                    f"degrees={[round(v, 2) for v in vision_pose_degrees]}"
                )
                return True

            self._log_event(
                f"move_arm_to_vision_pose skipped reason={reason}: "
                f"no compatible arm API"
            )
            return False

        except Exception as exc:
            self._log_event(f"move_arm_to_vision_pose failed reason={reason}: {exc}")
            return False

    def _clear_stale_target_state(self):
        """Clear cached target data after rejected / stale grab target.

        This prevents the state machine from repeatedly using the same old
        yolo_target_info / marker to enter OBSERVE -> GRAB_TARGET again.
        """
        self._last_valid_target_info = None
        self._target_lock_active = False
        self._locked_target_info = None
        self._locked_yolo_marker = None
        self._locked_detection_position = None
        self._outlier_count = 0
        self._lost_target_count = 0
        self._observe_started_at = None

    def _get_current_position(self):
        try:
            pose, _ = self.nav_processing.data_processor.get_processed_amcl_pose()
            return pose
        except Exception:
            return None

    def _elapsed(self):
        if self.state_started_at is None:
            return 0.0

        return time.monotonic() - self.state_started_at

    def _distance(self, first, second):
        return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5

    def _describe_marker(self, marker):
        if marker is None:
            return "None"

        pos = marker.pose.position

        return (
            f"frame={marker.header.frame_id} "
            f"pos=({pos.x:.3f},{pos.y:.3f},{pos.z:.3f})"
        )

    def _describe_point(self, point_msg):
        if point_msg is None:
            return "None"

        point = point_msg.point

        return (
            f"frame={point_msg.header.frame_id} "
            f"point=({point.x:.3f},{point.y:.3f},{point.z:.3f})"
        )

    def _safe_float(self, value):
        try:
            number = float(value)
            if not math.isfinite(number):
                return None
            return number
        except Exception:
            return None