import time
import os
import copy
import math
from enum import Enum

from trajectory_msgs.msg import JointTrajectoryPoint

from pros_car_py.grid_astar_planner import GridAStarPlanner
from pros_car_py.hybrid_astar_planner import HybridAStarPlanner


class TaskType(Enum):
    BEAR_RECOVERY = "task_1_bear_recovery"
    BRIDGE_RECOVERY = "task_2_bridge_recovery"
    DOOR_UNLOCK = "task_3_door_unlock"


class TaskState(Enum):
    IDLE = "idle"
    RECORD_HOME = "record_home"
    SET_VISION_TARGET = "set_vision_target"
    SEARCH_TARGET = "search_target"
    PLAN_TO_BEAR = "plan_to_bear"
    NAV_TO_BEAR = "nav_to_bear"
    APPROACH_TARGET = "approach_target"
    OBSERVE = "observe"
    GRAB_TARGET = "grab_target"
    RETURN_HOME = "return_home"
    PLAN_TO_TASK2_STAGING = "plan_to_task2_staging"
    NAV_TO_TASK2_STAGING = "nav_to_task2_staging"
    WAIT_TASK2_STAGING = "wait_task2_staging"
    ALIGN_TASK2_STAGING_YAW = "align_task2_staging_yaw"
    SET_BRIDGE_VISION_TARGET = "set_bridge_vision_target"
    ALIGN_BRIDGE_ENTRY = "align_bridge_entry"
    SEARCH_BEAR_ON_BRIDGE = "search_bear_on_bridge"
    EXPLORE_MAP = "explore_map"
    SEARCH_BRIDGE = "search_bridge"
    NAV_TO_BRIDGE_PRE_ENTRY = "nav_to_bridge_pre_entry"
    ASCEND_BRIDGE = "ascend_bridge"
    DESCEND_BRIDGE = "descend_bridge"
    PREPARE_DOOR_PRE_PRESS = "prepare_door_pre_press"
    PLAN_TO_DOOR_STAGING = "plan_to_door_staging"
    NAV_TO_DOOR_STAGING = "nav_to_door_staging"
    ALIGN_DOOR_STAGING_YAW = "align_door_staging_yaw"
    ALIGN_KNOB_YOLO = "align_knob_yolo"
    PLAN_TO_KNOB = "plan_to_knob"
    NAV_TO_KNOB = "nav_to_knob"
    UNLOCK_DOOR = "unlock_door"
    DRIVE_TO_KNOB_CONTACT = "drive_to_knob_contact"
    PRESS_DOOR_KNOB = "press_door_knob"
    PUSH_DOOR = "push_door"
    DONE = "done"
    FAILED = "failed"


class TaskController:
    """Runs high-level final-project tasks as a non-blocking state machine."""

    TARGET_DISTANCE_M = 0.7
    DEBUG_MAP_IMAGE_ENABLED = True
    DEBUG_MAP_IMAGE_DIR = "/tmp/pros_task_maps"

    # Bear 會在接近時被手臂/車體遮住，0.9m 太晚。
    # 提早在 1.05m 停下觀察，避免 YOLO/depth 開始跳掉。
    APPROACH_STOP_DISTANCE = 1.05

    # 抓取前必須把目標放到畫面正中間附近。
    # 注意：nav_processing.camera_nav() 的 centered_px 可能比較寬，
    # 但手臂抓取需要更嚴格的置中門檻。
    GRAB_ALIGN_DELTA_X = 45.0
    GRAB_CAMERA_DISTANCE_THRESHOLD = 0.394
    # 只有在目標已經置中時才允許前進靠近。
    # 這避免車子斜著靠近，導致手臂最後偏抓。
    APPROACH_ALIGN_DELTA_X =53.0

    ALIGN_PULSE_ENABLED = True
    ALIGN_ROTATE_PULSE_SECONDS = 0.15
    ALIGN_SETTLE_SECONDS = 0.25
    ALIGN_STALE_REPEAT_MAX = 8
    ALIGN_STUCK_SECONDS = 5.0
    ALIGN_PROGRESS_MIN_DELTA_X = 20.0

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
    RETURN_TIMEOUT_SECONDS = 999.0
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
    MAP_FIRST_NAVIGATION_ENABLED = True
    OBSTACLE_GUARD_USE_SCAN = False
    OBSTACLE_GUARD_SCAN_EMERGENCY_ONLY = False
    OBSTACLE_GUARD_SCAN_EMERGENCY_DISTANCE_M = 0.12
    OBSTACLE_GUARD_USE_CAMERA_DEPTH = True
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
    OBSTACLE_STUCK_SECONDS = 1.2
    OBSTACLE_STUCK_MIN_MOVE = 0.03
    OBSTACLE_STUCK_RECOVERY_SECONDS = 0.7
    STUCK_RECOVERY_ENABLED = False

    BEAR_LABEL = "bear"
    KNOB_LABEL = "knob"
    BRIDGE_LABEL = "bridge"

    DETECTION_MODE = "detection"
    SEGMENTATION_MODE = "segmentation"
    TASK2_SIMPLE_YOLO_BEAR_AFTER_STAGING = True
    TASK2_SIMPLE_DONE_AFTER_GRAB = False
    BRIDGE_ASCEND_ACTION = "FORWARD_BRIDGE"
    BRIDGE_BEAR_APPROACH_ACTION = "FORWARD_BRIDGE"
    BRIDGE_UPHILL_KEEP_THRUST_ENABLED = True
    BRIDGE_UPHILL_ACTION = "FORWARD_BRIDGE"
    BRIDGE_UPHILL_LEFT_ACTION = "FORWARD_BRIDGE_LEFT"
    BRIDGE_UPHILL_RIGHT_ACTION = "FORWARD_BRIDGE_RIGHT"
    BRIDGE_UPHILL_END_DISTANCE_M = 0.39
    BRIDGE_BEAR_CLOSE_PRECISION_ENABLED = True
    BRIDGE_BEAR_CLOSE_PRECISION_ENTER_DISTANCE = 0.39
    BRIDGE_BEAR_YOLO_EMERGENCY_STOP_ENABLED = True
    BRIDGE_BEAR_YOLO_EMERGENCY_STOP_DISTANCE = 0.39
    BRIDGE_BEAR_CLOSE_JUMP_DISTANCE = 0.90
    BRIDGE_BEAR_CLOSE_JUMP_STABLE_COUNT = 5
    BRIDGE_BEAR_NORMAL_SPEED_DISTANCE = 0.39
    BRIDGE_BEAR_CLOSE_PRECISION_EXIT_DISTANCE = 0.39
    BRIDGE_BEAR_NORMAL_SPEED_PAUSE_SECONDS = 2.0
    BRIDGE_BEAR_GRAB_ALIGN_DELTA_X = 85.0
    BRIDGE_BEAR_GRAB_Z_OFFSET_M = -0.01
    BRIDGE_BEAR_OBSERVE_DISTANCE_GUARD_ENABLED = True
    BRIDGE_BEAR_OBSERVE_MAX_DISTANCE_M = 0.389
    BRIDGE_BEAR_OBSERVE_DISTANCE_BAD_COUNT_MAX = 2
    BRIDGE_BEAR_AREA_FILTER_ENABLED = True
    BRIDGE_BEAR_AREA_MARGIN_M = 0.08
    BRIDGE_BEAR_AREA_POLYGON = [
        [1.2133876461384152, 0.946610707913933],
        [1.2169159413309767, 2.108463866739534],
        [0.607744089090695, 2.0893887531954354],
        [0.580656231312927, 1.0065501087341735],
    ]
    BRIDGE_BEAR_GATE_ACTION = "STOP"
    BRIDGE_BEAR_PROBE_SEARCH_ENABLED = True
    BRIDGE_BEAR_PROBE_FORWARD_ACTION = "FORWARD_BRIDGE_SLOW"
    BRIDGE_BEAR_PROBE_BACKWARD_ACTION = "BACKWARD_SLOW"
    BRIDGE_BEAR_PROBE_FRONT_BLOCK_DISTANCE = 0.35
    BRIDGE_BEAR_PROBE_PATTERN = [
        ("STOP", 0.15, "bridge_bear_probe_search_start"),
        ("FORWARD_BRIDGE_SLOW", 0.25, "bridge_bear_probe_forward"),
        ("STOP", 0.15, "bridge_bear_probe_search_start"),
        ("CLOCKWISE_ROTATION_FINE", 0.20, "bridge_bear_probe_rotate_right"),
        ("STOP", 0.15, "bridge_bear_probe_search_start"),
        ("BACKWARD_SLOW", 0.25, "bridge_bear_probe_backward"),
        ("STOP", 0.15, "bridge_bear_probe_search_start"),
        ("COUNTERCLOCKWISE_ROTATION_FINE", 0.20, "bridge_bear_probe_rotate_left"),
    ]
    BRIDGE_EXIT_Y_THRESHOLD = 2.90
    BRIDGE_EXIT_STABLE_COUNT = 3

    # Foxglove should publish /initialpose manually in local_unity/localization_unity.
    # Keep the publisher available in RosCommunicator, but do not auto-publish 0,0,0.
    INITIAL_POSE_ENABLED = False
    INITIAL_POSE_X = 0.0
    INITIAL_POSE_Y = 0.0
    INITIAL_POSE_YAW = 0.0  # radians, in map frame
    INITIAL_POSE_PUBLISH_COUNT = 3
    INITIAL_POSE_SETTLE_SECONDS = 1.0
    INITIAL_POSE_SKIP_IF_POSE_AVAILABLE = False

    WAIT_FOR_AMCL_POSE_ENABLED = True
    WAIT_FOR_AMCL_POSE_TIMEOUT_SECONDS = 30.0
    REQUIRE_AMCL_BEFORE_RECORD_HOME = True

    FIXED_WAYPOINT_REACHED_DISTANCE = 0.45
    FIXED_WAYPOINT_HEADING_THRESHOLD_DEG = 20.0

    BEAR_GLOBAL_NAV_ENABLED = True
    BEAR_MAP_STABLE_COUNT = 3
    BEAR_MAP_MAX_JUMP_M = 0.5
    GROUND_BEAR_MAP_MIN_CLEARANCE_M = 0.10
    BEAR_NAV_GOAL_STOP_DISTANCE = 0.8
    BEAR_FINAL_APPROACH_DISTANCE = 0.6
    BEAR_GLOBAL_NAV_REACHED_DISTANCE = 0.6
    BEAR_NAV_REPLAN_INTERVAL_SECONDS = 2.0
    
    RETURN_HOME_ASTAR_ENABLED = True
    RETURN_HOME_REACHED_DISTANCE = 0.10
    RETURN_HOME_FAST_ROTATION_ENABLED = True
    RETURN_HOME_FAST_ROTATION_ANGLE_DEG = 40.0
    RETURN_HOME_ASTAR_FAILED_COOLDOWN_SECONDS = 3.0
    RETURN_HOME_MAX_ANGLE_FOR_FORWARD_DEG = 45.0
    RETURN_HOME_FORWARD_ACTION = "FORWARD"
    RETURN_HOME_WAYPOINT_REACHED_DISTANCE_M = 0.15
    RETURN_HOME_FINAL_WAYPOINT_REACHED_DISTANCE_M = 0.10
    RETURN_HOME_STUCK_RELEASE_DISTANCE = 0.41
    RETURN_HOME_NEAR_HOME_NO_PROGRESS_SECONDS = 2.0
    RETURN_HOME_NEAR_HOME_PROGRESS_EPS = 0.02
    
    CAR_LENGTH_M = 0.35
    CAR_WIDTH_M = 0.25
    CARRY_CAR_LENGTH_M = 0.40
    CARRY_CAR_WIDTH_M = 0.25
    SAFETY_MARGIN_M = 0.08
    ASTAR_INFLATION_RADIUS_M = 0.0
    ASTAR_INFLATION_MODE = "width"  # "circumscribed" or "width"
    ASTAR_MIN_CORRIDOR_WIDTH_M = None
    ASTAR_WALL_CLEARANCE_M = None
    ASTAR_UNKNOWN_AS_OBSTACLE = True
    ASTAR_OCCUPIED_THRESHOLD = 50
    ASTAR_REPLAN_ON_BLOCKED = True
    ASTAR_BLOCKED_REPLAN_COUNT = 8
    ASTAR_REPLAN_INTERVAL_SECONDS = 2.0
    WAYPOINT_REACHED_DISTANCE_M = 0.1
    WAYPOINT_INTERMEDIATE_REACHED_DISTANCE_M = 0.35
    BEAR_PRE_GRAB_REACHED_DISTANCE = 0.18
    ASTAR_NAV_IGNORE_LIDAR_SIDE_GUARD = True
    ASTAR_NAV_REQUIRE_CAMERA_CONFIRM_FOR_SCAN_FRONT = True
    ASTAR_NAV_FRONT_EMERGENCY_STOP_DISTANCE = 0.15
    ASTAR_NAV_SCAN_CAMERA_DISAGREE_MARGIN = 0.50
    WAYPOINT_HEADING_THRESHOLD_DEG = 15.0
    WAYPOINT_MAX_ANGLE_FOR_FORWARD_DEG = 20.0
    WAYPOINT_SPACING_M = 0.15
    RECTANGULAR_FOOTPRINT_CHECK_ENABLED = True
    FOOTPRINT_CHECK_STEP_M = 0.05
    FOOTPRINT_YAW_STEP_DEG = 10.0
    FOOTPRINT_OCCUPIED_THRESHOLD = 50
    FOOTPRINT_UNKNOWN_AS_OBSTACLE = True
    FOOTPRINT_ROTATION_CHECK_ENABLED = True

    HYBRID_ASTAR_ENABLED = True
    HYBRID_YAW_BINS = 16
    HYBRID_STEP_M = 0.15
    HYBRID_YAW_STEP_DEG = 22.5
    HYBRID_MAX_ITERATIONS = 20000
    HYBRID_GOAL_TOLERANCE_M = 0.12
    HYBRID_BEAR_GOAL_TOLERANCE_M = 0.12
    HYBRID_RETURN_HOME_GOAL_TOLERANCE_M = 0.12
    HYBRID_CLEARANCE_COST_WEIGHT = 0.20
    HYBRID_START_RECOVERY_ENABLED = True
    HYBRID_START_RECOVERY_RADIUS_M = 0.15
    HYBRID_START_RECOVERY_STEP_M = 0.05
    HYBRID_START_RECOVERY_YAW_RANGE_DEG = 30.0
    HYBRID_START_RECOVERY_YAW_STEP_DEG = 15.0
    HYBRID_START_ALLOW_UNKNOWN = True
    HYBRID_DEBUG_IMAGE_ENABLED = True
    HYBRID_DEBUG_IMAGE_INTERVAL = 500
    HYBRID_DEBUG_MAX_IMAGES_PER_PLAN = 8

    # Semantic waypoints. Leave None until calibrated in /map frame.
    TASK2_BRIDGE_PRE_ENTRY_ENABLED = True
    BRIDGE_PRE_ENTRY = None
    BRIDGE_ENTRY_CENTER = None
    BRIDGE_EXIT_CENTER = None
    BEAR_SEARCH_AREA = None

    CHAIN_TASKS_AFTER_TASK1 = True
    CHAIN_TASK3_AFTER_TASK2 = True
    TASK2_STAGING_POSITION = [0.8949071669458816, 0.28174277532982515]
    TASK2_STAGING_YAW_DEG = 90.1
    TASK2_STAGING_REACHED_DISTANCE_M = 0.08
    TASK2_STAGING_FINAL_APPROACH_DISTANCE_M = 0.20
    TASK2_STAGING_PAUSE_SECONDS = 2.0
    TASK2_STAGING_LATERAL_TOLERANCE_M = 0.025
    TASK2_STAGING_FORWARD_TOLERANCE_M = 0.095
    TASK2_STAGING_POSITION_STABLE_COUNT = 4
    TASK2_STAGING_FINAL_DOCK_ENTER_DISTANCE_M = 0.20
    TASK2_STAGING_FINAL_DOCK_ACTION = "FORWARD_VERY_SLOW"
    TASK2_STAGING_FINAL_DOCK_MAX_ANGLE_DEG = 35.0
    TASK2_STAGING_YAW_TOLERANCE_DEG = 0.5
    TASK2_STAGING_YAW_DRIFT_DISTANCE_M = 0.06
    TASK2_STAGING_YAW_DRIFT_LATERAL_TOLERANCE_M = 0.045
    TASK2_STAGING_YAW_DRIFT_FORWARD_TOLERANCE_M = 0.060
    TASK2_STAGING_YAW_FINE_THRESHOLD_DEG = 10.0
    TASK2_STAGING_YAW_STABLE_COUNT = 3
    TASK2_STAGING_MAX_ANGLE_FOR_FORWARD_DEG = 15.0
    TASK2_START_BEAR_SEARCH_FORWARD_ACTION = "FORWARD_VERY_SLOW"
    BRIDGE_ENTRY_IMAGE_CENTER_X = 320.0
    BRIDGE_ENTRY_ALIGN_DELTA_X = 55.0
    BRIDGE_ENTRY_SLOW_DELTA_X = 180.0
    BRIDGE_ALIGN_PULSE_SECONDS = 0.25
    BRIDGE_ALIGN_SETTLE_SECONDS = 0.35
    BRIDGE_ALIGN_STUCK_SECONDS = 8.0
    BRIDGE_ALIGN_PROGRESS_MIN_DELTA_X = 10.0

    # Task3 direct door opening pose measured in /map frame.
    # source pose:
    # position x=3.2079232586192075
    # position y=1.6375321169030426
    # quaternion z=0.008995354371944667
    # quaternion w=0.9999595409813955
    # yaw_deg = 2 * atan2(z, w) = 1.0308055833 deg
    # If strict mode cannot align consistently, recalibrate this pose by placing
    # the car where the arm exactly presses the knob, then record /amcl_pose.
    DOOR_STAGING_POSITION = [3.2079232586192075, 1.6375321169030426]
    DOOR_STAGING_YAW_DEG = 1.0308055833
    DOOR_STAGING_REACHED_DISTANCE_M = 0.025
    DOOR_STAGING_FINAL_APPROACH_DISTANCE_M = 0.03
    DOOR_STAGING_EXACT_PRESS_DISTANCE_M = 0.020
    DOOR_STAGING_YAW_DRIFT_DISTANCE_M = 0.040
    DOOR_STAGING_YAW_TOLERANCE_DEG = 0.08
    DOOR_STAGING_YAW_FINE_THRESHOLD_DEG = 1.5
    DOOR_STAGING_LATERAL_TOLERANCE_M = 0.018
    DOOR_STAGING_FORWARD_TOLERANCE_M = 0.030
    DOOR_STAGING_FINAL_ACCEPT_DISTANCE_M = 0.0
    DOOR_STAGING_FINAL_FORWARD_ONLY_DISTANCE_M = 0.060
    DOOR_STAGING_POSITION_STABLE_COUNT = 4
    DOOR_STAGING_YAW_STABLE_COUNT = 4
    DOOR_STAGING_DIRECT_CREEP_ENTER_DISTANCE_M = 0.20
    DOOR_STAGING_FORWARD_ACTION = "FORWARD"
    DOOR_STAGING_FINAL_CREEP_ACTION = "FORWARD_VERY_SLOW"

    KNOB_MAP_STABLE_COUNT = 3
    KNOB_MAP_MAX_JUMP_M = 0.30
    KNOB_NAV_GOAL_STOP_DISTANCE = 0.45
    KNOB_FINAL_APPROACH_DISTANCE = 0.60
    KNOB_PRE_PRESS_REACHED_DISTANCE = 0.20
    KNOB_NAV_REPLAN_INTERVAL_SECONDS = 1.0

    KNOB_ALIGN_DELTA_X = 55.0
    KNOB_PRESS_DISTANCE = 0.45
    KNOB_APPROACH_FAST_UNTIL_DISTANCE = 0.80

    DOOR_KNOB_PRESS_DOWN_SECONDS = 2.0
    DOOR_PRE_PRESS_BEFORE_NAV_SECONDS = 7.0
    DOOR_PUSH_FORWARD_DISTANCE_M = 0.25
    DOOR_PUSH_FORWARD_MAX_SECONDS = 20.0
    DOOR_PUSH_ACTION = "RIGHT_FRONT_STRONG"

    # Task3 knob YOLO alignment before arm pre-press.
    TASK3_USE_KNOB_YOLO_BEFORE_PRE_PRESS = True
    TASK3_DISABLE_KNOB_YOLO_AFTER_ALIGN = True

    # Knob must be almost exactly centered before arm moves out.
    TASK3_KNOB_ALIGN_DELTA_X = 5.0
    TASK3_KNOB_ALIGN_STABLE_COUNT = 5

    # Distance target is not measured yet. If None, only x alignment is checked.
    TASK3_KNOB_TARGET_DISTANCE_M = 0.378
    TASK3_KNOB_DISTANCE_TOLERANCE_M = 0.02

    TASK3_KNOB_ALIGN_SEARCH_ACTION = "CLOCKWISE_ROTATION_FINE"
    TASK3_KNOB_ALIGN_FORWARD_ACTION = "FORWARD_VERY_SLOW"
    TASK3_KNOB_ALIGN_BACKWARD_ACTION = "BACKWARD_SLOW"

    # After YOLO alignment and pre-press, drive forward slightly to contact knob.
    TASK3_DRIVE_TO_KNOB_CONTACT_DISTANCE_M = 0.176
    TASK3_DRIVE_TO_KNOB_CONTACT_MAX_SECONDS = 3.5
    TASK3_DRIVE_TO_KNOB_CONTACT_ACTION = "FORWARD_VERY_SLOW"

    # Near-door map navigation only needs to get close enough for YOLO.
    TASK3_DOOR_NEAR_REACHED_DISTANCE_M = 0.15
    TASK3_DOOR_NEAR_YAW_TOLERANCE_DEG = 5.0

    DOOR_KNOB_USE_RAW_RADIAN_POSE = True
    DOOR_KNOB_PRE_PRESS_POSE_RAD = [-5.8415926, 9.4, 0.0]
    DOOR_KNOB_PRESS_DOWN_POSE_RAD = [-5.115926, 9.4, 0.0]
    DOOR_KNOB_RELEASE_POSE_RAD = [-3.1415926, 0.0, 1.5707963]

    DOOR_KNOB_PRE_PRESS_POSE = None
    DOOR_KNOB_PRESS_DOWN_POSE = None
    DOOR_KNOB_RELEASE_POSE = [-180.0, 0.0, 90.0]

    DOOR_KNOB_SEARCH_ACTION = "CLOCKWISE_ROTATION_SLOW"

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
        self._grab_auto_started = False

        # True 表示已經夾住 bear/目標，車子還沒回到起始點。
        # 這段期間禁止任何 vision pose / reset pose 打開夾爪。
        self._holding_target = False

        self._last_debug_at = 0.0

        self._initial_pose_published = False
        self._initial_pose_wait_until = 0.0
        self._wait_for_pose_started_at = None
        self._odom_fallback_warned = False

        self._bear_map_positions = []
        self._stable_bear_map_position = None
        self._active_path = None
        self._active_path_yaws = None
        self._active_path_index = 0
        self._active_path_goal = None
        self._active_path_context = None
        self._last_astar_plan_at = 0.0
        self._return_home_astar_failed_until = 0.0
        self._return_home_near_distance = None
        self._return_home_near_since = None
        self._astar_blocked_count = 0

        # Runtime state for stuck recovery.
        self._last_stuck_pose = None
        self._last_stuck_time = None
        self._stuck_started_at = None
        self._recovery_until = 0.0
        self._recovery_action = "STOP"
        self._stuck_recovery_stage = 0
        self._stuck_recovery_stage_pose = None
        self._stuck_recovery_stage_started_at = None

        # Used by visual approach stuck detection: if YOLO distance is still
        # decreasing, do not call it stuck even when /odom is noisy/static.
        self._target_progress_last_distance = None
        self._target_progress_last_time = None
        self._initial_pose_published = False
        self._initial_pose_wait_until = 0.0
        self._wait_for_pose_started_at = None
        self._odom_fallback_warned = False

        self._align_pulse_until = 0.0
        self._align_settle_until = 0.0
        self._align_last_info = None
        self._align_same_count = 0
        self._align_started_at = None
        self._align_best_abs_delta_x = None
        self._bridge_bear_normal_speed_pause_until = None
        self._bridge_bear_normal_speed_pause_done = False
        self._bridge_bear_grabbed = False
        self._bridge_exit_stable_count = 0
        self._bridge_bear_close_precision_mode = False
        self._bridge_bear_close_jump_count = 0
        self._bridge_bear_close_precision_entered_at = None
        self._bridge_bear_observe_bad_distance_count = 0
        self._task2_start_bear_search_forward = False
        self._task2_staging_position_aligned_count = 0
        self._task2_staging_yaw_aligned_count = 0
        self._bridge_bear_probe_step_index = 0
        self._bridge_bear_probe_step_started_at = None
        self._bridge_bear_probe_cycle_started = False

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

        self.log_path = os.environ.get("PROS_TASK_LOG_PATH", "/tmp/pros_task_debug.log")
        self._log_event("TaskController initialized")
        if self.MAP_FIRST_NAVIGATION_ENABLED:
            self._log_event(
                "map_first_navigation enabled "
                f"use_scan={self.OBSTACLE_GUARD_USE_SCAN} "
                f"scan_emergency_only={self.OBSTACLE_GUARD_SCAN_EMERGENCY_ONLY} "
                f"scan_emergency_distance={self.OBSTACLE_GUARD_SCAN_EMERGENCY_DISTANCE_M:.3f} "
                f"use_camera_depth={self.OBSTACLE_GUARD_USE_CAMERA_DEPTH}"
            )
        if not self.OBSTACLE_GUARD_USE_SCAN and not self.OBSTACLE_GUARD_SCAN_EMERGENCY_ONLY:
            self._log_event("scan_disabled_for_navigation use_scan=False scan_emergency_only=False")

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
        self._grab_auto_started = False
        self._holding_target = False

        self._last_stuck_pose = None
        self._last_stuck_time = None
        self._stuck_started_at = None
        self._recovery_until = 0.0
        self._recovery_action = "STOP"
        self._reset_stuck_recovery_stage()
        self._return_home_near_distance = None
        self._return_home_near_since = None
        self._target_progress_last_distance = None
        self._target_progress_last_time = None
        self._wait_for_pose_started_at = None
        self._odom_fallback_warned = False
        self._reset_align_state()
        self._bridge_bear_normal_speed_pause_until = None
        self._bridge_bear_normal_speed_pause_done = False
        self._bridge_bear_grabbed = False
        self._bridge_exit_stable_count = 0
        self._bridge_bear_close_precision_mode = False
        self._bridge_bear_close_jump_count = 0
        self._bridge_bear_close_precision_entered_at = None
        self._bridge_bear_observe_bad_distance_count = 0
        self._task2_start_bear_search_forward = False
        self._task2_staging_position_aligned_count = 0
        self._task2_staging_yaw_aligned_count = 0
        self._reset_bridge_bear_probe_search()
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
        self._bear_map_positions = []
        self._stable_bear_map_position = None
        self._reset_active_path()

        self._log_event("=" * 60)
        self._log_event(f"NEW RUN start task={task_type.value}")
        self._log_event("=" * 60)

        # 關鍵修改：
        # 任務一開始先把手臂抬到不擋相機的位置，避免靠近 bear 時手臂遮住相機。
        self._move_arm_to_vision_pose("task_start")

        self.nav_processing.reset_nav_process()

        if self.frontier_explorer:
            self.frontier_explorer.reset()

        self._publish_initial_pose_if_configured("task_start")

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

            if self.state == TaskState.PLAN_TO_BEAR:
                return self._plan_to_bear()

            if self.state == TaskState.NAV_TO_BEAR:
                return self._nav_to_bear()

            if self.state == TaskState.APPROACH_TARGET:
                return self._approach_target()

            if self.state == TaskState.OBSERVE:
                return self._observe_target()

            if self.state == TaskState.GRAB_TARGET:
                return self._grab_target()

            if self.state == TaskState.RETURN_HOME:
                return self._return_home()

            if self.state == TaskState.PLAN_TO_TASK2_STAGING:
                return self._plan_to_task2_staging()

            if self.state == TaskState.NAV_TO_TASK2_STAGING:
                return self._nav_to_task2_staging()

            if self.state == TaskState.WAIT_TASK2_STAGING:
                return self._wait_task2_staging()

            if self.state == TaskState.ALIGN_TASK2_STAGING_YAW:
                return self._align_task2_staging_yaw()

            if self.state == TaskState.SET_BRIDGE_VISION_TARGET:
                return self._set_bridge_vision_target()

            if self.state == TaskState.ALIGN_BRIDGE_ENTRY:
                return self._align_bridge_entry()

            if self.state == TaskState.SEARCH_BEAR_ON_BRIDGE:
                return self._search_bear_on_bridge()

            if self.state == TaskState.EXPLORE_MAP:
                return self._explore_map()

            if self.state == TaskState.SEARCH_BRIDGE:
                return self._search_bridge()

            if self.state == TaskState.NAV_TO_BRIDGE_PRE_ENTRY:
                return self._nav_to_bridge_pre_entry()

            if self.state == TaskState.ASCEND_BRIDGE:
                return self._ascend_bridge()

            if self.state == TaskState.DESCEND_BRIDGE:
                return self._descend_bridge()

            if self.state == TaskState.PREPARE_DOOR_PRE_PRESS:
                return self._prepare_door_pre_press()

            if self.state == TaskState.PLAN_TO_DOOR_STAGING:
                return self._plan_to_door_staging()

            if self.state == TaskState.NAV_TO_DOOR_STAGING:
                return self._nav_to_door_staging()

            if self.state == TaskState.ALIGN_DOOR_STAGING_YAW:
                return self._align_door_staging_yaw()

            if self.state == TaskState.ALIGN_KNOB_YOLO:
                return self._align_knob_yolo()

            if self.state == TaskState.PLAN_TO_KNOB:
                return self._plan_to_knob()

            if self.state == TaskState.NAV_TO_KNOB:
                return self._nav_to_knob()

            if self.state == TaskState.UNLOCK_DOOR:
                return self._unlock_door()

            if self.state == TaskState.DRIVE_TO_KNOB_CONTACT:
                return self._drive_to_knob_contact()

            if self.state == TaskState.PRESS_DOOR_KNOB:
                return self._press_door_knob()

            if self.state == TaskState.PUSH_DOOR:
                return self._push_door()

            return "STOP"

        except Exception as exc:
            self._transition(TaskState.FAILED, f"Task failed: {exc}")
            return "STOP"

    def _transition(self, state, message):
        previous_state = self.state
        self.state = state
        self.state_started_at = time.monotonic()
        self.status_message = message

        if state == TaskState.SEARCH_BEAR_ON_BRIDGE or previous_state == TaskState.SEARCH_BEAR_ON_BRIDGE:
            self._reset_bridge_bear_probe_search()

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

    def _save_simple_map_debug_image(self, context="", goal=None, extra_text=""):
        if not self.DEBUG_MAP_IMAGE_ENABLED:
            return

        try:
            import cv2
            import numpy as np

            grid = self.ros_communicator.get_latest_map()
            current_pose = self._get_current_pose(require_map_frame=True)
            current_position = self._get_current_position(require_map_frame=True)
            home_position = self.home_position
            active_path = self._active_path
            active_path_index = self._active_path_index

            if grid is None:
                self._log_event("simple_map_debug skipped no_map")
                return

            os.makedirs(self.DEBUG_MAP_IMAGE_DIR, exist_ok=True)

            width = int(grid.info.width)
            height = int(grid.info.height)
            resolution = float(grid.info.resolution)
            origin_x = float(grid.info.origin.position.x)
            origin_y = float(grid.info.origin.position.y)

            data = np.array(grid.data, dtype=np.int16).reshape((height, width))
            image = np.full((height, width, 3), 180, dtype=np.uint8)
            image[data >= int(self.ASTAR_OCCUPIED_THRESHOLD)] = (0, 0, 0)
            image[(data >= 0) & (data < int(self.ASTAR_OCCUPIED_THRESHOLD))] = (255, 255, 255)
            image = np.flipud(image)
            image = np.ascontiguousarray(image.copy()).astype(np.uint8)

            def world_to_image(point):
                if point is None or len(point) < 2 or resolution <= 0.0:
                    return None
                gx = int((float(point[0]) - origin_x) / resolution)
                gy = int((float(point[1]) - origin_y) / resolution)
                ix = gx
                iy = height - 1 - gy
                if ix < 0 or ix >= width or iy < 0 or iy >= height:
                    return None
                return (ix, iy)

            def draw_point(point, color, radius=4):
                nonlocal image
                image_point = world_to_image(point)
                if image_point is not None:
                    image = np.ascontiguousarray(image.copy()).astype(np.uint8)
                    cv2.circle(image, image_point, radius, color, -1)

            def draw_polyline(points, color, thickness=2, closed=True):
                nonlocal image
                image_points = [world_to_image(point) for point in points]
                image_points = [point for point in image_points if point is not None]
                if len(image_points) < 2:
                    return
                try:
                    image = np.ascontiguousarray(image.copy()).astype(np.uint8)
                    pts = np.array(image_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], closed, color, thickness)
                except Exception as exc:
                    self._log_event(f"simple_map_debug polyline skipped error={exc}")

            if active_path:
                path_points = [world_to_image(waypoint) for waypoint in active_path]
                path_points = [point for point in path_points if point is not None]
                if len(path_points) >= 2:
                    try:
                        image = np.ascontiguousarray(image.copy()).astype(np.uint8)
                        pts = np.array(path_points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(image, [pts], False, (255, 0, 0), 2)
                    except Exception as exc:
                        self._log_event(f"simple_map_debug path skipped error={exc}")

                try:
                    for index, waypoint in enumerate(active_path):
                        image_point = world_to_image(waypoint)
                        if image_point is None:
                            continue
                        image = np.ascontiguousarray(image.copy()).astype(np.uint8)
                        cv2.circle(image, image_point, 3, (0, 140, 255), -1)
                        cv2.putText(
                            image,
                            str(index),
                            (image_point[0] + 4, image_point[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (0, 80, 180),
                            1,
                            cv2.LINE_AA,
                        )
                except Exception as exc:
                    self._log_event(f"simple_map_debug waypoints skipped error={exc}")

                if 0 <= active_path_index < len(active_path):
                    draw_point(active_path[active_path_index], (0, 255, 255), radius=5)

            footprint_mode = "unknown"
            if current_pose is not None:
                try:
                    pose_position, pose_orientation = current_pose
                    current_yaw = self._yaw_from_quaternion(pose_orientation)
                    length, footprint_width, footprint_mode = self._footprint_dimensions()
                    corners = self._footprint_corners(
                        pose_position[0],
                        pose_position[1],
                        current_yaw,
                        length,
                        footprint_width,
                    )
                    draw_polyline(corners, (0, 200, 0), thickness=2, closed=True)
                except Exception as exc:
                    self._log_event(f"simple_map_debug footprint skipped error={exc}")

            draw_point(current_position, (0, 200, 0), radius=3)
            draw_point(home_position, (180, 0, 180), radius=5)
            draw_point(goal, (0, 0, 255), radius=5)

            text_lines = [
                f"state={self.state.value}",
                f"context={context}",
                f"current_position={current_position}",
                f"home_position={home_position}",
                f"goal={goal}",
                f"active_path_index={active_path_index}/{len(active_path) if active_path else 0}",
                f"footprint_mode={footprint_mode}",
            ]
            if extra_text:
                text_lines.append(f"extra={extra_text}")

            text_height = 24 * len(text_lines) + 12
            panel = np.full((text_height, width, 3), 245, dtype=np.uint8)
            panel = np.ascontiguousarray(panel.copy()).astype(np.uint8)
            for i, text in enumerate(text_lines):
                panel = np.ascontiguousarray(panel.copy()).astype(np.uint8)
                cv2.putText(
                    panel,
                    text[:160],
                    (8, 22 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            output = np.vstack([image, panel])
            output = np.ascontiguousarray(output.copy()).astype(np.uint8)
            safe_context = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(context))
            filename = f"{int(time.time() * 1000)}_{safe_context or 'map_debug'}.png"
            path = os.path.join(self.DEBUG_MAP_IMAGE_DIR, filename)
            if not cv2.imwrite(path, output):
                raise RuntimeError(f"cv2.imwrite returned false path={path}")
            self._log_event(f"simple_map_debug saved path={path}")
        except Exception as exc:
            self._log_event(f"simple_map_debug failed error={exc}")

    def _record_home(self):
        if self._initial_pose_wait_until and time.monotonic() < self._initial_pose_wait_until:
            remaining = self._initial_pose_wait_until - time.monotonic()
            self.status_message = f"Waiting {remaining:.1f}s for AMCL initial pose to settle."
            self._debug("waiting for AMCL initial pose settle before recording home", 0.5)
            return "STOP"

        if self.WAIT_FOR_AMCL_POSE_ENABLED and self._wait_for_pose_started_at is None:
            self._wait_for_pose_started_at = time.monotonic()

        pose = self._get_current_position(require_map_frame=self.REQUIRE_AMCL_BEFORE_RECORD_HOME)

        if pose is None:
            self.status_message = "Waiting for AMCL pose, please set initial pose in Foxglove."
            self._debug("waiting for AMCL pose; please publish 2D pose estimate in Foxglove", 2.0)
            self._log_event("waiting_for_amcl_pose please_set_initial_pose_in_foxglove")

            if (
                self.WAIT_FOR_AMCL_POSE_ENABLED
                and self._wait_for_pose_started_at is not None
                and time.monotonic() - self._wait_for_pose_started_at > self.WAIT_FOR_AMCL_POSE_TIMEOUT_SECONDS
            ):
                self._debug(
                    f"still waiting for AMCL pose after {self.WAIT_FOR_AMCL_POSE_TIMEOUT_SECONDS:.0f}s; "
                    "publish /initialpose from Foxglove",
                    2.0,
                )
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
                if self._bridge_pre_entry_configured():
                    self._transition(
                        TaskState.NAV_TO_BRIDGE_PRE_ENTRY,
                        f"Navigating to bridge pre-entry waypoint {self.BRIDGE_PRE_ENTRY}.",
                    )
                else:
                    self._transition(
                        TaskState.EXPLORE_MAP,
                        "Bridge pre-entry waypoint not configured; exploring map while searching bridge.",
                    )

        elif self.task_type == TaskType.DOOR_UNLOCK:
            self._knob_map_positions = []
            self._stable_knob_map_position = None
            self._clear_stale_target_state()

            self._door_pre_press_published = False
            self._door_pre_press_started_at = None
            self._door_press_started = False
            self._door_press_done = False
            self._door_press_started_at = None
            self._door_push_started_at = None
            self._door_push_start_position = None

            self._task3_knob_yolo_align_count = 0
            self._task3_knob_yolo_aligned_info = None
            self._task3_knob_yolo_enabled = False
            self._task3_knob_yolo_disabled = False
            self._drive_to_knob_contact_started_at = None
            self._drive_to_knob_contact_start_position = None

            self._move_arm_to_vision_pose("task3_start_before_knob_yolo")
            self._log_event("task3_knob_yolo_flow_enabled arm_kept_in_vision_pose")
            self._transition(
                TaskState.PLAN_TO_DOOR_STAGING,
                "Task3 started. Navigating near door first with arm in vision pose.",
            )

        return "STOP"

    def _search_target(self):
        if self._target_seen():
            if self.task_type == TaskType.BEAR_RECOVERY and self.BEAR_GLOBAL_NAV_ENABLED:
                bear_position = self._update_stable_bear_map_position()
                if bear_position is not None:
                    self._transition(TaskState.PLAN_TO_BEAR, f"Bear map position stable at {bear_position}. Planning A* path.")
                    return "STOP"

                self.status_message = "Bear seen; waiting for stable /map position before A*."
                self._debug("bear seen but map position is not stable yet", 0.5)
                return "STOP"

            if self.task_type == TaskType.DOOR_UNLOCK and self._target_label == self.KNOB_LABEL:
                knob_position = self._update_stable_knob_map_position()
                if knob_position is not None:
                    self._transition(TaskState.PLAN_TO_KNOB, f"Knob map position stable at {knob_position}. Planning A* path.")
                    return "STOP"

            self._outlier_count = 0
            self._lost_target_count = 0
            self._transition(TaskState.APPROACH_TARGET, f"Target {self._target_label} found. Approaching.")
            return "STOP"

        if self._elapsed() > self.SEARCH_TIMEOUT_SECONDS:
            self.status_message = f"Still searching for {self._target_label}; continuing rotation."
            self.state_started_at = time.monotonic()

        action = self.DOOR_KNOB_SEARCH_ACTION if self.task_type == TaskType.DOOR_UNLOCK else "CLOCKWISE_ROTATION_SLOW"
        return self._apply_obstacle_guard(action, None, "search_target")

    def _approach_target(self):
        door_knob_context = (
            self.task_type == TaskType.DOOR_UNLOCK
            and self._target_label == self.KNOB_LABEL
        )
        if (
            door_knob_context
            and self.TASK3_USE_KNOB_YOLO_BEFORE_PRE_PRESS
            and self.state != TaskState.ALIGN_KNOB_YOLO
        ):
            self._log_event(
                f"task3_ignore_legacy_knob_approach "
                f"state={self.state.value} "
                f"reason=using_align_knob_yolo_flow"
            )
            return "STOP"

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
            if self.task_type == TaskType.BRIDGE_RECOVERY and self._target_label == self.BEAR_LABEL:
                self._transition(TaskState.SEARCH_BEAR_ON_BRIDGE, f"Lost bridge bear. Searching on bridge again.")
            else:
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

        if door_knob_context:
            if not self._delta_x_aligned(delta_x, self.KNOB_ALIGN_DELTA_X):
                action = self._alignment_action_from_delta_x(delta_x, self.KNOB_ALIGN_DELTA_X)
                self._last_action = action
                self._log_event(
                    f"door_knob_align distance={distance:.3f} delta_x={delta_x:.1f} action={action}"
                )
                return self._apply_obstacle_guard(action, info, "door_knob_align")

            if distance > self.KNOB_PRESS_DISTANCE:
                if distance > self.KNOB_APPROACH_FAST_UNTIL_DISTANCE:
                    action = "FORWARD_SLOW"
                else:
                    action = "FORWARD_VERY_SLOW"

                self._last_action = action
                self._log_event(
                    f"door_knob_approach distance={distance:.3f} "
                    f"threshold={self.KNOB_PRESS_DISTANCE:.3f} delta_x={delta_x:.1f} action={action}"
                )
                return self._apply_obstacle_guard(action, info, "door_knob_approach")

            self._capture_locked_target(info, "door_knob_centered_close")
            self._log_event(
                f"door_knob_centered_close distance={distance:.3f} "
                f"threshold={self.KNOB_PRESS_DISTANCE:.3f} delta_x={delta_x:.1f}"
            )
            self._transition(
                TaskState.PRESS_DOOR_KNOB,
                "Door knob centered and close. Pressing knob down.",
            )
            return "STOP"

        bridge_bear_context = self.task_type == TaskType.BRIDGE_RECOVERY and self._target_label == self.BEAR_LABEL
        bridge_bear_yolo_emergency_context = (
            bridge_bear_context
            and self.BRIDGE_BEAR_YOLO_EMERGENCY_STOP_ENABLED
            and not self._bridge_bear_grabbed
            and distance is not None
            and distance <= self.BRIDGE_BEAR_YOLO_EMERGENCY_STOP_DISTANCE
        )
        if bridge_bear_context:
            gate_reason = "bridge_bear_yolo_emergency" if bridge_bear_yolo_emergency_context else "approach_target"
            if not self._bridge_bear_current_target_allowed(gate_reason):
                if bridge_bear_yolo_emergency_context:
                    self._log_event(
                        f"bridge_bear_yolo_emergency_ignored_outside_area "
                        f"distance={distance:.3f} delta_x={delta_x:.1f}"
                    )
                self._log_event("bridge_bear_approach_reject_outside_area")
                self._clear_stale_target_state()
                self.nav_processing.reset_nav_process()
                transition_reason = (
                    "Close bear is outside valid bridge area. Ignoring as below-bridge bear."
                    if bridge_bear_yolo_emergency_context
                    else "Detected bear is outside valid bridge area. Searching bridge bear again."
                )
                self._transition(
                    TaskState.SEARCH_BEAR_ON_BRIDGE,
                    transition_reason
                )
                return "STOP"

        align_pulse_seconds = self.BRIDGE_ALIGN_PULSE_SECONDS if bridge_bear_context else self.ALIGN_ROTATE_PULSE_SECONDS
        align_settle_seconds = self.BRIDGE_ALIGN_SETTLE_SECONDS if bridge_bear_context else self.ALIGN_SETTLE_SECONDS
        align_stuck_seconds = self.BRIDGE_ALIGN_STUCK_SECONDS if bridge_bear_context else self.ALIGN_STUCK_SECONDS
        align_progress_min_delta_x = self.BRIDGE_ALIGN_PROGRESS_MIN_DELTA_X if bridge_bear_context else self.ALIGN_PROGRESS_MIN_DELTA_X
        grab_align_delta_x = self.BRIDGE_BEAR_GRAB_ALIGN_DELTA_X if bridge_bear_context else self.GRAB_ALIGN_DELTA_X

        if bridge_bear_yolo_emergency_context:
            just_entered_close_precision = False

            if not self._bridge_bear_close_precision_mode:
                self._bridge_bear_close_precision_mode = True
                self._bridge_bear_close_jump_count = 0
                self._bridge_bear_close_precision_entered_at = time.monotonic()
                just_entered_close_precision = True
                self._reset_align_state()
                self._log_event(
                    f"bridge_bear_emergency_enter_close_precision "
                    f"distance={distance:.3f} "
                    f"emergency_threshold={self.BRIDGE_BEAR_YOLO_EMERGENCY_STOP_DISTANCE:.3f} "
                    f"delta_x={delta_x:.1f}"
                )

            if just_entered_close_precision:
                self.car_controller.update_action("STOP")
                self._last_action = "STOP"
                self.nav_processing.reset_nav_process()
                self._log_event(
                    f"bridge_bear_yolo_emergency_stop_once "
                    f"distance={distance:.3f} "
                    f"threshold={self.BRIDGE_BEAR_YOLO_EMERGENCY_STOP_DISTANCE:.3f} "
                    f"delta_x={delta_x:.1f}"
                )
                return "STOP"

        if (
            bridge_bear_context
            and self._bridge_bear_close_precision_mode
            and not self._bridge_bear_grabbed
            and distance >= self.BRIDGE_BEAR_CLOSE_JUMP_DISTANCE
        ):
            self._bridge_bear_close_jump_count += 1
            self._log_event(
                f"bridge_bear_close_target_jump_hold "
                f"count={self._bridge_bear_close_jump_count}/{self.BRIDGE_BEAR_CLOSE_JUMP_STABLE_COUNT} "
                f"distance={distance:.3f} "
                f"jump_threshold={self.BRIDGE_BEAR_CLOSE_JUMP_DISTANCE:.3f} "
                f"delta_x={delta_x:.1f}"
            )

            if self._bridge_bear_close_jump_count < self.BRIDGE_BEAR_CLOSE_JUMP_STABLE_COUNT:
                return "STOP"

            self._log_event(
                f"bridge_bear_close_target_jump_release_to_search "
                f"distance={distance:.3f} delta_x={delta_x:.1f}"
            )
            self._bridge_bear_close_precision_mode = False
            self._bridge_bear_close_jump_count = 0
            self._bridge_bear_close_precision_entered_at = None
            self._clear_stale_target_state()
            self.nav_processing.reset_nav_process()
            self._transition(
                TaskState.SEARCH_BEAR_ON_BRIDGE,
                "Bridge bear close target jumped far. Re-searching nearest bear."
            )
            return "STOP"
        else:
            if bridge_bear_context and self._bridge_bear_close_precision_mode:
                self._bridge_bear_close_jump_count = 0

        if (
            bridge_bear_context
            and not self._bridge_bear_grabbed
            and distance <= self.BRIDGE_BEAR_YOLO_EMERGENCY_STOP_DISTANCE
            and not self._bridge_bear_close_precision_mode
        ):
            self._bridge_bear_close_precision_mode = True
            self._log_event(
                f"bridge_uphill_blocked_by_yolo_emergency "
                f"distance={distance:.3f} "
                f"threshold={self.BRIDGE_BEAR_YOLO_EMERGENCY_STOP_DISTANCE:.3f}"
            )
            return "STOP"

        if (
            bridge_bear_context
            and self.BRIDGE_UPHILL_KEEP_THRUST_ENABLED
            and not self._bridge_bear_grabbed
            and not self._bridge_bear_close_precision_mode
            and distance > self.BRIDGE_BEAR_CLOSE_PRECISION_ENTER_DISTANCE
        ):
            action = self._bridge_uphill_keep_thrust_action(delta_x)
            self._last_action = action
            self._log_event(
                f"bridge_uphill_keep_thrust distance={distance:.3f} "
                f"delta_x={delta_x:.1f} action={action}"
            )
            return self._apply_obstacle_guard(
                action,
                info,
                "bridge_uphill_keep_thrust"
            )

        if bridge_bear_context and distance <= self.BRIDGE_UPHILL_END_DISTANCE_M:
            self._log_event(
                f"bridge_bear_close_precision distance={distance:.3f} "
                f"delta_x={delta_x:.1f}"
            )

        # 5. 第一優先：目標必須在畫面正中間附近。
        # 不管距離多近，只要還沒置中，就不 lock、不 observe、不 grab。
        if not self._delta_x_aligned(delta_x, grab_align_delta_x):
            now = time.monotonic()

            if self.ALIGN_PULSE_ENABLED and now >= self._align_pulse_until and now < self._align_settle_until:
                if (
                    bridge_bear_context
                    and not self._bridge_bear_close_precision_mode
                    and distance > self.BRIDGE_BEAR_CLOSE_PRECISION_ENTER_DISTANCE
                ):
                    action = self._bridge_uphill_keep_thrust_action(delta_x)
                    self._last_action = action
                    self._log_event(
                        f"bridge_uphill_no_stop_settle action={action} "
                        f"distance={distance:.3f} delta_x={delta_x:.1f}"
                    )
                    return self._apply_obstacle_guard(
                        action,
                        info,
                        "bridge_uphill_no_stop_settle"
                    )
                self._log_event("align_settle waiting for fresh target_info")
                self._debug("align settle waiting for fresh target_info", 0.5)
                return "STOP"

            if self.ALIGN_PULSE_ENABLED and now < self._align_pulse_until:
                return self._apply_obstacle_guard(
                    self._last_action,
                    info,
                    "approach_align_target_pulse",
                )

            current_align_info = (distance, delta_x)
            if self._align_last_info == current_align_info:
                self._align_same_count += 1
            else:
                self._align_last_info = current_align_info
                self._align_same_count = 1

            if self._align_same_count > self.ALIGN_STALE_REPEAT_MAX:
                self._log_event(
                    f"align_stale_target_info stop_waiting count={self._align_same_count} "
                    f"distance={distance:.3f} delta_x={delta_x:.1f}"
                )
                self._align_settle_until = now + align_settle_seconds
                return "STOP"

            abs_delta_x = abs(delta_x)
            if self._align_started_at is None:
                self._align_started_at = now
                self._align_best_abs_delta_x = abs_delta_x
            elif self._align_best_abs_delta_x is None:
                self._align_best_abs_delta_x = abs_delta_x
            elif abs_delta_x <= self._align_best_abs_delta_x - align_progress_min_delta_x:
                self._align_best_abs_delta_x = abs_delta_x
                self._align_started_at = now

            if (
                self._align_started_at is not None
                and now - self._align_started_at > align_stuck_seconds
                and abs_delta_x > grab_align_delta_x
            ):
                self._log_event(
                    f"align_stuck_no_progress reset_search distance={distance:.3f} "
                    f"delta_x={delta_x:.1f} best_abs_delta_x={self._format_float(self._align_best_abs_delta_x, 1)}"
                )
                self._clear_stale_target_state()
                self.nav_processing.reset_nav_process()
                if bridge_bear_context:
                    self._transition(TaskState.SEARCH_BEAR_ON_BRIDGE, "Bridge bear alignment stuck. Searching on bridge again.")
                else:
                    self._transition(TaskState.SEARCH_TARGET, "Alignment stuck. Searching target again.")
                return "STOP"

            if bridge_bear_context and abs(delta_x) > 180.0:
                action = "CLOCKWISE_ROTATION_SLOW" if delta_x > 0 else "COUNTERCLOCKWISE_ROTATION_SLOW"
            else:
                action = self._alignment_action_from_delta_x(delta_x, grab_align_delta_x)

            self.status_message = (
                f"Aligning {self._target_label}: "
                f"distance={distance:.2f}m delta_x={delta_x:.1f}"
            )
            self._last_action = action
            if self.ALIGN_PULSE_ENABLED:
                self._align_pulse_until = now + align_pulse_seconds
                self._align_settle_until = self._align_pulse_until + align_settle_seconds

            self._log_event(
                f"align_pulse delta_x={delta_x:.1f} action={action} "
                f"pulse={align_pulse_seconds:.2f} settle={align_settle_seconds:.2f}"
            )
            self._debug(
                f"aligning target distance={distance:.3f} delta_x={delta_x:.1f} action={action}",
                0.5,
            )
            return self._apply_obstacle_guard(action, info, "approach_align_target_pulse_start")

        self._reset_align_state()

        # 6. 第二優先：目標已置中，才允許往前靠近。
        # Bear / Bridge 要等 ArmController 判定 target 在手臂可達範圍內，才准抓。
        if self.task_type in (TaskType.BEAR_RECOVERY, TaskType.BRIDGE_RECOVERY):
            if not self._arm_target_reachable():
                self.status_message = (
                    f"{self._target_label} centered but outside arm reach. Moving closer."
                )

                if bridge_bear_context:
                    action = self._bridge_bear_approach_forward_action(distance)
                else:
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
            if not self._delta_x_aligned(delta_x, grab_align_delta_x):
                action = self._alignment_action_from_delta_x(delta_x, grab_align_delta_x)
                self._log_event(
                    f"grab_align_rejected distance={distance:.3f} delta_x={delta_x:.1f} "
                    f"threshold={grab_align_delta_x:.1f} action={action}"
                )
                self._last_action = action
                return self._apply_obstacle_guard(action, info, "approach_grab_align_rejected")

            # 7. 第三優先：已置中 + 手臂可達，但 camera distance 還太遠時，不准抓。
            # 之前 distance=0.47 就開始抓，實測夾不到，所以要繼續慢慢靠近。
            if distance > self.GRAB_CAMERA_DISTANCE_THRESHOLD:
                if bridge_bear_context:
                    action = self._bridge_bear_approach_forward_action(distance)
                else:
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

            if bridge_bear_context and not self._bridge_bear_current_target_allowed("before_observe_grab"):
                self._log_event("bridge_bear_grab_ready_rejected_outside_area")
                self._clear_stale_target_state()
                self.nav_processing.reset_nav_process()
                self._transition(
                    TaskState.SEARCH_BEAR_ON_BRIDGE,
                    "Bear became ready but is outside valid bridge area. Searching again."
                )
                return "STOP"

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
            if bridge_bear_context:
                self._log_event(
                    f"bridge_bear_grab_ready distance={distance:.3f} "
                    f"delta_x={delta_x:.1f}"
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
            if (
                self.task_type == TaskType.BRIDGE_RECOVERY
                and self._target_label == self.BEAR_LABEL
                and self.state == TaskState.OBSERVE
                and self.BRIDGE_BEAR_OBSERVE_DISTANCE_GUARD_ENABLED
            ):
                info = self._get_yolo_target_info()
                distance = None
                delta_x = None
                if info and info[0] == 1:
                    distance = self._safe_float(info[1])
                    delta_x = self._safe_float(info[2])

                if distance is not None and distance > self.BRIDGE_BEAR_OBSERVE_MAX_DISTANCE_M:
                    self._bridge_bear_observe_bad_distance_count += 1
                    self._log_event(
                        f"bridge_bear_observe_distance_too_far "
                        f"count={self._bridge_bear_observe_bad_distance_count}/"
                        f"{self.BRIDGE_BEAR_OBSERVE_DISTANCE_BAD_COUNT_MAX} "
                        f"distance={distance:.3f} "
                        f"threshold={self.BRIDGE_BEAR_OBSERVE_MAX_DISTANCE_M:.3f} "
                        f"delta_x={self._format_float(delta_x, 1)}"
                    )

                    if self._bridge_bear_observe_bad_distance_count >= self.BRIDGE_BEAR_OBSERVE_DISTANCE_BAD_COUNT_MAX:
                        self._observe_started_at = None
                        self._bridge_bear_observe_bad_distance_count = 0
                        self._clear_stale_target_state()
                        self.nav_processing.reset_nav_process()

                        self._log_event(
                            f"bridge_bear_observe_abort_return_to_approach "
                            f"distance={distance:.3f} "
                            f"threshold={self.BRIDGE_BEAR_OBSERVE_MAX_DISTANCE_M:.3f}"
                        )

                        self._transition(
                            TaskState.APPROACH_TARGET,
                            "Bridge bear moved too far during observation. Re-approaching before grab."
                        )
                        return "STOP"
                else:
                    self._bridge_bear_observe_bad_distance_count = 0

            return "STOP"

        if self.task_type in (TaskType.BEAR_RECOVERY, TaskType.BRIDGE_RECOVERY):
            self._bridge_bear_observe_bad_distance_count = 0
            self._grab_started = False
            self._grab_auto_started = False
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
            self._apply_bridge_bear_grab_z_offset()

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
                self._grab_auto_started = False
                self._observe_started_at = None

                # auto_control 回傳 False 代表 ArmController 判定目前 target 不可達。
                # 這時不能保留 _last_valid_target_info，否則下一個 tick 會用同一筆舊資料再次 observe/grab。
                self._clear_stale_target_state()

                self.nav_processing.reset_nav_process()
                self._move_arm_to_vision_pose("grab_rejected")
                if self.task_type == TaskType.BRIDGE_RECOVERY and self._target_label == self.BEAR_LABEL:
                    self._transition(
                        TaskState.SEARCH_BEAR_ON_BRIDGE,
                        "Arm target is not reachable. Searching bear on bridge again.",
                    )
                else:
                    self._transition(
                        TaskState.SEARCH_TARGET,
                        "Arm target is not reachable. Clearing stale target and searching again.",
                    )
                return "STOP"

            self._grab_auto_started = True
            self.status_message = "Grab sequence started. Waiting for arm motion."
            return "STOP"

        if self._elapsed() < self.GRAB_SECONDS:
            return "STOP"

        if not getattr(self, "_grab_auto_started", False):
            self._log_event("grab_wait_abort_no_auto_control")
            self._grab_started = False
            self._grab_auto_started = False
            self._observe_started_at = None
            self._clear_stale_target_state()
            self.nav_processing.reset_nav_process()
            self._move_arm_to_vision_pose("grab_wait_abort_no_auto_control")
            if self.task_type == TaskType.BRIDGE_RECOVERY and self._target_label == self.BEAR_LABEL:
                self._transition(
                    TaskState.SEARCH_BEAR_ON_BRIDGE,
                    "Grab wait elapsed without auto_control. Searching bear on bridge again.",
                )
            else:
                self._transition(
                    TaskState.SEARCH_TARGET,
                    "Grab wait elapsed without auto_control. Searching target again.",
                )
            return "STOP"

        self._return_started_at = time.monotonic()
        self.nav_processing.reset_nav_process()

        # 抓取流程結束後，改成「抱著目標回家姿態」。
        # 重點：不可以呼叫 _move_arm_to_vision_pose，因為 vision pose 會把 gripper 設成 init/max，導致放掉 bear。
        self._holding_target = True
        self._move_arm_to_carry_pose("grab_wait_complete_hold_target")
        self._grab_started = False
        self._grab_auto_started = False

        if self.task_type == TaskType.BRIDGE_RECOVERY:
            self._log_event("bridge_bear_grab_done")
            self._bridge_bear_grabbed = True
            self._bridge_bear_close_precision_mode = False
            self._bridge_bear_close_jump_count = 0
            self._bridge_bear_close_precision_entered_at = None
            self._reset_active_path()

            current_position = self._get_current_position(require_map_frame=True)
            current_y = None
            if current_position is not None and len(current_position) >= 2:
                current_y = current_position[1]

            if current_y is not None and current_y >= self.BRIDGE_EXIT_Y_THRESHOLD:
                self._log_event(
                    f"bridge_bear_grabbed_already_exited "
                    f"current_y={current_y:.3f} "
                    f"threshold={self.BRIDGE_EXIT_Y_THRESHOLD:.3f}"
                )
                self._bridge_exit_stable_count = 0
                self._return_started_at = time.monotonic()
                self._log_event("return_home_after_bridge_exit")
                self._transition(
                    TaskState.RETURN_HOME,
                    "Bridge bear grabbed after exit. Returning home."
                )
                return "STOP"

            self._log_event(
                f"bridge_bear_grabbed_need_descend "
                f"current_y={self._format_float(current_y, 3)} "
                f"threshold={self.BRIDGE_EXIT_Y_THRESHOLD:.3f}"
            )
            self._bridge_exit_stable_count = 0
            self._return_started_at = None
            self._transition(
                TaskState.DESCEND_BRIDGE,
                "Bridge bear grabbed. Driving down bridge while holding target."
            )
            return "STOP"

        self._log_event("grab_wait_complete returning_home holding_target=True")
        self._reset_active_path()
        self._return_home_near_distance = None
        self._return_home_near_since = None
        self._transition(TaskState.RETURN_HOME, "Returning home while holding target.")

        return "STOP"

    def _return_home(self):
        if not self.home_position:
            self._transition(TaskState.FAILED, "Cannot return home: no home pose recorded.")
            return "STOP"

        if self._return_started_at and time.monotonic() - self._return_started_at > self.RETURN_TIMEOUT_SECONDS:
            self._save_simple_map_debug_image(
                context="return_home_timeout",
                goal=self.home_position,
                extra_text="return home timeout",
            )
            self._transition(TaskState.FAILED, "Return home timed out.")
            return "STOP"

        current_position = self._get_current_position(require_map_frame=True)
        distance_to_home = self._distance(current_position, self.home_position) if current_position else None

        if distance_to_home is not None and distance_to_home <= self.RETURN_HOME_REACHED_DISTANCE:
            # 只有真的回到起始位置，才允許打開夾爪放下 bear。
            if getattr(self, "_holding_target", False):
                self._save_simple_map_debug_image(
                    context="return_home_release",
                    goal=self.home_position,
                    extra_text="release bear near home",
                )
                self._release_arm_gripper_at_home("return_home_arrived")
                self._holding_target = False
                self._transition_after_return_home_release(
                    "Returned home and released target. Task done."
                )
            else:
                self._transition(TaskState.DONE, "Returned home. Task done.")
            return "STOP"

        if self._should_return_home_stuck_release(distance_to_home):
            self._log_event(
                f"return_home_near_home_no_progress_release distance_to_home={self._format_float(distance_to_home, 3)}"
            )
            if getattr(self, "_holding_target", False):
                self._save_simple_map_debug_image(
                    context="return_home_near_home_no_progress_release",
                    goal=self.home_position,
                    extra_text=f"insurance release distance={self._format_float(distance_to_home, 3)}",
                )
                self._release_arm_gripper_at_home("return_home_near_home_no_progress_release")
                self._holding_target = False
                self._transition_after_return_home_release(
                    "Near home with no progress. Released target. Task done."
                )
            else:
                self._transition(TaskState.DONE, "Near home with no progress. Task done.")
            return "STOP"

        if self.RETURN_HOME_ASTAR_ENABLED:
            if not self._active_path or self._active_path_context != "return_home":
                now = time.monotonic()
                cooldown_active = now < self._return_home_astar_failed_until
                if cooldown_active:
                    self._log_event(
                        f"return_home astar_failed_cooldown_active until={self._return_home_astar_failed_until:.2f}"
                    )
                    plan_ok = False
                else:
                    plan_ok = self._plan_astar_path(self.home_position, "return_home")

                if not plan_ok:
                    if not cooldown_active:
                        self._return_home_astar_failed_until = now + float(self.RETURN_HOME_ASTAR_FAILED_COOLDOWN_SECONDS)
                    self._log_event("return_home astar_failed fallback_to_direct_low_speed")
                    action = self._navigate_to_waypoint(
                        self.home_position,
                        reached_distance=self.RETURN_HOME_REACHED_DISTANCE,
                        context="return_home_fallback_direct",
                    )
                else:
                    action = "STOP"
            else:
                action = self._follow_active_path_action(
                    reached_distance=self.RETURN_HOME_REACHED_DISTANCE,
                    context="return_home",
                )
        else:
            action = self._navigate_to_waypoint(
                self.home_position,
                reached_distance=self.RETURN_HOME_REACHED_DISTANCE,
                context="return_home_direct",
            )

        current_position = self._get_current_position(require_map_frame=True)
        distance = self._distance(current_position, self.home_position) if current_position else None
        self._debug(
            f"return_home current={current_position} home={self.home_position} "
            f"distance={self._format_float(distance, 3)} action={action}",
            0.5,
        )
        return action

    def _transition_after_return_home_release(self, done_message):
        if self.task_type == TaskType.BEAR_RECOVERY and self.CHAIN_TASKS_AFTER_TASK1:
            self._holding_target = False
            self._reset_active_path()
            self._log_event(
                f"task1_chain_to_task2_staging enabled=True"
            )
            self._log_event(
                "task2_staging_goal "
                f"position=[{self.TASK2_STAGING_POSITION[0]:.6f},{self.TASK2_STAGING_POSITION[1]:.6f}] "
                f"yaw_deg={self.TASK2_STAGING_YAW_DEG:.1f}"
            )
            self._transition(
                TaskState.PLAN_TO_TASK2_STAGING,
                "Task1 done. Bear released. Planning path to task2 staging pose.",
            )
            return

        if self.task_type == TaskType.BRIDGE_RECOVERY and self.CHAIN_TASK3_AFTER_TASK2:
            self._holding_target = False
            self._reset_active_path()
            self.nav_processing.reset_nav_process()
            self._clear_stale_target_state()

            self.task_type = TaskType.DOOR_UNLOCK
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

            self._log_event("task2_chain_to_task3 enabled=True")
            self._log_event("task3_knob_yolo_flow_enabled arm_kept_in_vision_pose")

            self._move_arm_to_vision_pose("task2_chain_to_task3_before_knob_yolo")
            self._transition(
                TaskState.PLAN_TO_DOOR_STAGING,
                "Task2 done. Starting Task3. Navigating near door first with arm in vision pose.",
            )
            return

        self._transition(TaskState.DONE, done_message)

    def _plan_to_task2_staging(self):
        self._holding_target = False
        self._reset_active_path()
        self._task2_staging_position_aligned_count = 0
        goal = list(self.TASK2_STAGING_POSITION)
        self._log_event(
            "task2_staging_goal "
            f"position=[{goal[0]:.6f},{goal[1]:.6f}] yaw_deg={self.TASK2_STAGING_YAW_DEG:.1f}"
        )
        if self._plan_astar_path(goal, "task2_staging"):
            if self._active_path:
                last = self._active_path[-1]
                if self._distance(last, goal) > 0.05:
                    self._active_path.append(list(goal))
                    if self._active_path_yaws:
                        self._active_path_yaws.append(self._active_path_yaws[-1])
                    self._log_event(
                        f"task2_staging_append_true_goal last={last} true_goal={goal}"
                    )
            path_len = len(self._active_path) if self._active_path else 0
            self._log_event(f"task2_staging_plan_success path_len={path_len}")
            self._transition(TaskState.NAV_TO_TASK2_STAGING, "Following path to task2 staging pose.")
            return "STOP"

        self._log_event("task2_staging_plan_failed fallback=direct")
        self._transition(TaskState.NAV_TO_TASK2_STAGING, "Task2 staging plan failed. Driving directly to staging pose.")
        return "STOP"

    def _task2_staging_pose_error(self, current_position=None):
        if current_position is None:
            current_position = self._get_current_position(require_map_frame=True)

        if current_position is None or len(current_position) < 2:
            return None, None, None

        goal = self.TASK2_STAGING_POSITION
        dx = float(goal[0]) - float(current_position[0])
        dy = float(goal[1]) - float(current_position[1])

        yaw = math.radians(float(self.TASK2_STAGING_YAW_DEG))
        forward_error = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral_error = -math.sin(yaw) * dx + math.cos(yaw) * dy
        distance = math.hypot(dx, dy)

        return distance, forward_error, lateral_error

    def _task2_staging_position_aligned(self, current_position=None):
        distance, forward_error, lateral_error = self._task2_staging_pose_error(current_position)

        if distance is None:
            return False, distance, forward_error, lateral_error

        aligned = (
            abs(lateral_error) <= float(self.TASK2_STAGING_LATERAL_TOLERANCE_M)
            and abs(forward_error) <= float(self.TASK2_STAGING_FORWARD_TOLERANCE_M)
        )

        self._log_event(
            f"task2_staging_axis_error "
            f"distance={self._format_float(distance, 3)} "
            f"forward_error={self._format_float(forward_error, 3)} "
            f"forward_tol={self.TASK2_STAGING_FORWARD_TOLERANCE_M:.3f} "
            f"lateral_error={self._format_float(lateral_error, 3)} "
            f"lateral_tol={self.TASK2_STAGING_LATERAL_TOLERANCE_M:.3f} "
            f"aligned={aligned}"
        )

        return aligned, distance, forward_error, lateral_error

    def _task2_staging_final_dock_action(self, current_position=None):
        raw_action, pose_position, distance, angle_error = self._fixed_waypoint_action(
            self.TASK2_STAGING_POSITION,
            reached_distance=0.0,
            context="task2_staging_direct",
        )

        aligned, axis_distance, forward_error, lateral_error = self._task2_staging_position_aligned(current_position)

        if aligned:
            self._log_event(
                f"task2_staging_final_dock_aligned "
                f"distance={self._format_float(axis_distance, 3)} "
                f"forward_error={self._format_float(forward_error, 3)} "
                f"lateral_error={self._format_float(lateral_error, 3)}"
            )
            return "STOP"

        if angle_error is not None and abs(angle_error) <= self.TASK2_STAGING_FINAL_DOCK_MAX_ANGLE_DEG:
            action = self.TASK2_STAGING_FINAL_DOCK_ACTION
        else:
            if angle_error is None:
                action = "STOP"
            elif angle_error > 0:
                action = "COUNTERCLOCKWISE_ROTATION_FINE"
            else:
                action = "CLOCKWISE_ROTATION_FINE"

        self._log_event(
            f"task2_staging_final_dock "
            f"distance={self._format_float(distance, 3)} "
            f"axis_distance={self._format_float(axis_distance, 3)} "
            f"angle_error={self._format_float(angle_error, 1)} "
            f"forward_error={self._format_float(forward_error, 3)} "
            f"lateral_error={self._format_float(lateral_error, 3)} "
            f"raw_action={raw_action} "
            f"action={action}"
        )

        return self._apply_obstacle_guard(
            action,
            None,
            "task2_staging_final_dock",
        )

    def _nav_to_task2_staging(self):
        current_position = self._get_current_position(require_map_frame=True)
        distance = self._distance(current_position, self.TASK2_STAGING_POSITION) if current_position else None

        aligned, axis_distance, forward_error, lateral_error = self._task2_staging_position_aligned(current_position)

        self._log_event(
            f"task2_staging_distance_check current={current_position} "
            f"target={self.TASK2_STAGING_POSITION} "
            f"distance={self._format_float(distance, 3)} "
            f"threshold={self.TASK2_STAGING_REACHED_DISTANCE_M:.3f} "
            f"forward_error={self._format_float(forward_error, 3)} "
            f"lateral_error={self._format_float(lateral_error, 3)}"
        )

        if aligned:
            self._task2_staging_position_aligned_count += 1
            self._log_event(
                f"task2_staging_position_aligned_sample "
                f"count={self._task2_staging_position_aligned_count}/{self.TASK2_STAGING_POSITION_STABLE_COUNT} "
                f"distance={self._format_float(distance, 3)} "
                f"forward_error={self._format_float(forward_error, 3)} "
                f"lateral_error={self._format_float(lateral_error, 3)}"
            )

            if self._task2_staging_position_aligned_count < self.TASK2_STAGING_POSITION_STABLE_COUNT:
                return "STOP"

            self._log_event("task2_staging_exact_position_confirmed")
            self._reset_active_path()
            self._log_event("task2_staging_position_reached_pause_start")
            self._transition(
                TaskState.WAIT_TASK2_STAGING,
                "Reached task2 staging exact position. Pausing before yaw alignment."
            )
            return "STOP"

        self._task2_staging_position_aligned_count = 0

        if distance is not None and distance <= self.TASK2_STAGING_FINAL_DOCK_ENTER_DISTANCE_M:
            return self._task2_staging_final_dock_action(current_position)

        if self._active_path and self._active_path_context == "task2_staging":
            return self._follow_active_path_action(
                reached_distance=self.TASK2_STAGING_REACHED_DISTANCE_M,
                context="task2_staging",
            )

        self._log_event(
            f"task2_staging_direct_final_approach distance={self._format_float(distance, 3)}"
        )
        return self._navigate_to_waypoint(
            self.TASK2_STAGING_POSITION,
            reached_distance=self.TASK2_STAGING_REACHED_DISTANCE_M,
            context="task2_staging_direct",
        )

    def _align_task2_staging_yaw(self):
        current_position = self._get_current_position(require_map_frame=True)
        aligned, distance, forward_error, lateral_error = self._task2_staging_position_aligned(current_position)

        if not aligned:
            yaw_position_ok = (
                distance is not None
                and distance <= self.TASK2_STAGING_YAW_DRIFT_DISTANCE_M
                and forward_error is not None
                and abs(forward_error) <= self.TASK2_STAGING_YAW_DRIFT_FORWARD_TOLERANCE_M
                and lateral_error is not None
                and abs(lateral_error) <= self.TASK2_STAGING_YAW_DRIFT_LATERAL_TOLERANCE_M
            )

            if not yaw_position_ok:
                self._task2_staging_yaw_aligned_count = 0
                self._log_event(
                    f"task2_staging_yaw_position_not_exact_return_nav "
                    f"distance={self._format_float(distance, 3)} "
                    f"forward_error={self._format_float(forward_error, 3)} "
                    f"lateral_error={self._format_float(lateral_error, 3)}"
                )
                self._transition(
                    TaskState.NAV_TO_TASK2_STAGING,
                    "Position drifted from exact task2 staging. Returning to staging position."
                )
                return "STOP"

            self._log_event(
                f"task2_staging_yaw_position_soft_accept "
                f"distance={self._format_float(distance, 3)} "
                f"forward_error={self._format_float(forward_error, 3)} "
                f"lateral_error={self._format_float(lateral_error, 3)}"
            )

        action, yaw_error, current_yaw = self._align_to_map_yaw_action(
            self.TASK2_STAGING_YAW_DEG,
            self.TASK2_STAGING_YAW_TOLERANCE_DEG,
            context="task2_staging",
        )

        if (
            action != "STOP"
            and yaw_error is not None
            and abs(yaw_error) <= self.TASK2_STAGING_YAW_FINE_THRESHOLD_DEG
        ):
            action = (
                "COUNTERCLOCKWISE_ROTATION_FINE"
                if yaw_error > 0.0
                else "CLOCKWISE_ROTATION_FINE"
            )
            self._log_event(
                f"task2_staging_yaw_fine_adjust "
                f"current_yaw={self._format_float(current_yaw, 2)} "
                f"target_yaw={self.TASK2_STAGING_YAW_DEG:.2f} "
                f"error={self._format_float(yaw_error, 2)} "
                f"action={action}"
            )

        self._log_event(
            f"task2_staging_yaw_align current_yaw={self._format_float(current_yaw, 1)} "
            f"target_yaw={self.TASK2_STAGING_YAW_DEG:.1f} error={self._format_float(yaw_error, 1)} "
            f"tolerance={self.TASK2_STAGING_YAW_TOLERANCE_DEG:.1f} action={action}"
        )

        if action == "STOP" and yaw_error is not None and abs(yaw_error) <= self.TASK2_STAGING_YAW_TOLERANCE_DEG:
            self._task2_staging_yaw_aligned_count += 1
            self._log_event(
                f"task2_staging_yaw_aligned_sample "
                f"count={self._task2_staging_yaw_aligned_count}/{self.TASK2_STAGING_YAW_STABLE_COUNT} "
                f"current_yaw={self._format_float(current_yaw, 2)} "
                f"target_yaw={self.TASK2_STAGING_YAW_DEG:.2f} "
                f"error={self._format_float(yaw_error, 2)} "
                f"tolerance={self.TASK2_STAGING_YAW_TOLERANCE_DEG:.2f}"
            )

            if self._task2_staging_yaw_aligned_count < self.TASK2_STAGING_YAW_STABLE_COUNT:
                return "STOP"

            self._log_event("task2_staging_yaw_aligned_confirmed_starting_task2")
            self.task_type = TaskType.BRIDGE_RECOVERY
            self._holding_target = False
            self._bridge_bear_grabbed = False
            self._bridge_exit_stable_count = 0
            self._bridge_bear_close_precision_mode = False
            self._bridge_bear_close_jump_count = 0
            self._bridge_bear_close_precision_entered_at = None
            self._reset_active_path()
            if self.TASK2_SIMPLE_YOLO_BEAR_AFTER_STAGING:
                self.nav_processing.reset_nav_process()
                self._clear_stale_target_state()
                self._set_vision(self.DETECTION_MODE, self.BEAR_LABEL)
                self._log_event("task2_simple_bear_detection_enabled mode=detection label=bear")
                self._task2_start_bear_search_forward = True
                self._log_event(
                    "task2_start_bear_search_forward enabled=True "
                    f"action={self.TASK2_START_BEAR_SEARCH_FORWARD_ACTION}"
                )
                self._transition(TaskState.SEARCH_BEAR_ON_BRIDGE, "Task2 staging yaw aligned. Searching bridge bear.")
                return "STOP"
            self._transition(TaskState.SET_BRIDGE_VISION_TARGET, "Task2 staging yaw aligned. Enabling bridge segmentation.")
            return "STOP"

        self._task2_staging_yaw_aligned_count = 0
        return self._apply_obstacle_guard(action, None, "task2_staging_yaw")

    def _wait_task2_staging(self):
        if self._elapsed() < self.TASK2_STAGING_PAUSE_SECONDS:
            return "STOP"
        self._log_event("task2_staging_pause_complete")
        self._task2_staging_yaw_aligned_count = 0
        self._transition(TaskState.ALIGN_TASK2_STAGING_YAW, "Task2 staging pause complete. Aligning yaw.")
        return "STOP"

    def _set_bridge_vision_target(self):
        self.task_type = TaskType.BRIDGE_RECOVERY
        self._set_vision(self.SEGMENTATION_MODE, self.BRIDGE_LABEL)
        self._log_event("task2_bridge_vision_enabled mode=segmentation label=bridge")
        self._transition(TaskState.ALIGN_BRIDGE_ENTRY, "Bridge vision enabled. Aligning bridge entry.")
        return "STOP"

    def _align_bridge_entry(self):
        delta_x = self._get_bridge_entry_delta_x()
        if delta_x is None:
            action = "COUNTERCLOCKWISE_ROTATION_SLOW"
            self._log_event("bridge_entry_align delta_x=None action=COUNTERCLOCKWISE_ROTATION_SLOW")
            return self._apply_obstacle_guard(action, None, "bridge_entry_align")

        if abs(delta_x) <= self.BRIDGE_ENTRY_ALIGN_DELTA_X:
            self._log_event("bridge_entry_aligned")
            self._log_event("ascend_bridge_start")
            self._transition(TaskState.ASCEND_BRIDGE, "Bridge entry aligned. Ascending bridge.")
            return "STOP"

        if delta_x > 0.0:
            action = "CLOCKWISE_ROTATION_SLOW" if abs(delta_x) > self.BRIDGE_ENTRY_SLOW_DELTA_X else "CLOCKWISE_ROTATION_FINE"
        else:
            action = "COUNTERCLOCKWISE_ROTATION_SLOW" if abs(delta_x) > self.BRIDGE_ENTRY_SLOW_DELTA_X else "COUNTERCLOCKWISE_ROTATION_FINE"

        self._log_event(f"bridge_entry_align delta_x={delta_x:.1f} action={action}")
        return self._apply_obstacle_guard(action, None, "bridge_entry_align")

    def _reset_bridge_bear_probe_search(self):
        self._bridge_bear_probe_step_index = 0
        self._bridge_bear_probe_step_started_at = None
        self._bridge_bear_probe_cycle_started = False

    def _bridge_bear_probe_front_blocked(self):
        snapshot = self._get_obstacle_snapshot()
        front_depth = snapshot.get("front") if snapshot else None

        if (
            front_depth is not None
            and front_depth < self.BRIDGE_BEAR_PROBE_FRONT_BLOCK_DISTANCE
        ):
            self._log_event(
                f"bridge_bear_probe_abort_obstacle "
                f"front={front_depth:.3f} "
                f"threshold={self.BRIDGE_BEAR_PROBE_FRONT_BLOCK_DISTANCE:.3f}"
            )
            return True

        return False

    def _run_bridge_bear_probe_search(self):
        if not self.BRIDGE_BEAR_PROBE_SEARCH_ENABLED:
            return "STOP"

        pattern = self.BRIDGE_BEAR_PROBE_PATTERN
        if not pattern:
            return "STOP"

        now = time.monotonic()

        if self._bridge_bear_probe_step_started_at is None:
            self._bridge_bear_probe_step_index = 0
            self._bridge_bear_probe_step_started_at = now
            self._bridge_bear_probe_cycle_started = True
            self._log_event("bridge_bear_probe_search_start")

        action, duration, log_name = pattern[self._bridge_bear_probe_step_index]
        elapsed = now - self._bridge_bear_probe_step_started_at

        if elapsed >= float(duration):
            self._bridge_bear_probe_step_index = (
                self._bridge_bear_probe_step_index + 1
            ) % len(pattern)
            self._bridge_bear_probe_step_started_at = now
            action, duration, log_name = pattern[self._bridge_bear_probe_step_index]
            if self._bridge_bear_probe_step_index == 0:
                self._log_event("bridge_bear_probe_search_start")
            elif log_name != "bridge_bear_probe_search_start":
                self._log_event(log_name)

        if self._is_forward_action(action) and self._bridge_bear_probe_front_blocked():
            return "STOP"

        if self._is_movement_action(action):
            return self._apply_obstacle_guard(action, None, "bridge_bear_probe_search")

        return action

    def _search_bear_on_bridge(self):
        self._log_event("search_bear_on_bridge")
        if self._target_seen():
            if not self._bridge_bear_current_target_allowed("search_bear_on_bridge"):
                self._log_event("bridge_bear_seen_but_rejected_by_area_filter")
                return self._run_bridge_bear_probe_search()

            self._log_event("bridge_bear_probe_found_target")
            self._log_event("bridge_bear_found_inside_area")
            self._reset_bridge_bear_probe_search()
            self._task2_start_bear_search_forward = False
            self._lost_target_count = 0
            self._outlier_count = 0
            self._transition(
                TaskState.APPROACH_TARGET,
                "Bridge bear found inside valid bridge area. Approaching target."
            )
            return "STOP"

        return self._run_bridge_bear_probe_search()

    def _ascend_bridge(self):
        if self._elapsed() < self.ASCEND_SECONDS:
            self._log_event(f"bridge_ascend_action action={self.BRIDGE_ASCEND_ACTION}")
            return self._apply_obstacle_guard(self.BRIDGE_ASCEND_ACTION, None, "ascend_bridge")
        self._log_event("ascend_bridge_done")
        self._set_vision(self.DETECTION_MODE, self.BEAR_LABEL)
        self._log_event("bridge_bear_detection_enabled mode=detection label=bear")
        self._transition(TaskState.SEARCH_BEAR_ON_BRIDGE, "Ascent complete. Searching bear on bridge.")
        return "STOP"

    def _descend_bridge(self):
        current_position = self._get_current_position(require_map_frame=True)
        current_y = None
        if current_position is not None and len(current_position) >= 2:
            current_y = current_position[1]

        self._log_event(
            f"descend_bridge_check current={current_position} "
            f"current_y={self._format_float(current_y, 3)} "
            f"exit_y_threshold={self.BRIDGE_EXIT_Y_THRESHOLD:.3f} "
            f"stable_count={self._bridge_exit_stable_count}/{self.BRIDGE_EXIT_STABLE_COUNT} "
            f"holding_target={self._holding_target} "
            f"bridge_bear_grabbed={self._bridge_bear_grabbed}"
        )

        if not self._holding_target or not self._bridge_bear_grabbed:
            self._log_event(
                f"descend_bridge_no_grab_stop "
                f"holding_target={self._holding_target} "
                f"bridge_bear_grabbed={self._bridge_bear_grabbed}"
            )
            return "STOP"

        if current_y is not None and current_y >= self.BRIDGE_EXIT_Y_THRESHOLD:
            self._bridge_exit_stable_count += 1
        else:
            self._bridge_exit_stable_count = 0

        if self._bridge_exit_stable_count >= self.BRIDGE_EXIT_STABLE_COUNT:
            self._log_event(
                f"bridge_exit_reached_after_grab "
                f"current_y={current_y:.3f} "
                f"threshold={self.BRIDGE_EXIT_Y_THRESHOLD:.3f}"
            )
            self._bridge_exit_stable_count = 0
            self._reset_active_path()
            self._return_started_at = time.monotonic()
            self._log_event("return_home_after_bridge_exit")
            self._transition(
                TaskState.RETURN_HOME,
                "Exited bridge with grabbed bear. Returning home."
            )
            return "STOP"

        return self._apply_obstacle_guard("FORWARD_SLOW", None, "descend_bridge")

    def _get_bridge_entry_delta_x(self):
        try:
            bridge_info = self.nav_processing.data_processor.get_yolo_bridge_info()
        except Exception:
            bridge_info = None

        if bridge_info is not None and len(bridge_info) >= 3:
            try:
                if float(bridge_info[0]) == 1.0:
                    return float(bridge_info[2]) - float(self.BRIDGE_ENTRY_IMAGE_CENTER_X)
            except Exception:
                pass

        info = self._get_yolo_target_info()
        if info is not None and len(info) >= 3:
            try:
                if float(info[0]) == 1.0:
                    return float(info[2])
            except Exception:
                pass

        return None

    def _should_return_home_stuck_release(self, distance_to_home):
        if distance_to_home is None:
            self._return_home_near_distance = None
            self._return_home_near_since = None
            return False

        if distance_to_home > self.RETURN_HOME_STUCK_RELEASE_DISTANCE:
            self._return_home_near_distance = None
            self._return_home_near_since = None
            return False

        camera_front = None
        try:
            camera_snapshot = self._get_camera_obstacle_snapshot()
            if camera_snapshot is not None:
                camera_front = camera_snapshot.get("front")
        except Exception:
            camera_front = None

        if camera_front is not None and camera_front <= 0.45:
            self._log_event(
                f"return_home_near_home_camera_blocked distance_to_home={distance_to_home:.3f} camera_front={camera_front:.3f}"
            )
            return True

        now = time.monotonic()
        if (
            self._return_home_near_distance is None
            or distance_to_home < self._return_home_near_distance - self.RETURN_HOME_NEAR_HOME_PROGRESS_EPS
        ):
            self._return_home_near_distance = distance_to_home
            self._return_home_near_since = now
            return False

        if self._return_home_near_since is None:
            self._return_home_near_since = now
            return False

        return now - self._return_home_near_since >= self.RETURN_HOME_NEAR_HOME_NO_PROGRESS_SECONDS

    def _explore_map(self):
        if self._target_seen():
            if self.task_type == TaskType.BRIDGE_RECOVERY and self._target_label == self.BRIDGE_LABEL:
                if self.frontier_explorer:
                    self.frontier_explorer.reset()

                self.nav_processing.reset_nav_process()
                if self._bridge_pre_entry_configured():
                    self._transition(
                        TaskState.NAV_TO_BRIDGE_PRE_ENTRY,
                        "Bridge seen during exploration. Navigating to bridge pre-entry first.",
                    )
                else:
                    self._transition(TaskState.ASCEND_BRIDGE, "Bridge found during exploration. Driving up bridge slowly.")
                return "STOP"

            if self.task_type == TaskType.BEAR_RECOVERY and self.BEAR_GLOBAL_NAV_ENABLED:
                bear_position = self._update_stable_bear_map_position()
                if bear_position is not None:
                    if self.frontier_explorer:
                        self.frontier_explorer.reset()

                    self.nav_processing.reset_nav_process()
                    self._transition(
                        TaskState.PLAN_TO_BEAR,
                        f"Bear seen during exploration with stable map position {bear_position}. Planning A* path.",
                    )
                    return "STOP"

                self.status_message = "Bear seen; waiting for stable /map position before A*."
                self._debug("bear seen during exploration but map position is not stable yet", 0.5)
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
            if self._bridge_pre_entry_configured():
                self._transition(
                    TaskState.NAV_TO_BRIDGE_PRE_ENTRY,
                    "Bridge found. Navigating to bridge pre-entry first.",
                )
            else:
                self._transition(TaskState.ASCEND_BRIDGE, "Bridge found. Driving up bridge slowly.")
            return "STOP"

        return self._apply_obstacle_guard("CLOCKWISE_ROTATION_SLOW", None, "search_bridge")

    def _plan_to_bear(self):
        bear_position = self._stable_bear_map_position or self._update_stable_bear_map_position()
        if bear_position is None:
            self._transition(TaskState.EXPLORE_MAP, "Bear map position lost before planning. Searching again.")
            return "STOP"

        goal = self._bear_pre_grab_goal(bear_position)
        if goal is None:
            self._transition(TaskState.EXPLORE_MAP, "Could not compute bear pre-grab goal. Searching again.")
            return "STOP"

        if not self._plan_astar_path(goal, "bear"):
            self._reset_active_path()
            self.nav_processing.reset_nav_process()
            self._log_event("astar_plan failed context=bear fallback_to_yolo_approach")
            self._transition(
                TaskState.APPROACH_TARGET,
                "A* failed because start is inside inflated obstacle. Falling back to YOLO approach.",
            )
            return "STOP"

        self._transition(TaskState.NAV_TO_BEAR, f"Following A* path to bear pre-grab goal {goal}.")
        return "STOP"

    def _nav_to_bear(self):
        bear_position = self._update_stable_bear_map_position() or self._stable_bear_map_position
        if bear_position is not None and self._should_replan_astar("bear"):
            goal = self._bear_pre_grab_goal(bear_position)
            if goal is not None:
                self._plan_astar_path(goal, "bear")

        current_position = self._get_current_position(require_map_frame=True)
        if current_position and bear_position and self._distance(current_position, bear_position) <= self.BEAR_FINAL_APPROACH_DISTANCE:
            self._reset_active_path()
            self.nav_processing.reset_nav_process()
            self._transition(TaskState.APPROACH_TARGET, "Near bear. Switching from A* to YOLO final approach.")
            return "STOP"

        action = self._follow_active_path_action(
            reached_distance=self.BEAR_PRE_GRAB_REACHED_DISTANCE,
            context="nav_to_bear",
            done_state=TaskState.APPROACH_TARGET,
            done_message="Reached bear pre-grab waypoint. Switching to YOLO final approach.",
        )
        return action

    def _plan_to_door_staging(self):
        if self.DOOR_STAGING_POSITION is None:
            self._log_event("door_staging_not_configured")
            self._transition(TaskState.FAILED, "Door staging is not configured for direct door opening.")
            return "STOP"

        self._reset_active_path()
        self._door_staging_position_aligned_count = 0
        goal = list(self.DOOR_STAGING_POSITION)
        self._log_event(f"door_staging_plan_start goal={goal}")
        if self._plan_astar_path(goal, "door_staging"):
            path_len = len(self._active_path) if self._active_path else 0
            self._log_event(f"door_staging_plan_success path_len={path_len}")
            self._transition(TaskState.NAV_TO_DOOR_STAGING, "Following path to direct door opening pose.")
            return "STOP"

        self._log_event("door_staging_plan_failed fallback=direct")
        self._transition(TaskState.NAV_TO_DOOR_STAGING, "Door staging plan failed. Driving directly to door opening pose.")
        return "STOP"

    def _door_staging_pose_error(self, current_position=None):
        if current_position is None:
            current_position = self._get_current_position(require_map_frame=True)

        if current_position is None or len(current_position) < 2:
            return None, None, None

        goal = self.DOOR_STAGING_POSITION
        dx = float(goal[0]) - float(current_position[0])
        dy = float(goal[1]) - float(current_position[1])

        yaw = math.radians(float(self.DOOR_STAGING_YAW_DEG))
        forward_error = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral_error = -math.sin(yaw) * dx + math.cos(yaw) * dy
        distance = math.hypot(dx, dy)

        return distance, forward_error, lateral_error

    def _door_staging_position_aligned(self, current_position=None):
        distance, forward_error, lateral_error = self._door_staging_pose_error(current_position)

        if distance is None:
            return False, distance, forward_error, lateral_error

        aligned = (
            abs(lateral_error) <= float(self.DOOR_STAGING_LATERAL_TOLERANCE_M)
            and abs(forward_error) <= float(self.DOOR_STAGING_FORWARD_TOLERANCE_M)
            and distance <= float(self.DOOR_STAGING_REACHED_DISTANCE_M)
        )

        self._log_event(
            f"door_staging_axis_error_strict "
            f"distance={self._format_float(distance, 3)} "
            f"distance_tol={self.DOOR_STAGING_REACHED_DISTANCE_M:.3f} "
            f"forward_error={self._format_float(forward_error, 3)} "
            f"forward_tol={self.DOOR_STAGING_FORWARD_TOLERANCE_M:.3f} "
            f"lateral_error={self._format_float(lateral_error, 3)} "
            f"lateral_tol={self.DOOR_STAGING_LATERAL_TOLERANCE_M:.3f} "
            f"aligned={aligned}"
        )

        return aligned, distance, forward_error, lateral_error

    def _door_staging_final_creep_action(self, current_position=None):
        aligned, distance, forward_error, lateral_error = self._door_staging_position_aligned(current_position)

        if aligned:
            self._log_event(
                f"door_staging_final_creep_aligned_strict "
                f"distance={self._format_float(distance, 3)} "
                f"forward_error={self._format_float(forward_error, 3)} "
                f"lateral_error={self._format_float(lateral_error, 3)}"
            )
            return "STOP"

        if distance is None:
            return "STOP"

        if distance <= self.DOOR_STAGING_FINAL_FORWARD_ONLY_DISTANCE_M:
            if lateral_error is not None and abs(lateral_error) > self.DOOR_STAGING_LATERAL_TOLERANCE_M:
                action = (
                    "CLOCKWISE_ROTATION_FINE"
                    if lateral_error > 0.0
                    else "COUNTERCLOCKWISE_ROTATION_FINE"
                )
            elif forward_error is not None and abs(forward_error) > self.DOOR_STAGING_FORWARD_TOLERANCE_M:
                action = self.DOOR_STAGING_FINAL_CREEP_ACTION
            else:
                action = "STOP"

            self._log_event(
                f"door_staging_final_creep_close_strict "
                f"distance={self._format_float(distance, 3)} "
                f"forward_error={self._format_float(forward_error, 3)} "
                f"lateral_error={self._format_float(lateral_error, 3)} "
                f"action={action}"
            )
            return self._apply_obstacle_guard(action, None, "door_staging_direct")

        raw_action, pose_position, distance2, angle_error = self._fixed_waypoint_action(
            self.DOOR_STAGING_POSITION,
            reached_distance=0.0,
            context="door_staging_direct",
        )

        if raw_action in ("FORWARD", "FORWARD_SLOW"):
            action = self.DOOR_STAGING_FINAL_CREEP_ACTION
        elif raw_action in ("CLOCKWISE_ROTATION_SLOW", "CLOCKWISE_ROTATION_MEDIAN"):
            action = "CLOCKWISE_ROTATION_FINE"
        elif raw_action in ("COUNTERCLOCKWISE_ROTATION_SLOW", "COUNTERCLOCKWISE_ROTATION_MEDIAN"):
            action = "COUNTERCLOCKWISE_ROTATION_FINE"
        else:
            action = raw_action

        self._log_event(
            f"door_staging_final_creep_strict "
            f"distance={self._format_float(distance, 3)} "
            f"angle_error={self._format_float(angle_error, 1)} "
            f"forward_error={self._format_float(forward_error, 3)} "
            f"lateral_error={self._format_float(lateral_error, 3)} "
            f"raw_action={raw_action} "
            f"action={action}"
        )

        return self._apply_obstacle_guard(action, None, "door_staging_direct")

    def _nav_to_door_staging(self):
        if self.DOOR_STAGING_POSITION is None:
            self._transition(TaskState.FAILED, "Door staging is not configured for direct door opening.")
            return "STOP"

        current_position = self._get_current_position(require_map_frame=True)
        distance = self._distance(current_position, self.DOOR_STAGING_POSITION) if current_position else None

        if (
            self.TASK3_USE_KNOB_YOLO_BEFORE_PRE_PRESS
            and distance is not None
            and distance <= self.TASK3_DOOR_NEAR_REACHED_DISTANCE_M
        ):
            self._reset_active_path()
            self.nav_processing.reset_nav_process()
            self._door_staging_yaw_aligned_count = 0
            self._log_event(
                f"task3_door_near_reached_start_yaw_or_knob_yolo "
                f"current={current_position} "
                f"target={self.DOOR_STAGING_POSITION} "
                f"distance={distance:.3f} "
                f"near_threshold={self.TASK3_DOOR_NEAR_REACHED_DISTANCE_M:.3f}"
            )
            self._transition(
                TaskState.ALIGN_DOOR_STAGING_YAW,
                "Reached near door. Rough-aligning yaw before knob YOLO.",
            )
            return "STOP"

        aligned, axis_distance, forward_error, lateral_error = self._door_staging_position_aligned(current_position)

        if aligned:
            self._door_staging_position_aligned_count += 1
            self._log_event(
                f"door_staging_position_aligned_sample_strict "
                f"count={self._door_staging_position_aligned_count}/{self.DOOR_STAGING_POSITION_STABLE_COUNT} "
                f"distance={self._format_float(axis_distance, 3)} "
                f"forward_error={self._format_float(forward_error, 3)} "
                f"lateral_error={self._format_float(lateral_error, 3)}"
            )

            if self._door_staging_position_aligned_count < self.DOOR_STAGING_POSITION_STABLE_COUNT:
                return "STOP"

            self._reset_active_path()
            self.nav_processing.reset_nav_process()
            self._door_staging_yaw_aligned_count = 0
            self._log_event(
                f"door_staging_exact_position_confirmed_strict "
                f"current={current_position} "
                f"target={self.DOOR_STAGING_POSITION} "
                f"distance={self._format_float(axis_distance, 3)} "
                f"forward_error={self._format_float(forward_error, 3)} "
                f"lateral_error={self._format_float(lateral_error, 3)}"
            )
            self._transition(
                TaskState.ALIGN_DOOR_STAGING_YAW,
                "Reached door staging position by axis error. Aligning yaw before direct door press."
            )
            return "STOP"

        self._door_staging_position_aligned_count = 0

        if distance is None or distance > self.DOOR_STAGING_REACHED_DISTANCE_M:
            self._log_event(
                f"door_staging_never_align_too_far "
                f"current={current_position} "
                f"target={self.DOOR_STAGING_POSITION} "
                f"distance={self._format_float(distance, 3)} "
                f"threshold={self.DOOR_STAGING_REACHED_DISTANCE_M:.3f}"
            )
            if distance is not None and distance <= self.DOOR_STAGING_DIRECT_CREEP_ENTER_DISTANCE_M:
                return self._door_staging_final_creep_action(current_position)

        if self._active_path and self._active_path_context == "door_staging":
            return self._follow_active_path_action(
                reached_distance=self.DOOR_STAGING_REACHED_DISTANCE_M,
                context="door_staging",
            )

        return self._navigate_to_waypoint(
            self.DOOR_STAGING_POSITION,
            reached_distance=self.DOOR_STAGING_REACHED_DISTANCE_M,
            context="door_staging_direct",
        )

    def _align_door_staging_yaw(self):
        current_position = self._get_current_position(require_map_frame=True)
        aligned, axis_distance, forward_error, lateral_error = self._door_staging_position_aligned(current_position)
        distance = axis_distance

        self._log_event(
            f"door_staging_yaw_distance_check "
            f"current={current_position} "
            f"target={self.DOOR_STAGING_POSITION} "
            f"distance={self._format_float(distance, 3)} "
            f"forward_error={self._format_float(forward_error, 3)} "
            f"lateral_error={self._format_float(lateral_error, 3)} "
            f"drift_threshold={self.DOOR_STAGING_YAW_DRIFT_DISTANCE_M:.3f}"
        )

        if distance is None:
            return "STOP"

        if self.TASK3_USE_KNOB_YOLO_BEFORE_PRE_PRESS and not self._door_pre_press_published:
            action, yaw_error, current_yaw = self._align_to_map_yaw_action(
                self.DOOR_STAGING_YAW_DEG,
                self.TASK3_DOOR_NEAR_YAW_TOLERANCE_DEG,
                context="task3_door_near_yaw",
            )

            self._log_event(
                f"task3_near_door_yaw_align "
                f"current_yaw={self._format_float(current_yaw, 2)} "
                f"target_yaw={self.DOOR_STAGING_YAW_DEG:.2f} "
                f"error={self._format_float(yaw_error, 2)} "
                f"tolerance={self.TASK3_DOOR_NEAR_YAW_TOLERANCE_DEG:.2f} "
                f"action={action}"
            )

            if action == "STOP" and yaw_error is not None and abs(yaw_error) <= self.TASK3_DOOR_NEAR_YAW_TOLERANCE_DEG:
                self._log_event("task3_near_door_yaw_ready_start_knob_yolo")
                self._transition(
                    TaskState.ALIGN_KNOB_YOLO,
                    "Near door yaw aligned. Enabling knob YOLO alignment.",
                )
                return "STOP"

            return self._apply_obstacle_guard(action, None, "task3_door_near_yaw")

        if not aligned and axis_distance is not None and axis_distance > self.DOOR_STAGING_YAW_DRIFT_DISTANCE_M:
            self._log_event(
                f"door_staging_yaw_drifted_return_nav "
                f"distance={self._format_float(axis_distance, 3)} "
                f"forward_error={self._format_float(forward_error, 3)} "
                f"lateral_error={self._format_float(lateral_error, 3)} "
                f"threshold={self.DOOR_STAGING_YAW_DRIFT_DISTANCE_M:.3f}"
            )
            self._transition(
                TaskState.NAV_TO_DOOR_STAGING,
                "Door staging position drifted too far while aligning yaw. Returning to staging.",
            )
            return "STOP"

        action, yaw_error, current_yaw = self._align_to_map_yaw_action(
            self.DOOR_STAGING_YAW_DEG,
            self.DOOR_STAGING_YAW_TOLERANCE_DEG,
            context="door_staging_yaw",
        )

        if (
            action != "STOP"
            and yaw_error is not None
            and abs(yaw_error) <= self.DOOR_STAGING_YAW_FINE_THRESHOLD_DEG
        ):
            action = (
                "COUNTERCLOCKWISE_ROTATION_FINE"
                if yaw_error > 0.0
                else "CLOCKWISE_ROTATION_FINE"
            )
            self._log_event(
                f"door_staging_yaw_fine_adjust "
                f"current_yaw={self._format_float(current_yaw, 2)} "
                f"target_yaw={self.DOOR_STAGING_YAW_DEG:.2f} "
                f"error={self._format_float(yaw_error, 2)} "
                f"action={action}"
            )

        self._log_event(
            f"door_staging_yaw_align "
            f"current_yaw={self._format_float(current_yaw, 1)} "
            f"target_yaw={self.DOOR_STAGING_YAW_DEG:.1f} "
            f"error={self._format_float(yaw_error, 1)} "
            f"tolerance={self.DOOR_STAGING_YAW_TOLERANCE_DEG:.1f} "
            f"action={action}"
        )

        if action == "STOP" and yaw_error is not None and abs(yaw_error) <= self.DOOR_STAGING_YAW_TOLERANCE_DEG:
            self._door_staging_yaw_aligned_count += 1
            self._log_event(
                f"door_staging_yaw_aligned_sample "
                f"count={self._door_staging_yaw_aligned_count}/{self.DOOR_STAGING_YAW_STABLE_COUNT} "
                f"current_yaw={self._format_float(current_yaw, 2)} "
                f"target_yaw={self.DOOR_STAGING_YAW_DEG:.2f} "
                f"error={self._format_float(yaw_error, 2)} "
                f"tolerance={self.DOOR_STAGING_YAW_TOLERANCE_DEG:.2f}"
            )

            if self._door_staging_yaw_aligned_count < self.DOOR_STAGING_YAW_STABLE_COUNT:
                return "STOP"

            exact_press_distance = float(getattr(
                self,
                "DOOR_STAGING_EXACT_PRESS_DISTANCE_M",
                self.DOOR_STAGING_REACHED_DISTANCE_M,
            ))

            aligned, axis_distance, forward_error, lateral_error = self._door_staging_position_aligned(current_position)

            if (
                not aligned
                or axis_distance is None
                or axis_distance > exact_press_distance
            ):
                self._door_staging_yaw_aligned_count = 0
                self._log_event(
                    f"door_staging_press_blocked_strict_position_error "
                    f"distance={self._format_float(axis_distance, 3)} "
                    f"exact_press_distance={exact_press_distance:.3f} "
                    f"forward_error={self._format_float(forward_error, 3)} "
                    f"lateral_error={self._format_float(lateral_error, 3)}"
                )
                self._transition(
                    TaskState.NAV_TO_DOOR_STAGING,
                    "Door press blocked because strict position is not exact enough. Re-approaching."
                )
                return "STOP"

            self._knob_map_positions = []
            self._stable_knob_map_position = None
            self._clear_stale_target_state()
            self._log_event(
                f"door_staging_press_position_confirmed "
                f"current={current_position} "
                f"target={self.DOOR_STAGING_POSITION} "
                f"distance={axis_distance:.3f} "
                f"yaw_error={self._format_float(yaw_error, 2)}"
            )
            self._log_event("door_staging_yaw_aligned_confirmed_direct_press")

            self._transition(
                TaskState.PRESS_DOOR_KNOB,
                "Door staging yaw aligned strictly. Pressing door knob directly without YOLO.",
            )
            return "STOP"

        self._door_staging_yaw_aligned_count = 0
        return self._apply_obstacle_guard(action, None, "door_staging_yaw")

    def _plan_to_knob(self):
        if self._vision_mode != self.DETECTION_MODE or self._target_label != self.KNOB_LABEL:
            self._set_vision(self.DETECTION_MODE, self.KNOB_LABEL)

        knob_position = self._update_stable_knob_map_position() or self._stable_knob_map_position
        if knob_position is None:
            if self.DOOR_STAGING_POSITION is not None:
                self._transition(TaskState.PLAN_TO_DOOR_STAGING, "Knob map position lost. Returning to door staging plan.")
            else:
                self._transition(TaskState.SEARCH_TARGET, "Knob map position lost. Searching knob.")
            return "STOP"

        goal = self._knob_pre_press_goal(knob_position)
        if goal is None:
            self._transition(TaskState.APPROACH_TARGET, "Already near knob. Switching to YOLO final approach.")
            return "STOP"

        self._log_event(f"knob_plan_start knob_position={knob_position} goal={goal}")
        if self._plan_astar_path(goal, "knob"):
            path_len = len(self._active_path) if self._active_path else 0
            self._log_event(f"knob_plan_success path_len={path_len} goal={goal}")
            self._transition(TaskState.NAV_TO_KNOB, "Following A* path to knob pre-press waypoint.")
            return "STOP"

        self._log_event("knob_plan_failed_fallback_to_visual")
        self._reset_active_path()
        self.nav_processing.reset_nav_process()
        self._transition(TaskState.APPROACH_TARGET, "Knob A* failed. Falling back to YOLO final approach.")
        return "STOP"

    def _nav_to_knob(self):
        if self._vision_mode != self.DETECTION_MODE or self._target_label != self.KNOB_LABEL:
            self._set_vision(self.DETECTION_MODE, self.KNOB_LABEL)

        knob_position = self._update_stable_knob_map_position() or self._stable_knob_map_position
        if knob_position is None:
            self._transition(TaskState.PLAN_TO_KNOB, "Knob map position lost during navigation. Replanning.")
            return "STOP"

        if self._should_replan_astar("knob"):
            goal = self._knob_pre_press_goal(knob_position)
            if goal is not None:
                self._plan_astar_path(goal, "knob")

        current_position = self._get_current_position(require_map_frame=True)
        if current_position and self._distance(current_position, knob_position) <= self.KNOB_FINAL_APPROACH_DISTANCE:
            self._reset_active_path()
            self.nav_processing.reset_nav_process()
            self._log_event(f"knob_final_approach current={current_position} knob={knob_position}")
            self._transition(TaskState.APPROACH_TARGET, "Near knob. Switching to YOLO final approach.")
            return "STOP"

        self._log_event(f"knob_nav_to_pre_press current={current_position} knob={knob_position}")
        if not self._active_path:
            self._transition(TaskState.PLAN_TO_KNOB, "No active knob path. Replanning.")
            return "STOP"

        return self._follow_active_path_action(
            reached_distance=self.KNOB_PRE_PRESS_REACHED_DISTANCE,
            context="knob",
            done_state=TaskState.APPROACH_TARGET,
            done_message="Reached knob pre-press waypoint. Switching to YOLO final approach.",
        )

    def _nav_to_bridge_pre_entry(self):
        if not self._bridge_pre_entry_configured():
            self._transition(
                TaskState.EXPLORE_MAP,
                "Bridge pre-entry waypoint is not configured; falling back to bridge search.",
            )
            return "STOP"

        action = self._navigate_to_waypoint(
            self.BRIDGE_PRE_ENTRY,
            reached_distance=self.FIXED_WAYPOINT_REACHED_DISTANCE,
            context="bridge_pre_entry",
        )

        current_position = self._get_current_position()
        distance = self._distance(current_position, self.BRIDGE_PRE_ENTRY) if current_position else None
        if current_position and distance is not None and distance < self.FIXED_WAYPOINT_REACHED_DISTANCE:
            self.nav_processing.reset_nav_process()
            self._transition(
                TaskState.SEARCH_BRIDGE,
                "Reached bridge pre-entry. Stopping global waypoint navigation and using bridge segmentation.",
            )
            return "STOP"

        self.status_message = (
            f"Navigating to bridge pre-entry {self.BRIDGE_PRE_ENTRY}; "
            f"distance={self._format_float(distance, 2)} action={action}"
        )
        return action

    def _unlock_door(self):
        self._transition(TaskState.PRESS_DOOR_KNOB, "Unlock door requested. Pressing knob down.")
        return "STOP"

    def _enable_knob_yolo_for_task3(self):
        self._set_vision(self.DETECTION_MODE, self.KNOB_LABEL)
        self._task3_knob_yolo_align_count = 0
        self._task3_knob_yolo_aligned_info = None
        self._task3_knob_yolo_enabled = True
        self._task3_knob_yolo_disabled = False
        self._log_event("task3_knob_yolo_enabled_before_pre_press")
        return True

    def _disable_knob_yolo_for_task3(self, reason=""):
        self._clear_stale_target_state()
        self.nav_processing.reset_nav_process()

        try:
            if hasattr(self.ros_communicator, "publish_target_label"):
                self.ros_communicator.publish_target_label("")
        except Exception as exc:
            self._log_event(f"task3_knob_yolo_disable_publish_label_failed error={exc}")

        self._task3_knob_yolo_disabled = True
        self._task3_knob_yolo_enabled = False
        self._log_event(f"task3_knob_yolo_disabled reason={reason}")
        return True

    def _align_knob_yolo(self):
        if self._vision_mode != self.DETECTION_MODE or self._target_label != self.KNOB_LABEL:
            self._enable_knob_yolo_for_task3()
            return "STOP"

        info = self._get_yolo_target_info()

        if not info or len(info) < 3 or info[0] != 1:
            self._task3_knob_yolo_align_count = 0
            self._log_event(f"task3_knob_yolo_lost info={info}")
            return self.TASK3_KNOB_ALIGN_SEARCH_ACTION

        distance = self._safe_float(info[1])
        delta_x = self._safe_float(info[2])

        if distance is None or delta_x is None:
            self._task3_knob_yolo_align_count = 0
            self._log_event(f"task3_knob_yolo_invalid info={info}")
            return "STOP"

        x_aligned = abs(delta_x) <= self.TASK3_KNOB_ALIGN_DELTA_X

        target_distance = getattr(self, "TASK3_KNOB_TARGET_DISTANCE_M", None)
        distance_aligned = True
        if target_distance is not None:
            distance_aligned = (
                abs(distance - float(target_distance))
                <= self.TASK3_KNOB_DISTANCE_TOLERANCE_M
            )

        if not x_aligned:
            self._task3_knob_yolo_align_count = 0
            action = self._alignment_action_from_delta_x(
                delta_x,
                self.TASK3_KNOB_ALIGN_DELTA_X,
            )
            self._log_event(
                f"task3_knob_yolo_align_x "
                f"distance={distance:.3f} "
                f"delta_x={delta_x:.1f} "
                f"x_threshold={self.TASK3_KNOB_ALIGN_DELTA_X:.1f} "
                f"action={action}"
            )
            return self._apply_obstacle_guard(action, info, "task3_knob_yolo_align_x")

        if target_distance is not None and not distance_aligned:
            self._task3_knob_yolo_align_count = 0

            if distance > float(target_distance) + self.TASK3_KNOB_DISTANCE_TOLERANCE_M:
                action = self.TASK3_KNOB_ALIGN_FORWARD_ACTION
            else:
                action = self.TASK3_KNOB_ALIGN_BACKWARD_ACTION

            self._log_event(
                f"task3_knob_yolo_align_distance "
                f"distance={distance:.3f} "
                f"target_distance={float(target_distance):.3f} "
                f"distance_tolerance={self.TASK3_KNOB_DISTANCE_TOLERANCE_M:.3f} "
                f"delta_x={delta_x:.1f} "
                f"action={action}"
            )
            return self._apply_obstacle_guard(action, info, "task3_knob_yolo_align_distance")

        self._task3_knob_yolo_align_count += 1
        self._task3_knob_yolo_aligned_info = list(info)

        self._log_event(
            f"task3_knob_yolo_aligned_sample "
            f"count={self._task3_knob_yolo_align_count}/{self.TASK3_KNOB_ALIGN_STABLE_COUNT} "
            f"distance={distance:.3f} "
            f"delta_x={delta_x:.1f} "
            f"x_threshold={self.TASK3_KNOB_ALIGN_DELTA_X:.1f} "
            f"target_distance={target_distance}"
        )

        if self._task3_knob_yolo_align_count < self.TASK3_KNOB_ALIGN_STABLE_COUNT:
            return "STOP"

        self._log_event(
            f"task3_knob_yolo_alignment_confirmed "
            f"distance={distance:.3f} "
            f"delta_x={delta_x:.1f}"
        )

        if self.TASK3_DISABLE_KNOB_YOLO_AFTER_ALIGN:
            self._disable_knob_yolo_for_task3("knob_yolo_alignment_confirmed")

        self._transition(
            TaskState.PREPARE_DOOR_PRE_PRESS,
            "Knob YOLO aligned. Disabled YOLO. Moving arm to pre-press pose.",
        )
        return "STOP"

    def _prepare_door_pre_press(self):
        if not self._door_pre_press_published:
            ok = self._publish_door_knob_raw_radian_pose(
                self.DOOR_KNOB_PRE_PRESS_POSE_RAD,
                reason="task3_pre_press_after_knob_yolo",
            )
            if not ok:
                self._transition(TaskState.FAILED, "Task3 pre-press pose after knob YOLO failed.")
                return "STOP"

            self._door_pre_press_published = True
            self._door_pre_press_started_at = time.monotonic()
            self._log_event(
                f"task3_pre_press_after_knob_yolo published positions={self.DOOR_KNOB_PRE_PRESS_POSE_RAD}"
            )
            return "STOP"

        if self._door_pre_press_started_at is None:
            self._door_pre_press_started_at = time.monotonic()
            return "STOP"

        elapsed = time.monotonic() - self._door_pre_press_started_at
        if elapsed < self.DOOR_PRE_PRESS_BEFORE_NAV_SECONDS:
            self._debug(
                f"waiting task3 pre_press after knob yolo elapsed={elapsed:.2f}",
                0.5,
            )
            return "STOP"

        self._transition(
            TaskState.DRIVE_TO_KNOB_CONTACT,
            "Task3 pre-press pose ready. Driving forward slightly to contact knob.",
        )
        return "STOP"

    def _drive_to_knob_contact(self):
        if self._drive_to_knob_contact_started_at is None:
            self._drive_to_knob_contact_started_at = time.monotonic()
            self._drive_to_knob_contact_start_position = self._get_current_position(require_map_frame=False)
            self._log_event(
                f"task3_drive_to_knob_contact_start "
                f"start_position={self._drive_to_knob_contact_start_position} "
                f"target_distance={self.TASK3_DRIVE_TO_KNOB_CONTACT_DISTANCE_M:.3f} "
                f"max_seconds={self.TASK3_DRIVE_TO_KNOB_CONTACT_MAX_SECONDS:.2f} "
                f"action={self.TASK3_DRIVE_TO_KNOB_CONTACT_ACTION}"
            )

        elapsed = time.monotonic() - self._drive_to_knob_contact_started_at
        current_position = self._get_current_position(require_map_frame=False)
        moved_distance = None

        if current_position is not None and self._drive_to_knob_contact_start_position is not None:
            moved_distance = self._distance(
                current_position,
                self._drive_to_knob_contact_start_position,
            )

        if (
            elapsed < self.TASK3_DRIVE_TO_KNOB_CONTACT_MAX_SECONDS
            and (
                moved_distance is None
                or moved_distance < self.TASK3_DRIVE_TO_KNOB_CONTACT_DISTANCE_M
            )
        ):
            self._log_event(
                f"task3_drive_to_knob_contact_forward "
                f"action={self.TASK3_DRIVE_TO_KNOB_CONTACT_ACTION} "
                f"elapsed={elapsed:.2f} "
                f"moved_distance={self._format_float(moved_distance, 3)} "
                f"target_distance={self.TASK3_DRIVE_TO_KNOB_CONTACT_DISTANCE_M:.3f}"
            )
            return self.TASK3_DRIVE_TO_KNOB_CONTACT_ACTION

        self.car_controller.update_action("STOP")
        self._log_event(
            f"task3_drive_to_knob_contact_done "
            f"elapsed={elapsed:.2f} "
            f"moved_distance={self._format_float(moved_distance, 3)}"
        )

        self._transition(
            TaskState.PRESS_DOOR_KNOB,
            "Reached knob contact offset. Pressing door knob down.",
        )
        return "STOP"

    def _publish_door_knob_raw_radian_pose(self, positions, reason=""):
        if positions is None or len(positions) < 3:
            self._log_event(f"door_knob_raw_pose_invalid reason={reason} positions={positions}")
            return False

        try:
            msg = JointTrajectoryPoint()
            msg.positions = [float(value) for value in positions]
            msg.velocities = [0.0] * len(msg.positions)

            publisher = getattr(self.ros_communicator, "publisher_joint_trajectory", None)
            if publisher is not None:
                publisher.publish(msg)
            elif hasattr(self.ros_communicator, "publish_robot_arm_angle"):
                self.ros_communicator.publish_robot_arm_angle(msg.positions)
            else:
                self._log_event(
                    f"door_knob_raw_pose_publish_failed reason={reason} error=no /robot_arm publisher"
                )
                return False

            self._log_event(
                f"door_knob_raw_pose_publish reason={reason} "
                f"positions={[round(float(v), 4) for v in msg.positions]}"
            )
            return True

        except Exception as exc:
            self._log_event(f"door_knob_raw_pose_publish_failed reason={reason} error={exc}")
            return False

    def _press_door_knob(self):
        if not self._door_press_started:
            self._door_press_started = True
            self._door_press_started_at = time.monotonic()
            self._log_event("door_knob_press_down_only_start")

            ok = self._publish_door_knob_raw_radian_pose(
                self.DOOR_KNOB_PRESS_DOWN_POSE_RAD,
                reason="door_knob_press_down_only_at_staging",
            )
            if not ok:
                self._door_press_started = False
                self._door_press_started_at = None
                self._transition(TaskState.FAILED, "Door knob press-down raw pose failed.")
                return "STOP"

            self._door_press_done = True
            return "STOP"

        elapsed = 0.0
        if self._door_press_started_at is not None:
            elapsed = time.monotonic() - self._door_press_started_at

        if elapsed < self.DOOR_KNOB_PRESS_DOWN_SECONDS:
            self._debug(
                f"holding door knob press_down before push elapsed={elapsed:.2f}",
                0.5,
            )
            return "STOP"

        self._log_event("door_knob_press_down_done_no_more_arm_motion")
        self._transition(
            TaskState.PUSH_DOOR,
            "Door knob pressed down. Arm will stay still. Pushing door forward.",
        )
        return "STOP"

    def _sync_door_knob_arm_config(self):
        arm = self.arm_controller
        for name in (
            "DOOR_KNOB_PRE_PRESS_POSE",
            "DOOR_KNOB_PRESS_DOWN_POSE",
            "DOOR_KNOB_RELEASE_POSE",
            "DOOR_KNOB_PRESS_DOWN_SECONDS",
        ):
            value = getattr(self, name, None)
            if value is not None:
                setattr(arm, name, value)

    def _press_door_knob_down_fallback(self):
        if self.DOOR_KNOB_PRE_PRESS_POSE is None or self.DOOR_KNOB_PRESS_DOWN_POSE is None:
            self._log_event("door_knob_press_pose_not_configured")
            return False

        arm = self.arm_controller
        if not hasattr(arm, "_smooth_move_to"):
            self._log_event("door_knob_press_down_failed no compatible arm API")
            return False

        try:
            self._log_event(f"door_knob_pre_press_pose {self.DOOR_KNOB_PRE_PRESS_POSE}")
            arm._smooth_move_to(self.DOOR_KNOB_PRE_PRESS_POSE, step=3.0, delay=0.1)
            time.sleep(0.2)

            self._log_event(f"door_knob_press_down_pose {self.DOOR_KNOB_PRESS_DOWN_POSE}")
            arm._smooth_move_to(self.DOOR_KNOB_PRESS_DOWN_POSE, step=2.0, delay=0.1)
            time.sleep(self.DOOR_KNOB_PRESS_DOWN_SECONDS)
            return True
        except Exception as exc:
            self._log_event(f"door_knob_press_down_failed error={exc}")
            return False

    def _push_door(self):
        if self._door_push_started_at is None:
            self._door_push_started_at = time.monotonic()
            self._door_push_start_position = self._get_current_position(require_map_frame=False)
            self._log_event(
                f"door_push_right_forward_start_no_obstacle_guard "
                f"start_position={self._door_push_start_position} "
                f"max_seconds={self.DOOR_PUSH_FORWARD_MAX_SECONDS:.2f} "
                f"action={self.DOOR_PUSH_ACTION}"
            )

        elapsed = time.monotonic() - self._door_push_started_at

        if elapsed < self.DOOR_PUSH_FORWARD_MAX_SECONDS:
            self._log_event(
                f"door_push_right_forward_direct_no_depth "
                f"action={self.DOOR_PUSH_ACTION} "
                f"elapsed={elapsed:.2f} "
                f"max_seconds={self.DOOR_PUSH_FORWARD_MAX_SECONDS:.2f}"
            )
            return self.DOOR_PUSH_ACTION

        self.car_controller.update_action("STOP")
        self._log_event(
            f"door_push_right_forward_done_hold_arm_still "
            f"elapsed={elapsed:.2f}"
        )

        self._log_event("door_task_done_no_arm_release")
        self._transition(
            TaskState.DONE,
            "Door pushed open by right-forward drive. Task 3 done. Arm left in press-down pose."
        )
        return "STOP"

    def _publish_initial_pose_if_configured(self, reason=""):
        if not self.INITIAL_POSE_ENABLED or self._initial_pose_published:
            return False

        if self.INITIAL_POSE_SKIP_IF_POSE_AVAILABLE and self._get_current_position() is not None:
            self._initial_pose_published = True
            self._log_event(f"initial_pose skipped reason={reason}: pose already available")
            return False

        if not hasattr(self.ros_communicator, "publish_initial_pose"):
            self._log_event(f"initial_pose skipped reason={reason}: publisher not available")
            return False

        count = max(1, int(self.INITIAL_POSE_PUBLISH_COUNT))
        for _ in range(count):
            self.ros_communicator.publish_initial_pose(
                self.INITIAL_POSE_X,
                self.INITIAL_POSE_Y,
                self.INITIAL_POSE_YAW,
            )

        self._initial_pose_published = True
        self._initial_pose_wait_until = time.monotonic() + max(0.0, self.INITIAL_POSE_SETTLE_SECONDS)
        self._log_event(
            f"initial_pose published reason={reason} count={count} "
            f"x={self.INITIAL_POSE_X:.3f} y={self.INITIAL_POSE_Y:.3f} yaw={self.INITIAL_POSE_YAW:.3f}"
        )
        return True

    def _update_stable_bear_map_position(self):
        msg = None
        try:
            msg = self.nav_processing.data_processor.get_yolo_target_map_position()
        except Exception:
            msg = None

        if msg is None or getattr(msg.header, "frame_id", "") != "map":
            return self._stable_bear_map_position

        point = msg.point
        try:
            position = [float(point.x), float(point.y)]
        except Exception:
            return self._stable_bear_map_position

        if not all(math.isfinite(value) for value in position):
            return self._stable_bear_map_position

        clearance = self._map_obstacle_clearance(position, self.GROUND_BEAR_MAP_MIN_CLEARANCE_M)
        if clearance is not None and clearance < self.GROUND_BEAR_MAP_MIN_CLEARANCE_M:
            self._log_event(
                f"reject_wall_bear reason=near_obstacle clearance={clearance:.3f} position={position}"
            )
            self._debug(
                f"reject_wall_bear reason=near_obstacle clearance={clearance:.3f} position={position}",
                0.5,
            )
            self._bear_map_positions = []
            self._stable_bear_map_position = None
            return None

        if self._bear_map_positions:
            last = self._bear_map_positions[-1]
            jump = self._distance(position, last)
            if jump > self.BEAR_MAP_MAX_JUMP_M:
                self._log_event(
                    f"bear_map_position ignored jump={jump:.3f} current={position} last={last}"
                )
                self._bear_map_positions = []
                self._stable_bear_map_position = None
                return None

        self._bear_map_positions.append(position)
        max_count = max(1, int(self.BEAR_MAP_STABLE_COUNT))
        self._bear_map_positions = self._bear_map_positions[-max_count:]

        if len(self._bear_map_positions) < max_count:
            self._debug(
                f"bear_map_position samples={len(self._bear_map_positions)}/{max_count} latest={position}",
                0.5,
            )
            return None

        avg_x = sum(item[0] for item in self._bear_map_positions) / len(self._bear_map_positions)
        avg_y = sum(item[1] for item in self._bear_map_positions) / len(self._bear_map_positions)
        self._stable_bear_map_position = [avg_x, avg_y]
        self._log_event(f"bear_map_position stable avg=({avg_x:.3f},{avg_y:.3f}) samples={self._bear_map_positions}")
        return self._stable_bear_map_position

    def _point_in_polygon(self, point, polygon):
        x = float(point[0])
        y = float(point[1])
        inside = False
        n = len(polygon)
        j = n - 1

        for i in range(n):
            xi, yi = float(polygon[i][0]), float(polygon[i][1])
            xj, yj = float(polygon[j][0]), float(polygon[j][1])

            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / ((yj - yi) if abs(yj - yi) > 1e-9 else 1e-9) + xi
                if x < x_cross:
                    inside = not inside

            j = i

        return inside

    def _get_current_yolo_target_map_xy_if_available(self):
        try:
            msg = self.nav_processing.data_processor.get_yolo_target_map_position()
        except Exception:
            msg = None

        if msg is None or getattr(msg.header, "frame_id", "") != "map":
            return None

        try:
            return [float(msg.point.x), float(msg.point.y)]
        except Exception:
            return None

    def _bridge_bear_position_in_area(self, position):
        if not self.BRIDGE_BEAR_AREA_FILTER_ENABLED:
            return True

        if position is None or len(position) < 2:
            self._log_event("bridge_bear_area_filter no_position")
            return False

        x = float(position[0])
        y = float(position[1])
        polygon = self.BRIDGE_BEAR_AREA_POLYGON
        margin = float(self.BRIDGE_BEAR_AREA_MARGIN_M)

        min_x = min(p[0] for p in polygon) - margin
        max_x = max(p[0] for p in polygon) + margin
        min_y = min(p[1] for p in polygon) - margin
        max_y = max(p[1] for p in polygon) + margin

        if x < min_x or x > max_x or y < min_y or y > max_y:
            self._log_event(
                f"bridge_bear_reject_outside_bbox "
                f"position=({x:.3f},{y:.3f}) "
                f"x_range=({min_x:.3f},{max_x:.3f}) "
                f"y_range=({min_y:.3f},{max_y:.3f})"
            )
            return False

        inside_polygon = self._point_in_polygon([x, y], polygon)

        if not inside_polygon:
            self._log_event(
                f"bridge_bear_reject_outside_polygon "
                f"position=({x:.3f},{y:.3f})"
            )
            return False

        self._log_event(
            f"bridge_bear_accept_inside_polygon position=({x:.3f},{y:.3f})"
        )
        return True

    def _bridge_bear_current_target_allowed(self, reason=""):
        if not (
            self.task_type == TaskType.BRIDGE_RECOVERY
            and self._target_label == self.BEAR_LABEL
        ):
            return True

        bear_position = self._get_current_yolo_target_map_xy_if_available()
        allowed = self._bridge_bear_position_in_area(bear_position)

        self._log_event(
            f"bridge_bear_area_gate reason={reason} "
            f"position={bear_position} allowed={allowed}"
        )

        return allowed

    def _update_stable_knob_map_position(self):
        if self.task_type != TaskType.DOOR_UNLOCK or self._target_label != self.KNOB_LABEL:
            return self._stable_knob_map_position

        msg = None
        try:
            msg = self.nav_processing.data_processor.get_yolo_target_map_position()
        except Exception:
            msg = None

        if msg is None or getattr(msg.header, "frame_id", "") != "map":
            return self._stable_knob_map_position

        try:
            position = [float(msg.point.x), float(msg.point.y)]
        except Exception:
            return self._stable_knob_map_position

        if not all(math.isfinite(value) for value in position):
            return self._stable_knob_map_position

        if self._knob_map_positions:
            last = self._knob_map_positions[-1]
            jump = self._distance(position, last)
            if jump > self.KNOB_MAP_MAX_JUMP_M:
                self._log_event(
                    f"knob_map_position_ignored_jump jump={jump:.3f} current={position} last={last}"
                )
                self._knob_map_positions = []
                self._stable_knob_map_position = None
                return None

        self._knob_map_positions.append(position)
        max_count = max(1, int(self.KNOB_MAP_STABLE_COUNT))
        self._knob_map_positions = self._knob_map_positions[-max_count:]

        if len(self._knob_map_positions) < max_count:
            self._log_event(
                f"knob_map_position_sample count={len(self._knob_map_positions)}/{max_count} latest={position}"
            )
            return None

        avg_x = sum(item[0] for item in self._knob_map_positions) / len(self._knob_map_positions)
        avg_y = sum(item[1] for item in self._knob_map_positions) / len(self._knob_map_positions)
        self._stable_knob_map_position = [avg_x, avg_y]

        self._log_event(
            f"knob_map_position_stable avg=({avg_x:.3f},{avg_y:.3f}) samples={self._knob_map_positions}"
        )
        return self._stable_knob_map_position

    def _map_obstacle_clearance(self, position, max_distance_m):
        grid = self.ros_communicator.get_latest_map()
        if grid is None or position is None:
            return None

        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return None

        origin = grid.info.origin.position
        cx = int(math.floor((float(position[0]) - origin.x) / resolution))
        cy = int(math.floor((float(position[1]) - origin.y) / resolution))
        width = int(grid.info.width)
        height = int(grid.info.height)
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            return 0.0

        radius_cells = max(1, int(math.ceil(float(max_distance_m) / resolution)))
        best = None
        for y in range(max(0, cy - radius_cells), min(height, cy + radius_cells + 1)):
            for x in range(max(0, cx - radius_cells), min(width, cx + radius_cells + 1)):
                value = grid.data[y * width + x]
                if value < 0 or int(value) >= self.ASTAR_OCCUPIED_THRESHOLD:
                    distance = math.hypot(x - cx, y - cy) * resolution
                    if best is None or distance < best:
                        best = distance

        if best is None:
            return float(max_distance_m)
        return best

    def _bear_pre_grab_goal(self, bear_position):
        current_position = self._get_current_position(require_map_frame=True)
        if current_position is None or bear_position is None:
            return None

        dx = bear_position[0] - current_position[0]
        dy = bear_position[1] - current_position[1]
        distance = math.hypot(dx, dy)
        if distance <= 0.01:
            return None

        stop_distance = max(0.0, float(self.BEAR_NAV_GOAL_STOP_DISTANCE))
        if distance <= stop_distance:
            return list(current_position)

        scale = (distance - stop_distance) / distance
        return [current_position[0] + dx * scale, current_position[1] + dy * scale]

    def _knob_pre_press_goal(self, knob_position):
        current_position = self._get_current_position(require_map_frame=True)
        if current_position is None or knob_position is None:
            return None

        dx = knob_position[0] - current_position[0]
        dy = knob_position[1] - current_position[1]
        distance = math.hypot(dx, dy)

        if distance <= 0.01:
            return None

        stop_distance = max(0.0, float(self.KNOB_NAV_GOAL_STOP_DISTANCE))

        if distance <= stop_distance:
            return list(current_position)

        scale = (distance - stop_distance) / distance
        goal = [
            current_position[0] + dx * scale,
            current_position[1] + dy * scale,
        ]
        return goal

    def _plan_astar_path(self, goal, context):
        grid = self.ros_communicator.get_latest_map()
        current_position = self._get_current_position(require_map_frame=True)
        current_pose = self._get_current_pose(require_map_frame=True)
        grid_ok = grid is not None
        current_ok = current_position is not None
        goal_ok = goal is not None

        if grid is not None:
            origin = grid.info.origin.position
            map_info = (
                f"width={grid.info.width} height={grid.info.height} "
                f"resolution={grid.info.resolution} origin=({origin.x:.2f},{origin.y:.2f})"
            )
        else:
            map_info = "None"

        self._log_event(
            f"astar_plan input_check context={context} "
            f"grid_ok={grid_ok} current_ok={current_ok} goal_ok={goal_ok} "
            f"map_info={map_info} current_position={current_position} goal={goal}"
        )

        if not grid_ok:
            self._log_event(
                "astar_plan no /map received in RosCommunicator; check /map subscriber QoS"
            )
            self._debug("astar_plan no /map received; check /map subscriber QoS", 1.0)

        inflation_radius = self._astar_inflation_radius()
        if grid is None or current_position is None or goal is None:
            self._log_event(
                f"astar_plan failed context={context}: "
                f"grid_ok={grid_ok} current_ok={current_ok} goal_ok={goal_ok}"
            )
            self._save_simple_map_debug_image(
                context=f"{context}_astar_failed",
                goal=goal,
                extra_text=f"inflation_radius={inflation_radius:.3f}",
            )
            return False

        if self.HYBRID_ASTAR_ENABLED:
            hybrid_success = self._try_hybrid_astar_path(
                grid=grid,
                current_pose=current_pose,
                current_position=current_position,
                goal=goal,
                context=context,
            )
            if hybrid_success:
                return True

            self._log_event(
                f"hybrid_astar failed context={context} fallback_to_grid_astar=True"
            )

        planner = GridAStarPlanner(
            unknown_as_obstacle=self.ASTAR_UNKNOWN_AS_OBSTACLE,
            occupied_threshold=self.ASTAR_OCCUPIED_THRESHOLD,
            inflation_radius_m=inflation_radius,
            waypoint_spacing_m=self.WAYPOINT_SPACING_M,
        )
        path = planner.plan_path(grid, current_position, goal)
        for warning in planner.last_warnings:
            self._log_event(f"astar_warning context={context} {warning}")
            self._debug(f"astar_warning context={context} {warning}", 1.0)

        if not path:
            self._reset_active_path()
            self._log_event(
                f"astar_plan failed context={context} start={current_position} goal={goal} "
                f"inflation_radius={inflation_radius:.3f}"
            )
            self._save_simple_map_debug_image(
                context=f"{context}_astar_failed",
                goal=goal,
                extra_text=f"inflation_radius={inflation_radius:.3f}",
            )
            return False

        if not self._validate_astar_path_footprint(grid, path, context):
            self._reset_active_path()
            self._save_simple_map_debug_image(
                context=f"{context}_footprint_failed",
                goal=goal,
                extra_text=f"footprint validation failed waypoints={len(path)}",
            )
            return False

        self._active_path = path
        self._active_path_yaws = None
        self._active_path_index = 0
        self._active_path_goal = list(goal)
        self._active_path_context = context
        self._last_astar_plan_at = time.monotonic()
        self._astar_blocked_count = 0
        self._log_event(
            f"astar_plan success context={context} start={current_position} goal={goal} "
            f"waypoints={len(path)} inflation_radius={inflation_radius:.3f} mode={self.ASTAR_INFLATION_MODE}"
        )
        self._save_simple_map_debug_image(
            context=f"{context}_astar_success",
            goal=goal,
            extra_text=f"inflation_radius={inflation_radius:.3f} waypoints={len(path)}",
        )
        return True

    def _try_hybrid_astar_path(self, grid, current_pose, current_position, goal, context):
        length, footprint_width, footprint_mode = self._footprint_dimensions()
        if current_pose is None:
            self._log_event(
                f"hybrid_astar input_check context={context} current_pose_ok=False "
                f"start_pose=None goal={goal} footprint_mode={footprint_mode} fallback_to_grid_astar=True"
            )
            return False

        current_pose_position, current_orientation = current_pose
        start_yaw = self._yaw_from_quaternion(current_orientation)
        start_pose = [float(current_pose_position[0]), float(current_pose_position[1]), float(start_yaw)]
        goal_xy = [float(goal[0]), float(goal[1])]
        goal_tolerance = self._hybrid_goal_tolerance(context)

        self._log_event(
            f"hybrid_astar input_check context={context} current_pose_ok=True "
            f"start_pose=({start_pose[0]:.3f},{start_pose[1]:.3f},{start_pose[2]:.1f}) "
            f"current_position={current_position} goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f}) "
            f"footprint_mode={footprint_mode} length={length:.3f} width={footprint_width:.3f} "
            f"yaw_bins={self.HYBRID_YAW_BINS} step_m={self.HYBRID_STEP_M:.3f} "
            f"yaw_step_deg={self.HYBRID_YAW_STEP_DEG:.1f} max_iterations={self.HYBRID_MAX_ITERATIONS} "
            f"goal_tolerance_m={goal_tolerance:.3f} "
            f"clearance_cost_weight={self.HYBRID_CLEARANCE_COST_WEIGHT:.3f}"
        )

        planner = HybridAStarPlanner(
            yaw_bins=self.HYBRID_YAW_BINS,
            step_m=self.HYBRID_STEP_M,
            yaw_step_deg=self.HYBRID_YAW_STEP_DEG,
            max_iterations=self.HYBRID_MAX_ITERATIONS,
            goal_tolerance_m=goal_tolerance,
            footprint_length_m=length,
            footprint_width_m=footprint_width,
            unknown_as_obstacle=self.FOOTPRINT_UNKNOWN_AS_OBSTACLE,
            occupied_threshold=self.FOOTPRINT_OCCUPIED_THRESHOLD,
            clearance_cost_weight=self.HYBRID_CLEARANCE_COST_WEIGHT,
            start_recovery_enabled=self.HYBRID_START_RECOVERY_ENABLED,
            start_recovery_radius_m=self.HYBRID_START_RECOVERY_RADIUS_M,
            start_recovery_step_m=self.HYBRID_START_RECOVERY_STEP_M,
            start_recovery_yaw_range_deg=self.HYBRID_START_RECOVERY_YAW_RANGE_DEG,
            start_recovery_yaw_step_deg=self.HYBRID_START_RECOVERY_YAW_STEP_DEG,
            start_allow_unknown=self.HYBRID_START_ALLOW_UNKNOWN,
            debug_image_enabled=self.HYBRID_DEBUG_IMAGE_ENABLED,
            debug_image_dir=self.DEBUG_MAP_IMAGE_DIR,
            debug_image_interval=self.HYBRID_DEBUG_IMAGE_INTERVAL,
            debug_max_images_per_plan=self.HYBRID_DEBUG_MAX_IMAGES_PER_PLAN,
            logger=self._log_event,
        )

        path, yaws = planner.plan_path(grid, start_pose, goal_xy, context=context)
        for warning in planner.last_warnings:
            self._log_event(f"hybrid_astar_warning context={context} {warning}")
            self._debug(f"hybrid_astar_warning context={context} {warning}", 1.0)

        if not path:
            self._log_event(
                f"hybrid_astar fail context={context} "
                f"start_pose=({start_pose[0]:.3f},{start_pose[1]:.3f},{start_pose[2]:.1f}) "
                f"goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f}) footprint_mode={footprint_mode} "
                f"iterations={planner.last_iterations} visited_count={planner.last_visited_count} "
                f"path_len=0 fallback_to_grid_astar=True debug_images={planner.last_debug_images}"
            )
            return False

        if str(context) == "return_home" and self._distance(path[-1], goal_xy) > 1e-6:
            path.append(list(goal_xy))
            if yaws:
                yaws.append(yaws[-1])
            self._log_event(
                f"hybrid_astar append_true_goal context={context} goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f})"
            )

        self._active_path = path
        self._active_path_yaws = yaws
        self._active_path_index = 0
        self._active_path_goal = list(goal_xy)
        self._active_path_context = context
        self._last_astar_plan_at = time.monotonic()
        self._astar_blocked_count = 0
        self._log_event(
            f"hybrid_astar success context={context} "
            f"start_pose=({start_pose[0]:.3f},{start_pose[1]:.3f},{start_pose[2]:.1f}) "
            f"goal=({goal_xy[0]:.3f},{goal_xy[1]:.3f}) footprint_mode={footprint_mode} "
            f"iterations={planner.last_iterations} visited_count={planner.last_visited_count} "
            f"path_len={len(path)} yaw_path_len={len(yaws) if yaws else 0} "
            f"fallback_to_grid_astar=False debug_images={planner.last_debug_images}"
        )
        self._save_simple_map_debug_image(
            context=f"{context}_hybrid_astar_success",
            goal=goal_xy,
            extra_text=f"hybrid waypoints={len(path)} yaws={len(yaws) if yaws else 0} footprint_mode={footprint_mode}",
        )
        return True

    def _hybrid_goal_tolerance(self, context):
        context = str(context)
        if context == "return_home":
            return float(self.HYBRID_RETURN_HOME_GOAL_TOLERANCE_M)
        if context == "nav_to_bear":
            return float(self.HYBRID_BEAR_GOAL_TOLERANCE_M)
        if context == "door_staging":
            return float(self.DOOR_STAGING_FINAL_APPROACH_DISTANCE_M)
        if context == "knob":
            return float(self.KNOB_PRE_PRESS_REACHED_DISTANCE)
        return float(self.HYBRID_GOAL_TOLERANCE_M)

    def _follow_active_path_action(self, reached_distance, context, done_state=None, done_message=""):
        if not self._active_path:
            self._log_event(f"follow_path no_active_path context={context}")
            return "STOP"

        current_position = self._get_current_position(require_map_frame=True)
        if current_position is None:
            self._log_event(f"follow_path no_map_pose context={context}")
            return "STOP"

        if self._active_path_goal and self._distance(current_position, self._active_path_goal) <= reached_distance:
            self._reset_active_path()
            if done_state is not None:
                self._transition(done_state, done_message)
            return "STOP"

        waypoint_skipped = False
        while self._active_path_index < len(self._active_path) - 1:
            waypoint = self._active_path[self._active_path_index]
            intermediate_reached_distance = self.WAYPOINT_REACHED_DISTANCE_M
            if context == "nav_to_bear":
                intermediate_reached_distance = self.WAYPOINT_INTERMEDIATE_REACHED_DISTANCE_M
            elif context == "return_home":
                intermediate_reached_distance = self.RETURN_HOME_WAYPOINT_REACHED_DISTANCE_M
                self._log_event(
                    f"return_home_intermediate_threshold index={self._active_path_index}/{len(self._active_path)} "
                    f"threshold={self._format_float(intermediate_reached_distance, 3)}"
                )

            waypoint_distance = self._distance(current_position, waypoint)
            if waypoint_distance > intermediate_reached_distance:
                break
            self._log_event(
                f"follow_path waypoint_skipped context={context} index={self._active_path_index}/{len(self._active_path)} "
                f"waypoint={waypoint} distance={self._format_float(waypoint_distance, 3)} "
                f"threshold={self._format_float(intermediate_reached_distance, 3)}"
            )
            waypoint_skipped = True
            self._active_path_index += 1

        waypoint = self._active_path[min(self._active_path_index, len(self._active_path) - 1)]
        waypoint_reached_distance = self.WAYPOINT_REACHED_DISTANCE_M
        if context == "nav_to_bear":
            if self._active_path_index < len(self._active_path) - 1:
                waypoint_reached_distance = self.WAYPOINT_INTERMEDIATE_REACHED_DISTANCE_M
            else:
                waypoint_reached_distance = self.BEAR_PRE_GRAB_REACHED_DISTANCE
        elif context == "return_home":
            if self._active_path_index < len(self._active_path) - 1:
                waypoint_reached_distance = self.RETURN_HOME_WAYPOINT_REACHED_DISTANCE_M
            else:
                waypoint_reached_distance = self.RETURN_HOME_FINAL_WAYPOINT_REACHED_DISTANCE_M
            self._log_event(
                f"return_home_waypoint_threshold index={self._active_path_index}/{len(self._active_path)} "
                f"threshold={self._format_float(waypoint_reached_distance, 3)}"
            )

        raw_action, _, distance, angle_error = self._fixed_waypoint_action(
            waypoint,
            reached_distance=waypoint_reached_distance,
            context=context,
        )
        if distance <= waypoint_reached_distance:
            self._log_event(
                f"follow_path waypoint_reached context={context} index={self._active_path_index}/{len(self._active_path)} "
                f"waypoint={waypoint} distance={self._format_float(distance, 3)} "
                f"threshold={self._format_float(waypoint_reached_distance, 3)} raw_action={raw_action}"
            )

            if self._active_path_index < len(self._active_path) - 1:
                self._active_path_index += 1
                return "STOP"

            if context == "return_home":
                distance_to_home = None
                if self.home_position is not None:
                    distance_to_home = self._distance(current_position, self.home_position)

                if distance_to_home is not None and distance_to_home <= reached_distance:
                    self._reset_active_path()
                    if done_state is not None:
                        self._transition(done_state, done_message)
                    return "STOP"

                self._log_event(
                    f"return_home final_waypoint_reached_but_not_home "
                    f"distance_to_home={self._format_float(distance_to_home, 3)} "
                    f"reached_distance={self._format_float(reached_distance, 3)} "
                    f"current_position={current_position} home_position={self.home_position}"
                )
                self._reset_active_path()
                if self.home_position is not None:
                    self._plan_astar_path(self.home_position, "return_home")
                return "STOP"

            self._reset_active_path()
            if done_state is not None:
                self._transition(done_state, done_message)
            return "STOP"

        safe_action = self._apply_obstacle_guard(raw_action, None, context)
        blocked = raw_action != safe_action

        if blocked:
            self._astar_blocked_count += 1
        elif raw_action == "FORWARD_SLOW":
            self._astar_blocked_count = 0

        self._log_event(
            f"follow_path context={context} index={self._active_path_index}/{len(self._active_path)} "
            f"waypoint={waypoint} distance={self._format_float(distance, 3)} "
            f"waypoint_reached_distance={self._format_float(waypoint_reached_distance, 3)} "
            f"angle_error={self._format_float(angle_error, 1)} raw_action={raw_action} "
            f"safe_action={safe_action} blocked={blocked} blocked_count={self._astar_blocked_count} "
            f"waypoint_skipped={waypoint_skipped}"
        )

        if (
            self.ASTAR_REPLAN_ON_BLOCKED
            and self._astar_blocked_count >= self.ASTAR_BLOCKED_REPLAN_COUNT
        ):
            self._log_event(f"follow_path blocked_replan_needed context={context}")
            self._astar_blocked_count = 0
            if self._active_path_goal and self._should_replan_astar(context, force=True):
                if self._plan_astar_path(self._active_path_goal, context):
                    return "STOP"
            return "STOP"

        return safe_action

    def _should_replan_astar(self, context, force=False):
        if not self.ASTAR_REPLAN_ON_BLOCKED and not force:
            return False
        if str(context) == "knob":
            interval = float(self.KNOB_NAV_REPLAN_INTERVAL_SECONDS)
        else:
            interval = max(float(self.ASTAR_REPLAN_INTERVAL_SECONDS), float(self.BEAR_NAV_REPLAN_INTERVAL_SECONDS))
        return force or time.monotonic() - self._last_astar_plan_at >= interval

    def _reset_active_path(self):
        self._active_path = None
        self._active_path_yaws = None
        self._active_path_index = 0
        self._active_path_goal = None
        self._active_path_context = None
        self._astar_blocked_count = 0

    def _astar_inflation_radius(self):
        if self.ASTAR_INFLATION_RADIUS_M is not None:
            return float(self.ASTAR_INFLATION_RADIUS_M)

        length = float(self.CAR_LENGTH_M)
        width = float(self.CAR_WIDTH_M)
        margin = float(self.SAFETY_MARGIN_M)
        if self.ASTAR_INFLATION_MODE == "width":
            return width / 2.0 + margin
        return math.sqrt((length / 2.0) ** 2 + (width / 2.0) ** 2) + margin

    def _footprint_dimensions(self):
        if getattr(self, "_holding_target", False):
            return float(self.CARRY_CAR_LENGTH_M), float(self.CARRY_CAR_WIDTH_M), "carry"
        return float(self.CAR_LENGTH_M), float(self.CAR_WIDTH_M), "normal"

    def _normalize_yaw_deg(self, yaw):
        return (float(yaw) + 180.0) % 360.0 - 180.0

    def _segment_yaw_deg(self, start, end):
        return math.degrees(math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0])))

    def _iter_yaw_samples(self, start_yaw, end_yaw):
        delta = self._normalize_yaw_deg(float(end_yaw) - float(start_yaw))
        step = max(float(self.FOOTPRINT_YAW_STEP_DEG), 1.0)
        count = max(1, int(math.ceil(abs(delta) / step)))
        for i in range(count + 1):
            yield self._normalize_yaw_deg(float(start_yaw) + delta * (i / count))

    def _world_to_grid_cell(self, grid, point):
        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return None
        origin = grid.info.origin.position
        gx = int((float(point[0]) - float(origin.x)) / resolution)
        gy = int((float(point[1]) - float(origin.y)) / resolution)
        return gx, gy

    def _footprint_corners(self, x, y, yaw_deg, length, width):
        yaw = math.radians(float(yaw_deg))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        half_length = float(length) / 2.0
        half_width = float(width) / 2.0
        corners = []
        for local_x, local_y in (
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ):
            world_x = float(x) + local_x * cos_yaw - local_y * sin_yaw
            world_y = float(y) + local_x * sin_yaw + local_y * cos_yaw
            corners.append((world_x, world_y))
        return corners

    def _footprint_pose_collides(self, grid, x, y, yaw_deg, length, width):
        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return True

        width_cells = int(grid.info.width)
        height_cells = int(grid.info.height)
        corners = self._footprint_corners(x, y, yaw_deg, length, width)
        corner_cells = [self._world_to_grid_cell(grid, corner) for corner in corners]
        if any(cell is None for cell in corner_cells):
            return True

        min_gx = min(cell[0] for cell in corner_cells)
        max_gx = max(cell[0] for cell in corner_cells)
        min_gy = min(cell[1] for cell in corner_cells)
        max_gy = max(cell[1] for cell in corner_cells)
        if min_gx < 0 or min_gy < 0 or max_gx >= width_cells or max_gy >= height_cells:
            return True

        yaw = math.radians(float(yaw_deg))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        half_length = float(length) / 2.0
        half_width = float(width) / 2.0
        origin = grid.info.origin.position

        for gy in range(min_gy, max_gy + 1):
            cell_y = float(origin.y) + (gy + 0.5) * resolution
            for gx in range(min_gx, max_gx + 1):
                cell_x = float(origin.x) + (gx + 0.5) * resolution
                dx = cell_x - float(x)
                dy = cell_y - float(y)
                local_x = dx * cos_yaw + dy * sin_yaw
                local_y = -dx * sin_yaw + dy * cos_yaw
                if abs(local_x) > half_length or abs(local_y) > half_width:
                    continue

                value = int(grid.data[gy * width_cells + gx])
                if value < 0 and self.FOOTPRINT_UNKNOWN_AS_OBSTACLE:
                    return True
                if value >= int(self.FOOTPRINT_OCCUPIED_THRESHOLD):
                    return True

        return False

    def _log_footprint_collision(self, context, waypoint_index, x, y, yaw_deg, footprint_mode):
        self._log_event(
            f"footprint_collision context={context} waypoint_index={waypoint_index} "
            f"pose=({float(x):.3f},{float(y):.3f},{float(yaw_deg):.1f}) "
            f"footprint_mode={footprint_mode} reason=occupied_or_unknown"
        )

    def _validate_astar_path_footprint(self, grid, path, context):
        if not self.RECTANGULAR_FOOTPRINT_CHECK_ENABLED:
            return True
        if grid is None or not path:
            return False

        pose = self._get_current_pose(require_map_frame=True)
        if pose is None:
            self._log_event(f"footprint_validation skipped_no_pose context={context}")
            return False

        current_position, current_orientation = pose
        current_yaw = self._yaw_from_quaternion(current_orientation)
        length, width, footprint_mode = self._footprint_dimensions()

        segment_yaws = []
        for i in range(len(path) - 1):
            segment_yaws.append(self._segment_yaw_deg(path[i], path[i + 1]))
        if not segment_yaws:
            segment_yaws.append(current_yaw)

        if self.FOOTPRINT_ROTATION_CHECK_ENABLED:
            for yaw in self._iter_yaw_samples(current_yaw, segment_yaws[0]):
                if self._footprint_pose_collides(grid, current_position[0], current_position[1], yaw, length, width):
                    self._log_footprint_collision(context, 0, current_position[0], current_position[1], yaw, footprint_mode)
                    return False

        check_step = max(float(self.FOOTPRINT_CHECK_STEP_M), 0.01)
        for i, waypoint in enumerate(path):
            yaw = segment_yaws[min(i, len(segment_yaws) - 1)]
            if self._footprint_pose_collides(grid, waypoint[0], waypoint[1], yaw, length, width):
                self._log_footprint_collision(context, i, waypoint[0], waypoint[1], yaw, footprint_mode)
                return False

            if i < len(path) - 1:
                next_waypoint = path[i + 1]
                yaw = segment_yaws[i]
                distance = self._distance(waypoint, next_waypoint)
                sample_count = max(1, int(math.ceil(distance / check_step)))
                for sample_index in range(1, sample_count + 1):
                    ratio = sample_index / sample_count
                    x = float(waypoint[0]) + (float(next_waypoint[0]) - float(waypoint[0])) * ratio
                    y = float(waypoint[1]) + (float(next_waypoint[1]) - float(waypoint[1])) * ratio
                    if self._footprint_pose_collides(grid, x, y, yaw, length, width):
                        self._log_footprint_collision(context, i, x, y, yaw, footprint_mode)
                        return False

                if self.FOOTPRINT_ROTATION_CHECK_ENABLED and i + 1 < len(segment_yaws):
                    next_yaw = segment_yaws[i + 1]
                    for rotation_yaw in self._iter_yaw_samples(yaw, next_yaw):
                        if self._footprint_pose_collides(grid, next_waypoint[0], next_waypoint[1], rotation_yaw, length, width):
                            self._log_footprint_collision(context, i + 1, next_waypoint[0], next_waypoint[1], rotation_yaw, footprint_mode)
                            return False

        self._log_event(
            f"footprint_validation success context={context} waypoints={len(path)} footprint_mode={footprint_mode} "
            f"length={length:.3f} width={width:.3f}"
        )
        return True

    def _bridge_pre_entry_configured(self):
        return self.TASK2_BRIDGE_PRE_ENTRY_ENABLED and self._valid_waypoint(self.BRIDGE_PRE_ENTRY)

    def _valid_waypoint(self, waypoint):
        if waypoint is None or len(waypoint) < 2:
            return False

        try:
            x = float(waypoint[0])
            y = float(waypoint[1])
        except Exception:
            return False

        return math.isfinite(x) and math.isfinite(y)

    def _navigate_to_waypoint(self, waypoint, reached_distance=None, context="waypoint"):
        raw_action, current_position, distance, angle_error = self._fixed_waypoint_action(
            waypoint,
            reached_distance=reached_distance,
            context=context,
        )

        safe_action = self._apply_obstacle_guard(raw_action, None, context)
        blocked = raw_action != safe_action
        self._log_event(
            f"fixed_waypoint context={context} current={current_position} goal={waypoint} "
            f"distance={self._format_float(distance, 3)} "
            f"angle_error={self._format_float(angle_error, 1)} "
            f"raw_action={raw_action} safe_action={safe_action} blocked={blocked}"
        )
        self._debug(
            f"fixed_waypoint context={context} current={current_position} goal={waypoint} "
            f"distance={self._format_float(distance, 3)} raw={raw_action} safe={safe_action} blocked={blocked}",
            0.5,
        )
        return safe_action

    def _fixed_waypoint_action(self, waypoint, reached_distance=None, context=""):
        pose = self._get_current_pose()
        if pose is None or not self._valid_waypoint(waypoint):
            return "STOP", None, None, None

        if reached_distance is None:
            reached_distance = self.FIXED_WAYPOINT_REACHED_DISTANCE

        current_position, orientation = pose
        goal = (float(waypoint[0]), float(waypoint[1]))
        distance = self._distance(current_position, goal)
        if distance < float(reached_distance):
            return "STOP", current_position, distance, 0.0

        angle_error = self._angle_to_waypoint(current_position, orientation, goal)
        forward_threshold = min(
            float(self.FIXED_WAYPOINT_HEADING_THRESHOLD_DEG),
            float(self.WAYPOINT_MAX_ANGLE_FOR_FORWARD_DEG),
        )
        return_home_context = str(context) in (
            "return_home",
            "return_home_direct",
            "return_home_fallback_direct",
        )
        if return_home_context:
            forward_threshold = float(self.RETURN_HOME_MAX_ANGLE_FOR_FORWARD_DEG)
        task2_staging_context = str(context) in (
            "task2_staging",
            "task2_staging_direct",
        )
        if task2_staging_context:
            forward_threshold = float(self.TASK2_STAGING_MAX_ANGLE_FOR_FORWARD_DEG)
        door_staging_context = str(context) in (
            "door_staging",
            "door_staging_direct",
            "knob",
        )
        if door_staging_context:
            forward_threshold = float(self.WAYPOINT_MAX_ANGLE_FOR_FORWARD_DEG)

        if abs(angle_error) <= forward_threshold:
            action = "FORWARD_SLOW"
            if return_home_context:
                action = self.RETURN_HOME_FORWARD_ACTION
            elif door_staging_context:
                action = self.DOOR_STAGING_FORWARD_ACTION
            if return_home_context:
                self._log_event(
                    f"return_home_forward_threshold angle_error={angle_error:.1f} "
                    f"threshold={forward_threshold:.1f} action={action}"
                )
            return action, current_position, distance, angle_error

        if angle_error > 0.0:
            action = "COUNTERCLOCKWISE_ROTATION_SLOW"
        else:
            action = "CLOCKWISE_ROTATION_SLOW"

        if (
            self.RETURN_HOME_FAST_ROTATION_ENABLED
            and return_home_context
            and abs(angle_error) > float(self.RETURN_HOME_FAST_ROTATION_ANGLE_DEG)
        ):
            if angle_error > 0.0:
                action = "COUNTERCLOCKWISE_ROTATION_MEDIAN"
            else:
                action = "CLOCKWISE_ROTATION_MEDIAN"
            self._log_event(
                f"return_home_fast_rotation context={context} angle_error={angle_error:.1f} action={action}"
            )

        if return_home_context:
            self._log_event(
                f"return_home_forward_threshold angle_error={angle_error:.1f} "
                f"threshold={forward_threshold:.1f} action={action}"
            )

        return action, current_position, distance, angle_error

    def _align_to_map_yaw_action(self, target_yaw_deg, tolerance_deg, context=""):
        pose = self._get_current_pose(require_map_frame=True)
        if pose is None:
            self._log_event(f"align_to_map_yaw no_pose context={context}")
            return "STOP", None, None

        _, orientation = pose
        current_yaw = self._yaw_from_quaternion(orientation)
        yaw_error = self._normalize_yaw_deg(float(target_yaw_deg) - float(current_yaw))
        if abs(yaw_error) <= float(tolerance_deg):
            return "STOP", yaw_error, current_yaw

        if yaw_error > 0.0:
            return "COUNTERCLOCKWISE_ROTATION_SLOW", yaw_error, current_yaw
        return "CLOCKWISE_ROTATION_SLOW", yaw_error, current_yaw

    def _get_current_pose(self, require_map_frame=False):
        pose = self._pose_from_pose_with_covariance_msg(
            getattr(self.ros_communicator, "latest_amcl_pose", None)
        )
        if pose is not None:
            return pose

        pose = self._pose_from_pose_with_covariance_msg(
            getattr(self.ros_communicator, "latest_slam_pose", None)
        )
        if pose is not None:
            return pose

        if require_map_frame:
            return None

        try:
            pose, orientation = self.nav_processing.data_processor.get_processed_amcl_pose()
            if pose is None or orientation is None:
                return None

            if not self._odom_fallback_warned:
                self._odom_fallback_warned = True
                self._log_event("warning using odom fallback for map waypoint may be inaccurate")
                self._debug("using odom fallback for map waypoint may be inaccurate", 0.0)

            return pose[:2], orientation
        except Exception:
            return None

    def _pose_from_pose_with_covariance_msg(self, msg):
        if msg is None or not hasattr(msg, "pose"):
            return None

        try:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            pose = [float(position.x), float(position.y)]
            quaternion = [
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ]
        except Exception:
            return None

        if not all(math.isfinite(value) for value in pose + quaternion):
            return None

        return pose, quaternion

    def _angle_to_waypoint(self, position, orientation, waypoint):
        target_yaw = math.degrees(
            math.atan2(float(waypoint[1]) - position[1], float(waypoint[0]) - position[0])
        )
        current_yaw = self._yaw_from_quaternion(orientation)
        return (target_yaw - current_yaw + 180.0) % 360.0 - 180.0

    def _yaw_from_quaternion(self, quaternion):
        x, y, z, w = quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

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
        return action in (
            "FORWARD",
            "FORWARD_SLOW",
            "FORWARD_VERY_SLOW",
            "FORWARD_BRIDGE",
            "FORWARD_BRIDGE_SLOW",
            "FORWARD_BRIDGE_LEFT",
            "FORWARD_BRIDGE_RIGHT",
            "LEFT_FRONT",
            "RIGHT_FRONT",
            "RIGHT_FRONT_STRONG",
        )

    def _is_backward_action(self, action):
        return action in (
            "BACKWARD",
            "BACKWARD_SLOW",
        )

    def _is_rotation_action(self, action):
        return action in (
            "CLOCKWISE_ROTATION_FINE",
            "COUNTERCLOCKWISE_ROTATION_FINE",
            "CLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
            "CLOCKWISE_ROTATION_MEDIAN",
            "COUNTERCLOCKWISE_ROTATION_MEDIAN",
        )

    def _is_movement_action(self, action):
        return self._is_forward_action(action) or self._is_backward_action(action) or self._is_rotation_action(action)

    def _is_left_turn_action(self, action):
        return action in (
            "COUNTERCLOCKWISE_ROTATION_FINE",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_MEDIAN",
        )

    def _is_right_turn_action(self, action):
        return action in (
            "CLOCKWISE_ROTATION_FINE",
            "CLOCKWISE_ROTATION_SLOW",
            "CLOCKWISE_ROTATION_MEDIAN",
        )

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

    def _scan_sector_min_depth_raw(self, scan_msg, min_deg, max_deg):
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
            if not math.isfinite(value) or value <= max(0.0, range_min):
                continue
            if math.isfinite(range_max) and value > range_max:
                continue

            angle = angle_min + index * angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))
            deg = math.degrees(angle)
            if min_deg <= deg <= max_deg:
                values.append(value)

        if not values:
            return None
        return min(values)

    def _get_scan_emergency_front_depth(self):
        scan_msg = self._get_latest_scan_msg()
        if scan_msg is None:
            return None
        front_deg = float(self.OBSTACLE_SCAN_FRONT_DEG)
        return self._scan_sector_min_depth_raw(scan_msg, -front_deg, front_deg)

    def _get_camera_obstacle_snapshot(self):
        if not self.OBSTACLE_GUARD_USE_CAMERA_DEPTH:
            return None

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
        if not self.OBSTACLE_USE_LIDAR_SCAN or not self.OBSTACLE_GUARD_USE_SCAN:
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

    def _is_astar_nav_context(self, context):
        return str(context) in (
            "nav_to_bear",
            "return_home",
            "follow_path",
            "door_staging",
            "door_staging_direct",
            "knob",
        )

    def _apply_stuck_guard(self, action, snapshot=None, context="", target_info=None):
        if not self.STUCK_RECOVERY_ENABLED:
            self._log_event(f"stuck_recovery_disabled context={context} action={action}")
            return action

        now = time.monotonic()

        target_approach_context = "approach" in str(context)
        if (
            target_approach_context
            and target_info is not None
            and self._is_target_centered_for_obstacle_guard(target_info)
            and self._target_distance_is_progressing(target_info, context)
        ):
            return action

        if self._stuck_recovery_stage != 0:
            return self._run_stuck_recovery_stage(action, context, now)

        if not self._is_forward_action(action) and not self._is_rotation_action(action):
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
            self._reset_stuck_recovery_stage()
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

        self._start_stuck_recovery_stage(1, current_xy, now, context, action)
        self._last_stuck_pose = current_xy
        self._last_stuck_time = now
        self._stuck_started_at = None

        self._log_event(
            f"stuck_detected context={context} raw_action={action} "
            f"moved={moved:.3f} recovery_action={self._recovery_action}"
        )
        self._debug(
            f"stuck_detected context={context} raw_action={action} recovery_action={self._recovery_action}",
            0.3,
        )

        return self._recovery_action

    def _reset_stuck_recovery_stage(self):
        self._stuck_recovery_stage = 0
        self._stuck_recovery_stage_pose = None
        self._stuck_recovery_stage_started_at = None
        self._recovery_until = 0.0
        self._recovery_action = "STOP"

    def _start_stuck_recovery_stage(self, stage, current_xy, now, context, raw_action):
        self._stuck_recovery_stage = int(stage)
        self._stuck_recovery_stage_pose = current_xy
        self._stuck_recovery_stage_started_at = now
        self._recovery_until = now + float(self.OBSTACLE_STUCK_RECOVERY_SECONDS)
        if stage == 1:
            self._recovery_action = "FORWARD_SLOW"
            stage_name = "try_forward"
        elif stage == 2:
            self._recovery_action = "BACKWARD_SLOW"
            stage_name = "try_backward"
        else:
            self._recovery_action = "STOP"
            stage_name = "reset_path_replan"

        self._log_event(
            f"stuck_recovery stage={stage_name} context={context} raw_action={raw_action} "
            f"recovery_action={self._recovery_action} duration={self.OBSTACLE_STUCK_RECOVERY_SECONDS:.2f}"
        )
        self._debug(
            f"stuck_recovery stage={stage_name} context={context} action={self._recovery_action}",
            0.0,
        )

    def _run_stuck_recovery_stage(self, action, context, now):
        current_position = self._get_current_position()
        current_xy = None
        if current_position is not None and len(current_position) >= 2:
            try:
                current_xy = (float(current_position[0]), float(current_position[1]))
            except Exception:
                current_xy = None

        if now < self._recovery_until:
            self._debug(
                f"stuck_recovery active stage={self._stuck_recovery_stage} context={context} action={self._recovery_action}",
                0.3,
            )
            return self._recovery_action

        moved = 0.0
        if current_xy is not None and self._stuck_recovery_stage_pose is not None:
            moved = self._distance(current_xy, self._stuck_recovery_stage_pose)

        if self._stuck_recovery_stage == 1:
            if moved >= self.OBSTACLE_STUCK_MIN_MOVE:
                self._last_stuck_pose = current_xy
                self._last_stuck_time = now
                self._stuck_started_at = None
                self._reset_stuck_recovery_stage()
                return action
            self._start_stuck_recovery_stage(2, current_xy, now, context, action)
            return self._recovery_action

        if self._stuck_recovery_stage == 2:
            self._start_stuck_recovery_stage(3, current_xy, now, context, action)
            self._log_event(
                f"stuck_recovery reset_path_replan context={context} moved_after_backward={moved:.3f} "
                f"active_path_goal={self._active_path_goal}"
            )
            goal = list(self._active_path_goal) if self._active_path_goal else None
            replan_context = self._active_path_context or context
            self._reset_active_path()
            self._reset_stuck_recovery_stage()
            if goal is not None:
                self._plan_astar_path(goal, replan_context)
            return "STOP"

        self._reset_stuck_recovery_stage()
        return action

    def _apply_obstacle_guard(self, action, target_info=None, context=""):
        if not self.OBSTACLE_GUARD_ENABLED:
            return action

        if not self._is_movement_action(action):
            return action

        if self.MAP_FIRST_NAVIGATION_ENABLED and self._is_map_first_context(context):
            return self._apply_map_first_obstacle_guard(action, target_info, context)

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

        astar_nav_context = self._is_astar_nav_context(context)
        if astar_nav_context and self.ASTAR_NAV_IGNORE_LIDAR_SIDE_GUARD:
            ignored_scan_false_positive = False
            if (
                self.ASTAR_NAV_REQUIRE_CAMERA_CONFIRM_FOR_SCAN_FRONT
                and scan_front is not None
                and camera_front is not None
                and scan_front < 0.30
                and camera_front > 1.0
                and camera_front - scan_front >= self.ASTAR_NAV_SCAN_CAMERA_DISAGREE_MARGIN
            ):
                ignored_scan_false_positive = True
                self._log_event(
                    f"astar_nav_guard ignore_scan_front_disagree "
                    f"scan_front={self._format_float(scan_front, 3)} "
                    f"camera_front={self._format_float(camera_front, 3)}"
                )

            scan_emergency = (
                scan_front is not None
                and scan_front <= self.ASTAR_NAV_FRONT_EMERGENCY_STOP_DISTANCE
            )
            camera_emergency = (
                camera_front is not None
                and camera_front <= self.ASTAR_NAV_FRONT_EMERGENCY_STOP_DISTANCE
            )
            safe_action = "STOP" if scan_emergency or camera_emergency else action

            self._log_event(
                f"astar_nav_guard context={context} source={source} "
                f"front={self._format_float(front_depth, 3)} "
                f"front_left={self._format_float(front_left_depth, 3)} "
                f"front_right={self._format_float(front_right_depth, 3)} "
                f"left={self._format_float(left_depth, 3)} "
                f"right={self._format_float(right_depth, 3)} "
                f"scan_front={self._format_float(scan_front, 3)} "
                f"camera_front={self._format_float(camera_front, 3)} "
                f"ignored_scan_false_positive={ignored_scan_false_positive} "
                f"scan_emergency={scan_emergency} camera_emergency={camera_emergency} "
                f"raw_action={action} safe_action={safe_action}"
            )
            if safe_action != action:
                return safe_action
            return self._apply_stuck_guard(action, snapshot, context, target_info)

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

    def _is_map_first_context(self, context):
        context = str(context)
        return self._is_astar_nav_context(context) or context.startswith("approach_")

    def _apply_map_first_obstacle_guard(self, action, target_info=None, context=""):
        emergency_distance = float(self.OBSTACLE_GUARD_SCAN_EMERGENCY_DISTANCE_M)
        scan_front = None
        if self.OBSTACLE_GUARD_USE_SCAN or self.OBSTACLE_GUARD_SCAN_EMERGENCY_ONLY:
            scan_front = self._get_scan_emergency_front_depth()
            if scan_front is not None and scan_front <= emergency_distance:
                self._log_event(
                    f"scan_emergency_stop front={scan_front:.3f} context={context} "
                    f"threshold={emergency_distance:.3f} raw_action={action}"
                )
                self._debug(
                    f"scan_emergency_stop front={scan_front:.3f} context={context} threshold={emergency_distance:.3f}",
                    0.3,
                )
                return "STOP"
        else:
            self._log_event(f"scan_disabled_for_navigation context={context} raw_action={action}")

        camera_snapshot = self._get_camera_obstacle_snapshot()
        camera_front = camera_snapshot.get("front") if camera_snapshot else None
        try:
            target_found = bool(target_info and len(target_info) >= 1 and float(target_info[0]) == 1.0)
        except Exception:
            target_found = False
        target_approach_context = str(context).startswith("approach_")

        self._log_event(
            f"scan_ignored context={context} scan_front={self._format_float(scan_front, 3)} "
            f"emergency_threshold={emergency_distance:.3f} raw_action={action}"
        )

        if (
            self.OBSTACLE_GUARD_USE_CAMERA_DEPTH
            and not target_found
            and camera_front is not None
            and camera_front <= self.OBSTACLE_EMERGENCY_STOP_DISTANCE
        ):
            self._log_event(
                f"obstacle_guard source=map_first/camera_depth context={context} "
                f"camera_front={camera_front:.3f} raw_action={action} safe_action=STOP"
            )
            return "STOP"

        if target_approach_context and target_found and camera_front is not None:
            self._log_event(
                f"obstacle_guard source=map_first/camera_depth context={context} "
                f"target_found=True camera_front={camera_front:.3f} raw_action={action} safe_action={action}"
            )
        else:
            self._log_event(
                f"obstacle_guard source=map_first context={context} "
                f"camera_front={self._format_float(camera_front, 3)} raw_action={action} safe_action={action}"
            )

        return self._apply_stuck_guard(action, camera_snapshot, context, target_info)

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

    def _bridge_bear_approach_forward_action(self, distance):
        try:
            distance = float(distance)
        except Exception:
            return "STOP"

        if getattr(self, "_bridge_bear_close_precision_mode", False):
            if (
                distance is not None
                and distance > self.BRIDGE_BEAR_CLOSE_PRECISION_EXIT_DISTANCE
            ):
                self._bridge_bear_close_precision_mode = False
                self._bridge_bear_close_precision_entered_at = None
                action = self.BRIDGE_BEAR_APPROACH_ACTION
                self._log_event(
                    f"bridge_bear_close_precision_exit_to_normal_speed "
                    f"distance={distance:.3f} "
                    f"exit_threshold={self.BRIDGE_BEAR_CLOSE_PRECISION_EXIT_DISTANCE:.3f} "
                    f"action={action}"
                )
                return action

            action = "FORWARD_VERY_SLOW"
            self._log_event(
                f"bridge_bear_close_precision_creep action={action} distance={distance:.3f}"
            )
            return action

        if distance > self.BRIDGE_BEAR_NORMAL_SPEED_DISTANCE:
            action = self.BRIDGE_BEAR_APPROACH_ACTION
            self._log_event(
                f"bridge_bear_normal_speed_approach "
                f"distance={distance:.3f} "
                f"threshold={self.BRIDGE_BEAR_NORMAL_SPEED_DISTANCE:.3f} "
                f"action={action}"
            )
            return action

        action = "FORWARD_VERY_SLOW"
        self._log_event(
            f"bridge_bear_slow_approach "
            f"distance={distance:.3f} "
            f"threshold={self.BRIDGE_BEAR_NORMAL_SPEED_DISTANCE:.3f} "
            f"action={action}"
        )
        return action

    def _bridge_uphill_keep_thrust_action(self, delta_x):
        if delta_x > self.BRIDGE_BEAR_GRAB_ALIGN_DELTA_X:
            return self.BRIDGE_UPHILL_RIGHT_ACTION

        if delta_x < -self.BRIDGE_BEAR_GRAB_ALIGN_DELTA_X:
            return self.BRIDGE_UPHILL_LEFT_ACTION

        return self.BRIDGE_UPHILL_ACTION

    def _reset_align_state(self):
        self._align_pulse_until = 0.0
        self._align_settle_until = 0.0
        self._align_last_info = None
        self._align_same_count = 0
        self._align_started_at = None
        self._align_best_abs_delta_x = None

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

    def _apply_bridge_bear_grab_z_offset(self):
        if self.task_type != TaskType.BRIDGE_RECOVERY or self._target_label != self.BEAR_LABEL:
            return False

        offset = float(self.BRIDGE_BEAR_GRAB_Z_OFFSET_M)
        if offset == 0.0:
            return False

        adjusted = False
        try:
            marker = getattr(self.ros_communicator, "latest_yolo_marker", None)
            if marker is not None and hasattr(marker, "pose") and hasattr(marker.pose, "position"):
                old_z = float(marker.pose.position.z)
                marker.pose.position.z = old_z + offset
                adjusted = True
                self._log_event(
                    f"bridge_bear_grab_z_offset marker old_z={old_z:.3f} "
                    f"new_z={marker.pose.position.z:.3f} offset={offset:.3f}"
                )
        except Exception as exc:
            self._log_event(f"bridge_bear_grab_z_offset marker_failed error={exc}")

        if not adjusted:
            try:
                point = self.ros_communicator.get_latest_yolo_detection_position()
                if point is not None and hasattr(point, "point"):
                    old_z = float(point.point.z)
                    point.point.z = old_z + offset
                    adjusted = True
                    self._log_event(
                        f"bridge_bear_grab_z_offset detection_point old_z={old_z:.3f} "
                        f"new_z={point.point.z:.3f} offset={offset:.3f}"
                    )
            except Exception as exc:
                self._log_event(f"bridge_bear_grab_z_offset detection_point_failed error={exc}")

        return adjusted


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
        self._bridge_bear_close_precision_mode = False
        self._bridge_bear_close_jump_count = 0
        self._bridge_bear_close_precision_entered_at = None
        self._reset_align_state()

    def _get_current_position(self, require_map_frame=False):
        pose = self._get_current_pose(require_map_frame=require_map_frame)
        if pose is None:
            return None
        return pose[0]

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
