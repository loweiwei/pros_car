# return 角度一律 radians
import math
import os
import time
import threading

import rclpy
import tf2_ros
import tf2_geometry_msgs

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped


class ArmController:
    """
    IK 計算版機械手臂控制。

    流程：
    1. 從 YOLO marker / detection position 取得目標點
    2. TF 轉換到 arm_ik_base
    3. 對 x/z 做抓取點補正
    4. 同時檢查 pre-grab / grab / lift 三個點是否都可達
    5. 用 2D IK 算 shoulder / elbow
    6. 打開夾爪 -> pre-grab -> approach -> close -> lift -> return
    """

    # ==========================
    # 抓取點補正參數
    # ==========================

    GRAB_X_OFFSET = 0.081
    GRAB_Z_OFFSET = 0.0795

    PRE_GRAB_X_BACKOFF = 0.00
    PRE_GRAB_Z_LIFT = 0.010

    LIFT_Z_OFFSET = 0.020

    MIN_GRAB_X = 0.10
    MIN_GRAB_Z = -0.090

    REACH_MARGIN = 0.01
    SOFT_GRIP_CLOSE_ANGLE = 10.0

    DOOR_KNOB_PRE_PRESS_POSE = None
    DOOR_KNOB_PRESS_DOWN_POSE = None
    DOOR_KNOB_RELEASE_POSE = [-180.0, 0.0, 90.0]
    DOOR_KNOB_PRESS_DOWN_SECONDS = 1.0

    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.target_marker = None

        self.log_path = os.environ.get("PROS_TASK_LOG_PATH", "/tmp/pros_task_debug.log")
        self.last_grab_started = False
        self.last_grab_error = None

        # 建立 TF2 監聽器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self.ros_communicator,
        )

        # 第一個馬達的基準座標系
        self.base_link_name = "arm_ik_base"

        # ==========================================
        # 手臂關節設定
        # ==========================================
        # joint 0: Shoulder
        # joint 1: Elbow
        # joint 2: Gripper / Finger
        #
        # 注意：
        # joint_angles 內部用 degrees
        # publish 時會轉成 radians
        self.joint_limits = [
            {
                "length": 0.08089007,
                "min_angle": -180,
                "max_angle": 0,
                "init": -180,
                "offset": 270,
                "dir": -1.0,
            },  # Joint 0 Shoulder

            {
                "length": 0.11,
                "min_angle": -240,
                "max_angle": 0,
                "init": 0,
                "offset": -120,
                "dir": -1.0,
            },  # Joint 1 Elbow

            {
                "length": 0.00,
                "min_angle": 0,
                "max_angle": 90,
                "init": 90,
                "offset": 0.0,
                "dir": 1.0,
            },  # Joint 2 Gripper
        ]

        self.joint_angles = [joint["init"] for joint in self.joint_limits]
        self.manual_step = 3.0

        print(f"🦾 Arm Controller Initialized: {len(self.joint_limits)} Joints Managed.")
        self._log_event(
            f"initialized joints={len(self.joint_limits)} base_link={self.base_link_name}"
        )

    # ==========================================
    # Log
    # ==========================================
    def _log_event(self, message):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(self.log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{timestamp} [ArmController] {message}\n")
        except Exception:
            pass

    # ==========================================
    # 手動控制
    # ==========================================
    def manual_control(self, index, key):
        """處理手動按鍵輸入，並根據 index 控制特定關節。"""

        if key == "b":
            self.joint_angles = [joint["init"] for joint in self.joint_limits]
            self._clamp_and_publish()
            self._visualize_arm_lines()
            print("手臂已重置為初始角度。")
            return False

        if key == "q":
            print("結束手臂手動控制。")
            return True

        if 0 <= index < len(self.joint_limits):
            if key == "i":
                self.joint_angles[index] += self.manual_step
            elif key == "k":
                self.joint_angles[index] -= self.manual_step
            else:
                print(
                    f"按鍵 '{key}' 無效，請使用 'i'(增加), 'k'(減少), "
                    f"'b'(重置), 或 'q'(取消)。"
                )
                return False

            self._clamp_and_publish()
            self._visualize_arm_lines()

        else:
            print(f"索引 {index} 無效，請確保其在範圍內 0-{len(self.joint_limits) - 1}。")

        return False

    # ==========================================
    # 自動抓取入口
    # ==========================================
    def auto_control(self, key=None, mode="auto_arm_control"):
        """自動抓取 /yolo/target_marker 的目標。"""

        self.last_grab_started = False
        self.last_grab_error = None

        if key == "q":
            self.target_marker = None
            self._log_event(f"auto_control cancel key={key} mode={mode}")
            return False

        if key == "b":
            self.joint_angles = [joint["init"] for joint in self.joint_limits]
            self._clamp_and_publish()
            self._visualize_arm_lines()
            print("手臂已重置為初始角度。")
            self._log_event("auto_control reset arm")
            return False

        if key != "g":
            print(f"按鍵 '{key}' 無效，請使用 'g'(抓取), 'b'(重置), 或 'q'(取消)。")
            self.last_grab_error = "invalid_key"
            self._log_event(f"auto_control invalid_key key={key} mode={mode}")
            return False

        # 取得最新目標位置
        target_marker = getattr(self.ros_communicator, "latest_yolo_marker", None)

        if not target_marker:
            target_point = self.ros_communicator.get_latest_yolo_detection_position()

            if target_point:
                target_marker = Marker()
                target_marker.header = target_point.header
                target_marker.pose.position = target_point.point
                target_marker.pose.orientation.w = 1.0

        if not target_marker:
            print("尚未收到 YOLO 目標，等待中...")
            self._log_event("auto_control failed: no yolo marker or detection position")
            self.last_grab_error = "no_yolo_target"
            return False

        marker_pos = target_marker.pose.position
        self._log_event(
            f"auto_control key={key} mode={mode} "
            f"marker_frame={target_marker.header.frame_id} "
            f"marker_pos=({marker_pos.x:.3f},{marker_pos.y:.3f},{marker_pos.z:.3f})"
        )

        self.target_marker = target_marker

        try:
            x_target, z_target, raw_distance, max_reach = self.get_latest_target_arm_metrics(
                target_marker
            )

            points = self._compute_grab_points(x_target, z_target)
            distances = self._compute_grab_distances(points)

            max_allowed_reach = max_reach - self.REACH_MARGIN

            self._log_event(
                "reach_check "
                f"raw_distance={raw_distance:.3f} "
                f"grab_distance={distances['grab']:.3f} "
                f"pre_distance={distances['pre']:.3f} "
                f"lift_distance={distances['lift']:.3f} "
                f"max_needed_distance={distances['max_needed']:.3f} "
                f"max_reach={max_reach:.3f} "
                f"max_allowed_reach={max_allowed_reach:.3f} "
                f"raw_x={x_target:.3f} raw_z={z_target:.3f} "
                f"grab_x={points['grab_x']:.3f} grab_z={points['grab_z']:.3f}"
            )

            if distances["max_needed"] > max_allowed_reach:
                self.last_grab_error = "target_out_of_reach"

                print(
                    f"⚠️ 目標抓取流程超出手臂可達範圍: "
                    f"needed={distances['max_needed']:.3f}, "
                    f"allowed={max_allowed_reach:.3f}, "
                    f"max_reach={max_reach:.3f}"
                )

                self._log_event(
                    "grab_rejected_out_of_reach "
                    f"max_needed_distance={distances['max_needed']:.3f} "
                    f"max_allowed_reach={max_allowed_reach:.3f} "
                    f"max_reach={max_reach:.3f} "
                    f"raw_x={x_target:.3f} raw_z={z_target:.3f} "
                    f"grab_x={points['grab_x']:.3f} grab_z={points['grab_z']:.3f} "
                    f"pre_x={points['pre_grab_x']:.3f} pre_z={points['pre_grab_z']:.3f} "
                    f"lift_x={points['lift_x']:.3f} lift_z={points['lift_z']:.3f}"
                )

                return False

            self.last_grab_started = True

            threading.Thread(
                target=self._execute_grab_sequence,
                args=(x_target, z_target),
                daemon=True,
            ).start()

            return True

        except Exception as exc:
            print(f"⚠️ 座標轉換或 IK 準備失敗: {exc}")
            self.last_grab_error = "tf_or_ik_prepare_failed"
            self._log_event(f"auto_control failed target_frame={self.base_link_name} error={exc}")
            return False

    # ==========================================
    # YOLO target -> arm base
    # ==========================================
    def get_latest_target_arm_metrics(self, target_marker=None):
        if target_marker is None:
            target_marker = getattr(self.ros_communicator, "latest_yolo_marker", None)

            if not target_marker:
                target_point = self.ros_communicator.get_latest_yolo_detection_position()

                if target_point:
                    target_marker = Marker()
                    target_marker.header = target_point.header
                    target_marker.pose.position = target_point.point
                    target_marker.pose.orientation.w = 1.0

        if not target_marker:
            raise RuntimeError("no yolo marker or detection position")

        target_map = PointStamped()
        target_map.header.frame_id = target_marker.header.frame_id
        target_map.header.stamp = self.ros_communicator.get_clock().now().to_msg()
        target_map.point = target_marker.pose.position

        transform = self.tf_buffer.lookup_transform(
            self.base_link_name,
            target_map.header.frame_id,
            rclpy.time.Time(),
        )

        target_base = tf2_geometry_msgs.do_transform_point(target_map, transform)

        x_target = target_base.point.x
        z_target = target_base.point.z
        arm_distance = math.sqrt(x_target**2 + z_target**2)
        max_reach = self.get_max_reach()

        print(f"🎯 目標相對基座座標: X={x_target:.3f}, Z={z_target:.3f}")

        self._log_event(
            f"tf_success source_frame={target_map.header.frame_id} "
            f"target_frame={self.base_link_name} "
            f"base_x={x_target:.3f} base_z={z_target:.3f} "
            f"arm_distance={arm_distance:.3f} max_reach={max_reach:.3f}"
        )

        if x_target < 0:
            self._log_event(
                f"arm_target_behind_base x={x_target:.3f} z={z_target:.3f}; "
                f"check camera_optical_frame -> {self.base_link_name} axes"
            )

        return x_target, z_target, arm_distance, max_reach

    def get_max_reach(self):
        return self.joint_limits[0]["length"] + self.joint_limits[1]["length"]

    def is_latest_target_reachable(self):
        """
        給 TaskController 用的可達範圍判斷。

        這裡會同時檢查：
        - pre-grab 點
        - grab 點
        - lift 點

        避免「grab 點可達，但 lift 點不可達」造成抓取開始後直接 abort。
        """
        try:
            x_target, z_target, raw_distance, max_reach = self.get_latest_target_arm_metrics()

            points = self._compute_grab_points(x_target, z_target)
            distances = self._compute_grab_distances(points)

            max_allowed_reach = max_reach - self.REACH_MARGIN
            reachable = distances["max_needed"] <= max_allowed_reach

            self._log_event(
                "arm_reach_check "
                f"raw_distance={raw_distance:.3f} "
                f"grab_distance={distances['grab']:.3f} "
                f"pre_distance={distances['pre']:.3f} "
                f"lift_distance={distances['lift']:.3f} "
                f"max_needed_distance={distances['max_needed']:.3f} "
                f"max_reach={max_reach:.3f} "
                f"max_allowed_reach={max_allowed_reach:.3f} "
                f"reachable={reachable} "
                f"grab_x={points['grab_x']:.3f} grab_z={points['grab_z']:.3f}"
            )

            return reachable

        except Exception as exc:
            self._log_event(f"arm_reach_check_failed error={exc}")
            return False

    # ==========================================
    # 抓取點補正
    # ==========================================
    def _compute_grab_points(self, x_target, z_target):
        """
        將 YOLO / TF 的 raw target 轉成真正給 IK 的抓取點。

        原因：
        - YOLO marker 通常是熊的中心，不一定是夾爪入口點。
        - TF 後的 z 可能是負數，直接丟 IK 會往地板抓。
        - pre-grab / lift 都不能離最大可達距離太近。
        """

        raw_grab_x = x_target + self.GRAB_X_OFFSET
        raw_grab_z = z_target + self.GRAB_Z_OFFSET

        # x 不要太小，避免手臂往正下方折
        grab_x = max(self.MIN_GRAB_X, raw_grab_x)

        # z 不允許是負的，避免往地板抓
        grab_z = max(self.MIN_GRAB_Z, raw_grab_z)

        # 預備位置：在目標前上方
        pre_grab_x = max(self.MIN_GRAB_X, grab_x - self.PRE_GRAB_X_BACKOFF)
        pre_grab_z = grab_z + self.PRE_GRAB_Z_LIFT

        # 夾住後抬起
        lift_x = grab_x
        lift_z = grab_z + self.LIFT_Z_OFFSET

        return {
            "raw_grab_x": raw_grab_x,
            "raw_grab_z": raw_grab_z,
            "grab_x": grab_x,
            "grab_z": grab_z,
            "pre_grab_x": pre_grab_x,
            "pre_grab_z": pre_grab_z,
            "lift_x": lift_x,
            "lift_z": lift_z,
        }

    def _compute_grab_distances(self, points):
        grab_distance = math.sqrt(points["grab_x"] ** 2 + points["grab_z"] ** 2)
        pre_distance = math.sqrt(points["pre_grab_x"] ** 2 + points["pre_grab_z"] ** 2)
        lift_distance = math.sqrt(points["lift_x"] ** 2 + points["lift_z"] ** 2)

        return {
            "grab": grab_distance,
            "pre": pre_distance,
            "lift": lift_distance,
            "max_needed": max(grab_distance, pre_distance, lift_distance),
        }

    def move_to_carry_pose(self):
        """回到不擋相機的搬運姿態，但保持夾爪關閉。"""
        try:
            self._log_event("move_to_carry_pose_start")

            close_angle = getattr(
                self,
                "SOFT_GRIP_CLOSE_ANGLE",
                self.joint_limits[2]["min_angle"],
            )
            elbow_init = self.joint_limits[1]["init"]
            shoulder_init = self.joint_limits[0]["init"]

            # 先確定夾爪關閉，再收 elbow/shoulder。
            self._smooth_move_to([None, None, close_angle], step=5.0, delay=0.1)
            self._smooth_move_to([None, elbow_init, None], step=5.0, delay=0.1)
            self._smooth_move_to([shoulder_init, None, None], step=5.0, delay=0.1)

            self._log_event(
                f"move_to_carry_pose_done shoulder={shoulder_init:.1f} "
                f"elbow={elbow_init:.1f} gripper={close_angle:.1f}"
            )
            return True

        except Exception as exc:
            self._log_event(f"move_to_carry_pose_failed error={exc}")
            return False

    def release_gripper_at_home(self):
        """只有車子回到起始位置後，TaskController 才能呼叫這個放下目標。"""
        try:
            self._log_event("release_gripper_at_home_start")

            release_angle = self.joint_limits[2]["max_angle"]
            self._smooth_move_to([None, None, release_angle], step=5.0, delay=0.1)
            time.sleep(0.5)

            self._log_event(f"release_gripper_at_home_done gripper={release_angle:.1f}")
            return True

        except Exception as exc:
            self._log_event(f"release_gripper_at_home_failed error={exc}")
            return False

    def press_door_knob_down(self):
        """
        Door knob special action:
        move to calibrated pre-press pose,
        move to calibrated press-down pose,
        hold press-down pose.
        It does NOT grab, lift, or carry.
        """
        if self.DOOR_KNOB_PRE_PRESS_POSE is None or self.DOOR_KNOB_PRESS_DOWN_POSE is None:
            self._log_event("door_knob_press_pose_not_configured")
            return False

        try:
            self._log_event(f"door_knob_pre_press_pose {self.DOOR_KNOB_PRE_PRESS_POSE}")
            self._smooth_move_to(self.DOOR_KNOB_PRE_PRESS_POSE, step=3.0, delay=0.1)
            time.sleep(0.2)

            self._log_event(f"door_knob_press_down_pose {self.DOOR_KNOB_PRESS_DOWN_POSE}")
            self._smooth_move_to(self.DOOR_KNOB_PRESS_DOWN_POSE, step=2.0, delay=0.1)
            time.sleep(self.DOOR_KNOB_PRESS_DOWN_SECONDS)
            return True

        except Exception as exc:
            self._log_event(f"door_knob_press_down_failed error={exc}")
            return False

    def release_door_knob_after_push(self):
        try:
            pose = self.DOOR_KNOB_RELEASE_POSE
            if pose is None:
                pose = [
                    self.joint_limits[0]["init"],
                    self.joint_limits[1]["init"],
                    self.joint_limits[2]["max_angle"],
                ]

            self._log_event(f"door_knob_release_pose {pose}")
            self._smooth_move_to(pose, step=3.0, delay=0.1)
            return True

        except Exception as exc:
            self._log_event(f"door_knob_release_failed error={exc}")
            return False


    def _execute_grab_sequence(self, x_target, z_target):
        """背景執行的完整抓取流程：YOLO/TF + offset + IK。

        重點：
        不要一開始就把 pre / grab / lift 的 IK 全部算完。
        要「走到 pre 之後，再算 grab」，
        避免 IK 分支從 -175 度突然跳到 -41 度。
        """

        points = self._compute_grab_points(x_target, z_target)
        distances = self._compute_grab_distances(points)
        max_reach = self.get_max_reach()
        max_allowed_reach = max_reach - self.REACH_MARGIN

        if distances["max_needed"] > max_allowed_reach:
            self.last_grab_error = "target_out_of_reach"

            self._log_event(
                "grab_sequence_abort_out_of_reach "
                f"max_needed_distance={distances['max_needed']:.3f} "
                f"max_allowed_reach={max_allowed_reach:.3f} "
                f"max_reach={max_reach:.3f} "
                f"grab_x={points['grab_x']:.3f} grab_z={points['grab_z']:.3f} "
                f"pre_x={points['pre_grab_x']:.3f} pre_z={points['pre_grab_z']:.3f} "
                f"lift_x={points['lift_x']:.3f} lift_z={points['lift_z']:.3f}"
            )
            return

        # 只做可達性檢查，不要把結果存起來用。
        # 因為 _calculate_2d_ik() 會根據 self.joint_angles 選最接近的解，
        # 如果一開始全算完，後面的 grab/lift 可能選到錯誤分支。
        try:
            self._calculate_2d_ik(points["pre_grab_x"], points["pre_grab_z"])
            self._calculate_2d_ik(points["grab_x"], points["grab_z"])
            self._calculate_2d_ik(points["lift_x"], points["lift_z"])
        except ValueError as exc:
            self.last_grab_error = "target_out_of_reach"
            self._log_event(f"grab_sequence_abort {exc}")
            return

        self._log_event(
            "grab_sequence_start "
            f"raw_x={x_target:.3f} raw_z={z_target:.3f} "
            f"raw_grab_x={points['raw_grab_x']:.3f} "
            f"raw_grab_z={points['raw_grab_z']:.3f} "
            f"grab_x={points['grab_x']:.3f} grab_z={points['grab_z']:.3f} "
            f"pre_x={points['pre_grab_x']:.3f} pre_z={points['pre_grab_z']:.3f} "
            f"lift_x={points['lift_x']:.3f} lift_z={points['lift_z']:.3f} "
            f"grab_distance={distances['grab']:.3f} "
            f"pre_distance={distances['pre']:.3f} "
            f"lift_distance={distances['lift']:.3f} "
            f"max_needed_distance={distances['max_needed']:.3f} "
            f"max_allowed_reach={max_allowed_reach:.3f} "
            f"offsets=(x={self.GRAB_X_OFFSET:.3f}, z={self.GRAB_Z_OFFSET:.3f}, "
            f"min_x={self.MIN_GRAB_X:.3f}, min_z={self.MIN_GRAB_Z:.3f}, "
            f"backoff={self.PRE_GRAB_X_BACKOFF:.3f}, "
            f"pre_lift={self.PRE_GRAB_Z_LIFT:.3f}, "
            f"lift={self.LIFT_Z_OFFSET:.3f}, "
            f"margin={self.REACH_MARGIN:.3f})"
        )

        # 1. 打開夾爪
        print("🔧 [1/7] 打開夾爪...")
        self._log_event("grab_open_gripper")
        target_open = [None, None, self.joint_limits[2]["max_angle"]]
        self._smooth_move_to(target_open, step=5.0, delay=0.1)
        time.sleep(0.5)

        # 2. 移動到預備抓取位置
        # 這裡才算 pre IK
        print("🤖 [2/7] 移動到預備抓取位置...")
        try:
            pre_deg1, pre_deg2 = self._calculate_2d_ik(
                points["pre_grab_x"],
                points["pre_grab_z"],
            )
        except ValueError as exc:
            self.last_grab_error = "pre_grab_ik_failed"
            self._log_event(f"grab_pre_grab_ik_failed {exc}")
            return

        self._log_event(
            f"grab_move_to_pre_grab "
            f"x={points['pre_grab_x']:.3f} z={points['pre_grab_z']:.3f} "
            f"shoulder={pre_deg1:.1f} elbow={pre_deg2:.1f}"
        )
        self._smooth_move_to([pre_deg1, pre_deg2, None], step=3.0, delay=0.1)
        time.sleep(0.3)

        # 3. 慢慢靠近真正抓取點
        # 重點：pre 已經走完後，self.joint_angles 已更新。
        # 所以 grab IK 會選比較接近 pre 的解，不會突然跳分支。
        print("🎯 [3/7] 慢慢靠近目標...")
        try:
            grab_deg1, grab_deg2 = self._calculate_2d_ik(
                points["grab_x"],
                points["grab_z"],
            )
        except ValueError as exc:
            self.last_grab_error = "grab_ik_failed"
            self._log_event(f"grab_approach_ik_failed {exc}")
            return

        self._log_event(
            f"grab_approach_target "
            f"x={points['grab_x']:.3f} z={points['grab_z']:.3f} "
            f"shoulder={grab_deg1:.1f} elbow={grab_deg2:.1f}"
        )
        self._smooth_move_to([grab_deg1, grab_deg2, None], step=2.0, delay=0.12)
        time.sleep(0.4)

        # 4. 關閉夾爪
        print("✊ [4/7] 夾取目標...")
        self._log_event("grab_close_gripper")
        target_close = [
            None,
            None,
            getattr(
                self,
                "SOFT_GRIP_CLOSE_ANGLE",
                self.joint_limits[2]["min_angle"],
            ),
        ]
        self._smooth_move_to(target_close, step=5.0, delay=0.1)
        time.sleep(1.0)

        # 5. 夾住後抬起
        # 重點：grab 已經走完後，才算 lift IK。
        print("⬆️ [5/7] 抬起目標...")
        try:
            lift_deg1, lift_deg2 = self._calculate_2d_ik(
                points["lift_x"],
                points["lift_z"],
            )
        except ValueError as exc:
            self.last_grab_error = "lift_ik_failed"
            self._log_event(f"grab_lift_ik_failed {exc}")
            return

        self._log_event(
            f"grab_lift_target "
            f"x={points['lift_x']:.3f} z={points['lift_z']:.3f} "
            f"shoulder={lift_deg1:.1f} elbow={lift_deg2:.1f}"
        )
        self._smooth_move_to([lift_deg1, lift_deg2, None], step=3.0, delay=0.1)
        time.sleep(0.4)

        # 6. 夾住目標，回到搬運姿態，但不要打開夾爪。
        # 真正放下目標的時機改由 TaskController._return_home() 判斷：
        # 只有車子回到記錄的 home_position 後，才呼叫 release_gripper_at_home()。
        print("🏠 [6/6] 夾住目標回到搬運姿態，保持夾爪關閉...")
        self._log_event("grab_return_carry_with_target_keep_closed")

        self.move_to_carry_pose()
        time.sleep(0.5)

        print("✅ 抓取任務完成，已保持夾爪關閉，等待車子回到起始位置")
        self._log_event("grab_sequence_done_keep_holding_target")
 

    # ==========================================
    # 2D IK
    # ==========================================
    def _calculate_2d_ik(self, x, z):
        """
        計算 2D 逆向運動學，回傳 shoulder / elbow 角度。

        這版會嘗試多個 IK 分支，只接受 joint_limits 內的角度，
        避免出現 shoulder=376 這種不可能到達的角度。
        """

        L1 = self.joint_limits[0]["length"]
        L2 = self.joint_limits[1]["length"]

        D = math.sqrt(x**2 + z**2)

        if D > (L1 + L2):
            self._log_event(
                f"ik_target_out_of_reach distance={D:.3f} "
                f"max_reach={(L1 + L2):.3f} x={x:.3f} z={z:.3f}"
            )
            raise ValueError(
                f"IK target out of reach: distance={D:.3f}, max_reach={(L1 + L2):.3f}"
            )

        if D < abs(L1 - L2):
            self._log_event(
                f"ik_target_too_close distance={D:.3f} "
                f"min_reach={abs(L1 - L2):.3f} x={x:.3f} z={z:.3f}"
            )
            raise ValueError(
                f"IK target too close: distance={D:.3f}, min_reach={abs(L1 - L2):.3f}"
            )

        cos_theta2 = (D**2 - L1**2 - L2**2) / (2 * L1 * L2)
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))

        alpha = math.atan2(z, x)

        cos_beta = (L1**2 + D**2 - L2**2) / (2 * L1 * D)
        cos_beta = max(-1.0, min(1.0, cos_beta))
        beta = math.acos(cos_beta)

        theta2_abs = math.acos(cos_theta2)

        def valid_angle_candidates(angle, min_limit, max_limit, tolerance=3.0):
            base_angle = angle % 360.0
            raw_candidates = [base_angle - 360.0, base_angle, base_angle + 360.0]

            valid = []

            for cand in raw_candidates:
                if min_limit <= cand <= max_limit:
                    valid.append(cand)

                elif min_limit - tolerance <= cand < min_limit:
                    valid.append(min_limit)

                elif max_limit < cand <= max_limit + tolerance:
                    valid.append(max_limit)

            deduped = []
            for cand in valid:
                if cand not in deduped:
                    deduped.append(cand)

            return deduped

        possible_solutions = []

        candidate_defs = [
            ("elbow_up_alpha_minus_beta", alpha - beta, -theta2_abs),
            ("elbow_up_alpha_plus_beta", alpha + beta, -theta2_abs),
            ("elbow_down_alpha_minus_beta", alpha - beta, theta2_abs),
            ("elbow_down_alpha_plus_beta", alpha + beta, theta2_abs),
        ]

        for name, theta1_rad, theta2_rad in candidate_defs:
            raw_deg1 = math.degrees(theta1_rad) + self.joint_limits[0]["offset"]
            raw_deg2 = math.degrees(theta2_rad) + self.joint_limits[1]["offset"]

            deg1_candidates = valid_angle_candidates(
                raw_deg1,
                self.joint_limits[0]["min_angle"],
                self.joint_limits[0]["max_angle"],
            )

            deg2_candidates = valid_angle_candidates(
                raw_deg2,
                self.joint_limits[1]["min_angle"],
                self.joint_limits[1]["max_angle"],
            )

            self._log_event(
                f"ik_branch_raw "
                f"x={x:.3f} z={z:.3f} distance={D:.3f} "
                f"current_shoulder={self.joint_angles[0]:.1f} "
                f"current_elbow={self.joint_angles[1]:.1f} "
                f"name={name} "
                f"raw_shoulder={raw_deg1:.1f} raw_elbow={raw_deg2:.1f} "
                f"shoulder_candidates={[round(v, 1) for v in deg1_candidates]} "
                f"elbow_candidates={[round(v, 1) for v in deg2_candidates]}"
            )

            for deg1 in deg1_candidates:
                for deg2 in deg2_candidates:
                    score = abs(deg1 - self.joint_angles[0]) + abs(
                        deg2 - self.joint_angles[1]
                    )

                    # 防止低處抓取時跑到 shoulder=-50/-60 的錯邊。
                    wrong_low_branch = z < -0.02 and deg1 > -120.0

                    if wrong_low_branch:
                        self._log_event(
                            f"ik_candidate_rejected_wrong_low_branch "
                            f"x={x:.3f} z={z:.3f} "
                            f"name={name} shoulder={deg1:.1f} elbow={deg2:.1f}"
                        )
                        continue

                    guarded_score = score

                    self._log_event(
                        f"ik_candidate "
                        f"x={x:.3f} z={z:.3f} "
                        f"name={name} "
                        f"shoulder={deg1:.1f} elbow={deg2:.1f} "
                        f"score={score:.1f} "
                        f"wrong_low_branch={wrong_low_branch} "
                        f"guarded_score={guarded_score:.1f}"
                    )

                    possible_solutions.append(
                        (guarded_score, name, deg1, deg2, raw_deg1, raw_deg2)
                    )

        if not possible_solutions:
            self._log_event(
                f"ik_no_valid_solution x={x:.3f} z={z:.3f} distance={D:.3f}"
            )
            raise ValueError(
                f"IK has no valid joint solution: x={x:.3f}, z={z:.3f}, distance={D:.3f}"
            )

        possible_solutions.sort(key=lambda item: item[0])
        score, name, deg1, deg2, raw_deg1, raw_deg2 = possible_solutions[0]

        self._log_event(
            f"ik_selected "
            f"x={x:.3f} z={z:.3f} "
            f"selected_name={name} "
            f"selected_shoulder={deg1:.1f} "
            f"selected_elbow={deg2:.1f} "
            f"selected_score={score:.1f} "
            f"current_before=({self.joint_angles[0]:.1f},{self.joint_angles[1]:.1f})"
        )
        print(f"🧮 IK 計算完成: Shoulder={deg1:.1f}°, Elbow={deg2:.1f}°")

        self._log_event(
            f"ik_result solution={name} x={x:.3f} z={z:.3f} distance={D:.3f} "
            f"raw_shoulder={raw_deg1:.1f} raw_elbow={raw_deg2:.1f} "
            f"shoulder={deg1:.1f} elbow={deg2:.1f} score={score:.1f}"
        )

        return deg1, deg2

    # ==========================================
    # 平滑移動
    # ==========================================
    def _smooth_move_to(self, target_angles, step=2.0, delay=0.05):
        """
        將手臂平滑地移動到目標角度。

        target_angles: [j0_target, j1_target, j2_target]
        若某個 joint 填 None，代表該軸不動。
        """

        safe_targets = list(target_angles)

        for i, target in enumerate(safe_targets):
            if target is None:
                continue

            min_a = self.joint_limits[i]["min_angle"]
            max_a = self.joint_limits[i]["max_angle"]

            if target < min_a or target > max_a:
                clamped = max(min_a, min(max_a, target))

                self._log_event(
                    f"smooth_move_target_clamped joint={i} "
                    f"target={target:.2f} clamped={clamped:.2f} "
                    f"limit=({min_a:.2f},{max_a:.2f})"
                )

                safe_targets[i] = clamped

        target_angles = safe_targets

        while True:
            all_reached = True

            for i in range(len(self.joint_angles)):
                if target_angles[i] is None:
                    continue

                diff = target_angles[i] - self.joint_angles[i]

                if abs(diff) <= step:
                    self.joint_angles[i] = target_angles[i]
                else:
                    self.joint_angles[i] += step if diff > 0 else -step
                    all_reached = False

            self._clamp_and_publish()
            self._visualize_arm_lines()

            if all_reached:
                break

            time.sleep(delay)

    def _normalize_angle(self, angle, min_limit, max_limit):
        """
        嘗試加減 360 度，尋找是否有多轉或少轉一圈後，
        剛好能落入 [min_limit, max_limit] 物理極限內的同界角。
        """

        base_angle = angle % 360.0
        candidates = [base_angle - 360.0, base_angle, base_angle + 360.0]

        for cand in candidates:
            if min_limit <= cand <= max_limit:
                return cand

        return max(min_limit, min(max_limit, angle))

    # ==========================================
    # 視覺化手臂
    # ==========================================
    def _visualize_arm_lines(self):
        """根據目前角度和長度，算出 3 個點並發布視覺化線條。"""

        L1 = self.joint_limits[0]["length"]
        L2 = self.joint_limits[1]["length"]

        th1 = math.radians(self.joint_angles[0] - self.joint_limits[0]["offset"])
        th2 = math.radians(self.joint_angles[1] - self.joint_limits[1]["offset"])

        p0 = Point(x=0.0, y=0.0, z=0.0)

        p1 = Point(
            x=L1 * math.cos(th1),
            y=0.0,
            z=L1 * math.sin(th1),
        )

        p2 = Point(
            x=p1.x + L2 * math.cos(th1 + th2),
            y=0.0,
            z=p1.z + L2 * math.sin(th1 + th2),
        )

        marker = Marker()
        marker.header.frame_id = self.base_link_name
        marker.header.stamp = self.ros_communicator.get_clock().now().to_msg()
        marker.ns = "arm_kinematics"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.scale.x = 0.02
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        marker.points = [p0, p1, p2]

        try:
            self.ros_communicator.publish_arm_visual_lines(marker)
        except Exception as exc:
            self._log_event(f"publish_arm_visual_lines failed error={exc}")

    # ==========================================
    # 發布手臂角度
    # ==========================================
    def _clamp_and_publish(self):
        """確保所有數值在安全範圍內，並轉換為 radians 後發布。"""

        for i in range(len(self.joint_limits)):
            min_a = self.joint_limits[i]["min_angle"]
            max_a = self.joint_limits[i]["max_angle"]

            self.joint_angles[i] = max(
                min_a,
                min(max_a, self.joint_angles[i]),
            )

        joint_pos_radians = [
            math.radians(
                float(self.joint_angles[i]) * self.joint_limits[i].get("dir", 1.0)
            )
            for i in range(len(self.joint_angles))
        ]

        self._log_event(
            f"publish_arm degrees={[round(angle, 2) for angle in self.joint_angles]} "
            f"radians={[round(angle, 3) for angle in joint_pos_radians]}"
        )

        self.ros_communicator.publish_robot_arm_angle(joint_pos_radians)
