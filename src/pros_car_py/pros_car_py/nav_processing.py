from pros_car_py.nav2_utils import (
    get_yaw_from_quaternion,
    get_direction_vector,
    get_angle_to_target,
    calculate_angle_point,
    cal_distance,
)
import math
import os


class Nav2Processing:
    BRIDGE_PATH_MIN_CONFIDENCE = 0.55
    BRIDGE_PATH_MIN_AREA = 1000.0
    BRIDGE_ENTRY_STOP_DISTANCE = 0.55
    BRIDGE_ENTRY_TOO_CLOSE_DISTANCE = 0.40
    BRIDGE_PATH_ALIGN_DELTA_X = 35.0
    BRIDGE_PATH_ALIGN_ANGLE_DEG = 8.0

    def __init__(self, ros_communicator, data_processor):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.finishFlag = False
        self.global_plan_msg = None
        self.index = 0
        self.index_length = 0
        self.recordFlag = 0
        self.goal_published_flag = False
        self._camera_nav_delta_ema = None
        self._camera_nav_depth_ema = None
        self._camera_nav_last_action = "STOP"
        self._camera_nav_invert_turn = os.environ.get("PROS_CAMERA_NAV_INVERT_TURN", "0") == "1"

    def reset_nav_process(self):
        self.finishFlag = False
        self.recordFlag = 0
        self.goal_published_flag = False
        self._camera_nav_delta_ema = None
        self._camera_nav_depth_ema = None
        self._camera_nav_last_action = "STOP"

    def finish_nav_process(self):
        self.finishFlag = True
        self.recordFlag = 1

    def get_finish_flag(self):
        return self.finishFlag

    def get_action_from_nav2_plan(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True
        orientation_points, coordinates = (
            self.data_processor.get_processed_received_global_plan()
        )
        action_key = "STOP"
        if not orientation_points or not coordinates:
            action_key = "STOP"
        else:
            try:
                z, w = orientation_points[0]
                plan_yaw = get_yaw_from_quaternion(z, w)
                car_position, car_orientation = (
                    self.data_processor.get_processed_amcl_pose()
                )
                car_orientation_z, car_orientation_w = (
                    car_orientation[2],
                    car_orientation[3],
                )
                goal_position = self.ros_communicator.get_latest_goal()
                target_distance = cal_distance(car_position, goal_position)
                if target_distance < 0.5:
                    action_key = "STOP"
                    self.finishFlag = True
                else:
                    car_yaw = get_yaw_from_quaternion(
                        car_orientation_z, car_orientation_w
                    )
                    diff_angle = (plan_yaw - car_yaw) % 360.0
                    if diff_angle < 30.0 or (diff_angle > 330 and diff_angle < 360):
                        action_key = "FORWARD"
                    elif diff_angle > 30.0 and diff_angle < 180.0:
                        action_key = "COUNTERCLOCKWISE_ROTATION"
                    elif diff_angle > 180.0 and diff_angle < 330.0:
                        action_key = "CLOCKWISE_ROTATION"
                    else:
                        action_key = "STOP"
            except:
                action_key = "STOP"
        return action_key

    def get_action_from_nav2_plan_no_dynamic_p_2_p(self, goal_coordinates=None):
        if goal_coordinates is not None and not self.goal_published_flag:
            self.ros_communicator.publish_goal_pose(goal_coordinates)
            self.goal_published_flag = True

        # 只抓第一次路径
        if self.recordFlag == 0:
            if not self.check_data_availability():
                return "STOP"
            else:
                print("Get first path")
                self.index = 0
                self.global_plan_msg = (
                    self.data_processor.get_processed_received_global_plan_no_dynamic()
                )
                self.recordFlag = 1
                action_key = "STOP"

        car_position, car_orientation = self.data_processor.get_processed_amcl_pose()

        goal_position = self.ros_communicator.get_latest_goal()
        target_distance = cal_distance(car_position, goal_position)

        # 抓最近的物標(可調距離)
        target_x, target_y = self.get_next_target_point(car_position)

        if target_x is None or target_distance < 0.5:
            self.ros_communicator.reset_nav2()
            self.finish_nav_process()
            return "STOP"

        # 計算角度誤差
        diff_angle = self.calculate_diff_angle(
            car_position, car_orientation, target_x, target_y
        )
        if diff_angle < 20 and diff_angle > -20:
            action_key = "FORWARD"
        elif diff_angle < -20 and diff_angle > -180:
            action_key = "CLOCKWISE_ROTATION"
        elif diff_angle > 20 and diff_angle < 180:
            action_key = "COUNTERCLOCKWISE_ROTATION"
        return action_key

    def check_data_availability(self):
        return (
            self.data_processor.get_processed_received_global_plan_no_dynamic()
            and self.data_processor.get_processed_amcl_pose()
            and self.ros_communicator.get_latest_goal()
        )

    def get_next_target_point(self, car_position, min_required_distance=0.5):
        """
        選擇距離車輛 min_required_distance 以上最短路徑然後返回 target_x, target_y
        """
        if self.global_plan_msg is None or self.global_plan_msg.poses is None:
            print("Error: global_plan_msg is None or poses is missing!")
            return None, None
        while self.index < len(self.global_plan_msg.poses) - 1:
            target_x = self.global_plan_msg.poses[self.index].pose.position.x
            target_y = self.global_plan_msg.poses[self.index].pose.position.y
            distance_to_target = cal_distance(car_position, (target_x, target_y))

            if distance_to_target < min_required_distance:
                self.index += 1
            else:
                self.ros_communicator.publish_selected_target_marker(
                    x=target_x, y=target_y
                )
                return target_x, target_y

        return None, None

    def calculate_diff_angle(self, car_position, car_orientation, target_x, target_y):
        target_pos = [target_x, target_y]
        diff_angle = calculate_angle_point(
            car_orientation[2], car_orientation[3], car_position[:2], target_pos
        )
        return diff_angle

    def filter_negative_one(self, depth_list):
        return [depth for depth in depth_list if depth != -1.0]

    def bridge_depth_segment_nav(self):
        path_info = self.data_processor.get_yolo_path_info()
        if path_info is None or len(path_info) < 9:
            return "STOP"

        try:
            found = float(path_info[0]) == 1.0
            bottom_center_x = float(path_info[3])
            angle = float(path_info[4])
            width = float(path_info[5])
            area = float(path_info[7])
            confidence = float(path_info[8])
        except Exception:
            return "STOP"

        if (
            not found
            or not math.isfinite(bottom_center_x)
            or not math.isfinite(angle)
            or not math.isfinite(width)
            or not math.isfinite(area)
            or not math.isfinite(confidence)
            or width <= 0.0
            or confidence < self.BRIDGE_PATH_MIN_CONFIDENCE
            or area < self.BRIDGE_PATH_MIN_AREA
        ):
            return "STOP"

        image_center_x = width / 2.0
        delta_x = bottom_center_x - image_center_x

        entry_depth = None
        entry_info = self.data_processor.get_yolo_bridge_entry_info()
        if entry_info is not None and len(entry_info) >= 4:
            try:
                depth = float(entry_info[3])
                if math.isfinite(depth) and depth > 0.0:
                    entry_depth = depth
            except Exception:
                entry_depth = None

        if entry_depth is not None:
            if entry_depth < self.BRIDGE_ENTRY_TOO_CLOSE_DISTANCE:
                return "STOP"
            if (
                self.BRIDGE_ENTRY_TOO_CLOSE_DISTANCE <= entry_depth <= 0.60
                and abs(delta_x) <= self.BRIDGE_PATH_ALIGN_DELTA_X
                and abs(angle) <= self.BRIDGE_PATH_ALIGN_ANGLE_DEG
            ):
                return "STOP_READY_FOR_ASCEND"
            if entry_depth <= self.BRIDGE_ENTRY_STOP_DISTANCE:
                return "STOP"

        if abs(delta_x) > self.BRIDGE_PATH_ALIGN_DELTA_X:
            if delta_x > 0:
                return "CLOCKWISE_ROTATION_FINE"
            return "COUNTERCLOCKWISE_ROTATION_FINE"

        if abs(angle) > self.BRIDGE_PATH_ALIGN_ANGLE_DEG:
            if angle > 0:
                return "CLOCKWISE_ROTATION_FINE"
            return "COUNTERCLOCKWISE_ROTATION_FINE"

        return "FORWARD_VERY_SLOW"

    def camera_nav(self):
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()

        if camera_multi_depth is None or yolo_target_info is None:
            return "STOP"

        action = "STOP"

        if yolo_target_info[0] == 1:
            depth = float(yolo_target_info[1])
            delta_x = float(yolo_target_info[2])

            alpha = 0.35
            self._camera_nav_delta_ema = self._ema(
                self._camera_nav_delta_ema, delta_x, alpha
            )

            if depth > 0.0:
                self._camera_nav_depth_ema = self._ema(
                    self._camera_nav_depth_ema, depth, alpha
                )

            delta_x = self._camera_nav_delta_ema
            depth = self._camera_nav_depth_ema if self._camera_nav_depth_ema is not None else depth

            centered_px = 120.0

            if delta_x > centered_px:
                action = "CLOCKWISE_ROTATION_FINE"
            elif delta_x < -centered_px:
                action = "COUNTERCLOCKWISE_ROTATION_FINE"
            else:
                if depth <= 0.0 or depth < 0.9:
                    action = "STOP"
                else:
                    action = "FORWARD_VERY_SLOW"

        else:
            self._camera_nav_delta_ema = None
            self._camera_nav_depth_ema = None
            action = "STOP"

        if self._camera_nav_invert_turn:
            if action == "CLOCKWISE_ROTATION_FINE":
                action = "COUNTERCLOCKWISE_ROTATION_FINE"
            elif action == "COUNTERCLOCKWISE_ROTATION_FINE":
                action = "CLOCKWISE_ROTATION_FINE"

        self._camera_nav_last_action = action
        return action


    def _ema(self, current, value, alpha):
        if current is None:
            return value
        return current * (1.0 - alpha) + value * alpha


    def camera_nav_unity(self):
        yolo_target_info = self.data_processor.get_yolo_target_info()
        camera_multi_depth = self.data_processor.get_camera_x_multi_depth()

        if camera_multi_depth is None or yolo_target_info is None:
            return "STOP"

        action = "STOP"

        if yolo_target_info[0] == 1:
            depth = float(yolo_target_info[1])
            delta_x = float(yolo_target_info[2])

            centered_px = 120.0

            if delta_x > centered_px:
                action = "CLOCKWISE_ROTATION_FINE"
            elif delta_x < -centered_px:
                action = "COUNTERCLOCKWISE_ROTATION_FINE"
            else:
                if depth <= 0.0 or depth < 0.9:
                    action = "STOP"
                else:
                    action = "FORWARD_VERY_SLOW"
        else:
            action = "STOP"

        return action


    def stop_nav(self):
        return "STOP"
