from rclpy.node import Node
from pros_car_py.car_models import DeviceDataTypeEnum, CarCControl
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point
from std_msgs.msg import String, Header
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, Imu, CompressedImage
from trajectory_msgs.msg import JointTrajectoryPoint
import orjson
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Bool
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from nav2_msgs.srv import ClearEntireCostmap
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import rclpy
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import time
import math


class RosCommunicator(Node):
    def __init__(self):
        super().__init__("RosCommunicator")
        self._last_debug_log = {}
        self.latest_data = {}

        # subscribeamcl_pose
        self.latest_amcl_pose = None
        self.subscriber_amcl = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.subscriber_amcl_callback, 1
        )

        self.latest_slam_pose = None
        self.subscriber_slam_pose = self.create_subscription(
            PoseWithCovarianceStamped, "/pose", self.subscriber_slam_pose_callback, 1
        )

        self.latest_odom = None
        self.subscriber_odom = self.create_subscription(
            Odometry, "/odom", self.subscriber_odom_callback, 1
        )

        # subscribe goal_pose
        self.latest_goal_pose = None
        self.target_pose = None
        self.subscriber_goal = self.create_subscription(
            PoseStamped, "/goal_pose", self.subscriber_goal_callback, 10
        )

        # subscribe lidar
        self.latest_lidar = None
        self.subscriber_lidar = self.create_subscription(
            LaserScan, "/scan", self.subscriber_lidar_callback, 1
        )

        # subscribe global_plan
        self.latest_received_global_plan = None
        self.subscriber_received_global_plan = self.create_subscription(
            Path, "/received_global_plan", self.received_global_plan_callback, 1
        )

        self.latest_map = None
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.subscriber_map = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, map_qos
        )

        # Subscribe to YOLO detected object coordinates
        self.latest_yolo_coordinates = None
        self.subscriber_yolo_detection_position = self.create_subscription(
            PointStamped,
            "/yolo/detection/position",
            self.yolo_detection_position_callback,
            10,
        )

        # Subscribe to YOLO detected object coordinates
        self.latest_yolo_offset = None
        self.subscriber_yolo_offset = self.create_subscription(
            String,
            "/yolo/object/offset",
            self.yolo_detection_offset_callback,
            10,
        )

        self.latest_yolo_detection_status = None
        self.subscriber_yolo_detection_status = self.create_subscription(
            Bool, "/yolo/detection/status", self.yolo_detection_status_callback, 10
        )

        self.latest_yolo_target_map_position = None
        self.subscriber_yolo_target_map_position = self.create_subscription(
            PointStamped,
            "/yolo/target_map_position",
            self.yolo_target_map_position_callback,
            10,
        )

        self.latest_imu_data = None
        self.imu_sub = self.create_subscription(
            Imu, "/imu/data", self.imu_data_callback, 10
        )

        self.latest_mediapipe_data = None
        self.mediapipe_sub = self.create_subscription(
            Point, "/mediapipe_data", self.mediapipe_data_callback, 10
        )

        self.latest_yolo_target_info = None
        self.yolo_target_info_sub = self.create_subscription(
            Float32MultiArray, "/yolo/target_info", self.yolo_target_info_callback, 1
        )

        self.latest_yolo_bridge_info = None
        self.yolo_bridge_info_sub = self.create_subscription(
            Float32MultiArray, "/yolo/bridge_info", self.yolo_bridge_info_callback, 1
        )

        self.latest_yolo_path_info = None
        self.yolo_path_info_sub = self.create_subscription(
            Float32MultiArray, "/yolo/path_info", self.yolo_path_info_callback, 1
        )

        self.latest_yolo_bridge_entry_info = None
        self.yolo_bridge_entry_info_sub = self.create_subscription(
            Float32MultiArray,
            "/yolo/bridge_entry_info",
            self.yolo_bridge_entry_info_callback,
            1,
        )

        self.latest_yolo_pre_bridge_goal = None
        self.yolo_pre_bridge_goal_sub = self.create_subscription(
            PoseStamped,
            "/yolo/pre_bridge_goal",
            self.yolo_pre_bridge_goal_callback,
            1,
        )

        self.latest_camera_x_multi_depth = None
        self.camera_x_multi_depth_sub = self.create_subscription(
            Float32MultiArray,
            "/camera/x_multi_depth_values",
            self.camera_x_multi_depth_callback,
            10,
        )

        self.latest_cmd_vel = None
        self.subscriber_cmd_vel = self.create_subscription(
            Twist, "/cmd_vel", self.subscriber_cmd_vel_callback, 1
        )

        # publish car_C_rear_wheel and car_C_front_wheel
        self.publisher_rear = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_rear_wheel, 1
        )
        self.publisher_forward = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_front_wheel, 1
        )
        self.publisher_cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        # publish goal_pose
        self.publisher_goal_pose = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.publisher_initial_pose = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

        # publish robot arm angle
        self.publisher_joint_trajectory = self.create_publisher(
            JointTrajectoryPoint, DeviceDataTypeEnum.robot_arm, 10
        )

        self.publisher_coordinates = self.create_publisher(
            PointStamped, "/coordinates", 10
        )

        self.publisher_target_label = self.create_publisher(String, "/target_label", 10)
        self.publisher_yolo_model_mode = self.create_publisher(String, "/yolo/model_mode", 10)

        self.crane_state_publisher = self.create_publisher(String, "crane_state", 10)

        self.publisher_confirmed_path = self.create_publisher(
            Path, "/confirmed_initial_plan", 10
        )

        self.publisher_target_marker = self.create_publisher(
            Marker, "/selected_target_marker", 10
        )

        # 創清除 costmap Service
        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap, "/global_costmap/clear"
        )
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear"
        )

        self.publisher_received_global_plan = self.create_publisher(
            Path, "/received_global_plan", 10
        )
        self.publisher_plan = self.create_publisher(Path, "/plan", 10)

        self.clear_global_costmap_client = self.create_client(
            ClearEntireCostmap, "/global_costmap/clear"
        )
        self.clear_local_costmap_client = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear"
        )

        self.navigate_to_pose_action_client = ActionClient(
            self, NavigateToPose, "/navigate_to_pose"
        )

        # ======== 在 __init__ 裡面新增 ========
        # 訂閱 YOLO 算出的目標 3D 位置 Marker
        self.latest_yolo_marker = None
        self.subscriber_yolo_marker = self.create_subscription(
            Marker, "/yolo/target_marker", self.yolo_target_marker_callback, 10
        )

        self.cv_image = None
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 1
        )
        
        # 發布手臂關節視覺化線條
        self.publisher_arm_visual = self.create_publisher(
            Marker, "/arm_visual_lines", 10
        )
        
        self.subscriber_clicked_point = self.create_subscription(
            PointStamped, "/clicked_point", self.clicked_point_callback, 10
        )

        self.marker_pub = self.create_publisher(Marker, "/clicked_point_marker", 1)
    
    def image_callback(self, msg):
        """接收影像並進行物體檢測"""
        # 將 ROS 影像消息轉換為 OpenCV 格式
        try:
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

    # 新增 callback
    def clicked_point_callback(self, msg):
        # 為了偷懶，我們直接把它偽裝成 YOLO marker 塞給系統
        mock_marker = Marker()
        mock_marker.header = msg.header
        mock_marker.header.stamp = self.get_clock().now().to_msg()
        mock_marker.ns = 'yolo_target'
        mock_marker.id = 0
        mock_marker.type = Marker.SPHERE
        mock_marker.action = Marker.ADD
        mock_marker.pose.position = msg.point
        mock_marker.pose.orientation.w = 1.0
        mock_marker.scale.x = 0.08  # 網球大小約 8 公分
        mock_marker.scale.y = 0.08
        mock_marker.scale.z = 0.08
        mock_marker.color.a = 1.0   # 不透明度
        mock_marker.color.r = 0.8   # 螢光黃/綠色
        mock_marker.color.g = 1.0
        mock_marker.color.b = 0.0
        self.latest_yolo_marker = mock_marker
        self.marker_pub.publish(mock_marker)

    # ======== 在 class 內新增這兩個函式 ========
    def yolo_target_marker_callback(self, msg):
        self.latest_yolo_marker = msg

    def publish_arm_visual_lines(self, marker_msg):
        self.publisher_arm_visual.publish(marker_msg)

    def clear_received_global_plan(self):
        """
        清空 /received_global_plan 话题
        """
        empty_path = Path()
        empty_path.header.frame_id = "map"
        self.publisher_received_global_plan.publish(empty_path)
        self.get_logger().info("Published empty Path to /received_global_plan")

    def clear_plan(self):
        """
        清空 /plan 话题
        """
        empty_path = Path()
        empty_path.header.frame_id = "map"
        self.publisher_plan.publish(empty_path)
        self.get_logger().info("Published empty Path to /plan")

    def reset_nav2(self):
        """
        clear plan
        """
        self.clear_received_global_plan()
        self.clear_plan()
        self.get_logger().info("Nav2 Reset Completed")

    # amcl_pose callback and get_latest_amcl_pose
    def subscriber_amcl_callback(self, msg):
        self.latest_amcl_pose = msg

    def subscriber_slam_pose_callback(self, msg):
        self.latest_slam_pose = msg

    def subscriber_odom_callback(self, msg):
        self.latest_odom = msg

    def get_latest_amcl_pose(self):
        if self.latest_amcl_pose is not None:
            self._debug_log_once_per("pose_source", "Using /amcl_pose as robot pose", 5.0)
            return self.latest_amcl_pose
        if self.latest_slam_pose is not None:
            self._debug_log_once_per("pose_source", "Using SLAM /pose as robot pose", 5.0)
            return self.latest_slam_pose
        if self.latest_odom is not None:
            self._debug_log_once_per("pose_source", "Using /odom as robot pose fallback", 5.0)
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header = self.latest_odom.header
            pose_msg.pose = self.latest_odom.pose
            return pose_msg
        self.get_logger().warn("No AMCL, SLAM /pose, or odom pose data received yet.")
        return None

    def _debug_log_once_per(self, key, message, interval_seconds):
        now = time.monotonic()
        last = self._last_debug_log.get(key, 0.0)
        if now - last >= interval_seconds:
            self.get_logger().info(f"[debug] {message}")
            self._last_debug_log[key] = now

    # goal callback and get_latest_goal
    def subscriber_goal_callback(self, msg):
        self.latest_goal_pose = msg.pose
        position = msg.pose.position
        target = [position.x, position.y, position.z]
        self.target_pose = target

    def get_goal_pose(self):
        """提供給 nav_processing 索取完整的目標姿態"""
        return self.latest_goal_pose

    def get_latest_goal(self):
        if self.target_pose is None:
            self.get_logger().warn("No goal pose data received yet.")
        return self.target_pose

    # lidar callback and get_latest_lidar
    def subscriber_lidar_callback(self, msg):
        self.latest_lidar = msg

    def get_latest_lidar(self):
        if self.latest_lidar is None:
            self.get_logger().warn("No Lidar data received yet.")
        return self.latest_lidar

    # Compatibility aliases for TaskController safety layer.
    # /scan is currently stored as latest_lidar in this project.
    def get_latest_scan(self):
        return self.latest_lidar

    # Same /scan data, alternate name for readability.
    def get_latest_laser_scan(self):
        return self.latest_lidar

    # Direct /odom getter for future stuck/debug logic.
    def get_latest_odom(self):
        return self.latest_odom
        
    # received_global_plan callback and get_latest_received_global_plan
    def received_global_plan_callback(self, msg):
        self.latest_received_global_plan = msg

    def map_callback(self, msg):
        self.latest_map = msg
        origin = msg.info.origin.position
        self._debug_log_once_per(
            "map",
            f"received /map width={msg.info.width} height={msg.info.height} "
            f"resolution={msg.info.resolution} origin=({origin.x:.2f},{origin.y:.2f})",
            5.0,
        )

    def get_latest_map(self):
        return self.latest_map

    def get_latest_received_global_plan(self):
        if self.latest_received_global_plan is None:
            self.get_logger().warn("No received global plan data received yet.")
            return None
        return self.latest_received_global_plan
    
    def subscriber_cmd_vel_callback(self, msg):
        self.latest_cmd_vel = msg

    def get_latest_cmd_vel(self):
        return self.latest_cmd_vel

    # 4. 新增一個「直接發布數值」的方法 (繞過 ACTION_MAPPINGS)
    def publish_raw_car_control(self, velocities, publish_rear=True, publish_front=True):
        """
        直接發布四輪速度，不透過字串對應表。
        velocities 格式預期為: [rear_left, rear_right, front_left, front_right]
        """
        msg = Float32MultiArray()
        
        if publish_rear:
            msg.data = [float(velocities[0]), float(velocities[1])]
            self.publisher_rear.publish(msg)
            
        if publish_front:
            msg.data = [float(velocities[2]), float(velocities[3])]
            self.publisher_forward.publish(msg)

    def publish_car_control(self, action_key, publish_rear=True, publish_front=True):
        msg = Float32MultiArray()
        if action_key not in ACTION_MAPPINGS:
            self.get_logger().warn(f"Unknown car action '{action_key}', publishing STOP instead.")
            action_key = "STOP"
            if action_key not in ACTION_MAPPINGS:
                return
        velocities = ACTION_MAPPINGS[action_key]
        self._debug_log_once_per(
            "car_action",
            f"Publishing car action={action_key}, velocities={velocities}",
            1.0,
        )
        self._vel1, self._vel2, self._vel3, self._vel4 = velocities
        msg.data = [self._vel1, self._vel2]
        if publish_rear == True:
            self.publisher_rear.publish(msg)
        msg.data = [self._vel3, self._vel4]
        if publish_front == True:
            self.publisher_forward.publish(msg)
        self.publish_cmd_vel_from_action(action_key)

    def publish_cmd_vel_from_action(self, action_key):
        twist = Twist()
        if action_key in (
            "FORWARD",
            "FORWARD_SLOW",
            "FORWARD_VERY_SLOW",
            "FORWARD_BRIDGE",
            "FORWARD_BRIDGE_LEFT",
            "FORWARD_BRIDGE_RIGHT",
        ):
            if action_key == "FORWARD_VERY_SLOW":
                twist.linear.x = 0.14
            elif action_key == "FORWARD_SLOW":
                twist.linear.x = 0.18
            elif action_key in ("FORWARD_BRIDGE", "FORWARD_BRIDGE_LEFT", "FORWARD_BRIDGE_RIGHT"):
                twist.linear.x = 0.24
                if action_key == "FORWARD_BRIDGE_LEFT":
                    twist.angular.z = 0.25
                elif action_key == "FORWARD_BRIDGE_RIGHT":
                    twist.angular.z = -0.25
            else:
                twist.linear.x = 0.3
        elif action_key in ("BACKWARD", "BACKWARD_SLOW"):
            twist.linear.x = -0.18 if action_key == "BACKWARD_SLOW" else -0.3
        elif action_key in (
            "CLOCKWISE_ROTATION",
            "CLOCKWISE_ROTATION_SLOW",
            "CLOCKWISE_ROTATION_MEDIAN",
            "CLOCKWISE_ROTATION_FINE",
        ):
            twist.angular.z = -0.25 if action_key == "CLOCKWISE_ROTATION_FINE" else (-0.5 if action_key == "CLOCKWISE_ROTATION_SLOW" else -0.9)
        elif action_key in (
            "COUNTERCLOCKWISE_ROTATION",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_MEDIAN",
            "COUNTERCLOCKWISE_ROTATION_FINE",
        ):
            twist.angular.z = 0.25 if action_key == "COUNTERCLOCKWISE_ROTATION_FINE" else (0.5 if action_key == "COUNTERCLOCKWISE_ROTATION_SLOW" else 0.9)
        elif action_key == "LEFT_FRONT":
            twist.linear.x = 0.15
            twist.angular.z = 0.4
        elif action_key == "RIGHT_FRONT":
            twist.linear.x = 0.15
            twist.angular.z = -0.4
        elif action_key == "RIGHT_FRONT_STRONG":
            twist.linear.x = 0.22
            twist.angular.z = -0.55
        elif action_key == "STOP":
            pass
        else:
            return
        self.publisher_cmd_vel.publish(twist)

    # publish goal_pose
    def publish_goal_pose(self, goal):
        goal_pose = PoseStamped()
        goal_pose.header = Header()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = "map"
        goal_pose.pose.position.x = goal[0]
        goal_pose.pose.position.y = goal[1]
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0
        self.publisher_goal_pose.publish(goal_pose)

    def publish_initial_pose(self, x, y, yaw):
        initial_pose = PoseWithCovarianceStamped()
        initial_pose.header = Header()
        initial_pose.header.stamp = self.get_clock().now().to_msg()
        initial_pose.header.frame_id = "map"
        initial_pose.pose.pose.position.x = float(x)
        initial_pose.pose.pose.position.y = float(y)
        initial_pose.pose.pose.position.z = 0.0

        half_yaw = float(yaw) * 0.5
        initial_pose.pose.pose.orientation.z = math.sin(half_yaw)
        initial_pose.pose.pose.orientation.w = math.cos(half_yaw)

        # AMCL common initial covariance: some x/y/yaw uncertainty, other axes unused.
        initial_pose.pose.covariance[0] = 0.25
        initial_pose.pose.covariance[7] = 0.25
        initial_pose.pose.covariance[35] = 0.06853891945200942

        self.publisher_initial_pose.publish(initial_pose)
        self.get_logger().info(
            f"Published /initialpose x={float(x):.3f} y={float(y):.3f} yaw={float(yaw):.3f}"
        )

    # publish robot arm angle
    def publish_robot_arm_angle(self, angle):
        joint_trajectory_point = JointTrajectoryPoint()
        joint_trajectory_point.positions = angle
        joint_trajectory_point.velocities = [0.0] * len(angle)
        self.publisher_joint_trajectory.publish(joint_trajectory_point)

    def publish_coordinates(self, x, y, z, frame_id="map"):
        coordinate_msg = PointStamped()
        coordinate_msg.header.stamp = self.get_clock().now().to_msg()
        coordinate_msg.header.frame_id = frame_id
        coordinate_msg.point.x = x
        coordinate_msg.point.y = y
        coordinate_msg.point.z = z
        self.publisher_coordinates.publish(coordinate_msg)

    def mediapipe_data_callback(self, msg):
        self.latest_mediapipe_data = msg

    def get_latest_mediapipe_data(self):
        if self.latest_mediapipe_data is None:
            self.get_logger().warn("No Mediapipe data received yet.")
            return None
        return self.latest_mediapipe_data

    def yolo_target_info_callback(self, msg):
        self.latest_yolo_target_info = msg
        self.latest_data["target_info"] = msg

    def get_latest_yolo_target_info(self):
        if self.latest_yolo_target_info is None:
            return None
        return self.latest_yolo_target_info

    def yolo_bridge_info_callback(self, msg):
        self.latest_yolo_bridge_info = msg
        self.latest_data["bridge_info"] = msg

    def get_latest_yolo_bridge_info(self):
        return self.latest_yolo_bridge_info

    def yolo_path_info_callback(self, msg):
        self.latest_yolo_path_info = msg
        self.latest_data["path_info"] = msg

    def get_latest_yolo_path_info(self):
        return self.latest_yolo_path_info

    def yolo_bridge_entry_info_callback(self, msg):
        self.latest_yolo_bridge_entry_info = msg
        self.latest_data["bridge_entry_info"] = msg

    def get_latest_yolo_bridge_entry_info(self):
        return self.latest_yolo_bridge_entry_info

    def yolo_pre_bridge_goal_callback(self, msg):
        self.latest_yolo_pre_bridge_goal = msg
        self.latest_data["pre_bridge_goal"] = msg

    def get_latest_yolo_pre_bridge_goal(self):
        return self.latest_yolo_pre_bridge_goal

    def get_latest_data(self, key):
        return self.latest_data.get(key)

    def camera_x_multi_depth_callback(self, msg):
        self.latest_camera_x_multi_depth = msg
        self.latest_data["camera_x_multi_depth"] = msg

    def get_latest_camera_x_multi_depth(self):
        if self.latest_camera_x_multi_depth is None:
            return None
        return self.latest_camera_x_multi_depth

    # YOLO coordinates callback
    def yolo_detection_position_callback(self, msg):
        """Callback to receive YOLO detected object coordinates."""
        self.latest_yolo_coordinates = msg

    def get_latest_yolo_detection_position(self):
        """Getter for the latest YOLO detected object coordinates."""
        if self.latest_yolo_coordinates is None:
            return None
        return self.latest_yolo_coordinates

    def yolo_target_map_position_callback(self, msg):
        self.latest_yolo_target_map_position = msg
        self.latest_data["target_map_position"] = msg

    def get_latest_yolo_target_map_position(self):
        return self.latest_yolo_target_map_position

    def yolo_detection_offset_callback(self, msg):
        try:
            offsets = orjson.loads(msg.data)
            if not offsets:
                self.latest_yolo_offset = None
                return
            offset = offsets[0].get("offset_flu")
            if not offset or len(offset) < 3:
                self.latest_yolo_offset = None
                return
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "camera_optical_frame"
            point_msg.point.x = float(offset[0])
            point_msg.point.y = float(offset[1])
            point_msg.point.z = float(offset[2])
            self.latest_yolo_offset = point_msg
        except Exception as exc:
            self._debug_log_once_per(
                "yolo_offset_parse",
                f"Could not parse /yolo/object/offset: {exc}",
                2.0,
            )

    def get_latest_yolo_detection_offset(self):
        if self.latest_yolo_offset is None:
            return None
        return self.latest_yolo_offset

    def publish_target_label(self, label):
        target_label_msg = String()
        target_label_msg.data = label
        self.publisher_target_label.publish(target_label_msg)

    def publish_yolo_model_mode(self, mode):
        mode_msg = String()
        mode_msg.data = mode
        self.publisher_yolo_model_mode.publish(mode_msg)

    # 天車
    def publish_crane_state(self, state):
        control_signal = {"type": "crane", "data": dict(crane_state=state)}
        crane_state_msg = String()
        crane_state_msg.data = orjson.dumps(control_signal).decode()
        self.crane_state_publisher.publish(crane_state_msg)

    def yolo_detection_status_callback(self, msg):
        self.latest_yolo_detection_status = msg

    def get_latest_yolo_detection_status(self):
        if self.latest_yolo_detection_status is None:
            return None
        return self.latest_yolo_detection_status

    def imu_data_callback(self, msg):
        self.latest_imu_data = msg

    def get_latest_imu_data(self):
        if self.latest_imu_data is None:
            return None
        return self.latest_imu_data

    def publish_confirmed_initial_plan(self, path_msg: Path):
        """
        確認路徑使用
        """
        self.publisher_confirmed_path.publish(path_msg)

    def publish_selected_target_marker(self, x, y, z=0.0):
        """
        在 foxglove 畫紅點
        """
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = 0.2  # 球體大小
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0  # 透明度
        marker.color.r = 1.0  # 顏色
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.publisher_target_marker.publish(marker)
