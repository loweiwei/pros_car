import math
import time
from collections import deque


class FrontierExplorer:
    """Selects frontier goals from /map and drives toward them.

    This intentionally does not require /received_global_plan. In SLAM-only mode
    Nav2 may not be running, so exploration uses direct heading control toward
    the selected frontier while SLAM keeps updating /map.
    """

    FREE_THRESHOLD = 20
    OCCUPIED_THRESHOLD = 65
    MIN_FRONTIER_DISTANCE = 1.0
    MAX_FRONTIER_DISTANCE = 8.0
    GOAL_REACHED_DISTANCE = 0.6
    GOAL_TIMEOUT_SECONDS = 35.0
    MIN_CLUSTER_SIZE = 5

    def __init__(self, ros_communicator, data_processor, nav_processing):
        self.ros_communicator = ros_communicator
        self.data_processor = data_processor
        self.nav_processing = nav_processing
        self.current_goal = None
        self.goal_started_at = None
        self.blacklisted_goals = []
        self.status_message = "Frontier explorer idle."
        self._last_debug_at = 0.0

    def reset(self):
        self.current_goal = None
        self.goal_started_at = None
        self.blacklisted_goals = []
        self.nav_processing.reset_nav_process()
        self.status_message = "Frontier explorer reset."

    def _debug(self, message, interval=1.0):
        now = time.monotonic()
        if now - self._last_debug_at >= interval:
            print(f"[FrontierExplorer][debug] {message}")
            self._last_debug_at = now

    def tick(self):
        current_position = self._get_current_position()
        if current_position is None:
            self.status_message = "Waiting for robot pose before exploration."
            self._debug("no robot pose; cannot explore", 2.0)
            return "STOP"

        if self.current_goal is None:
            goal = self._select_frontier_goal(current_position)
            if goal is None:
                self.status_message = "No frontier available yet; rotating to reveal map."
                self._debug("no frontier candidate; rotating", 2.0)
                return "CLOCKWISE_ROTATION_SLOW"
            self._start_goal(goal)
            return "STOP"

        if self._distance(current_position, self.current_goal) < self.GOAL_REACHED_DISTANCE:
            self.status_message = "Frontier reached. Selecting next frontier."
            self._clear_goal()
            return "STOP"

        if self.goal_started_at and time.monotonic() - self.goal_started_at > self.GOAL_TIMEOUT_SECONDS:
            self.status_message = "Frontier timed out. Blacklisting and selecting another."
            self.blacklisted_goals.append(self.current_goal)
            self._clear_goal()
            return "STOP"

        action = self.get_action_to_goal(self.current_goal, self.GOAL_REACHED_DISTANCE)
        self.status_message = f"Exploring frontier at {self.current_goal}."
        self._debug(f"goal={self.current_goal} action={action}", 1.0)
        return action

    def get_action_to_goal(self, goal, reached_distance=0.6):
        pose = self._get_current_pose()
        if pose is None:
            self.status_message = "Waiting for robot pose before direct navigation."
            self._debug(f"no pose for direct goal={goal}", 2.0)
            return "STOP"

        position, orientation = pose
        if self._distance(position, goal) < reached_distance:
            return "STOP"

        diff_angle = self._angle_to_goal(position, orientation, goal)
        self._debug(
            f"direct_nav pos=({position[0]:.2f},{position[1]:.2f}) goal=({goal[0]:.2f},{goal[1]:.2f}) dist={self._distance(position, goal):.2f} diff_angle={diff_angle:.1f}",
            1.0,
        )
        if -20.0 <= diff_angle <= 20.0:
            return "FORWARD_SLOW"
        if diff_angle > 20.0:
            return "COUNTERCLOCKWISE_ROTATION_SLOW"
        return "CLOCKWISE_ROTATION_SLOW"

    def _start_goal(self, goal):
        self.current_goal = goal
        self.goal_started_at = time.monotonic()
        self.nav_processing.reset_nav_process()
        self.status_message = f"New frontier goal: {goal}."
        self._debug(f"selected new frontier goal={goal}", 0.0)

    def _clear_goal(self):
        self.current_goal = None
        self.goal_started_at = None
        self.nav_processing.reset_nav_process()

    def _select_frontier_goal(self, current_position):
        grid = self.ros_communicator.get_latest_map()
        if grid is None:
            self.status_message = "Waiting for /map OccupancyGrid."
            self._debug("no /map received", 2.0)
            return None

        width = grid.info.width
        height = grid.info.height
        data = grid.data
        if width == 0 or height == 0 or not data:
            return None

        frontier_cells = []
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                index = self._index(x, y, width)
                if not self._is_free(data[index]):
                    continue
                if self._has_unknown_neighbor(data, x, y, width):
                    frontier_cells.append((x, y))

        clusters = self._cluster_frontiers(frontier_cells)
        candidates = []
        for cluster in clusters:
            if len(cluster) < self.MIN_CLUSTER_SIZE:
                continue
            center_cell = self._cluster_center(cluster)
            world = self._map_to_world(grid, center_cell[0], center_cell[1])
            distance = self._distance(current_position, world)
            if distance < self.MIN_FRONTIER_DISTANCE or distance > self.MAX_FRONTIER_DISTANCE:
                continue
            if self._is_blacklisted(world):
                continue
            score = len(cluster) - distance
            candidates.append((score, world))

        if not candidates:
            self._debug(
                f"frontier_cells={len(frontier_cells)} clusters={len(clusters)} candidates=0 current_position={current_position}",
                2.0,
            )
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        self._debug(
            f"frontier_cells={len(frontier_cells)} clusters={len(clusters)} candidates={len(candidates)} best={candidates[0][1]}",
            1.0,
        )
        return candidates[0][1]

    def _cluster_frontiers(self, frontier_cells):
        frontier_set = set(frontier_cells)
        clusters = []
        while frontier_set:
            start = frontier_set.pop()
            cluster = [start]
            queue = deque([start])
            while queue:
                x, y = queue.popleft()
                for nx in (x - 1, x, x + 1):
                    for ny in (y - 1, y, y + 1):
                        if (nx, ny) in frontier_set:
                            frontier_set.remove((nx, ny))
                            queue.append((nx, ny))
                            cluster.append((nx, ny))
            clusters.append(cluster)
        return clusters

    def _cluster_center(self, cluster):
        x_sum = sum(cell[0] for cell in cluster)
        y_sum = sum(cell[1] for cell in cluster)
        return round(x_sum / len(cluster)), round(y_sum / len(cluster))

    def _map_to_world(self, grid, x, y):
        resolution = grid.info.resolution
        origin = grid.info.origin.position
        return [origin.x + (x + 0.5) * resolution, origin.y + (y + 0.5) * resolution]

    def world_to_grid(self, x, y):
        grid = self.ros_communicator.get_latest_map()
        if grid is None:
            self._debug("world_to_grid failed: no /map received", 2.0)
            return None

        resolution = grid.info.resolution
        origin = grid.info.origin.position
        if resolution <= 0.0:
            return None

        cx = int(math.floor((float(x) - origin.x) / resolution))
        cy = int(math.floor((float(y) - origin.y) / resolution))
        if cx < 0 or cy < 0 or cx >= grid.info.width or cy >= grid.info.height:
            return None

        return cx, cy

    def grid_to_world(self, cx, cy):
        grid = self.ros_communicator.get_latest_map()
        if grid is None:
            self._debug("grid_to_world failed: no /map received", 2.0)
            return None

        return self._map_to_world(grid, int(cx), int(cy))

    def _has_unknown_neighbor(self, data, x, y, width):
        for nx in (x - 1, x, x + 1):
            for ny in (y - 1, y, y + 1):
                if nx == x and ny == y:
                    continue
                if data[self._index(nx, ny, width)] == -1:
                    return True
        return False

    def _is_blacklisted(self, goal):
        return any(self._distance(goal, item) < 0.8 for item in self.blacklisted_goals)

    def _get_current_position(self):
        pose = self._get_current_pose()
        if pose is None:
            return None
        return pose[0]

    def _get_current_pose(self):
        try:
            processed = self.data_processor.get_processed_amcl_pose()
            if processed is None:
                return None
            pose, orientation = processed
            return pose[:2], orientation
        except Exception:
            return None

    def _angle_to_goal(self, position, orientation, goal):
        target_yaw = math.degrees(math.atan2(goal[1] - position[1], goal[0] - position[0]))
        current_yaw = self._yaw_from_quaternion(orientation)
        return (target_yaw - current_yaw + 180.0) % 360.0 - 180.0

    def _yaw_from_quaternion(self, quaternion):
        x, y, z, w = quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.degrees(math.atan2(siny_cosp, cosy_cosp))

    def _is_free(self, value):
        return 0 <= value <= self.FREE_THRESHOLD

    def _index(self, x, y, width):
        return y * width + x

    def _distance(self, first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])
