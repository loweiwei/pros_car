import heapq
import math
import os
import time
from collections import deque


class HybridAStarPlanner:
    """Experimental SE(2) lattice planner for nav_msgs/OccupancyGrid.

    State is (x_cell, y_cell, yaw_bin). The robot can rotate in place, so the
    primitive set is intentionally small: forward one step, rotate left, rotate
    right. Each generated candidate is footprint-checked before it can enter the
    open set.
    """

    def __init__(
        self,
        yaw_bins=16,
        step_m=0.20,
        yaw_step_deg=22.5,
        max_iterations=20000,
        goal_tolerance_m=0.12,
        footprint_length_m=0.45,
        footprint_width_m=0.30,
        unknown_as_obstacle=True,
        occupied_threshold=50,
        clearance_cost_weight=0.20,
        start_recovery_enabled=True,
        start_recovery_radius_m=0.15,
        start_recovery_step_m=0.05,
        start_recovery_yaw_range_deg=30.0,
        start_recovery_yaw_step_deg=15.0,
        start_allow_unknown=True,
        debug_image_enabled=False,
        debug_image_dir="/tmp/pros_task_maps",
        debug_image_interval=500,
        debug_max_images_per_plan=8,
        logger=None,
    ):
        self.yaw_bins = max(4, int(yaw_bins))
        self.step_m = float(step_m)
        self.yaw_step_deg = float(yaw_step_deg)
        self.max_iterations = int(max_iterations)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.footprint_length_m = float(footprint_length_m)
        self.footprint_width_m = float(footprint_width_m)
        self.unknown_as_obstacle = bool(unknown_as_obstacle)
        self.occupied_threshold = int(occupied_threshold)
        self.clearance_cost_weight = float(clearance_cost_weight)
        self.start_recovery_enabled = bool(start_recovery_enabled)
        self.start_recovery_radius_m = float(start_recovery_radius_m)
        self.start_recovery_step_m = float(start_recovery_step_m)
        self.start_recovery_yaw_range_deg = float(start_recovery_yaw_range_deg)
        self.start_recovery_yaw_step_deg = float(start_recovery_yaw_step_deg)
        self.start_allow_unknown = bool(start_allow_unknown)
        self.debug_image_enabled = bool(debug_image_enabled)
        self.debug_image_dir = debug_image_dir
        self.debug_image_interval = max(1, int(debug_image_interval))
        self.debug_max_images_per_plan = max(0, int(debug_max_images_per_plan))
        self.logger = logger

        self.last_warnings = []
        self.last_iterations = 0
        self.last_visited_count = 0
        self.last_path = None
        self.last_yaws = None
        self.last_debug_images = []
        self.last_collision_state = None
        self.last_collision_footprint = None
        self.last_adjusted_start_pose = None

    def plan_path(self, grid, start_pose, goal_world, context="hybrid_astar"):
        self._reset_debug()
        if grid is None or grid.info.width <= 0 or grid.info.height <= 0:
            self.last_warnings.append("no valid /map OccupancyGrid")
            return None, None

        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            self.last_warnings.append("invalid map resolution")
            return None, None

        start_cell = self.world_to_grid(grid, start_pose[0], start_pose[1])
        goal_cell = self.world_to_grid(grid, goal_world[0], goal_world[1])
        if start_cell is None or goal_cell is None:
            self.last_warnings.append(f"start or goal outside map start={start_cell} goal={goal_cell}")
            self._save_debug_image(grid, start_pose, goal_world, set(), set(), None, None, "failed_outside_map", context)
            return None, None

        start_yaw_bin = self._yaw_to_bin(float(start_pose[2]))
        start_state = (start_cell[0], start_cell[1], start_yaw_bin)

        if self._state_collides(grid, start_state):
            self.last_warnings.append(f"start footprint collides state={start_state}")
            self._log(
                "hybrid_astar start_pose_collides "
                f"original=({float(start_pose[0]):.3f},{float(start_pose[1]):.3f},{float(start_pose[2]):.1f})"
            )
            self.last_collision_state = start_state
            self.last_collision_footprint = self._footprint_corners_for_state(grid, start_state)
            adjusted_start_pose, adjusted_start_state = self._recover_start_pose(grid, start_pose)
            if adjusted_start_state is None:
                self._log(
                    "hybrid_astar start_recovery_failed "
                    f"original=({float(start_pose[0]):.3f},{float(start_pose[1]):.3f},{float(start_pose[2]):.1f})"
                )
                self._save_debug_image(grid, start_pose, goal_world, set(), set(), start_state, None, "failed_start_collision", context)
                return None, None

            self._log(
                "hybrid_astar adjusted_start_pose "
                f"original=({float(start_pose[0]):.3f},{float(start_pose[1]):.3f},{float(start_pose[2]):.1f}) "
                f"adjusted=({adjusted_start_pose[0]:.3f},{adjusted_start_pose[1]:.3f},{adjusted_start_pose[2]:.1f})"
            )
            self.last_adjusted_start_pose = adjusted_start_pose
            start_pose = adjusted_start_pose
            start_state = adjusted_start_state

        blocked = self._build_blocked_grid(grid)
        clearance = self._build_clearance_grid(grid, blocked)
        open_heap = []
        heapq.heappush(open_heap, (self._heuristic(grid, start_state, goal_world), 0.0, start_state))
        came_from = {}
        g_score = {start_state: 0.0}
        open_states = {start_state}
        closed_states = set()
        debug_images_saved = 0
        final_state = None

        for iteration in range(1, self.max_iterations + 1):
            if not open_heap:
                break

            _, current_cost, current = heapq.heappop(open_heap)
            if current in closed_states:
                continue
            open_states.discard(current)
            closed_states.add(current)
            self.last_iterations = iteration

            if self._distance_to_goal(grid, current, goal_world) <= self.goal_tolerance_m:
                final_state = current
                break

            if (
                self.debug_image_enabled
                and debug_images_saved < self.debug_max_images_per_plan
                and iteration % self.debug_image_interval == 0
            ):
                if self._save_debug_image(
                    grid, start_pose, goal_world, open_states, closed_states, current, None,
                    f"expand_{iteration}", context
                ):
                    debug_images_saved += 1

            for candidate, primitive_cost in self._neighbors(grid, current):
                if candidate in closed_states:
                    continue
                if self._state_collides(grid, candidate):
                    self.last_collision_state = candidate
                    self.last_collision_footprint = self._footprint_corners_for_state(grid, candidate)
                    continue

                step_cost = primitive_cost + self._clearance_cost(grid, clearance, candidate)
                tentative = current_cost + step_cost
                if tentative >= g_score.get(candidate, float("inf")):
                    continue

                came_from[candidate] = current
                g_score[candidate] = tentative
                priority = tentative + self._heuristic(grid, candidate, goal_world)
                heapq.heappush(open_heap, (priority, tentative, candidate))
                open_states.add(candidate)

        self.last_visited_count = len(closed_states)

        if final_state is None:
            self.last_warnings.append(
                f"Hybrid A* failed iterations={self.last_iterations} visited={self.last_visited_count}"
            )
            self._save_debug_image(grid, start_pose, goal_world, open_states, closed_states, None, None, "failed_summary", context)
            return None, None

        state_path = self._reconstruct_path(came_from, final_state)
        path, yaws = self._states_to_path(grid, state_path)
        self.last_path = path
        self.last_yaws = yaws
        self._save_debug_image(grid, start_pose, goal_world, open_states, closed_states, final_state, (path, yaws), "success_summary", context)
        return path, yaws

    def _reset_debug(self):
        self.last_warnings = []
        self.last_iterations = 0
        self.last_visited_count = 0
        self.last_path = None
        self.last_yaws = None
        self.last_debug_images = []
        self.last_collision_state = None
        self.last_collision_footprint = None
        self.last_adjusted_start_pose = None

    def _recover_start_pose(self, grid, start_pose):
        if not self.start_recovery_enabled:
            return None, None

        step = max(float(self.start_recovery_step_m), float(grid.info.resolution), 0.01)
        max_radius = max(float(self.start_recovery_radius_m), 0.0)
        yaw_step = max(float(self.start_recovery_yaw_step_deg), 1.0)
        yaw_range = max(float(self.start_recovery_yaw_range_deg), 0.0)

        radii = []
        radius = step
        while radius <= max_radius + 1e-9:
            radii.append(radius)
            radius += step

        yaw_offsets = [0.0]
        yaw = yaw_step
        while yaw <= yaw_range + 1e-9:
            yaw_offsets.extend([yaw, -yaw])
            yaw += yaw_step

        samples_per_radius = 16
        for radius in radii:
            for sample_index in range(samples_per_radius):
                angle = 2.0 * math.pi * (sample_index / samples_per_radius)
                x = float(start_pose[0]) + math.cos(angle) * radius
                y = float(start_pose[1]) + math.sin(angle) * radius
                cell = self.world_to_grid(grid, x, y)
                if cell is None:
                    continue
                for yaw_offset in yaw_offsets:
                    yaw_deg = float(start_pose[2]) + yaw_offset
                    state = (cell[0], cell[1], self._yaw_to_bin(yaw_deg))
                    if not self._state_collides(grid, state, allow_unknown=self.start_allow_unknown):
                        adjusted_yaw = self._bin_to_yaw(state[2])
                        return [self.grid_to_world(grid, cell[0], cell[1])[0], self.grid_to_world(grid, cell[0], cell[1])[1], adjusted_yaw], state

        return None, None

    def _neighbors(self, grid, state):
        x_cell, y_cell, yaw_bin = state
        yaw_deg = self._bin_to_yaw(yaw_bin)
        yaw = math.radians(yaw_deg)
        current_world = self.grid_to_world(grid, x_cell, y_cell)
        next_world = [
            current_world[0] + math.cos(yaw) * self.step_m,
            current_world[1] + math.sin(yaw) * self.step_m,
        ]
        next_cell = self.world_to_grid(grid, next_world[0], next_world[1])
        if next_cell is not None:
            yield (next_cell[0], next_cell[1], yaw_bin), max(self.step_m, float(grid.info.resolution))

        yaw_step_bins = max(1, int(round(abs(self.yaw_step_deg) / (360.0 / self.yaw_bins))))
        rotate_cost = max(0.05, abs(self.yaw_step_deg) / 90.0) * max(self.step_m, float(grid.info.resolution))
        yield (x_cell, y_cell, (yaw_bin + yaw_step_bins) % self.yaw_bins), rotate_cost
        yield (x_cell, y_cell, (yaw_bin - yaw_step_bins) % self.yaw_bins), rotate_cost

    def _state_collides(self, grid, state, allow_unknown=None):
        if allow_unknown is None:
            allow_unknown = False
        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return True

        width_cells = int(grid.info.width)
        height_cells = int(grid.info.height)
        corners = self._footprint_corners_for_state(grid, state)
        corner_cells = [self.world_to_grid(grid, corner[0], corner[1]) for corner in corners]
        if any(cell is None for cell in corner_cells):
            return True

        min_gx = min(cell[0] for cell in corner_cells)
        max_gx = max(cell[0] for cell in corner_cells)
        min_gy = min(cell[1] for cell in corner_cells)
        max_gy = max(cell[1] for cell in corner_cells)
        if min_gx < 0 or min_gy < 0 or max_gx >= width_cells or max_gy >= height_cells:
            return True

        center = self.grid_to_world(grid, state[0], state[1])
        yaw = math.radians(self._bin_to_yaw(state[2]))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        half_length = self.footprint_length_m / 2.0
        half_width = self.footprint_width_m / 2.0
        origin = grid.info.origin.position

        for gy in range(min_gy, max_gy + 1):
            cell_y = float(origin.y) + (gy + 0.5) * resolution
            for gx in range(min_gx, max_gx + 1):
                cell_x = float(origin.x) + (gx + 0.5) * resolution
                dx = cell_x - center[0]
                dy = cell_y - center[1]
                local_x = dx * cos_yaw + dy * sin_yaw
                local_y = -dx * sin_yaw + dy * cos_yaw
                if abs(local_x) > half_length or abs(local_y) > half_width:
                    continue

                value = int(grid.data[gy * width_cells + gx])
                if value < 0 and self.unknown_as_obstacle and not allow_unknown:
                    return True
                if value >= self.occupied_threshold:
                    return True

        return False

    def _footprint_corners_for_state(self, grid, state):
        center = self.grid_to_world(grid, state[0], state[1])
        yaw = math.radians(self._bin_to_yaw(state[2]))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        half_length = self.footprint_length_m / 2.0
        half_width = self.footprint_width_m / 2.0
        corners = []
        for local_x, local_y in (
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ):
            corners.append((
                center[0] + local_x * cos_yaw - local_y * sin_yaw,
                center[1] + local_x * sin_yaw + local_y * cos_yaw,
            ))
        return corners

    def _build_blocked_grid(self, grid):
        blocked = []
        for value in grid.data:
            if int(value) < 0:
                blocked.append(self.unknown_as_obstacle)
            else:
                blocked.append(int(value) >= self.occupied_threshold)
        return blocked

    def _build_clearance_grid(self, grid, blocked):
        width = int(grid.info.width)
        height = int(grid.info.height)
        distance_cells = [None] * (width * height)
        queue = deque()
        for index, is_blocked in enumerate(blocked):
            if is_blocked:
                distance_cells[index] = 0
                queue.append(index)

        while queue:
            index = queue.popleft()
            x = index % width
            y = index // width
            next_distance = distance_cells[index] + 1
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                next_index = ny * width + nx
                if distance_cells[next_index] is not None:
                    continue
                distance_cells[next_index] = next_distance
                queue.append(next_index)

        max_distance = max(width, height)
        return [float(value if value is not None else max_distance) * float(grid.info.resolution) for value in distance_cells]

    def _clearance_cost(self, grid, clearance, state):
        if self.clearance_cost_weight <= 0.0:
            return 0.0
        width = int(grid.info.width)
        index = state[1] * width + state[0]
        if index < 0 or index >= len(clearance):
            return self.clearance_cost_weight * 100.0
        clearance_m = max(float(clearance[index]), 0.0)
        return self.clearance_cost_weight / (clearance_m + float(grid.info.resolution))

    def _heuristic(self, grid, state, goal_world):
        current = self.grid_to_world(grid, state[0], state[1])
        return math.hypot(current[0] - float(goal_world[0]), current[1] - float(goal_world[1]))

    def _distance_to_goal(self, grid, state, goal_world):
        return self._heuristic(grid, state, goal_world)

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _states_to_path(self, grid, states):
        path = []
        yaws = []
        last_xy = None
        for state in states:
            world = self.grid_to_world(grid, state[0], state[1])
            yaw = self._bin_to_yaw(state[2])
            if last_xy is not None and math.hypot(world[0] - last_xy[0], world[1] - last_xy[1]) < 1e-9:
                if yaws:
                    yaws[-1] = yaw
                continue
            path.append(world)
            yaws.append(yaw)
            last_xy = world
        return path, yaws

    def world_to_grid(self, grid, x, y):
        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return None
        origin = grid.info.origin.position
        cx = int(math.floor((float(x) - float(origin.x)) / resolution))
        cy = int(math.floor((float(y) - float(origin.y)) / resolution))
        if cx < 0 or cy < 0 or cx >= int(grid.info.width) or cy >= int(grid.info.height):
            return None
        return cx, cy

    def grid_to_world(self, grid, cx, cy):
        resolution = float(grid.info.resolution)
        origin = grid.info.origin.position
        return [float(origin.x) + (int(cx) + 0.5) * resolution, float(origin.y) + (int(cy) + 0.5) * resolution]

    def _yaw_to_bin(self, yaw_deg):
        yaw = float(yaw_deg) % 360.0
        return int(round(yaw / (360.0 / self.yaw_bins))) % self.yaw_bins

    def _bin_to_yaw(self, yaw_bin):
        return (int(yaw_bin) % self.yaw_bins) * (360.0 / self.yaw_bins)

    def _save_debug_image(self, grid, start_pose, goal, open_states, closed_states, current_state, final_path, label, context):
        if not self.debug_image_enabled:
            return False
        try:
            import cv2
            import numpy as np

            os.makedirs(self.debug_image_dir, exist_ok=True)
            width = int(grid.info.width)
            height = int(grid.info.height)
            resolution = float(grid.info.resolution)
            origin = grid.info.origin.position

            data = np.array(grid.data, dtype=np.int16).reshape((height, width))
            image = np.full((height, width, 3), 180, dtype=np.uint8)
            image[data >= self.occupied_threshold] = (0, 0, 0)
            image[(data >= 0) & (data < self.occupied_threshold)] = (255, 255, 255)
            image = np.flipud(image)
            image = np.ascontiguousarray(image.copy()).astype(np.uint8)

            def cell_to_image(x_cell, y_cell):
                return int(x_cell), height - 1 - int(y_cell)

            def world_to_image(point):
                cell = self.world_to_grid(grid, point[0], point[1])
                if cell is None:
                    return None
                return cell_to_image(cell[0], cell[1])

            for state in list(closed_states)[:: max(1, len(closed_states) // 2500 or 1)]:
                cv2.circle(image, cell_to_image(state[0], state[1]), 1, (210, 210, 0), -1)
            for state in list(open_states)[:: max(1, len(open_states) // 1500 or 1)]:
                cv2.circle(image, cell_to_image(state[0], state[1]), 1, (0, 180, 255), -1)

            def draw_point(point, color, radius=4):
                image_point = world_to_image(point)
                if image_point is not None:
                    cv2.circle(image, image_point, radius, color, -1)

            def draw_polyline(points, color, thickness=1, closed=True):
                image_points = [world_to_image(point) for point in points]
                image_points = [point for point in image_points if point is not None]
                if len(image_points) >= 2:
                    pts = np.array(image_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], closed, color, thickness)

            def draw_heading(point, yaw_deg, color):
                image_point = world_to_image(point)
                if image_point is None:
                    return
                length_px = max(8, int(0.35 / max(resolution, 1e-6)))
                yaw = math.radians(yaw_deg)
                end = (int(image_point[0] + math.cos(yaw) * length_px), int(image_point[1] - math.sin(yaw) * length_px))
                cv2.arrowedLine(image, image_point, end, color, 2, tipLength=0.25)

            draw_point(start_pose, (0, 200, 0), 4)
            draw_heading(start_pose, start_pose[2], (0, 160, 0))
            draw_polyline(self._footprint_corners_world(start_pose[0], start_pose[1], start_pose[2]), (0, 160, 0), 1, True)
            draw_point(goal, (0, 0, 255), 5)

            if current_state is not None:
                current_world = self.grid_to_world(grid, current_state[0], current_state[1])
                draw_point(current_world, (255, 0, 255), 4)
                draw_heading(current_world, self._bin_to_yaw(current_state[2]), (255, 0, 255))
                draw_polyline(self._footprint_corners_for_state(grid, current_state), (255, 0, 255), 1, True)

            if self.last_collision_footprint:
                draw_polyline(self.last_collision_footprint, (0, 0, 200), 1, True)

            if final_path is not None:
                path, yaws = final_path
                image_points = [world_to_image(point) for point in path]
                image_points = [point for point in image_points if point is not None]
                if len(image_points) >= 2:
                    pts = np.array(image_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], False, (255, 0, 0), 2)
                for index, point in enumerate(path):
                    image_point = world_to_image(point)
                    if image_point is None:
                        continue
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
                sample_step = max(1, len(path) // 8)
                for index in range(0, len(path), sample_step):
                    yaw = yaws[min(index, len(yaws) - 1)] if yaws else 0.0
                    draw_heading(path[index], yaw, (255, 0, 0))
                    draw_polyline(self._footprint_corners_world(path[index][0], path[index][1], yaw), (120, 0, 180), 1, True)

            panel_lines = [
                f"hybrid_astar {label}",
                f"context={context}",
                f"start=({start_pose[0]:.2f},{start_pose[1]:.2f},{start_pose[2]:.1f}) goal=({float(goal[0]):.2f},{float(goal[1]):.2f})",
                f"iterations={self.last_iterations} visited={self.last_visited_count} open={len(open_states)} closed={len(closed_states)}",
                f"footprint=({self.footprint_length_m:.2f},{self.footprint_width_m:.2f}) yaw_bins={self.yaw_bins} step={self.step_m:.2f}",
            ]
            text_height = 24 * len(panel_lines) + 12
            panel = np.full((text_height, width, 3), 245, dtype=np.uint8)
            for i, text in enumerate(panel_lines):
                cv2.putText(panel, text[:160], (8, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            output = np.vstack([image, panel])
            safe_context = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(context))
            filename = f"{int(time.time() * 1000)}_{safe_context}_hybrid_{label}.png"
            path = os.path.join(self.debug_image_dir, filename)
            if not cv2.imwrite(path, output):
                raise RuntimeError(f"cv2.imwrite returned false path={path}")
            self.last_debug_images.append(path)
            self._log(f"hybrid_astar_debug saved path={path}")
            return True
        except Exception as exc:
            self._log(f"hybrid_astar_debug failed label={label} error={exc}")
            return False

    def _footprint_corners_world(self, x, y, yaw_deg):
        yaw = math.radians(float(yaw_deg))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        half_length = self.footprint_length_m / 2.0
        half_width = self.footprint_width_m / 2.0
        corners = []
        for local_x, local_y in (
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ):
            corners.append((
                float(x) + local_x * cos_yaw - local_y * sin_yaw,
                float(y) + local_x * sin_yaw + local_y * cos_yaw,
            ))
        return corners

    def _log(self, message):
        if self.logger is None:
            return
        try:
            self.logger(message)
        except Exception:
            pass
