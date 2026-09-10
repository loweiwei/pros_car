import heapq
import math


class GridAStarPlanner:
    """Small 2D A* planner for nav_msgs/OccupancyGrid."""

    def __init__(
        self,
        unknown_as_obstacle=True,
        occupied_threshold=50,
        inflation_radius_m=0.25,
        waypoint_spacing_m=0.25,
    ):
        self.unknown_as_obstacle = bool(unknown_as_obstacle)
        self.occupied_threshold = int(occupied_threshold)
        self.inflation_radius_m = float(inflation_radius_m)
        self.waypoint_spacing_m = float(waypoint_spacing_m)
        self.last_warnings = []
        self.last_raw_obstacle_mask = None
        self.last_inflated_obstacle_mask = None
        self.last_start_grid = None
        self.last_adjusted_start_grid = None
        self.last_goal_grid = None
        self.last_adjusted_goal_grid = None
        self.last_path_grid = None

    def plan_path(self, grid, start_world, goal_world):
        self._reset_debug()
        if grid is None or grid.info.width <= 0 or grid.info.height <= 0:
            self.last_warnings.append("no valid /map OccupancyGrid")
            return None

        start = self.world_to_grid(grid, start_world[0], start_world[1])
        goal = self.world_to_grid(grid, goal_world[0], goal_world[1])
        self.last_start_grid = start
        self.last_goal_grid = goal
        if start is None or goal is None:
            self.last_warnings.append(f"start or goal outside map start={start} goal={goal}")
            return None

        blocked = self._build_blocked_grid(grid)
        inflated = self._inflate_blocked(grid, blocked)
        self.last_raw_obstacle_mask = blocked
        self.last_inflated_obstacle_mask = inflated

        if self._is_blocked(blocked, grid.info.width, grid.info.height, start):
            self.last_warnings.append(f"start is inside raw obstacle start={start}")
            return None

        if self._is_blocked(inflated, grid.info.width, grid.info.height, start):
            adjusted_start, radius_cells = self._nearest_free_with_radius(
                inflated,
                grid.info.width,
                grid.info.height,
                start,
                max_radius=10,
            )
            if adjusted_start is None:
                self.last_warnings.append(
                    f"start is inside inflated obstacle and no nearby free cell found start={start}"
                )
                return None

            self.last_warnings.append(
                f"start adjusted out of inflated obstacle original_start={start} "
                f"adjusted_start={adjusted_start} radius_cells={radius_cells}"
            )
            self.last_adjusted_start_grid = adjusted_start
            start = adjusted_start

        if self._is_blocked(inflated, grid.info.width, grid.info.height, goal):
            self.last_warnings.append(f"goal is inside inflated obstacle goal={goal}; trying nearest free cell")
            nearest = self._nearest_free(inflated, grid.info.width, grid.info.height, goal)
            if nearest is None:
                self.last_warnings.append("no nearby free goal cell found")
                return None
            self.last_adjusted_goal_grid = nearest
            goal = nearest

        path_cells = self._astar(inflated, grid.info.width, grid.info.height, start, goal)
        if not path_cells:
            self.last_warnings.append(f"A* failed start={start} goal={goal}")
            return None
        self.last_path_grid = path_cells

        if self._path_may_be_too_close_to_wall(blocked, grid.info.width, grid.info.height, path_cells):
            self.last_warnings.append("path may be too close to wall")

        return self._cells_to_waypoints(grid, path_cells)

    def _reset_debug(self):
        self.last_warnings = []
        self.last_raw_obstacle_mask = None
        self.last_inflated_obstacle_mask = None
        self.last_start_grid = None
        self.last_adjusted_start_grid = None
        self.last_goal_grid = None
        self.last_adjusted_goal_grid = None
        self.last_path_grid = None

    def world_to_grid(self, grid, x, y):
        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return None

        origin = grid.info.origin.position
        cx = int(math.floor((float(x) - origin.x) / resolution))
        cy = int(math.floor((float(y) - origin.y) / resolution))
        if cx < 0 or cy < 0 or cx >= grid.info.width or cy >= grid.info.height:
            return None
        return cx, cy

    def grid_to_world(self, grid, cx, cy):
        resolution = float(grid.info.resolution)
        origin = grid.info.origin.position
        return [origin.x + (cx + 0.5) * resolution, origin.y + (cy + 0.5) * resolution]

    def _build_blocked_grid(self, grid):
        blocked = []
        for value in grid.data:
            if value < 0:
                blocked.append(self.unknown_as_obstacle)
            else:
                blocked.append(int(value) >= self.occupied_threshold)
        return blocked

    def _inflate_blocked(self, grid, blocked):
        width = grid.info.width
        height = grid.info.height
        resolution = float(grid.info.resolution)
        radius_cells = int(math.ceil(max(0.0, self.inflation_radius_m) / resolution))
        if radius_cells <= 0:
            return list(blocked)

        offsets = []
        radius_sq = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_sq:
                    offsets.append((dx, dy))

        inflated = list(blocked)
        for y in range(height):
            for x in range(width):
                if not blocked[self._index(x, y, width)]:
                    continue
                for dx, dy in offsets:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        inflated[self._index(nx, ny, width)] = True
        return inflated

    def _astar(self, blocked, width, height, start, goal):
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        neighbors = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct_path(came_from, current)
            closed.add(current)

            for dx, dy, cost in neighbors:
                nx = current[0] + dx
                ny = current[1] + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                neighbor = (nx, ny)
                if self._is_blocked(blocked, width, height, neighbor):
                    continue
                if dx != 0 and dy != 0:
                    if self._is_blocked(blocked, width, height, (current[0] + dx, current[1])):
                        continue
                    if self._is_blocked(blocked, width, height, (current[0], current[1] + dy)):
                        continue

                tentative = g_score[current] + cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + self._heuristic(neighbor, goal)
                heapq.heappush(open_heap, (priority, neighbor))

        return None

    def _nearest_free(self, blocked, width, height, origin):
        max_radius = max(width, height)
        result, _ = self._nearest_free_with_radius(blocked, width, height, origin, max_radius)
        return result

    def _nearest_free_with_radius(self, blocked, width, height, origin, max_radius):
        ox, oy = origin
        for radius in range(1, max_radius):
            candidates = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    nx = ox + dx
                    ny = oy + dy
                    if 0 <= nx < width and 0 <= ny < height and not self._is_blocked(blocked, width, height, (nx, ny)):
                        candidates.append((dx * dx + dy * dy, (nx, ny)))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1], radius
        return None, None

    def _path_may_be_too_close_to_wall(self, blocked, width, height, path_cells):
        for x, y in path_cells:
            for ny in range(max(0, y - 1), min(height, y + 2)):
                for nx in range(max(0, x - 1), min(width, x + 2)):
                    if blocked[self._index(nx, ny, width)]:
                        return True
        return False

    def _cells_to_waypoints(self, grid, path_cells):
        if not path_cells:
            return []

        resolution = float(grid.info.resolution)
        spacing_cells = max(1, int(round(max(self.waypoint_spacing_m, resolution) / resolution)))
        simplified = [path_cells[0]]
        last_direction = None
        last_added_index = 0

        for index in range(1, len(path_cells)):
            prev = path_cells[index - 1]
            current = path_cells[index]
            direction = (current[0] - prev[0], current[1] - prev[1])
            if direction != last_direction or index - last_added_index >= spacing_cells:
                simplified.append(current)
                last_added_index = index
                last_direction = direction

        if simplified[-1] != path_cells[-1]:
            simplified.append(path_cells[-1])

        return [self.grid_to_world(grid, x, y) for x, y in simplified]

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _heuristic(self, first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def _is_blocked(self, blocked, width, height, cell):
        x, y = cell
        if x < 0 or y < 0 or x >= width or y >= height:
            return True
        return bool(blocked[self._index(x, y, width)])

    def _index(self, x, y, width):
        return y * width + x
