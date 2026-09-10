# pros_car: Robot Control and Task Decision Layer

`pros_car` is the robot-side control module for PROS. It contains vehicle control, arm control, Nav2 processing, frontier exploration, custom A* planning, and the high-level task state machine.

## 路徑說明

本 README 中的檔案路徑都相對於 `pros_car` repository 根目錄，例如 `src/pros_car_py/pros_car_py/task_controller.py` 表示此 repo 內的 `src/pros_car_py/pros_car_py/task_controller.py`。

## Portfolio Focus

| What to review | File |
| --- | --- |
| Main entry point | [`src/pros_car_py/pros_car_py/main2.py`](./src/pros_car_py/pros_car_py/main2.py) |
| Task state machine | [`src/pros_car_py/pros_car_py/task_controller.py`](./src/pros_car_py/pros_car_py/task_controller.py) |
| Hybrid A* planner | [`src/pros_car_py/pros_car_py/hybrid_astar_planner.py`](./src/pros_car_py/pros_car_py/hybrid_astar_planner.py) |
| Grid A* fallback planner | [`src/pros_car_py/pros_car_py/grid_astar_planner.py`](./src/pros_car_py/pros_car_py/grid_astar_planner.py) |
| Navigation processing | [`src/pros_car_py/pros_car_py/nav_processing.py`](./src/pros_car_py/pros_car_py/nav_processing.py) |
| ROS topic interface | [`src/pros_car_py/pros_car_py/ros_communicator.py`](./src/pros_car_py/pros_car_py/ros_communicator.py) |
| Frontier exploration | [`src/pros_car_py/pros_car_py/frontier_explorer.py`](./src/pros_car_py/pros_car_py/frontier_explorer.py) |
| Arm and IK logic | [`src/pros_car_py/pros_car_py/arm_controller_2D.py`](./src/pros_car_py/pros_car_py/arm_controller_2D.py), [`src/pros_car_py/pros_car_py/ik_solver.py`](./src/pros_car_py/pros_car_py/ik_solver.py) |

## Key Ideas

- [`src/pros_car_py/pros_car_py/main2.py`](./src/pros_car_py/pros_car_py/main2.py) assembles the ROS communicator, data processor, Nav2 processing, car controller, arm controller, frontier explorer, and task controller.
- [`src/pros_car_py/pros_car_py/task_controller.py`](./src/pros_car_py/pros_car_py/task_controller.py) runs the autonomous demo as a non-blocking state machine, so perception, localization, map updates, timeouts, and safety checks remain active during execution.
- The planner first attempts Hybrid A* with robot-footprint collision checking, then falls back to Grid A* when needed.
- Long-distance movement uses map planning; short-distance final approach uses YOLO offset and depth feedback.

## 整體任務流程的兩個版本

以下兩個指令都是啟動整個任務控制系統，不是只執行 Task 2。差別在於 Task 2 的橋上熊回收流程是否啟用。

| 版本 | 執行指令 | 說明 |
| --- | --- | --- |
| 完整流程版 | `ros2 run pros_car_py robot_control` | 啟動完整任務系統；Task 2 設計包含上橋、橋上搜尋熊、抓取、下橋與返回起點。 |
| Demo 當天版本 | `ros2 run pros_car_py robot_control_skip_task2_bear` | 啟動 demo 當天使用的任務系統；Task 2 主要展示上橋與下橋，未完成橋上抓熊回收。 |

這兩個入口定義在 [`src/pros_car_py/setup.py`](./src/pros_car_py/setup.py)。Demo 當天若要重現展示流程，使用 `robot_control_skip_task2_bear`；若要看完整設計流程，使用 `robot_control`。

## Class Diagram

![pros_car](https://github.com/alianlbj23/pros_car/blob/main/img/pros_car.drawio.png?raw=true)

## 使用說明
## 🚀 環境初始化
1. 執行 [`car_control.sh`](./car_control.sh) 進入環境：
   ```bash
   ./car_control.sh
   ```
2. 在環境內輸入 `r` 來執行建置與設定：
   ```bash
   r  # 進行 colcon build 並執行 . ./install/setup.bash
   ```

## 🚗 車輛控制
執行以下指令來開始車輛控制：
```bash
ros2 run pros_car_py robot_control
```
執行後，畫面將會顯示控制介面。

### 🔹 車輛手動控制
| 鍵盤按鍵 | 功能描述 |
|---------|---------|
| `w` | **前進** |
| `s` | **後退** |
| `a` | **左斜走** |
| `d` | **右斜走** |
| `e` | **左自轉** |
| `r` | **右自轉** |
| `z` | **停止** |
| `q` | **回到主選單** |

## 🤖 手動機械臂控制
1. 進入機械臂控制模式後，選擇 **0~4 號關節** 來調整角度。
2. 角度調整指令：
   | 鍵盤按鍵 | 功能描述 |
   |---------|---------|
   | `i` | **增加角度** |
   | `k` | **減少角度** |
   | `q` | **回到關節選擇** |

## 📍 自動導航模式
共有 **兩種自動導航模式**：

### 1️⃣ 手動導航 (`manual_auto_nav`)
- **功能**：接收 **Foxglove** 所發送的 `/goal_pose` **座標** 來進行導航。

### 2️⃣ 目標導航 (`target_auto_nav`)
- **功能**：由 `car_controller.py` 內部自動 `publish` `/goal_pose` **座標**，進行自動導航。

📢 **注意**：在使用導航模式時，**按下 `q`** 即可立即停止車輛移動並退出導航模式。

---

# pros_car Usage Guide

## 🚀 Environment Setup
1. Enter the environment by running [`car_control.sh`](./car_control.sh):
   ```bash
   ./car_control.sh
   ```
2. Inside the environment, enter `r` to build and set up:
   ```bash
   r  # Run colcon build and source setup.bash
   ```

## 🚗 Vehicle Control
Start vehicle control by running:
```bash
ros2 run pros_car_py robot_control
```
Once started, the control interface will appear.

### 🔹 Manual Vehicle Control
| Key | Action |
|---------|---------|
| `w` | **Move forward** |
| `s` | **Move backward** |
| `a` | **Move diagonally left** |
| `d` | **Move diagonally right** |
| `e` | **Rotate left** |
| `r` | **Rotate right** |
| `z` | **Stop** |
| `q` | **Return to the main menu** |

## 🤖 Manual Arm Control
1. Enter **joint control mode**, then select a joint (0~4) to adjust its angle.
2. Use the following keys to control the joint angles:
   | Key | Action |
   |---------|---------|
   | `i` | **Increase angle** |
   | `k` | **Decrease angle** |
   | `q` | **Return to joint selection** |

## 📍 Autonomous Navigation Modes
There are **two autonomous navigation modes**:

### 1️⃣ Manual Auto Navigation (`manual_auto_nav`)
- **Function**: Receives `/goal_pose` coordinates from **Foxglove** and navigates accordingly.

### 2️⃣ Target Auto Navigation (`target_auto_nav`)
- **Function**: [`src/pros_car_py/pros_car_py/car_controller.py`](./src/pros_car_py/pros_car_py/car_controller.py) internally **publishes** `/goal_pose` coordinates for automatic navigation.

📢 **Note**: Press `q` at any time to **stop the vehicle immediately** and exit navigation mode.
