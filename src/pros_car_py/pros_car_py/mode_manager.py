import urwid
from pros_car_py.base_mode import BaseMode
import threading
import time
from pros_car_py.task_controller import TaskType, TaskState


class VehicleMode(BaseMode):
    def enter(self):
        text = urwid.Text("Vehicle Mode\nPress 'q' to return to main menu.")
        filler = urwid.Filler(text, valign="top")

        self.app.loop.widget = filler
        self.app.loop.unhandled_input = self.handle_input

    def handle_input(self, key):
        if key == "q":
            self.app.car_controller.manual_control(key)
            self.app.main_menu()
        else:
            self.app.car_controller.manual_control(key)


class ArmMode(BaseMode):
    submodes = ["0", "1", "2", "3", "4"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.arm_controller.manual_control(int(submode), key)

        self.show_submode_screen(
            message=f"Arm Mode: Submode {submode}\nPress 'q' to go back.", on_key=on_key
        )


class CraneMode(BaseMode):
    submodes = ["0", "1", "2", "3", "4", "5", "6", "99"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.crane_controller.manual_control(int(submode), key)

        self.show_submode_screen(
            message=f"Crane Mode: Submode {submode}\nPress 'q' to go back.",
            on_key=on_key,
        )


class AutoNavMode(BaseMode):
    submodes = ["manual_auto_nav", "target_auto_nav", "custom_nav"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.car_controller.auto_control(submode, key)
            if key == "q":
                self.app.car_controller.auto_control(submode, key=key)

        self.show_submode_screen(
            message=f"AutoNav Mode: Submode {submode}\nPress 'q' to go back.",
            on_key=on_key,
        )


class AutoArmMode(BaseMode):
    submodes = ["auto_arm_human"]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        def on_key(key):
            self.app.arm_controller.auto_control(mode=submode, key=key)
            if key == "q":
                self.app.arm_controller.auto_control(mode=submode, key=key)

        self.show_submode_screen(
            message=f"AutoArm Mode: Submode {submode}\nPress 'q' to go back.",
            on_key=on_key,
        )


class AutoTaskMode(BaseMode):
    submodes = [task.value for task in TaskType]

    def enter(self):
        self.app.horizontal_select(self.submodes, self.handle_submode_select)

    def handle_submode_select(self, submode):
        task_type = TaskType(submode)
        self.app.task_controller.start(task_type)
        self._render_task_screen(task_type)
        self._schedule_tick(task_type)

    def _render_task_screen(self, task_type):
        controller = self.app.task_controller
        message = (
            f"Automatic Task Mode: {task_type.value}\n"
            "Press 'q' to stop and return to task menu.\n\n"
            f"State: {controller.state.value}\n"
            f"Status: {controller.status_message}\n"
        )
        self.app.loop.widget = urwid.Filler(urwid.Text(message), valign="top")
        self.app.loop.unhandled_input = self._handle_input

    def _handle_input(self, key):
        if key == "q":
            self.app.task_controller.stop()
            self.enter()

    def _schedule_tick(self, task_type):
        def tick(loop, user_data):
            controller = self.app.task_controller
            if not controller.is_running():
                self._render_task_screen(task_type)
                return

            action = controller.tick()
            self.app.car_controller.update_action(action)
            self._render_task_screen(task_type)
            loop.set_alarm_in(0.1, tick)

        self.app.loop.set_alarm_in(0.1, tick)

    def exit(self):
        if self.app.task_controller.state not in (TaskState.IDLE, TaskState.DONE):
            self.app.task_controller.stop()
