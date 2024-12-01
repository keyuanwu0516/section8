#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from asl_tb3_lib.control import BaseController
from asl_tb3_msgs.msg import TurtleBotControl
from std_msgs.msg import Bool

class PerceptionController(BaseController):
    def __init__(self):
        super().__init__('perception_controller')
        self.declare_parameter('active', True)
        self._last_stop_time = None
        self._ignore_detection_until = None
        self._is_stopping = False

        self._detector_sub = self.create_subscription(
            Bool,
            '/detector_bool',
            self._detector_callback,
            10
        )

    @property
    def active(self):
        return self.get_parameter('active').get_parameter_value().bool_value

    def _detector_callback(self, msg):
        current_time = self.get_clock().now().nanoseconds / 1e9

        if self._ignore_detection_until is not None and current_time < self._ignore_detection_until:
            return

        if msg.data and not self._is_stopping:
            self.set_parameters([rclpy.Parameter('active', value=False)])
            self._last_stop_time = current_time
            self._is_stopping = True

    def compute_control(self):
        control_msg = TurtleBotControl()

        if self.active:
            control_msg.v = 0.0
            control_msg.omega = 0.5
        else:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self._last_stop_time is None:
                self._last_stop_time = current_time

            if current_time - self._last_stop_time < 5.0:
                control_msg.v = 0.0
                control_msg.omega = 0.0
            else:
                self.set_parameters([rclpy.Parameter('active', value=True)])
                self._last_stop_time = None
                self._ignore_detection_until = current_time + 3.0
                self._is_stopping = False
                control_msg.v = 0.0
                control_msg.omega = 0.5

        return control_msg

def main(args=None):
    rclpy.init(args=args)
    controller = PerceptionController()
    rclpy.spin(controller)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
