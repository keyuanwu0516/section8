#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool
from asl_tb3_lib.grids import StochOccupancyGrid2D
from asl_tb3_msgs.msg import TurtleBotState
import numpy as np
from scipy.signal import convolve2d
import time


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explore')

        # Subscribers
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(TurtleBotState, '/state', self.state_callback, 10)
        self.create_subscription(Bool, '/nav_success', self.nav_success_callback, 10)
        self.create_subscription(Bool, '/detector_bool', self.detector_callback, 10)

        # Publishers
        self.target_pub = self.create_publisher(TurtleBotState, '/cmd_nav', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parameters and attributes
        self.occupancy = None
        self.current_state = None
        self.current_goal = None
        self.is_navigating = False
        self.exploration_complete = False
        self.visited_frontiers = set()
        self.stop_detected = False 

        # Configurable parameters
        self.window_size = 13
        self.min_unknown_ratio = 0.2
        self.min_unoccupied_ratio = 0.3

        self.get_logger().info("FrontierExplorer initialized and running.")

    def map_callback(self, msg):
        """Update the occupancy grid when the map is received."""
        grid_probs = np.array(msg.data, dtype=np.float32).reshape(msg.info.height, msg.info.width)
        if np.all(grid_probs == -1):  
            self.get_logger().warn("Map is completely unknown. Waiting for updates.")
            return

        grid_probs = np.where(grid_probs == -1, -1, grid_probs / 100.0)  
        self.occupancy = StochOccupancyGrid2D(
            resolution=msg.info.resolution,
            size_xy=np.array([msg.info.width, msg.info.height]),
            origin_xy=np.array([msg.info.origin.position.x, msg.info.origin.position.y]),
            window_size=self.window_size,
            probs=grid_probs,
        )

        self.get_logger().info("Map updated.")
        if self.current_state is not None:
            self.trigger_exploration()

    def state_callback(self, msg):
        """Update the robot's current state."""
        self.current_state = np.array([msg.x, msg.y, msg.theta])
        if self.occupancy and not self.exploration_complete and not self.stop_detected:
            self.trigger_exploration()

    def nav_success_callback(self, msg):
        """Handle navigation success or failure."""
        if msg.data:  
            self.get_logger().info("Navigation succeeded. Continuing exploration...")
        else:  
            self.get_logger().warn("Navigation failed. Replanning...")
        self.is_navigating = False
        if not self.exploration_complete and not self.stop_detected:
            self.trigger_exploration()

    def detector_callback(self, msg):
        """Handle stop sign detection."""
        if msg.data:  
            self.get_logger().info("Stop sign detected! Halting the robot for 5 seconds.")
            self.stop_robot()
            self.stop_detected = True
            time.sleep(5)  
            self.stop_detected = False
            self.get_logger().info("Resuming exploration.")

    def trigger_exploration(self):
        """Identify and navigate to the nearest unvisited frontier."""
        if self.exploration_complete or self.is_navigating or self.stop_detected:
            self.get_logger().info("Exploration already complete, in progress, or halted.")
            return

        if self.occupancy is None or self.current_state is None:
            self.get_logger().warn("Cannot explore: missing map or state data.")
            return

        frontier_states, distances = self.explore_frontiers()
        if frontier_states is None or len(frontier_states) == 0:
            self.get_logger().info("Exploration complete: No more frontiers found.")
            self.exploration_complete = True
            self.stop_robot()
            return


        unvisited_indices = [i for i, f in enumerate(frontier_states) if tuple(f) not in self.visited_frontiers]
        if not unvisited_indices:
            self.get_logger().info("No unvisited frontiers left. Exploration complete.")
            self.exploration_complete = True
            self.stop_robot()
            return

        nearest_index = unvisited_indices[np.argmin([distances[i] for i in unvisited_indices])]
        self.current_goal = frontier_states[nearest_index]
        self.visited_frontiers.add(tuple(self.current_goal))

        self.get_logger().info(f"Navigating to frontier: {self.current_goal}")
        self.navigate_to_frontier(self.current_goal)

    def explore_frontiers(self):
        """Identify frontier cells and return potential exploration targets."""
        if self.occupancy is None or self.current_state is None:
            return None, None

        kernel = np.ones((self.window_size, self.window_size))
        probs = self.occupancy.probs


        unknown_mask = (probs == -1).astype(float)
        occupied_mask = (probs >= 0.5).astype(float)
        unoccupied_mask = ((probs >= 0) & (probs < 0.5)).astype(float)


        unknown_count = convolve2d(unknown_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        occupied_count = convolve2d(occupied_mask, kernel, mode='same', boundary='fill', fillvalue=0)
        unoccupied_count = convolve2d(unoccupied_mask, kernel, mode='same', boundary='fill', fillvalue=0)


        total_cells = self.window_size * self.window_size


        valid_frontier_mask = (
            (unknown_count / total_cells >= self.min_unknown_ratio) &
            (occupied_count == 0) &
            (unoccupied_count / total_cells >= self.min_unoccupied_ratio)
        )

        frontier_indices = np.argwhere(valid_frontier_mask)
        if len(frontier_indices) == 0:
            return None, None

        frontier_states = np.array([self.occupancy.grid2state(idx[::-1]) for idx in frontier_indices])
        distances = np.linalg.norm(frontier_states - self.current_state[:2], axis=1)

        return frontier_states, distances

    def navigate_to_frontier(self, frontier):
        """Navigate the robot to a frontier."""
        delta_x = frontier[0] - self.current_state[0]
        delta_y = frontier[1] - self.current_state[1]
        distance = np.sqrt(delta_x**2 + delta_y**2)

        if distance < 0.1:  
            self.stop_robot()
            self.get_logger().info("Reached frontier.")
            return

        angle_to_frontier = np.arctan2(delta_y, delta_x)
        angle_diff = np.arctan2(np.sin(angle_to_frontier - self.current_state[2]),
                                np.cos(angle_to_frontier - self.current_state[2]))


        twist = Twist()
        twist.linear.x = min(0.1, distance)  
        twist.angular.z = 0.5 * angle_diff  
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"Published velocity: linear={twist.linear.x}, angular={twist.angular.z}")

    def stop_robot(self):
        """Stop the robot."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info("Robot stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
