#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import splrep, splev
import rclpy
from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from asl_tb3_lib.grids import StochOccupancyGrid2D


class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########
        ##raise NotImplementedError("is_free not implemented")
        x_array = np.array(x)
        if not (self.statespace_lo[0] <= x_array[0] <= self.statespace_hi[0] and
                self.statespace_lo[1] <= x_array[1] <= self.statespace_hi[1]):
            return False
        return self.occupancy.is_free(x_array)
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########
        ##raise NotImplementedError
        return np.linalg.norm(np.array(x1) - np.array(x2))
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########
        ##raise NotImplementedError("get_neighbors not implemented")
        directions = [
            (0, self.resolution),  # Up
            (0, -self.resolution),  # Down
            (self.resolution, 0),  # Right
            (-self.resolution, 0),  # Left
            (self.resolution, self.resolution),  # Up-Right (NE)
            (-self.resolution, self.resolution),  # Up-Left (NW)
            (self.resolution, -self.resolution),  # Down-Right (SE)
            (-self.resolution, -self.resolution)  # Down-Left (SW)
        ]

        for direction in directions:
            neighbor = (x[0] + direction[0], x[1] + direction[1])
            neighbor = self.snap_to_grid(neighbor)  
            if self.is_free(neighbor):  # Only include neighbors that are free
                neighbors.append(neighbor)
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########
        ##raise NotImplementedError("solve not implemented")
        while len(self.open_set) > 0:
            x_current = self.find_best_est_cost_through()
            
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue  
                
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh)
                
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)  
                elif tentative_cost_to_arrive >= self.cost_to_arrive.get(x_neigh, float('inf')):
                    continue  
                
                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
    
        return False  
        ########## Code ends here ##########

class CustomNavigator(BaseNavigator):
    V_PREV_THRESH = 0.0001  

    def __init__(self):
        super().__init__()
        
        self.declare_parameter("kp", 2.0) 
        self.kpx = 1.0
        self.kpy = 1.0
        self.kdx = 0.5
        self.kdy = 0.5
        self.V_max = 0.5
        self._om_max = 1.0

        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0

        self.get_logger().info("Custom Navigator Node Initialized with Path Planning and Tracking Control")

    @property
    def kp(self) -> float:
        return self.get_parameter("kp").value

    @property
    def om_max(self):
        return self._om_max

    @om_max.setter
    def om_max(self, value):
        self._om_max = value


    def compute_heading_control(self, state: TurtleBotState, goal: TurtleBotState) -> TurtleBotControl:

        heading_error = wrap_angle(goal.theta - state.theta)
        omega = self.kp * heading_error

        control_message = TurtleBotControl()
        control_message.omega = omega
        return control_message

    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t: float) -> TurtleBotControl:

        dt = t - self.t_prev

        x_d = splev(t, plan.path_x_spline, der=0)
        y_d = splev(t, plan.path_y_spline, der=0)
        xd_d = splev(t, plan.path_x_spline, der=1)
        yd_d = splev(t, plan.path_y_spline, der=1)
        xdd_d = splev(t, plan.path_x_spline, der=2)
        ydd_d = splev(t, plan.path_y_spline, der=2)


        v_prev = max(self.V_prev, self.V_PREV_THRESH)
        if v_prev < 1e-6: 
            self.get_logger().warn("Very low velocity detected, using safety fallback")
            return self.compute_safety_fallback_control()
        xd = v_prev * np.cos(state.theta)
        yd = v_prev * np.sin(state.theta)

        u1 = xdd_d + self.kpx * (x_d - state.x) + self.kdx * (xd_d - xd)
        u2 = ydd_d + self.kpy * (y_d - state.y) + self.kdy * (yd_d - yd)

        J_inv = np.array([
            [np.cos(state.theta), np.sin(state.theta)],
            [-np.sin(state.theta) / v_prev, np.cos(state.theta) / v_prev]
        ])

        u = np.array([u1, u2])
        control_inputs = np.dot(J_inv, u)

        V_dot = control_inputs[0]
        om = control_inputs[1]

        V = self.V_prev + V_dot * dt
        V = max(V, self.V_PREV_THRESH)

        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        self.t_prev = t
        self.V_prev = V
        self.om_prev = om

        control_message = TurtleBotControl()
        control_message.v = V
        control_message.omega = om
        return control_message

    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState, occupancy: StochOccupancyGrid2D, resolution: float, horizon: float = 10.0) -> TrajectoryPlan:
        """ Compute a trajectory plan using A* and cubic spline fitting """

        # Define state space boundaries with a buffer around initial and goal positions
        min_x = state.x - horizon
        max_x = state.x + horizon
        min_y = state.y - horizon
        max_y = state.y + horizon

        astar = AStar(
            statespace_lo=(min_x, min_y),      # Adjusted lower bound
            statespace_hi=(max_x, max_y),      # Adjusted upper bound
            x_init=(state.x, state.y),
            x_goal=(goal.x, goal.y),
            occupancy=occupancy,
            resolution=0.1
        )

        if not astar.solve() or len(astar.path) < 4:
            self.get_logger().warn("Path planning failed or path is too short")
            return None

        self.V_prev = 0.0
        self.t_prev = 0.0

        path = np.asarray(astar.path)

        v_desired = 0.15
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        times = distances / v_desired
        ts = np.concatenate([[0], np.cumsum(times)])

        spline_alpha = 0.05
        path_x_spline = splrep(ts, path[:, 0], s=spline_alpha)
        path_y_spline = splrep(ts, path[:, 1], s=spline_alpha)

        return TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1]
        )

def main(args=None):
    rclpy.init(args=args)
    navigator = CustomNavigator()
    rclpy.spin(navigator)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
