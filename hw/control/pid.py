import numpy as np
from control_loop import ControlLoop
import sys
sys.path.append('../../duckietown-sim/')
from gym_duckietown.envs import DuckietownEnv

'''
Compute the angle of the line between a position and a goal.
Input:
    goal: [x,y] vector
    pos: [x,y] vector
Output:
    angle: 0-2pi
Might be useful for PID error terms.
'''
def angle_to_goal(goal, cur_pos):
    vec_to_goal = goal - cur_pos
    angle = np.arctan2(-vec_to_goal[1], vec_to_goal[0]) % (2*np.pi)
    if angle > np.pi:
        return angle - 2*np.pi   
    return angle

'''
ControlLoop PID Implementation
Use PID to compute actions to reach the next waypoint.
State: [x, y, heading angle] (numpy array)
Action: [velocity, steering angle]
'''
class PID(ControlLoop):

    def __init__(self, last_error1, last_error2, total_error1, total_error2, waypoints, dt, threshold=0.1, debug=False): # Feel free to add more arguments
        ControlLoop.__init__(self, waypoints=waypoints, dt=dt, threshold=threshold, debug=debug)

        ######################################
        #       Initialize PID constants     #
        ######################################
        self.last_error1 = 0
        self.last_error2 = 0
        self.total_error1 = 0
        self.total_error2 = 0

    def distance(self, x, y):
        return np.sqrt(np.square(y[0] - x[0]) + np.square(y[1] - x[1]))

    '''
    Use 2 PID loops (one for throttle, one for steering)
    to compute an action given a state and a goal
      Input: state = [x, y, heading angle]
      Access to self.cur_waypoint = [target_x, target_y]
      Output: action = [throttle, steering]
    Refer to PID_demo in demos/week 8 if you want to reference an example PID loop
    '''
    def calc_action(self, state):
        cur_pos, cur_angle = state[:2], state[2]

        if cur_angle > np.pi:
            cur_angle = cur_angle - 2*np.pi

        angle = 0
        derror1 = 0
        derror2 = 0

        action1 = 0
        action2 = 0

        kp2 = 0.5
        kd2 = 0.05
        ki2 = 0
        kp1 = 1 #1 
        kd1 = 0.8 #0.8
        ki1 = 0

        if self.reached_waypoint(state):
            self.next_waypoint();
        ##############################################################
        elif self.last_error2 != 0 and abs(self.last_error2) < 0.005:
            error1 = self.distance(cur_pos, self.cur_waypoint)
            derror1 = (error1 - self.last_error1)/self.dt
            self.total_error1 += error1 * self.dt

            derror1 = max(min(derror1, 1), -1)
            
            action1 = kp1 * error1 + kd1 * derror1 + ki1 * self.total_error1

            self.last_error1 = error1
        else:
            angle = angle_to_goal(self.cur_waypoint, cur_pos)

            error2 = angle - cur_angle 
            derror2 = (error2 - self.last_error2)/self.dt
            self.total_error2 += error2 * self.dt

            derror2 = max(min(derror2, 1), -1)

            action2 = kp2 * error2 + kd2 * derror2 + ki2 * self.total_error2

            self.last_error2 = error2
        ###############################################################
        ######################################
        #       Implement PID Algorithm      #
        ######################################
        ''''
        Calculate error:
            2 error terms corresponding to the 2 action dimensions
            the first should be minimized by setting velocity, options include 
              - Distance to next waypoint
              - Just setting a constant velocity instead of using PID
            the second should be minimized by setting steering, options include 
              - Crosstrack error (shortest distance to the path connecting the last and next waypoints)
              - Heading error (difference between current heading and angle to goal)
        Compute PID terms:
            Proportional - constant * error
            Integral - integrate the error over time (by tracking the total error)
            Derivative - find the emperical derivative of error (by tracking the last error)
        Compute action:
            For each error term, compute a weighted sum of the its PID terms.
            Return action = [velocity, steering angle]
        Note: Keeping error, constants, and action as 2-long np vectors will make code neat!
        Make helper functions or instance variables as needed
        '''
        """
        error1 = self.distance(cur_pos, self.cur_waypoint)
        derror1 = (error1 - self.last_error1)/self.dt
        self.total_error1 += error1 * self.dt
        
        angle = angle_to_goal(self.cur_waypoint, cur_pos)
            
        error2 = angle - cur_angle 
        derror2 = (error2 - self.last_error2)/self.dt
        self.total_error2 += error2 * self.dt

        derror1 = max(min(derror1, 1), -1)
        derror2 = max(min(derror2, 1), -1)
            
        kp1 = 1 #1 
        kd1 = 0.8 #0.8
        ki1 = 0
        action1 = kp1 * error1 + kd1 * derror1 + ki1 * self.total_error1

        kp2 = 0.5
        kd2 = 0.05
        ki2 = 0
        action2 = kp2 * error2 + kd2 * derror2 + ki2 * self.total_error2

        self.last_error1 = error1
        self.last_error2 = error2
        """
        if self.debug:
            print("State: {}, Action: {}, Errors: {}".format([cur_angle, angle], self.cur_waypoint, [kp2*self.last_error2, kd2*derror2, ki2 * self.total_error2]))
            #print("State: {}, Error: {}".format(state, [error1, error2]))
        return [action1/4, action2]

    def set_waypoint(self, waypoint):
        ControlLoop.set_waypoint(self, waypoint)

        ######################################
        #           Reset PID Loop           #
        ######################################
        '''
        Reset PID terms each time a waypoint is set
        If not reset, derivative term will spike
        '''
        self.last_error1 = 0
        self.last_error2 = 0
        self.total_error1 = 0
        self.total_error2 = 0

    # Make more methods if you wish!

if __name__ == "__main__":
    # Example trajectory, as would be generated by path planning
    trajectory = [np.array([2., 1.]), np.array([2., 3.]), np.array([1., 3.]), np.array([1.,1.])]

    env = DuckietownEnv(map_name='udem1', user_tile_start=(1, 1), init_angle=0, domain_rand=False)
    env.reset()
    env.render(top_down=True)

    # TODO: Pass any args you need
    
    control_loop = PID(0, 0, 0, 0, waypoints=trajectory, dt=env.delta_time, threshold=0.1, debug=True)

    while not control_loop.is_finished:
        cur_pos = [env.cur_pos[0], env.cur_pos[2]]
        cur_angle = env.cur_angle % (2*np.pi)
        state = cur_pos + [cur_angle]

        action = control_loop.get_action(state)

        env.step(action)
        env.render(top_down=True)

    print("FINISHED")


######################################
#           CHECKOFF                 #
######################################
'''
Send us a screen recording of your PID control loop in action, along with this file (pid.py).
Also send any changes to other files you used, including your Q and R matrices in lqr.py.

Since iLQR is very computationally intensive and as such, difficult to tune in this highly nonlinear environment,
you will not be required to fully follow the trajectory. You should however at least be able see the car moving towards the first waypoint.
4-6 sentences about your process for choosing Q and R and any insights about the algorithm you learned by exploring the code will suffice.
If you do get it running, we'd love to see!
'''
