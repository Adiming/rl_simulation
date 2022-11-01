import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


MAX_EPISODE_LEN = 20*100

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        # connect to pybullet using GUI mode
        p.connect(p.GUI)
        # adjust the view angle 
        p.resetDebugVisualizerCamera(cameraDistance=0.2, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0.7,-0.35,0.5])
        # p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=90, cameraPitch=-40, cameraTargetPosition=[0.8,0,0.2])
        
        # action space: target cartesian position of ee + a joint variable for both fingers
        # the range of each variable is -1 to 1
        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        # observation space: target cartesian position of ee + joints variable of each figures
        # the range of each variable is -1 to 1
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

        self.urdfRootPath=pybullet_data.getDataPath()


    # determin what will happen with each env.step(action)
    # action: target cartesian position of ee + a joint variable for both fingers
    def step(self, action):
        # for avoiding a sudden action spring in rendering
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # the gripper orientation is considered to be perpendicular to the ground
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        # dv = 0.005
        dv = 0.00
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        # fingers = action[3]
        fingers = 0.0

        # read the current Cartesian position (pybullet.getLinkState()) of the gripper
        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        # add the small variation toward the target Cartesian position
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]
        # calculating target joint variables for the robot
        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]
        # apply those joint variables
        # p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])
        p.setJointMotorControlArray(self.pandaUid, list(range(7)), p.POSITION_CONTROL, list(jointPoses))
        # p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, 0, force = 100)
        # p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, 0, force = 100)


        # stepSimulation will perform all the actions in a single forward dynamics simulation step 
        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])

        # In the step() function we also determine the reward
        # if the robot grasp the object and pick it up to a certain height (0.45) the agent gets 1 reward 
        # if state_object[2]>0.45:
        if state_object[2]<0.:
            reward = 1
            done = True
        else:
            reward = 0
            done = False

        self.step_counter += 1

        if self.step_counter > MAX_EPISODE_LEN:
            reward = 0
            done = True

        info = {'object_position': state_object}
        self.observation = state_robot + state_fingers
        return np.array(self.observation).astype(np.float32), reward, done, info

    # reset the pybullet env
    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        
        planeUid = p.loadURDF(os.path.join(self.urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        tableUid = p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        tablesqUid = p.loadURDF(os.path.join(self.urdfRootPath, "table_square/table_square.urdf"),basePosition=[0.85,0,-0.25])

        self.pandaUid = p.loadURDF(os.path.join(self.urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)

        p.setGravity(0,0,-10)
        

        # place the target platform in the scene
        platformUid = p.loadURDF(os.path.join(self.urdfRootPath, "custom_object/platform/platform.urdf"),basePosition=[0.7,0,0.4],
                            baseOrientation=[0.5,0.5,0.5,0.5],useFixedBase=True)

        # random.uniform: randomly pick a value in the given range
        # state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        # self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=state_object)

        # place the target gear in the scene
        self.objectUid = p.loadURDF(os.path.join(self.urdfRootPath, "custom_object/gear_medium/gear_medium.urdf"),basePosition=[0.7,0.2,0.4],
                            baseOrientation=[-0.707,0,0,0.707])

        # rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        rest_poses = [0.275,1.115,0.131,-0.065,-0.129,1.157,2.637]

        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        #     # p.resetJointState(self.pandaUid,i, jointPoses[i])
        p.resetJointState(self.pandaUid, 9, 0.04)
        p.resetJointState(self.pandaUid,10, 0.04)

        # p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, 0, force = 100)
        # p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, 0, force = 100)

        # get the position state (0 index) of the end effector
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        # get the position state (0 index) of both fingers
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = state_robot + state_fingers
        # enable the graphical rendering
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        return np.array(self.observation).astype(np.float32)

    def initialpose(self):
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])
        state_durations = [1,1]
        control_dt = 1./100.
        
        p.setTimestep = control_dt
        state_t = 0.
        current_state = 0

        while current_state<=1:
            state_t += control_dt

            if current_state == 0:
                p.setJointMotorControl2(self.pandaUid, 9, p.POSITION_CONTROL, 0, force = 200)
                p.setJointMotorControl2(self.pandaUid, 10, p.POSITION_CONTROL, 0, force = 200)

            if current_state == 1:
                gearPosition = [0.7,0,0.45]
                # gripping position
                newPosition = [gearPosition[0],
                                gearPosition[1],
                                gearPosition[2]]

                # calculating target joint variables for the robot
                jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]
                # print(jointPoses)

                p.setJointMotorControlArray(self.pandaUid, list(range(7)), p.POSITION_CONTROL, list(jointPoses))

                # abovePose = [-0.015,0.928,0.134,-0.163,-0.121,1.091,2.365]
                # for i in range(7):
                #     p.resetJointState(self.pandaUid,i, abovePose[i])

            if state_t >state_durations[current_state]:
                current_state += 1
                state_t = 0
            
            p.stepSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)


    # the output can be feed in CNN network which outputs the pose of the object
    def render(self, mode='human'):
        # place the camera at a desired position and orientation
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        # get the camera image, need view_matrix and proj_matrix
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
