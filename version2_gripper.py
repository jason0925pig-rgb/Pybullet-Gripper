import time
import numpy as np
import pybullet as p
import pybullet_data


class PR2Hand:
    GRASP_JOINTS = [0, 2]

    def __init__(self, urdf_path="pr2_gripper.urdf"):
        self.urdf_path = urdf_path
        self.gripper_id = None
        self.hand_base_constraint = None
        self.num_joints = 0
        self.open = True
        self.box_id = None

    # -------------------------------
    # Finger Control
    # -------------------------------
    def grasp(self):
        """Close PR2 gripper."""
        for i in [0, 2]:
            p.setJointMotorControl2(
                self.gripper_id, i, p.POSITION_CONTROL,
                targetPosition=0.0, maxVelocity=1, force=10
            )
        self.open = False

    def open_gripper(self):
        """Open PR2 gripper."""
        for i in [0, 2]:
            p.setJointMotorControl2(
                self.gripper_id, i, p.POSITION_CONTROL,
                targetPosition=0.54, maxVelocity=1, force=10
            )
        self.open = True

    # -------------------------------
    # Setup and Initialization
    # -------------------------------
    def connect(self):
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #-------------------------------------相机高度初始化
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.3]
        )
        #-------------------------------------目标高度？
        #-------------------------------------改5.6后爪子上下跳窜确认为爪子高度
        init_pos = [0, 0, 0.6]
        #-------------------------------------爪子角度
        #-------------------------------------尝试np.pi / 3 后爪子更改角度
        init_ori = p.getQuaternionFromEuler([0, np.pi / 2, 0])

        self.gripper_id = p.loadURDF(
            self.urdf_path, init_pos, init_ori,
            globalScaling=1, useFixedBase=False
        )

        self.num_joints = p.getNumJoints(self.gripper_id)

        #joint_positions = [0.550569, 0.0, 0.549657, 0.0]
        #-------------------------------------盲猜爪子开合joint坐标，改[10, 0.0, 10, 0.0]后发现相反打开且劈叉
        #-------------------------------------这里全改10
        joint_positions = [0.550569, 0.0, 0.549657, 0.0]
        for i, pos in enumerate(joint_positions):
            p.resetJointState(self.gripper_id, i, pos)

        for i in range(self.num_joints):
            info = p.getJointInfo(self.gripper_id, i)
            print(
                i,
                info[1].decode(),
                "lower =", info[8],
                "upper =", info[9]
            )

        # Cube exists from the beginning
        self.box_id = p.loadURDF(
            "cube_small.urdf",
            [0, 0, 0.025],
            globalScaling=1
        )

        p.setRealTimeSimulation(1)

        self.hand_base_constraint = p.createConstraint(
            self.gripper_id, -1, -1, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            init_pos
        )

        p.changeConstraint(
                    self.hand_base_constraint,
                    init_pos,
                    jointChildFrameOrientation=init_ori,
                    maxForce=50
                )

        print("Gripper loaded with ID:", self.gripper_id)

    # -------------------------------
    # Demo Script
    # -------------------------------
    def run_demo(self):
        self.connect()

        hand_ori = p.getQuaternionFromEuler([0, np.pi / 2, 0])

        # Visualize world axes
        p.addUserDebugLine([0, 0, 0], [0, 0, 0], [1, 0, 0], 1, 0)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0], [0, 1, 0], 1, 0)
        p.addUserDebugLine([0, 0, 0], [0, 0, 0], [0, 0, 1], 1, 0)

        print("\033[31mWill grasp and lift an object\033[0m")
        time.sleep(0.5)

        # -------------------------------
        # Step 1: Open
        # -------------------------------
        print("\033[31mWill open the gripper\033[0m")

        self.open_gripper()
        time.sleep(1.0)

        # -------------------------------
        # Step 2: Slowly move down
        # -------------------------------
        start_pos = np.array([0, 0, 0.6])
        grasp_pos = np.array([0, 0, 0.5])
        steps_down = 200

        print("\033[31mMoving down to grasp position\033[0m")

        for interp_pos in np.linspace(start_pos, grasp_pos, steps_down):

            # Keep open during descent
            self.open_gripper()

            p.changeConstraint(
                self.hand_base_constraint,
                interp_pos,
                jointChildFrameOrientation=hand_ori,
                maxForce=50
            )

            p.stepSimulation()
            time.sleep(0.01)

        # -------------------------------
        # Step 3: Close
        # -------------------------------
        print("\033[31mWill grasp\033[0m")

        self.grasp()
        time.sleep(1.5)

        # -------------------------------
        # Step 4: Lift
        # -------------------------------
        print("\033[31mWill lift\033[0m")

        start_pos = np.array([0, 0, 0.33])
        end_pos = np.array([0, 0, 0.6])
        steps = 200

        for interp_pos in np.linspace(start_pos, end_pos, steps):

            # Keep closed during lift
            self.grasp()

            p.changeConstraint(
                self.hand_base_constraint,
                interp_pos,
                jointChildFrameOrientation=hand_ori,
                maxForce=50
            )

            p.stepSimulation()
            time.sleep(0.01)

        print("\033[31mGrasp and lift done\033[0m")

        time.sleep(2)
        p.disconnect()


if __name__ == "__main__":
    hand = PR2Hand()
    hand.run_demo()
    
