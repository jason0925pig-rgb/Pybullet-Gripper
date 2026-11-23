import time
import numpy as np
import pybullet as p
import pybullet_data


class PR2Hand:
    GRASP_JOINTS = [0, 2]

    # ---- 随机云参数 ----
    BOX_POS = np.array([0, 0, 0.025])   # 方块中心 = 半球球心
    R_MIN = 0.33                        # 最近半径(= 抓取时爪子到方块的距离)
    R_MAX = 0.6                         # 最远半径(= 原来的起始高度)
    LIFT_H = 0.27                       # 上抬高度 0.6 - 0.33
    CONTACT_T = 3.0                     # 接触超过 3 秒 -> label 1

    def __init__(self, urdf_path="pr2_gripper.urdf", gui=True):
        self.urdf_path = urdf_path
        self.gripper_id = None
        self.hand_base_constraint = None
        self.num_joints = 0
        self.open = True
        self.box_id = None
        self.plane_id = None
        self.gui = gui
        self.dt = 1.0 / 240.0
        # 接触统计
        self.touched = False
        self.contact_time = 0.0       # 当前连续接触时间
        self.max_contact_time = 0.0   # 本次尝试中最长连续接触时间

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
        if self.gui:
            cid = p.connect(p.SHARED_MEMORY)
            if cid < 0:
                p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        #-------------------------------------相机高度初始化
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,   # 半球半径最大 0.6,相机拉远一点
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

        # 重力 + 地面(否则方块悬空,抓取/摩擦没有意义)
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        self.plane_id = p.loadURDF("plane.urdf")

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
            self.BOX_POS.tolist(),
            globalScaling=1
        )

        # 摩擦系数(不设置容易打滑)
        p.changeDynamics(self.box_id, -1, lateralFriction=1.0)
        for j in range(-1, self.num_joints):
            p.changeDynamics(self.gripper_id, j, lateralFriction=1.0)

        # 用 stepSimulation 手动推进,保证每次尝试可复现
        p.setRealTimeSimulation(0)

        self.hand_base_constraint = p.createConstraint(
            self.gripper_id, -1, -1, -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            init_pos,
            childFrameOrientation=init_ori
        )

        p.changeConstraint(
                    self.hand_base_constraint,
                    init_pos,
                    jointChildFrameOrientation=init_ori,
                    maxForce=50
                )

        print("Gripper loaded with ID:", self.gripper_id)

    # -------------------------------
    # Random Pose Cloud
    # -------------------------------
    def sample_pose(self):
        """在方块上半球壳 (R_MIN~R_MAX) 内随机取点,姿态对准方块。
        返回 pos, (yaw, roll, pitch)"""
        r = np.random.uniform(self.R_MIN ** 3, self.R_MAX ** 3) ** (1.0 / 3.0)  # 体积均匀
        phi = np.random.uniform(0, 2 * np.pi)
        cos_t = np.random.uniform(0, 1)          # 只取上半球(下面除外)
        sin_t = np.sqrt(1 - cos_t ** 2)
        direction = np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])  # 方块 -> 爪子
        pos = self.BOX_POS + r * direction

        # 爪子接近轴 = 夹爪局部 +x (原来 pitch=pi/2 时 x 轴指向 -z)
        # 让 +x 轴指向方块: d = -direction
        d = -direction
        yaw = np.arctan2(d[1], d[0])
        pitch = np.arctan2(-d[2], np.hypot(d[0], d[1]))
        roll = np.random.uniform(-np.pi, np.pi)  # 绕接近轴自转,不改变"对准"
        return pos, (yaw, roll, pitch)

    def reset_trial(self, pos, ori):
        """不重新加载模型,直接把爪子/方块复位到新一次尝试的状态"""
        p.changeConstraint(
            self.hand_base_constraint,
            pos,
            jointChildFrameOrientation=ori,
            maxForce=50
        )
        p.resetBasePositionAndOrientation(self.gripper_id, pos, ori)
        p.resetBaseVelocity(self.gripper_id, [0, 0, 0], [0, 0, 0])
        for i, jp in enumerate([0.550569, 0.0, 0.549657, 0.0]):
            p.resetJointState(self.gripper_id, i, jp, targetVelocity=0)

        p.resetBasePositionAndOrientation(self.box_id, self.BOX_POS.tolist(), [0, 0, 0, 1])
        p.resetBaseVelocity(self.box_id, [0, 0, 0], [0, 0, 0])

        self.touched = False
        self.contact_time = 0.0
        self.max_contact_time = 0.0

    def is_overlapping(self, margin=0.01):
        """生成后是否与方块(或地面)重叠"""
        p.performCollisionDetection()
        hit_box = p.getClosestPoints(self.gripper_id, self.box_id, margin)
        hit_plane = p.getClosestPoints(self.gripper_id, self.plane_id, 0.0)
        return len(hit_box) > 0 or len(hit_plane) > 0

    def step_sim(self):
        """推进一步仿真,并统计爪子与方块的接触"""
        p.stepSimulation()
        if self.gui:
            time.sleep(self.dt)

        contacts = p.getContactPoints(bodyA=self.gripper_id, bodyB=self.box_id)
        if len(contacts) > 0:
            self.touched = True
            self.contact_time += self.dt
            self.max_contact_time = max(self.max_contact_time, self.contact_time)
        else:
            self.contact_time = 0.0

    def get_label(self):
        if self.max_contact_time > self.CONTACT_T:
            return 1
        if self.touched:
            return 0.5
        return 0

    # -------------------------------
    # Demo Script
    # -------------------------------
    def run_demo(self, n_samples=100):
        self.connect()

        # Visualize world axes
        if self.gui:
            p.addUserDebugLine([0, 0, 0], [0.5, 0, 0], [1, 0, 0], 1, 0)
            p.addUserDebugLine([0, 0, 0], [0, 0.5, 0], [0, 1, 0], 1, 0)
            p.addUserDebugLine([0, 0, 0], [0, 0, 0.5], [0, 0, 1], 1, 0)

        time.sleep(0.5)

        count = 0
        while count < n_samples:
            # 随机生成爪子位姿(yaw / roll / pitch,对准方块)
            pos, (yaw, roll, pitch) = self.sample_pose()
            hand_ori = p.getQuaternionFromEuler([roll, pitch, yaw])

            self.reset_trial(pos, hand_ori)
            if self.is_overlapping():
                continue  # 与方块重叠,重新生成

            # -------------------------------
            # Step 1: Open
            # -------------------------------
            self.open_gripper()
            for _ in range(int(1.0 / self.dt)):
                self.step_sim()

            # -------------------------------
            # Step 2: Slowly move toward the cube (沿对准方向靠近到 R_MIN)
            # -------------------------------
            start_pos = np.array(pos)
            r = np.linalg.norm(start_pos - self.BOX_POS)
            grasp_pos = self.BOX_POS + (start_pos - self.BOX_POS) / r * self.R_MIN
            steps_down = 200

            for interp_pos in np.linspace(start_pos, grasp_pos, steps_down):

                # Keep open during descent
                self.open_gripper()

                p.changeConstraint(
                    self.hand_base_constraint,
                    interp_pos,
                    jointChildFrameOrientation=hand_ori,
                    maxForce=50
                )

                self.step_sim()

            # -------------------------------
            # Step 3: Close
            # -------------------------------
            self.grasp()
            for _ in range(int(1.5 / self.dt)):
                self.step_sim()

            # -------------------------------
            # Step 4: Lift (世界坐标系竖直上抬)
            # -------------------------------
            end_pos = grasp_pos + np.array([0, 0, self.LIFT_H])
            steps = 480   # 2 秒仿真时间,保证 3 秒接触阈值可达

            for interp_pos in np.linspace(grasp_pos, end_pos, steps):

                # Keep closed during lift
                self.grasp()

                p.changeConstraint(
                    self.hand_base_constraint,
                    interp_pos,
                    jointChildFrameOrientation=hand_ori,
                    maxForce=50
                )

                self.step_sim()

            # 上抬完再保持 1 秒,观察方块是否掉落
            for _ in range(int(1.0 / self.dt)):
                self.step_sim()

            # -------------------------------
            # Output: x | y | z | yaw | roll | pitch | label
            # -------------------------------
            label = self.get_label()
            print(
                f"x: {pos[0]:.4f} | y: {pos[1]:.4f} | z: {pos[2]:.4f} | "
                f"yaw: {yaw:.4f} | roll: {roll:.4f} | pitch: {pitch:.4f} | "
                f"label: {label}"
            )
            count += 1

        p.disconnect()


if __name__ == "__main__":
    hand = PR2Hand(gui=True)   # 批量采数据时改成 gui=False,速度快很多
    hand.run_demo(n_samples=100)
