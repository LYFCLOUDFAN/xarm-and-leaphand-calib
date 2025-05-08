import time
import numpy as np
import torch 
from xarm6_interface import XARM6_IP, XARM6LEFT_IP
from xarm.wrapper import XArmAPI
from xarm6_interface.arm_mplib import min_jerk_interpolator_with_alpha 
from scipy.spatial.transform import Rotation as R

class XArm6RealWorld:
    def __init__(self, ip=XARM6LEFT_IP,is_radian=True, default_speed=0.5):
        self.arm = XArmAPI(ip, is_radian=is_radian)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(5000)

        # self.arm.reset(wait=True)
        self.default_is_radian = is_radian
        self.default_speed = default_speed
        self.default_cmd_timestep = 1.0 / 500.0
        # self.default_joint_values = np.array(
        #     [0, -67, -40, 0, 65, 0]
        # ) * np.pi / 180
        self.default_joint_values = np.array(
            [2.1, -45.1, -53.8, 2.5, 96.3, 2.2]
        ) * np.pi / 180
        # self.set_joint_values(self.default_joint_values, speed=default_speed, is_radian=True, wait=True)

    def set_joint_values(self, joint_values, speed=None, is_radian=None, wait=True):
        if speed is None:
            speed = self.default_speed
        if is_radian is None:
            is_radian = self.default_is_radian
        self.arm.set_servo_angle(angle=self.to_list(joint_values), speed=speed, wait=wait, is_radian=is_radian)

    def set_joint_values_sequence(self, way_point_positions, planning_timestep=0.05, cmd_timestep=None, speed=None, is_radian=None):
        if speed is None:
            speed = self.default_speed
        if is_radian is None:
            is_radian = self.default_is_radian
        if cmd_timestep is None:
            cmd_timestep = self.default_cmd_timestep
        self.arm.set_mode(6)
        self.arm.set_state(0)
        
        joint_values_seq = min_jerk_interpolator_with_alpha(
            way_point_positions, planning_timestep, cmd_timestep
        )
        for joint_values in joint_values_seq:
            self.arm.set_servo_angle(angle=self.to_list(joint_values), speed=speed, is_radian=is_radian)
            time.sleep(1.5*cmd_timestep)  # 1.5 is a magic number
        self.arm.set_servo_angle(angle=self.to_list(joint_values_seq[-1]), speed=speed, is_radian=is_radian)
        time.sleep(2)  # 1.5 is a magic number
                
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def get_position_se3(self, is_radian=None):
        """
        Get the pose represented in SE(3) form (homogeneous transformation matrix).
        
        :param is_radian: Whether the returned rx/ry/rz values are in radians.
                          Default is self.default_is_radian.
        :return: tuple((code, se3_matrix)), where:
            - code: Status code (0 for success).
            - se3_matrix: 4x4 homogeneous transformation matrix if code is 0, otherwise None.
        """
        if is_radian is None:
            is_radian = self.default_is_radian

        # 获取轴角表示的位姿
        code, pose = self.arm.get_position_aa(is_radian=is_radian)
        if code != 0:
            return code, None  # 如果获取失败，返回错误码和 None

        x, y, z = np.array(pose[:3])*0.001      # 提取平移部分
        rx, ry, rz = pose[3:6]    # 提取轴角旋转部分

        # 如果返回的旋转不是弧度，转换为弧度
        if not is_radian:
            rx, ry, rz = np.deg2rad([rx, ry, rz])

        # 将轴角表示转换为旋转矩阵
        rotation_matrix = R.from_rotvec([rx, ry, rz]).as_matrix()

        # 构造 SE(3) 齐次变换矩阵
        se3_matrix = np.eye(4)  # 初始化 4x4 单位矩阵
        se3_matrix[:3, :3] = rotation_matrix  # 设置旋转部分
        se3_matrix[:3, 3] = [x, y, z]         # 设置平移部分
        # input("Press Enter to continue...")

        return code, se3_matrix

    def get_joint_values(self, is_radian=None):
        if is_radian is None:
            is_radian = self.default_is_radian
        state, joint_values = self.arm.get_servo_angle(is_radian=is_radian)
        if state != 0:
            raise ValueError("Failed to get joint values")
        return joint_values[:6]

    def to_list(self, joint_values):
        if isinstance(joint_values, list):
            return joint_values
        if isinstance(joint_values, np.ndarray):
            return joint_values.tolist()
        elif isinstance(joint_values, torch.Tensor):
            return joint_values.flatten().cpu().detach().numpy().tolist()

    
    def close(self):
        # 先让机械臂 reset，然后再断开连接
        try:
            self.arm.reset(wait=True)
        except Exception:
            pass
        self.arm.disconnect()
    
    def get_position(self, is_radian=None):
        if is_radian is None:
            is_radian = self.default_is_radian

        # 获取轴角表示的位姿
        code, pose = self.arm.get_position_aa(is_radian=is_radian)
        # print(f"pose is {pose}")
        if code != 0:
            return code, None  # 如果获取失败，返回错误码和 None

        x, y, z = np.array(pose[:3])*0.001      # 提取平移部分
        rx, ry, rz = pose[3:6]    # 提取轴角旋转部分

        # 如果返回的旋转不是弧度，转换为弧度
        if not is_radian:
            rx, ry, rz = np.deg2rad([rx, ry, rz])

        # 将轴角表示转换为旋转矩阵
        rotation_matrix = R.from_rotvec([rx, ry, rz]).as_matrix()

        # 构造 SE(3) 齐次变换矩阵
        se3_matrix = np.eye(4)  # 初始化 4x4 单位矩阵
        se3_matrix[:3, :3] = rotation_matrix  # 设置旋转部分
        se3_matrix[:3, 3] = [x, y, z]         # 设置平移部分
        # input("Press Enter to continue...")
        z_offset = np.array([0, 0, -0.185])
        z_offset = se3_matrix[:3, :3].dot(z_offset)
        se3_matrix[:3, 3] += z_offset
        # input("how are you? tap enter to ensure you are fine!")

        return code, se3_matrix
    
    def set_position_from_matrix(self, transform_matrix, speed=None, mvacc=None, is_radian=True, wait=False):
        """
        根据 4x4 齐次变换矩阵调用 set_position
        
        Args:
            arm: xArm 实例对象
            transform_matrix: 4x4 齐次变换矩阵 (numpy 数组)
            speed: 移动速度
            mvacc: 移动加速度
            is_radian: 是否使用弧度单位 (默认为 True)
            wait: 是否等待动作完成 (默认为 False)
        """
        # 提取平移部分
        x, y, z = transform_matrix[:3, 3]

        # 提取旋转矩阵部分
        rotation_matrix = transform_matrix[:3, :3]

        # 将旋转矩阵转换为 roll, pitch, yaw
        r = R.from_matrix(rotation_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=not is_radian)

        # 调用 set_position
        self.arm.set_position(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=mvacc,
            is_radian=is_radian, wait=wait
        )

        
    def set_tool_position(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0,
                        speed=None, mvacc=None, mvtime=None, is_radian=None,
                        wait=False, timeout=None, radius=None, **kwargs):
        return self.arm.set_tool_position(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, mvacc=mvacc, mvtime=mvtime, is_radian=is_radian,
            wait=wait, timeout=timeout, radius=radius, **kwargs
        )

    def set_tool_posotion(self):
        return self.arm.set_tool_position()


    

if __name__ == "__main__":
    from loguru import logger as lgr
    xarm = XArm6RealWorld()

    lgr.info("Current joint values: {}".format(xarm.get_joint_values()))

    xarm.close()