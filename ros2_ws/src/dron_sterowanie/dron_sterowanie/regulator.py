import time
import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators

from dron_sterowanie.pid import PID
from dron_sterowanie.math_utils import euler_from_quaternion


class AttitudeControllerNode(Node):
    """
    Kaskadowy sterownik drona X3:
    - Pętla nadrzędna (X, Y, Z) w układzie świata + transformacja przez kąt Yaw na zadane kąty Pitch/Roll.
    - Pętla podrzędna (Roll, Pitch, Yaw Rate) + Mikser X z desaturacją (priorytet orientacji).
    """
    def __init__(self):
        super().__init__('regulator_node')

        self.max_motor_speed = 800.0
        self.min_motor_speed = 0.0
        self.max_tilt = 0.35  # ok. 20 stopni

        # Bieżąca pozycja drona z odometrii oraz zatrzaskiwany pułap
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.held_z = None

        # Pętla podrzędna (orientacja) oraz wysokość Z
        self.pid_roll = PID(110.0, 5.0, 28.0, out_limit=140.0, i_limit=30.0)
        self.pid_pitch = PID(110.0, 5.0, 28.0, out_limit=140.0, i_limit=30.0)
        self.pid_yaw_rate = PID(180.0, 0.0, 5.0, out_limit=80.0, i_limit=20.0)
        self.pid_z = PID(75.0, 8.0, 65.0, out_limit=120.0, i_limit=40.0)

        # Pętla nadrzędna (pozycja XY w układzie świata -> wirtualne sterowanie przed obrotem)
        self.pid_x = PID(0.12, 0.0, 0.05, out_limit=self.max_tilt, i_limit=0.10)
        self.pid_y = PID(0.12, 0.0, 0.05, out_limit=self.max_tilt, i_limit=0.10)

        default_parameters = [
            ('use_rp_pid', True), ('use_yaw_pid', True), ('use_z_pid', True),
            ('use_pos_pid', False),
            ('base_throttle', 660.4),
            ('target_roll', 0.0), ('target_pitch', 0.0), ('target_yaw_rate', 0.0), ('target_vz', 0.0),
            ('target_x', 0.0), ('target_y', 0.0), ('target_z', 0.0),
            ('kp_rp', 110.0), ('ki_rp', 5.0), ('kd_rp', 28.0),
            ('kp_yaw', 180.0), ('ki_yaw', 0.0), ('kd_yaw', 5.0),
            ('kp_z', 75.0), ('ki_z', 8.0), ('kd_z', 65.0),
            ('kp_xy', 0.12), ('ki_xy', 0.0), ('kd_xy', 0.05),
        ]
        self.declare_parameters(namespace='', parameters=default_parameters)
        self._read_initial_parameters()

        self.add_on_set_parameters_callback(self.parameters_callback)

        self.last_time = None

        self.odom_sub = self.create_subscription(Odometry, '/model/X3/odometry', self.odom_callback, 10)
        self.motor_pub = self.create_publisher(Actuators, '/X3/command/motor_speed', 10)

        self.get_logger().info('Kaskadowy sterownik drona (Attitude + Position + Desaturacja) uruchomiony!')

    def _read_initial_parameters(self):
        self.use_rp_pid = bool(self.get_parameter('use_rp_pid').value)
        self.use_yaw_pid = bool(self.get_parameter('use_yaw_pid').value)
        self.use_z_pid = bool(self.get_parameter('use_z_pid').value)
        self.use_pos_pid = bool(self.get_parameter('use_pos_pid').value)

        self.base_throttle = float(self.get_parameter('base_throttle').value)
        self.target_roll = float(self.get_parameter('target_roll').value)
        self.target_pitch = float(self.get_parameter('target_pitch').value)
        self.target_yaw_rate = float(self.get_parameter('target_yaw_rate').value)
        self.target_vz = float(self.get_parameter('target_vz').value)

        self.target_x = float(self.get_parameter('target_x').value)
        self.target_y = float(self.get_parameter('target_y').value)
        self.target_z = float(self.get_parameter('target_z').value)

        self.pid_roll.kp = self.pid_pitch.kp = float(self.get_parameter('kp_rp').value)
        self.pid_roll.ki = self.pid_pitch.ki = float(self.get_parameter('ki_rp').value)
        self.pid_roll.kd = self.pid_pitch.kd = float(self.get_parameter('kd_rp').value)

        self.pid_yaw_rate.kp = float(self.get_parameter('kp_yaw').value)
        self.pid_yaw_rate.ki = float(self.get_parameter('ki_yaw').value)
        self.pid_yaw_rate.kd = float(self.get_parameter('kd_yaw').value)

        self.pid_z.kp = float(self.get_parameter('kp_z').value)
        self.pid_z.ki = float(self.get_parameter('ki_z').value)
        self.pid_z.kd = float(self.get_parameter('kd_z').value)

        self.pid_x.kp = self.pid_y.kp = float(self.get_parameter('kp_xy').value)
        self.pid_x.ki = self.pid_y.ki = float(self.get_parameter('ki_xy').value)
        self.pid_x.kd = self.pid_y.kd = float(self.get_parameter('kd_xy').value)

    def parameters_callback(self, params):
        param_map = {p.name: p.value for p in params}

        # 1. Obsługa trybu pozycji i zatrzaskiwania bieżącego punktu
        if 'use_pos_pid' in param_map:
            new_pos_mode = bool(param_map['use_pos_pid'])
            if new_pos_mode:
                latched_xy = ('target_x' not in param_map) and ('target_y' not in param_map)
                if not self.use_pos_pid or latched_xy:
                    self.pid_x.reset()
                    self.pid_y.reset()

                if 'target_x' not in param_map and self.current_x is not None:
                    self.target_x = self.current_x
                if 'target_y' not in param_map and self.current_y is not None:
                    self.target_y = self.current_y
                if 'target_z' not in param_map and self.current_z is not None:
                    self.target_z = self.current_z
                    self.held_z = self.current_z

                self.target_vz = 0.0
                self.get_logger().info(
                    f'Tryb POSITION -> cel: X={self.target_x:.2f}, Y={self.target_y:.2f}, Z={self.target_z:.2f}'
                )
            elif self.use_pos_pid:
                self.pid_x.reset()
                self.pid_y.reset()
                self.get_logger().info('Wyłączono tryb POSITION (powrót do trybu prędkościowego).')
            self.use_pos_pid = new_pos_mode

        # 2. Obsługa pozostałych parametrów
        for p in params:
            name, val = p.name, p.value
            if name == 'use_pos_pid':
                continue
            elif name == 'use_rp_pid':
                self.use_rp_pid = bool(val)
            elif name == 'use_yaw_pid':
                self.use_yaw_pid = bool(val)
            elif name == 'use_z_pid':
                self.use_z_pid = bool(val)
                if self.use_z_pid and self.current_z is not None:
                    self.held_z = self.current_z
            elif name == 'base_throttle':
                self.base_throttle = float(val)

            elif name == 'target_roll':
                self.target_roll = float(val)
            elif name == 'target_pitch':
                self.target_pitch = float(val)
            elif name == 'target_yaw_rate':
                self.target_yaw_rate = float(val)
            elif name == 'target_vz':
                self.target_vz = float(val)
                if not self.use_pos_pid and self.current_z is not None:
                    self.held_z = self.current_z

            elif name == 'target_x':
                self.target_x = float(val)
            elif name == 'target_y':
                self.target_y = float(val)
            elif name == 'target_z':
                self.target_z = float(val)
                if self.use_pos_pid:
                    self.held_z = max(0.05, self.target_z)
                    self.target_vz = 0.0

            elif name == 'kp_rp': self.pid_roll.kp = self.pid_pitch.kp = float(val)
            elif name == 'ki_rp': self.pid_roll.ki = self.pid_pitch.ki = float(val)
            elif name == 'kd_rp': self.pid_roll.kd = self.pid_pitch.kd = float(val)
            elif name == 'kp_yaw': self.pid_yaw_rate.kp = float(val)
            elif name == 'ki_yaw': self.pid_yaw_rate.ki = float(val)
            elif name == 'kd_yaw': self.pid_yaw_rate.kd = float(val)
            elif name == 'kp_z': self.pid_z.kp = float(val)
            elif name == 'ki_z': self.pid_z.ki = float(val)
            elif name == 'kd_z': self.pid_z.kd = float(val)
            elif name == 'kp_xy': self.pid_x.kp = self.pid_y.kp = float(val)
            elif name == 'ki_xy': self.pid_x.ki = self.pid_y.ki = float(val)
            elif name == 'kd_xy': self.pid_x.kd = self.pid_y.kd = float(val)

        return SetParametersResult(successful=True)

    def odom_callback(self, msg: Odometry):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Detekcja resetu symulacji Gazebo
        if dt < 0.0:
            self.get_logger().warn('Wykryto restart symulacji Gazebo! Czyszczenie pamięci regulatora.')
            self.held_z = None
            self.use_pos_pid = False
            self.target_vz = 0.0
            self.target_roll = 0.0
            self.target_pitch = 0.0
            self.target_yaw_rate = 0.0

        if dt <= 0.0 or dt > 0.5:
            for pid in (self.pid_roll, self.pid_pitch, self.pid_yaw_rate, self.pid_z, self.pid_x, self.pid_y):
                pid.reset()
            return

        pos = msg.pose.pose.position
        self.current_x = pos.x
        self.current_y = pos.y
        self.current_z = pos.z
        if self.held_z is None:
            self.held_z = pos.z

        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        current_yaw_rate = msg.twist.twist.angular.z

        # ------------------- 1. STEROWANIE WYSOKOŚCIĄ Z -------------------
        if self.use_z_pid:
            if self.use_pos_pid:
                self.held_z = max(0.05, self.target_z)
            elif abs(self.target_vz) > 1e-4:
                self.held_z = max(0.05, self.held_z + self.target_vz * dt)

            if self.held_z < 0.35 and self.target_vz <= 0.0 and pos.z < 0.35:
                self.pid_z.reset()
                u_z = 0.0
            else:
                e_z = self.held_z - pos.z
                u_z_corr = self.pid_z.compute(e_z, dt)
                tilt_factor = max(0.6, math.cos(roll) * math.cos(pitch))
                u_z = (self.base_throttle + u_z_corr) / tilt_factor
        else:
            self.pid_z.reset()
            u_z = self.base_throttle

        # ------------------- 2. PĘTLA NADRZĘDNA POZYCJI XY (KASKADA) -------------------
        if self.use_pos_pid and u_z > 0.0:
            if pos.z > 0.35:
                # 1. Uchyb i PID w układzie świata
                e_x_world = self.target_x - pos.x
                e_y_world = self.target_y - pos.y
                u_x = self.pid_x.compute(e_x_world, dt)
                u_y = self.pid_y.compute(e_y_world, dt)

                # 2. Transformacja do układu lokalnego drona (obrót o Yaw + znak Roll)
                cos_y = math.cos(yaw)
                sin_y = math.sin(yaw)
                raw_pitch = u_x * cos_y + u_y * sin_y
                raw_roll = u_x * sin_y - u_y * cos_y

                # 3. Limit końcowych kątów zadanych po obrocie [rad]
                eff_target_pitch = max(-self.max_tilt, min(self.max_tilt, raw_pitch))
                eff_target_roll = max(-self.max_tilt, min(self.max_tilt, raw_roll))
            else:
                self.pid_x.reset()
                self.pid_y.reset()
                eff_target_pitch = 0.0
                eff_target_roll = 0.0
        else:
            self.pid_x.reset()
            self.pid_y.reset()
            eff_target_pitch = self.target_pitch
            eff_target_roll = self.target_roll

        # ------------------- 3. PĘTLA PODRZĘDNA ORIENTACJI (ROLL / PITCH) -------------------
        if self.use_rp_pid and u_z > 0.0:
            u_phi = self.pid_roll.compute(eff_target_roll - roll, dt)
            u_theta = self.pid_pitch.compute(eff_target_pitch - pitch, dt)
        else:
            self.pid_roll.reset()
            self.pid_pitch.reset()
            u_phi = u_theta = 0.0

        # ------------------- 4. STEROWANIE YAW RATE -------------------
        if self.use_yaw_pid and u_z > 0.0:
            u_psi = self.pid_yaw_rate.compute(self.target_yaw_rate - current_yaw_rate, dt)
        else:
            self.pid_yaw_rate.reset()
            u_psi = 0.0

        # ------------------- 5. MIKSER X + DESATURACJA (PRIORYTET ORIENTACJI) -------------------
        if u_z <= 0.0:
            speeds = [0.0, 0.0, 0.0, 0.0]
        else:
            raw_speeds = [
                u_z - u_phi - u_theta - u_psi,
                u_z + u_phi + u_theta - u_psi,
                u_z + u_phi - u_theta + u_psi,
                u_z - u_phi + u_theta + u_psi,
            ]
            # Jeśli silniki wychodzą poza [0, 800], przesuwamy ciąg wspólny u_z,
            # zachowując w 100% różnicowe momenty Roll, Pitch i Yaw
            max_w = max(raw_speeds)
            min_w = min(raw_speeds)
            if max_w > self.max_motor_speed:
                shift = max_w - self.max_motor_speed
                raw_speeds = [w - shift for w in raw_speeds]
            elif min_w < self.min_motor_speed:
                shift = self.min_motor_speed - min_w
                raw_speeds = [w + shift for w in raw_speeds]

            speeds = [float(max(self.min_motor_speed, min(self.max_motor_speed, w))) for w in raw_speeds]

        out_msg = Actuators()
        out_msg.velocity = speeds
        self.motor_pub.publish(out_msg)

    def stop_motors(self):
        stop_msg = Actuators()
        stop_msg.velocity = [0.0, 0.0, 0.0, 0.0]
        for _ in range(3):
            self.motor_pub.publish(stop_msg)
            time.sleep(0.02)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=rclpy.SignalHandlerOptions.NO)
    node = AttitudeControllerNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.stop_motors()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
