import math
import time
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators


class PID:
    """
    Dyskretny regulator PID z ograniczeniem wyjścia (out_limit),
    ograniczeniem całki (i_limit - anti-windup) oraz klasycznym różniczkowaniem uchybu
    wygładzonym filtrem dolnoprzepustowym (LPF).
    """
    def __init__(self, kp: float, ki: float, kd: float, out_limit: float = 150.0, i_limit: float = 40.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i_limit = i_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_deriv = 0.0
        self.first_run = True

    def reset(self):
        """Zeruje pamięć regulatora."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_deriv = 0.0
        self.first_run = True

    def compute(self, error: float, dt: float) -> float:
        """Oblicza sygnał sterujący PID na podstawie uchybu (error) i kroku czasowego (dt)."""
        if dt <= 0.0:
            return 0.0

        # 1. Człon proporcjonalny (P)
        p_term = self.kp * error

        # 2. Człon całkujący (I) z zabezpieczeniem anti-windup
        self.integral += error * dt
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        i_term = self.ki * self.integral

        # 3. Człon różniczkujący (D) z klasycznej pochodnej uchybu + filtr LPF
        if self.first_run:
            raw_deriv = 0.0
            self.first_run = False
        else:
            raw_deriv = (error - self.prev_error) / dt
        self.prev_error = error

        self.filtered_deriv = 0.35 * raw_deriv + 0.65 * self.filtered_deriv
        d_term = self.kd * self.filtered_deriv

        output = p_term + i_term + d_term
        return max(-self.out_limit, min(self.out_limit, output))


def euler_from_quaternion(x: float, y: float, z: float, w: float):
    """Przelicza kwaternion orientacji (x, y, z, w) na kąty Eulera: roll, pitch, yaw [rad]."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class AttitudeControllerNode(Node):
    """
    Sterownik drona X3 realizujący:
    1. Lot kierunkowy (przód/tył/lewo/prawo) poprzez zadawanie kątów target_pitch i target_roll.
    2. Lot pionowy (góra/dół) poprzez zadawanie prędkości target_vz oraz automatyczną
       stabilizację wysokości (Altitude Hold) na poziomie, do którego dron doleciał.
    3. Obrót wokół własnej osi poprzez zadawanie prędkości kątowej target_yaw_rate.
    """
    def __init__(self):
        super().__init__('regulator_node')

        self.max_motor_speed = 800.0
        self.min_motor_speed = 0.0

        # Zmienne przechowujące aktualną oraz zatrzaskiwaną wysokość drona [m]
        self.current_z = None
        self.held_z = None  # Poziom wysokości, który dron ma utrzymywać

        # 1. Inicjalizacja 4 regulatorów PID (Roll, Pitch, Yaw Rate, Wysokość Z)
        self.pid_roll = PID(110.0, 5.0, 28.0, out_limit=140.0, i_limit=30.0)
        self.pid_pitch = PID(110.0, 5.0, 28.0, out_limit=140.0, i_limit=30.0)
        self.pid_yaw_rate = PID(180.0, 0.0, 5.0, out_limit=80.0, i_limit=20.0)
        self.pid_z = PID(75.0, 8.0, 65.0, out_limit=120.0, i_limit=40.0)

        # 2. Deklaracja parametrów ROS 2
        default_parameters = [
            ('use_rp_pid', True),
            ('use_yaw_pid', True),
            ('use_z_pid', True),
            ('base_throttle', 660.4),
            ('target_roll', 0.0),
            ('target_pitch', 0.0),
            ('target_yaw_rate', 0.0),
            ('target_vz', 0.0),
            ('kp_rp', 110.0), ('ki_rp', 5.0), ('kd_rp', 28.0),
            ('kp_yaw', 180.0), ('ki_yaw', 0.0), ('kd_yaw', 5.0),
            ('kp_z', 75.0), ('ki_z', 8.0), ('kd_z', 65.0),
        ]
        self.declare_parameters(namespace='', parameters=default_parameters)

        # 3. Pobranie wartości początkowych parametrów
        self._read_initial_parameters()

        # 4. Rejestracja callbacku dla dynamicznych zmian parametrów
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.last_time = None
        self.step_counter = 0

        self.odom_sub = self.create_subscription(
            Odometry, '/model/X3/odometry', self.odom_callback, 10
        )
        self.motor_pub = self.create_publisher(
            Actuators, '/X3/command/motor_speed', 10
        )

        self.get_logger().info('Sterownik (RP + YawRate + Altitude Hold) uruchomiony!')

    def _read_initial_parameters(self):
        """Wczytuje parametry przy starcie węzła."""
        self.use_rp_pid = bool(self.get_parameter('use_rp_pid').value)
        self.use_yaw_pid = bool(self.get_parameter('use_yaw_pid').value)
        self.use_z_pid = bool(self.get_parameter('use_z_pid').value)

        self.base_throttle = float(self.get_parameter('base_throttle').value)
        self.target_roll = float(self.get_parameter('target_roll').value)
        self.target_pitch = float(self.get_parameter('target_pitch').value)
        self.target_yaw_rate = float(self.get_parameter('target_yaw_rate').value)
        self.target_vz = float(self.get_parameter('target_vz').value)

        self.pid_roll.kp = self.pid_pitch.kp = float(self.get_parameter('kp_rp').value)
        self.pid_roll.ki = self.pid_pitch.ki = float(self.get_parameter('ki_rp').value)
        self.pid_roll.kd = self.pid_pitch.kd = float(self.get_parameter('kd_rp').value)

        self.pid_yaw_rate.kp = float(self.get_parameter('kp_yaw').value)
        self.pid_yaw_rate.ki = float(self.get_parameter('ki_yaw').value)
        self.pid_yaw_rate.kd = float(self.get_parameter('kd_yaw').value)

        self.pid_z.kp = float(self.get_parameter('kp_z').value)
        self.pid_z.ki = float(self.get_parameter('ki_z').value)
        self.pid_z.kd = float(self.get_parameter('kd_z').value)

    def parameters_callback(self, params):
        """Zdarzeniowa obsługa zmiany parametrów za pomocą jawnych instrukcji if/elif."""
        for p in params:
            name, val = p.name, p.value

            if name == 'use_rp_pid':
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
                # W momencie zmiany komendy pionowej (np. na 0.0 -> "zatrzymaj się"),
                # zatrzaskujemy dokładnie tę wysokość, na której dron aktualnie się znajduje!
                if self.current_z is not None:
                    self.held_z = self.current_z
                    self.get_logger().info(f'Zatrzaśnięto aktualny poziom wysokości: Z = {self.held_z:.2f} m')

            elif name == 'kp_rp':
                self.pid_roll.kp = self.pid_pitch.kp = float(val)
            elif name == 'ki_rp':
                self.pid_roll.ki = self.pid_pitch.ki = float(val)
            elif name == 'kd_rp':
                self.pid_roll.kd = self.pid_pitch.kd = float(val)
            elif name == 'kp_yaw':
                self.pid_yaw_rate.kp = float(val)
            elif name == 'ki_yaw':
                self.pid_yaw_rate.ki = float(val)
            elif name == 'kd_yaw':
                self.pid_yaw_rate.kd = float(val)
            elif name == 'kp_z':
                self.pid_z.kp = float(val)
            elif name == 'ki_z':
                self.pid_z.ki = float(val)
            elif name == 'kd_z':
                self.pid_z.kd = float(val)

            self.get_logger().info(f'[ZMIANA PARAMETRU] {name} = {val}')

        return SetParametersResult(successful=True)

    def stop_motors(self):
        """Bezpieczne zatrzymanie silników przy wyłączeniu węzła (Ctrl+C)."""
        stop_msg = Actuators()
        stop_msg.velocity = [0.0, 0.0, 0.0, 0.0]
        for _ in range(3):
            self.motor_pub.publish(stop_msg)
            time.sleep(0.02)
        self.get_logger().info('Silniki bezpiecznie zatrzymane.')

    def odom_callback(self, msg: Odometry):
        # 1. Obliczenie kroku czasowego dt [s]
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0.0 or dt > 0.5:
            self.pid_roll.reset()
            self.pid_pitch.reset()
            self.pid_yaw_rate.reset()
            self.pid_z.reset()
            return

        # 2. Odczyt wysokości, orientacji (kątów Eulera) i prędkości kątowej wokół osi Z
        pos = msg.pose.pose.position
        self.current_z = pos.z

        # Przy pierwszym pomiarze zapamiętujemy startowy poziom wysokości drona
        if self.held_z is None:
            self.held_z = pos.z

        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        current_yaw_rate = msg.twist.twist.angular.z

        # 3. Sterowanie WYSOKOŚCIĄ (wznoszenie/opadanie z prędkością target_vz lub utrzymanie poziomu held_z)
        if self.use_z_pid:
            # Jeśli zadano prędkość pionową (target_vz != 0), przesuwamy poziom docelowy w górę lub w dół.
            # Jeśli target_vz == 0.0, zmienna self.held_z pozostaje stała i dron utrzymuje osiągnięty pułap!
            if abs(self.target_vz) > 1e-4:
                self.held_z = max(0.05, self.held_z + self.target_vz * dt)

            # Jeśli dron ma stać na ziemi (held_z < 0.15 m oraz brak komendy lotu w górę), wyłączamy ciąg
            if self.held_z < 0.15 and self.target_vz <= 0.0 and pos.z < 0.15:
                self.pid_z.reset()
                u_z = 0.0
            else:
                e_z = self.held_z - pos.z
                u_z_corr = self.pid_z.compute(e_z, dt)
                # Kompensacja spadku pionowej siły nośnej podczas przechyłu (lotu w przód/boki)
                tilt_factor = max(0.6, math.cos(roll) * math.cos(pitch))
                u_z = (self.base_throttle + u_z_corr) / tilt_factor
        else:
            self.pid_z.reset()
            u_z = self.base_throttle

        # 4. Sterowanie ORIENTACJĄ POZIOMĄ (lot przód/tył przez target_pitch, lewo/prawo przez target_roll)
        if self.use_rp_pid and u_z > 0.0:
            e_roll = self.target_roll - roll
            e_pitch = self.target_pitch - pitch
            u_phi = self.pid_roll.compute(e_roll, dt)
            u_theta = self.pid_pitch.compute(e_pitch, dt)
        else:
            self.pid_roll.reset()
            self.pid_pitch.reset()
            u_phi = 0.0
            u_theta = 0.0

        # 5. Sterowanie PRĘDKOŚCIĄ KĄTOWĄ OBROTU (Yaw Rate)
        if self.use_yaw_pid and u_z > 0.0:
            e_yaw_rate = self.target_yaw_rate - current_yaw_rate
            u_psi = self.pid_yaw_rate.compute(e_yaw_rate, dt)
        else:
            self.pid_yaw_rate.reset()
            u_psi = 0.0

        # 6. Kinematyka odwrotna (Mikser układu X) i saturacja do [0.0, 800.0] [rad/s]
        if u_z <= 0.0:
            clamped_speeds = [0.0, 0.0, 0.0, 0.0]
        else:
            w1 = u_z - u_phi - u_theta - u_psi  # rotor_0: prawy przód (CCW)
            w2 = u_z + u_phi + u_theta - u_psi  # rotor_1: lewy tył (CCW)
            w3 = u_z + u_phi - u_theta + u_psi  # rotor_2: lewy przód (CW)
            w4 = u_z - u_phi + u_theta + u_psi  # rotor_3: prawy tył (CW)

            clamped_speeds = [
                float(max(self.min_motor_speed, min(self.max_motor_speed, w)))
                for w in (w1, w2, w3, w4)
            ]

        out_msg = Actuators()
        out_msg.velocity = clamped_speeds
        self.motor_pub.publish(out_msg)

        # 7. Log diagnostyczny raz na sekundę
        self.step_counter += 1
        if self.step_counter >= 50:
            self.step_counter = 0
            self.get_logger().info(
                f'Wysokość: Z={pos.z:.2f}m (Utrzymywany poziom={self.held_z:.2f}m, Vz_zad={self.target_vz:.2f}m/s) | '
                f'Roll={roll:.2f}/{self.target_roll:.2f}, Pitch={pitch:.2f}/{self.target_pitch:.2f} rad'
            )


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
