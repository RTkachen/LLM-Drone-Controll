import time
import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators

# Wczytujemy nasze moduły inżynierskie
from dron_sterowanie.pid import PID
from dron_sterowanie.math_utils import euler_from_quaternion


class AttitudeControllerNode(Node):
    """
    Węzeł spinający logikę lotu. Zarządza komunikacją z ROS 2 (odometria i parametry)
    oraz wywołuje wyliczenia PID w odpowiedniej kolejności.
    """
    def __init__(self):
        super().__init__('regulator_node')

        self.max_motor_speed = 800.0
        self.min_motor_speed = 0.0

        # current_z to wysokość na ten moment, held_z to docelowy pułap "zatrzaśnięty" do zawisu
        self.current_z = None
        self.held_z = None

        # Inicjalizacja instancji matematycznych regulatorów PID (Kp, Ki, Kd podane na start)
        self.pid_roll = PID(110.0, 5.0, 28.0, out_limit=140.0, i_limit=30.0)
        self.pid_pitch = PID(110.0, 5.0, 28.0, out_limit=140.0, i_limit=30.0)
        self.pid_yaw_rate = PID(180.0, 0.0, 5.0, out_limit=80.0, i_limit=20.0)
        self.pid_z = PID(75.0, 8.0, 65.0, out_limit=120.0, i_limit=40.0)

        # Deklaracja infrastruktury parametrów ROS 2
        default_parameters = [
            ('use_rp_pid', True), ('use_yaw_pid', True), ('use_z_pid', True),
            ('base_throttle', 660.4),
            ('target_roll', 0.0), ('target_pitch', 0.0), ('target_yaw_rate', 0.0), ('target_vz', 0.0),
            ('kp_rp', 110.0), ('ki_rp', 5.0), ('kd_rp', 28.0),
            ('kp_yaw', 180.0), ('ki_yaw', 0.0), ('kd_yaw', 5.0),
            ('kp_z', 75.0), ('ki_z', 8.0), ('kd_z', 65.0),
        ]
        self.declare_parameters(namespace='', parameters=default_parameters)
        self._read_initial_parameters()

        # Zarejestrowanie funkcji callback reagującej na zdarzenie zmiany parametrów z terminala/AI
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.last_time = None
        self.step_counter = 0

        # Subskrybent (odbiór danych o locie z Gazebo) i Publisher (wysyłka poleceń na silniki)
        self.odom_sub = self.create_subscription(Odometry, '/model/X3/odometry', self.odom_callback, 10)
        self.motor_pub = self.create_publisher(Actuators, '/X3/command/motor_speed', 10)

        self.get_logger().info('Sterownik drona uruchomiony!')

    def _read_initial_parameters(self):
        """Pobiera początkowe stany wszystkich zmiennych z bufora parametrów ROS 2."""
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
        """Obsługuje dynamiczne zmiany podczas lotu wysyłane przez most AI lub użytkownika."""
        for p in params:
            name, val = p.name, p.value

            # Przełączniki systemowe
            if name == 'use_rp_pid': self.use_rp_pid = bool(val)
            elif name == 'use_yaw_pid': self.use_yaw_pid = bool(val)
            elif name == 'use_z_pid':
                self.use_z_pid = bool(val)
                # W momencie włączenia stabilizacji pionowej natychmiast celujemy w bieżącą wysokość
                if self.use_z_pid and self.current_z is not None:
                    self.held_z = self.current_z
            elif name == 'base_throttle': self.base_throttle = float(val)

            # Komendy ruchu sterującego (trajektorii)
            elif name == 'target_roll': self.target_roll = float(val)
            elif name == 'target_pitch': self.target_pitch = float(val)
            elif name == 'target_yaw_rate': self.target_yaw_rate = float(val)
            elif name == 'target_vz':
                self.target_vz = float(val)
                # Gdy wyłączamy komendę wznoszenia (target_vz -> 0.0), zatrzaskujemy
                # tę konkretną wysokość do utrzymania jako "punkt docelowy" zawisu (held_z).
                if self.current_z is not None:
                    self.held_z = self.current_z

            # Nadpisywanie parametrów dla obiektu PID_RP
            elif name == 'kp_rp': self.pid_roll.kp = self.pid_pitch.kp = float(val)
            elif name == 'ki_rp': self.pid_roll.ki = self.pid_pitch.ki = float(val)
            elif name == 'kd_rp': self.pid_roll.kd = self.pid_pitch.kd = float(val)

            # Nadpisywanie parametrów dla obiektu PID_YAW
            elif name == 'kp_yaw': self.pid_yaw_rate.kp = float(val)
            elif name == 'ki_yaw': self.pid_yaw_rate.ki = float(val)
            elif name == 'kd_yaw': self.pid_yaw_rate.kd = float(val)

            # Nadpisywanie parametrów dla obiektu PID_Z
            elif name == 'kp_z': self.pid_z.kp = float(val)
            elif name == 'ki_z': self.pid_z.ki = float(val)
            elif name == 'kd_z': self.pid_z.kd = float(val)

        return SetParametersResult(successful=True)

    def odom_callback(self, msg: Odometry):
        """Serce sterownika wywoływane automatycznie za każdym nowym pomiarem odometrii (np. 50 Hz)."""
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        # Obliczanie realnego czasu (dt), jaki upłynął od ostatniej iteracji pętli
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # --- DETEKCJA RESETU SYMULATORA GAZEBO ---
        # Jeśli dt < 0, oznacza to, że zegar w Gazebo został cofnięty (Reset World).
        if dt < 0.0:
            self.get_logger().warn('Wykryto restart symulacji Gazebo! Czyszczenie pamięci regulatora.')
            self.held_z = None          # Zapominamy starą wysokość
            self.target_vz = 0.0        # Zerujemy komendy prędkości
            self.target_roll = 0.0
            self.target_pitch = 0.0
            self.target_yaw_rate = 0.0

        # Jeśli czas przeskoczył zbyt gwałtownie, resetujemy bufory całek (I) i pochodnych (D)
        if dt <= 0.0 or dt > 0.5:
            for pid in (self.pid_roll, self.pid_pitch, self.pid_yaw_rate, self.pid_z):
                pid.reset()
            return

        # Zczytywanie stanu świata z symulacji
        pos = msg.pose.pose.position
        self.current_z = pos.z
        if self.held_z is None:
            self.held_z = pos.z

        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
        current_yaw_rate = msg.twist.twist.angular.z

        # ------------------- STEROWANIE WYSOKOŚCIĄ Z -------------------
        if self.use_z_pid:
            # Manewr wznoszenia/opadania: presuwamy wirtualny punkt w górę lub dół (z prędkością Vz)
            if abs(self.target_vz) > 1e-4:
                self.held_z = max(0.05, self.held_z + self.target_vz * dt)

            # Logika "Lądowania / Postoju na ziemi" - jeśli dron leży, gasimy silniki by nie buczały
            if self.held_z < 0.35 and self.target_vz <= 0.0 and pos.z < 0.35:
                self.pid_z.reset()
                u_z = 0.0
            else:
                e_z = self.held_z - pos.z
                u_z_corr = self.pid_z.compute(e_z, dt)

                # Zjawisko odchylania ciągu: gdy przechylasz drona do lotu w przód, ucieka mu wektor
                # ciągu pionowego. Tilt_factor kompensuje to, dodając moc proporcjonalnie do kąta.
                tilt_factor = max(0.6, math.cos(roll) * math.cos(pitch))
                u_z = (self.base_throttle + u_z_corr) / tilt_factor
        else:
            self.pid_z.reset()
            u_z = self.base_throttle

        # ------------------- STEROWANIE POZIOME ROLL I PITCH -------------------
        # Aktywne dopiero wtedy, gdy dron ma wyliczony ciąg wznoszący (u_z > 0), by nie sterował leżąc
        if self.use_rp_pid and u_z > 0.0:
            u_phi = self.pid_roll.compute(self.target_roll - roll, dt)
            u_theta = self.pid_pitch.compute(self.target_pitch - pitch, dt)
        else:
            self.pid_roll.reset()
            self.pid_pitch.reset()
            u_phi = u_theta = 0.0

        # ------------------- STEROWANIE YAW RATE (OS Z) -------------------
        if self.use_yaw_pid and u_z > 0.0:
            u_psi = self.pid_yaw_rate.compute(self.target_yaw_rate - current_yaw_rate, dt)
        else:
            self.pid_yaw_rate.reset()
            u_psi = 0.0

        # ------------------- MIKSER KINEMATYCZNY (UKŁAD X) -------------------
        if u_z <= 0.0:
            speeds = [0.0, 0.0, 0.0, 0.0]
        else:
            # Miksujemy poszczególne zadania regulacji obciążając każdy z 4 silników w układzie "Krzyża"
            w1 = u_z - u_phi - u_theta - u_psi
            w2 = u_z + u_phi + u_theta - u_psi
            w3 = u_z + u_phi - u_theta + u_psi
            w4 = u_z - u_phi + u_theta + u_psi
            # Docinanie obrotów by nie wylecieć poza okno fizycznie wspierane w Gazebo (saturacja)
            speeds = [float(max(self.min_motor_speed, min(self.max_motor_speed, w))) for w in (w1, w2, w3, w4)]

        out_msg = Actuators()
        out_msg.velocity = speeds
        self.motor_pub.publish(out_msg)

    def stop_motors(self):
        """Bezpiecznie ubija komendy przesyłane na silniki przy twardym zamknięciu węzła."""
        stop_msg = Actuators()
        stop_msg.velocity = [0.0, 0.0, 0.0, 0.0]
        for _ in range(3):
            self.motor_pub.publish(stop_msg)
            time.sleep(0.02)

def main(args=None):
    # Wyłączenie standardowej obsługi sygnałów, by móc zdefiniować wlasny blok na KeyboardInterrupt
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
