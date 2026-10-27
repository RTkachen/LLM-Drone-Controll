import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators


class PID:
    """
    Klasa realizująca pojedynczy dyskretny regulator PID z ograniczeniem wyjścia
    oraz zabezpieczeniem członu całkującego przed nasyceniem (anti-windup).
    """
    def __init__(self, kp: float, ki: float, kd: float, out_limit: float = 200.0):
        # Wzmocnienia regulatora: proporcjonalne (Kp), całkujące (Ki), różniczkujące (Kd)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # Maksymalna bezwzględna wartość sygnału wyjściowego [rad/s]
        self.out_limit = out_limit

        # Pamięć regulatora między kolejnymi krokami czasowymi
        self.integral = 0.0      # Suma (całka) uchybu w czasie
        self.prev_error = 0.0    # Uchyb z poprzedniego kroku (do obliczenia pochodnej)

    def compute(self, error: float, dt: float) -> float:
        """
        Oblicza wartość sygnału sterującego U na podstawie bieżącego uchybu (error)
        oraz kroku czasowego (dt [s]).
        """
        if dt <= 0.0:
            return 0.0

        # 1. Człon proporcjonalny: P = Kp * e
        p_term = self.kp * error

        # 2. Człon całkujący: I = Ki * suma(e * dt)
        # Zabezpieczenie anti-windup przycina całkę do granic [-out_limit, out_limit]
        self.integral += error * dt
        self.integral = max(-self.out_limit, min(self.out_limit, self.integral))
        i_term = self.ki * self.integral

        # 3. Człon różniczkujący: D = Kd * (de / dt)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        d_term = self.kd * derivative

        # Suma sygnałów P + I + D ograniczona do dopuszczalnego zakresu
        output = p_term + i_term + d_term
        return max(-self.out_limit, min(self.out_limit, output))


def euler_from_quaternion(x: float, y: float, z: float, w: float):
    """
    Przelicza orientację zapisaną jako kwaternion (x, y, z, w)
    na kąty Eulera: roll (phi), pitch (theta), yaw (psi) wyrażone w radianach [rad].
    """
    # Roll (phi) - obrót wokół osi podłużnej X (przechył na boki: lewo/prawo)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (theta) - obrót wokół osi poprzecznej Y (pochylenie dziobu: góra/dół)
    sinp = 2.0 * (w * y - z * x)
    # Zabezpieczenie przed wyjściem poza dziedzinę funkcji arcus sinus [-1.0, 1.0]
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (psi) - obrót wokół osi pionowej Z (kierunek kompasowy dziobu)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class AttitudeControllerNode(Node):
    """
    Węzeł ROS 2 realizujący stabilizację orientacji drona (Roll, Pitch, Yaw)
    przy zadanym ciągu bazowym (base_throttle) oraz miksowanie sygnałów na układ X.
    """
    def __init__(self):
        super().__init__('regulator_pid_node')

        # Deklaracja parametrów konfiguracyjnych węzła (możliwych do zmiany w locie)
        self.declare_parameter('use_pid', True)          # True = stabilizacja włączona, False = brak korekcji (sam ciąg)
        self.declare_parameter('base_throttle', 680.0)   # Bazowa prędkość silników Uz [rad/s] (~662 rad/s to zawis)
        self.declare_parameter('target_roll', 0.0)       # Zadany kąt przechyłu Roll (phi_zad) [rad]
        self.declare_parameter('target_pitch', 0.0)      # Zadany kąt pochylenia Pitch (theta_zad) [rad]
        self.declare_parameter('target_yaw', 0.0)        # Zadany kąt obrotu Yaw (psi_zad) [rad]

        # Fizyczne ograniczenia prędkości kątowej silników w symulatorze [rad/s]
        self.max_motor_speed = 800.0
        self.min_motor_speed = 0.0

        # Inicjalizacja trzech niezależnych regulatorów PID dla poszczególnych osi obrotu
        self.pid_roll = PID(kp=150.0, ki=10.0, kd=40.0, out_limit=150.0)
        self.pid_pitch = PID(kp=150.0, ki=10.0, kd=40.0, out_limit=150.0)
        self.pid_yaw = PID(kp=80.0, ki=0.0, kd=20.0, out_limit=100.0)

        # Zmienna przechowująca znacznik czasu poprzedniej wiadomości odometrii
        self.last_time = None

        # Subskrypcja stanu rzeczywistego drona (odometrii z Gazebo)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/model/X3/odometry',
            self.odom_callback,
            10
        )

        # Publikacja obliczonych prędkości silników (do węzła zakłóceń motor_disturbance)
        self.motor_pub = self.create_publisher(
            Actuators,
            '/X3/command/motor_speed',
            10
        )

        self.get_logger().info('Węzeł regulatora PID orientacji uruchomiony pomyślnie.')

    def odom_callback(self, msg: Odometry):
        """
        Główna pętla sterowania wywoływana automatycznie po każdym odebraniu nowej
        wiadomosci z odometrią (domyślnie 50 razy na sekundę - 50 Hz).
        """
        # 1. Wyznaczenie kroku czasowego dt [s] pomiędzy kolejnymi pomiarami
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        if dt <= 0.0:
            return

        # 2. Pobranie aktualnych wartości parametrów zadanych
        use_pid = self.get_parameter('use_pid').get_parameter_value().bool_value
        u_z = self.get_parameter('base_throttle').get_parameter_value().double_value
        roll_zad = self.get_parameter('target_roll').get_parameter_value().double_value
        pitch_zad = self.get_parameter('target_pitch').get_parameter_value().double_value
        yaw_zad = self.get_parameter('target_yaw').get_parameter_value().double_value

        # 3. Odczyt orientacji (kwaternionu) z odometrii i przeliczenie na kąty Eulera [rad]
        q = msg.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)

        # 4. Wyznaczenie wirtualnych sygnałów sterujących (momentów korekcyjnych)
        if use_pid:
            # Obliczenie uchybów kątowych: e = wartość_zadana - wartość_rzeczywista [rad]
            e_roll = roll_zad - roll
            e_pitch = pitch_zad - pitch
            # Normalizacja uchybu kąta Yaw do najkrótszej drogi obrotu w przedziale [-pi, pi]
            e_yaw = math.atan2(math.sin(yaw_zad - yaw), math.cos(yaw_zad - yaw))

            # Wyliczenie korekt sterujących z regulatorów PID [rad/s]
            u_phi = self.pid_roll.compute(e_roll, dt)
            u_theta = self.pid_pitch.compute(e_pitch, dt)
            u_psi = self.pid_yaw.compute(e_yaw, dt)
        else:
            # Tryb otwartej pętli (bez korekcji PID) - na wszystkie silniki trafia wyłącznie ciąg u_z
            u_phi = 0.0
            u_theta = 0.0
            u_psi = 0.0

        # 5. Kinematyka odwrotna (Mikser układu X) - przeliczenie sygnałów na 4 silniki [rad/s]:
        # w1 -> rotor_0: prawy przód (obroty przeciwnie do wskazówek zegara - CCW)
        w1 = u_z - u_phi - u_theta - u_psi
        # w2 -> rotor_1: lewy tył (obroty przeciwnie do wskazówek zegara - CCW)
        w2 = u_z + u_phi + u_theta - u_psi
        # w3 -> rotor_2: lewy przód (obroty zgodnie ze wskazówkami zegara - CW)
        w3 = u_z + u_phi - u_theta + u_psi
        # w4 -> rotor_3: prawy tył (obroty zgodnie ze wskazówkami zegara - CW)
        w4 = u_z - u_phi + u_theta + u_psi

        # 6. Nasycenie (saturacja) wyliczonych prędkości do fizycznych limitów [0.0, 800.0] [rad/s]
        raw_speeds = [w1, w2, w3, w4]
        clamped_speeds = [
            float(max(self.min_motor_speed, min(self.max_motor_speed, w)))
            for w in raw_speeds
        ]

        # 7. Zbudowanie wiadomości i wysłanie komendy do węzła wykonawczego
        out_msg = Actuators()
        out_msg.velocity = clamped_speeds
        self.motor_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AttitudeControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
