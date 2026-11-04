import math
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from actuator_msgs.msg import Actuators


class PID:
    """
    Klasa realizująca pojedynczy dyskretny regulator PID z ograniczeniem wyjścia,
    zabezpieczeniem anti-windup oraz ochroną przed uderzeniem różniczkującym (derivative kick).
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
        self.first_run = True    # Flaga chroniąca przed szarpnięciem członu D przy włączeniu PID

    def reset(self):
        """
        Zeruje pamięć wewnętrzną regulatora.
        Ustawia flagę first_run = True, aby po ponownym włączeniu nie wystąpił skok pochodnej.
        """
        self.integral = 0.0
        self.prev_error = 0.0
        self.first_run = True

    def compute(self, error: float, dt: float) -> float:
        """
        Oblicza wartość sygnału sterującego U na podstawie bieżącego uchybu (error)
        oraz kroku czasowego (dt [s]).
        """
        if dt <= 0.0:
            return 0.0

        # 1. Człon proporcjonalny: P = Kp * e
        p_term = self.kp * error

        # 2. Człon całkujący: I = Ki * suma(e * dt) z ograniczeniem anti-windup
        self.integral += error * dt
        self.integral = max(-self.out_limit, min(self.out_limit, self.integral))
        i_term = self.ki * self.integral

        # 3. Człon różniczkujący: D = Kd * (de / dt)
        # W pierwszym kroku po starcie lub ponownym włączeniu przyjmujemy pochodną = 0.0,
        # aby uniknąć gwałtownego uderzenia w silniki (Derivative Kick).
        if self.first_run:
            derivative = 0.0
            self.first_run = False
        else:
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
    Uniwersalny węzeł ROS 2 realizujący:
    1. Stabilizację kątów przechyłu Roll i Pitch (włączaną parametrem use_rp_pid).
    2. Regulację prędkości kątowej obrotu wokół własnej osi Yaw (włączaną parametrem use_yaw_pid).
    3. Miksowanie sygnałów sterujących z ciągiem bazowym (base_throttle) na 4 silniki układu X.
    """
    def __init__(self):
        super().__init__('regulator_node')

        # --- DEKLARACJA PARAMETRÓW KONFIGURACYJNYCH WĘZŁA ---
        self.declare_parameter('use_rp_pid', True)        # True = stabilizacja kątów Roll i Pitch włączona
        self.declare_parameter('use_yaw_pid', False)      # True = regulator prędkości kątowej Yaw włączony

        self.declare_parameter('base_throttle', 665.0)    # Bazowa prędkość silników Uz [rad/s] (~660.4 rad/s to zawis)
        self.declare_parameter('target_roll', 0.0)        # Zadany kąt przechyłu Roll (phi_zad) [rad]
        self.declare_parameter('target_pitch', 0.0)       # Zadany kąt pochylenia Pitch (theta_zad) [rad]
        self.declare_parameter('target_yaw_rate', 0.8)    # Zadana PRĘDKOŚĆ KĄTOWA obrotu Yaw (omega_z_zad) [rad/s]

        # Fizyczne ograniczenia prędkości kątowej silników w symulatorze Gazebo [rad/s]
        self.max_motor_speed = 800.0
        self.min_motor_speed = 0.0

        # Inicjalizacja dwóch regulatorów PID dla kątów orientacji poziomej (Roll i Pitch)
        self.pid_roll = PID(kp=150.0, ki=10.0, kd=40.0, out_limit=150.0)
        self.pid_pitch = PID(kp=150.0, ki=10.0, kd=40.0, out_limit=150.0)

        # Inicjalizacja regulatora dla PRĘDKOŚCI KĄTOWEJ obrotu (Yaw Rate)
        # Ki = 0.0, ponieważ brak tarcia obrotowego w osi Z sprawia, że obiekt sam całkuje moment obrotowy.
        # Dzięki Ki = 0.0 po osiągnięciu zadanej prędkości moment spada do zera i dron nie przyspiesza dalej.
        self.pid_yaw_rate = PID(kp=200.0, ki=0.0, kd=5.0, out_limit=100.0)

        self.last_time = None
        self.step_counter = 0

        # Subskrypcja stanu rzeczywistego drona (odometrii z Gazebo - 50 Hz)
        self.odom_sub = self.create_subscription(
            Odometry,
            '/model/X3/odometry',
            self.odom_callback,
            10
        )

        # Publikacja obliczonych prędkości silników (do węzła motor_disturbance)
        self.motor_pub = self.create_publisher(
            Actuators,
            '/X3/command/motor_speed',
            10
        )

        self.get_logger().info('Węzeł sterownika (Roll/Pitch + Yaw Rate) uruchomiony pomyślnie.')

    def stop_motors(self):
        """
        Wysyła kilkukrotnie komendę zatrzymania wszystkich 4 silników [0.0, 0.0, 0.0, 0.0],
        aby mieć pewność, że węzeł motor_disturbance wyzeruje obroty przed zamknięciem.
        """
        stop_msg = Actuators()
        stop_msg.velocity = [0.0, 0.0, 0.0, 0.0]
        for _ in range(3):
            self.motor_pub.publish(stop_msg)
            time.sleep(0.02)
        self.get_logger().info('Wysłano [0.0, 0.0, 0.0, 0.0] - silniki bezpiecznie zatrzymane.')

    def odom_callback(self, msg: Odometry):
        """
        Główna pętla sterowania wywoływana po każdym odebraniu odometrii (50 Hz).
        """
        # 1. Wyznaczenie kroku czasowego dt [s]
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Jeśli symulacja została zresetowana (dt ujemne) lub krok jest zbyt duży, resetujemy pamięć
        if dt <= 0.0 or dt > 0.5:
            self.pid_roll.reset()
            self.pid_pitch.reset()
            self.pid_yaw_rate.reset()
            return

        # 2. Pobranie aktualnych wartości przełączników i parametrów zadanych
        use_rp_pid = self.get_parameter('use_rp_pid').get_parameter_value().bool_value
        use_yaw_pid = self.get_parameter('use_yaw_pid').get_parameter_value().bool_value

        u_z = self.get_parameter('base_throttle').get_parameter_value().double_value
        roll_zad = self.get_parameter('target_roll').get_parameter_value().double_value
        pitch_zad = self.get_parameter('target_pitch').get_parameter_value().double_value
        yaw_rate_zad = self.get_parameter('target_yaw_rate').get_parameter_value().double_value

        # 3. Odczyt orientacji z odometrii i przeliczenie na kąty Eulera [rad]
        q = msg.pose.pose.orientation
        roll, pitch, _ = euler_from_quaternion(q.x, q.y, q.z, q.w)

        # 4. Odczyt rzeczywistej PRĘDKOŚCI KĄTOWEJ obrotu wokół osi Z [rad/s]
        current_yaw_rate = msg.twist.twist.angular.z

        # 5. Wyznaczenie sygnałów sterujących dla osi Roll i Pitch
        if use_rp_pid:
            e_roll = roll_zad - roll
            e_pitch = pitch_zad - pitch
            u_phi = self.pid_roll.compute(e_roll, dt)
            u_theta = self.pid_pitch.compute(e_pitch, dt)
        else:
            self.pid_roll.reset()
            self.pid_pitch.reset()
            u_phi = 0.0
            u_theta = 0.0

        # 6. Wyznaczenie sygnału sterującego dla prędkości kątowej Yaw
        if use_yaw_pid:
            e_yaw_rate = yaw_rate_zad - current_yaw_rate
            u_psi = self.pid_yaw_rate.compute(e_yaw_rate, dt)
        else:
            self.pid_yaw_rate.reset()
            u_psi = 0.0

        # 7. Kinematyka odwrotna (Mikser układu X) [rad/s]
        w1 = u_z - u_phi - u_theta - u_psi  # rotor_0: prawy przód (CCW)
        w2 = u_z + u_phi + u_theta - u_psi  # rotor_1: lewy tył (CCW)
        w3 = u_z + u_phi - u_theta + u_psi  # rotor_2: lewy przód (CW)
        w4 = u_z - u_phi + u_theta + u_psi  # rotor_3: prawy tył (CW)

        # 8. Saturacja wyliczonych prędkości do [0.0, 800.0] [rad/s]
        raw_speeds = [w1, w2, w3, w4]
        clamped_speeds = [
            float(max(self.min_motor_speed, min(self.max_motor_speed, w)))
            for w in raw_speeds
        ]

        # 9. Publikacja komendy do silników
        out_msg = Actuators()
        out_msg.velocity = clamped_speeds
        self.motor_pub.publish(out_msg)

        # 10. Wypisywanie diagnostyki w terminalu raz na sekundę (co 50 próbek przy 50 Hz)
        self.step_counter += 1
        if self.step_counter >= 50:
            self.step_counter = 0
            self.get_logger().info(
                f'RP_PID: {use_rp_pid} (Roll={roll:.2f}, Pitch={pitch:.2f} rad) | '
                f'YAW_PID: {use_yaw_pid} (Zadana={yaw_rate_zad:.2f}, Aktualna={current_yaw_rate:.2f} rad/s)'
            )


def main(args=None):
    # Wyłączenie domyślnego zamykania kontekstu przez C++ przy Ctrl+C
    rclpy.init(args=args, signal_handler_options=rclpy.SignalHandlerOptions.NO)
    node = AttitudeControllerNode()

    try:
        # Pętla spin_once wraca do Pythona co maksymalnie 0.05 s, dzięki czemu Ctrl+C
        # natychmiast przerywa pętlę bez zawieszania procesu i bez zakleszczenia pamięci C++.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.get_logger().info('Wykryto Ctrl+C! Zatrzymywanie silników drona...')
        node.stop_motors()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
