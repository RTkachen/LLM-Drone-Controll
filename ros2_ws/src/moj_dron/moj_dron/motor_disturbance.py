import random
import rclpy
from rclpy.node import Node
from actuator_msgs.msg import Actuators


class MotorDisturbanceNode(Node):
    def __init__(self):
        # Inicjalizacja węzła ROS2 pod nazwą 'motor_disturbance_node'
        super().__init__('motor_disturbance_node')

        # Parametry z notatek opiekuna
        self.alpha = 0.2          # Współczynnik filtru dolnoprzepustowego [- (od 0.0 do 1.0)] (wygładzanie i inercja)
        self.base_stddev = 10.0   # Bazowe odchylenie standardowe szumu Gaussa [rad/s]
        self.max_speed = 800.0    # Fizyczny limit prędkości kątowej silników X3 w Gazebo [rad/s]

        # Listy przechowujące prędkości [rad/s] dla 4 silników: [rotor_0, rotor_1, rotor_2, rotor_3]
        self.target_speeds = [0.0, 0.0, 0.0, 0.0]    # Prędkości zadane przez sterownik
        self.filtered_speeds = [0.0, 0.0, 0.0, 0.0]  # Prędkości po nałożeniu szumu i filtracji

        # Subskrypcja czystego sygnału sterującego (z przyszłego regulatora PID)
        # Kolejka (qos_profile) ustawiona na 10 wiadomości
        self.sub = self.create_subscription(
            Actuators,
            '/X3/command/motor_speed',
            self.cmd_callback,
            10
        )

        # Publikacja zaszumionego i przefiltrowanego sygnału do Gazebo
        self.pub = self.create_publisher(
            Actuators,
            '/X3/gazebo/command/motor_speed',
            10
        )

        # Pętla główna wywoływana z częstotliwością 100 Hz (okres T = 0.01 [s])
        self.timer = self.create_timer(0.01, self.update_motors)
        self.get_logger().info('Węzeł zaburzeń silników (Gauss + LPF) uruchomiony!')

    def cmd_callback(self, msg: Actuators):
        # Pobranie pierwszych 4 wartości prędkości kątowych [rad/s] z odebranej wiadomości
        if len(msg.velocity) >= 4:
            self.target_speeds = list(msg.velocity[:4])

    def update_motors(self):
        out_msg = Actuators()
        out_velocities = []

        for i in range(4):
            u = self.target_speeds[i]  # Aktualna prędkość zadana dla i-tego silnika [rad/s]

            # Strefa nieczułości (deadband): poniżej 1.0 [rad/s] silnik uznajemy za wyłączony,
            # aby stojący dron nie kręcił losowo śmigłami od samego szumu.
            if u <= 1.0:
                noisy_u = 0.0
            else:
                # Wzór na liniowe skalowanie odchylenia standardowego szumu (sigma) w funkcji obrotów:
                # sigma(u) = sigma_bazowe * (0.5 + 0.5 * (u / u_max)) [rad/s]
                # Dla u -> 0 szum wynosi 50% wartości bazowej (5 rad/s).
                # Dla u = 800 (max) szum osiąga 100% wartości bazowej (10 rad/s).
                sigma = self.base_stddev * (0.5 + 0.5 * (u / self.max_speed))

                # Losowanie zaburzenia z rozkładu normalnego (Gaussa) o średniej 0.0 [rad/s] i odchyleniu 'sigma' [rad/s]
                noise = random.gauss(0.0, sigma)

                # Nałożenie wylosowanego szumu na sygnał zadany: x[k] = u[k] + n[k] [rad/s]
                noisy_u = u + noise

            # Dyskretny filtr dolnoprzepustowy I rzędu (LPF - Low-Pass Filter / wykładnicza średnia krocząca EMA):
            # Wzór: y[k] = alpha * x[k] + (1 - alpha) * y[k-1] [rad/s]
            # Gdzie:
            #   y[k]   - nowa przefiltrowana prędkość w kroku k
            #   x[k]   - obecna zaszumiona prędkość (noisy_u)
            #   y[k-1] - poprzednia przefiltrowana prędkość z kroku k-1
            #   alpha  - waga nowej próbki (0.2 oznacza 20% nowego sygnału i 80% historii, co symuluje bezwładność wirnika)
            self.filtered_speeds[i] = (
                self.alpha * noisy_u + (1.0 - self.alpha) * self.filtered_speeds[i]
            )

            # Nasycenie (clamping / saturacja) sygnału do fizycznych granic silnika: [0.0, 800.0] [rad/s]
            # Wzór: val = min(u_max, max(0.0, y[k]))
            val = max(0.0, min(self.max_speed, self.filtered_speeds[i]))
            out_velocities.append(float(val))

        # Przypisanie obliczonych prędkości do wiadomości i wysłanie do Gazebo
        out_msg.velocity = out_velocities
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)               # Inicjalizacja komunikacji ROS2
    node = MotorDisturbanceNode()       # Utworzenie obiektu węzła (wywołuje __init__)
    rclpy.spin(node)                    # Nieskończona pętla utrzymująca węzeł przy życiu (obsługa timera i subskrypcji)
    node.destroy_node()                 # Sprzątanie zasobów węzła po zamknięciu (np. po Ctrl+C)
    rclpy.shutdown()                    # Zamknięcie kontekstu ROS2


if __name__ == '__main__':
    main()
