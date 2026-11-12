import random
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from actuator_msgs.msg import Actuators


class MotorDisturbanceNode(Node):
    def __init__(self):
        super().__init__('motor_disturbance_node')

        # 1. Deklaracja parametrów w ROS 2
        self.declare_parameter('alpha', 0.2)          # Współczynnik filtru LPF [-] (0.0 - 1.0)
        self.declare_parameter('base_stddev', 30.0)   # Bazowe odchylenie standardowe szumu Gaussa [rad/s]
        self.declare_parameter('max_speed', 800.0)    # Fizyczny limit prędkości silników [rad/s]

        # 2. Wczytanie wartości początkowych do szybkich zmiennych lokalnych obiektu
        self.alpha = self.get_parameter('alpha').value
        self.base_stddev = self.get_parameter('base_stddev').value
        self.max_speed = self.get_parameter('max_speed').value

        # 3. Rejestracja callbacku dla dynamicznej zmiany parametrów (zamiast pollingu)
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.target_speeds = [0.0, 0.0, 0.0, 0.0]
        self.filtered_speeds = [0.0, 0.0, 0.0, 0.0]

        self.sub = self.create_subscription(
            Actuators,
            '/X3/command/motor_speed',
            self.cmd_callback,
            10
        )

        self.pub = self.create_publisher(
            Actuators,
            '/X3/gazebo/command/motor_speed',
            10
        )

        self.timer = self.create_timer(0.01, self.update_motors)
        self.get_logger().info(
            f'Węzeł zaburzeń uruchomiony (alpha={self.alpha}, base_stddev={self.base_stddev} rad/s).'
        )

    def parameters_callback(self, params):
        """Callback wywoływany zdarzeniowo tylko w momencie zmiany parametru."""
        for param in params:
            if param.name == 'alpha':
                if not (0.0 <= param.value <= 1.0):
                    self.get_logger().warn('Odrzucono parametr alpha (musi być w przedziale 0.0 - 1.0)!')
                    return SetParametersResult(successful=False, reason='alpha out of range [0, 1]')
                self.alpha = float(param.value)
                self.get_logger().info(f'Zaktualizowano alpha = {self.alpha}')

            elif param.name == 'base_stddev':
                if param.value < 0.0:
                    return SetParametersResult(successful=False, reason='base_stddev must be >= 0')
                self.base_stddev = float(param.value)
                self.get_logger().info(f'Zaktualizowano base_stddev = {self.base_stddev} rad/s')

            elif param.name == 'max_speed':
                if param.value <= 0.0:
                    return SetParametersResult(successful=False, reason='max_speed must be > 0')
                self.max_speed = float(param.value)
                self.get_logger().info(f'Zaktualizowano max_speed = {self.max_speed} rad/s')

        return SetParametersResult(successful=True)

    def cmd_callback(self, msg: Actuators):
        if len(msg.velocity) >= 4:
            self.target_speeds = list(msg.velocity[:4])

    def update_motors(self):
        out_msg = Actuators()
        out_velocities = []

        for i in range(4):
            u = self.target_speeds[i]

            if u <= 1.0:
                noisy_u = 0.0
            else:
                sigma = self.base_stddev * (0.5 + 0.5 * (u / self.max_speed))
                noise = random.gauss(0.0, sigma)
                noisy_u = u + noise

            # Filtr dolnoprzepustowy I rzędu: y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
            self.filtered_speeds[i] = (
                self.alpha * noisy_u + (1.0 - self.alpha) * self.filtered_speeds[i]
            )

            val = max(0.0, min(self.max_speed, self.filtered_speeds[i]))
            out_velocities.append(float(val))

        out_msg.velocity = out_velocities
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotorDisturbanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
