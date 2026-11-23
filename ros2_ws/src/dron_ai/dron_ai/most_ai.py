import http.server
import json
import threading
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters


class MostAINode(Node):
    def __init__(self):
        super().__init__('most_ai_node')
        self.get_logger().info('Most AI (tryb Velocity + Position + GET /state) uruchomiony.')

        self.param_client = self.create_client(SetParameters, '/regulator_node/set_parameters')
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Oczekiwanie na /regulator_node/set_parameters...')

        self.current_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.current_mode = "velocity"
        self.maneuver_id = 0
        self.lock = threading.Lock()

        self.odom_sub = self.create_subscription(
            Odometry, '/model/X3/odometry', self.odom_callback, 10
        )

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        self.current_pos = {
            "x": round(float(p.x), 2),
            "y": round(float(p.y), 2),
            "z": round(float(p.z), 2),
        }

    def get_state_dict(self) -> dict:
        return {
            "mode": self.current_mode,
            "position": self.current_pos,
        }

    def send_parameters_to_controller(self, param_dict: dict):
        request = SetParameters.Request()
        for name, val in param_dict.items():
            p = Parameter()
            p.name = name
            val_msg = ParameterValue()
            # bool jest podklasą int w Pythonie, więc sprawdzamy bool PRZED int
            if isinstance(val, bool):
                val_msg.type = ParameterType.PARAMETER_BOOL
                val_msg.bool_value = val
            elif isinstance(val, float):
                val_msg.type = ParameterType.PARAMETER_DOUBLE
                val_msg.double_value = val
            elif isinstance(val, int):
                val_msg.type = ParameterType.PARAMETER_INTEGER
                val_msg.integer_value = val
            elif isinstance(val, str):
                val_msg.type = ParameterType.PARAMETER_STRING
                val_msg.string_value = val
            else:
                continue
            p.value = val_msg
            request.parameters.append(p)

        self.param_client.call_async(request)

    def execute_maneuver(self, json_data: dict):
        with self.lock:
            self.maneuver_id += 1
            my_id = self.maneuver_id

        mode = str(json_data.get("mode", "velocity")).lower()

        # ---------------- TRYB POZYCYJNY ----------------
        if mode == "position":
            self.current_mode = "position"
            hold_current = bool(json_data.get("hold_current", False))

            if hold_current:
                self.get_logger().info("POSITION HOLD -> zatrzaskiwanie bieżącej pozycji [X, Y, Z].")
                self.send_parameters_to_controller({
                    "use_pos_pid": True,
                    "target_yaw_rate": 0.0,
                })
            else:
                tx = float(json_data.get("target_x", self.current_pos["x"]))
                ty = float(json_data.get("target_y", self.current_pos["y"]))
                tz = float(json_data.get("target_z", self.current_pos["z"]))
                self.get_logger().info(f"POSITION GOTO -> X={tx:.2f}, Y={ty:.2f}, Z={tz:.2f}")
                self.send_parameters_to_controller({
                    "use_pos_pid": True,
                    "target_x": tx,
                    "target_y": ty,
                    "target_z": tz,
                    "target_yaw_rate": 0.0,
                })
            return

        # ---------------- TRYB PRĘDKOŚCIOWY ----------------
        self.current_mode = "velocity"
        target_roll = float(json_data.get("target_roll", 0.0))
        target_pitch = float(json_data.get("target_pitch", 0.0))
        target_vz = float(json_data.get("target_vz", 0.0))
        target_yaw_rate = float(json_data.get("target_yaw_rate", 0.0))
        duration = float(json_data.get("duration_sec", 2.0))
        auto_hold = bool(json_data.get("auto_hold_after", True))

        self.get_logger().info(
            f"VELOCITY -> R={target_roll}, P={target_pitch}, Vz={target_vz}, Yaw={target_yaw_rate}, t={duration}s"
        )

        self.send_parameters_to_controller({
            "use_pos_pid": False,
            "target_roll": target_roll,
            "target_pitch": target_pitch,
            "target_vz": target_vz,
            "target_yaw_rate": target_yaw_rate,
        })

        if duration > 0.0:
            time.sleep(duration)
            if my_id != self.maneuver_id:
                return  # Nadeszła nowsza komenda

            # 1. Wyrównanie kątów i zatrzymanie Vz
            self.send_parameters_to_controller({
                "target_roll": 0.0,
                "target_pitch": 0.0,
                "target_vz": 0.0,
                "target_yaw_rate": 0.0,
            })

            # 2. Krótka pauza na wytracenie pędu i automatyczne zakotwiczenie pozycji
            if auto_hold:
                time.sleep(0.4)
                if my_id != self.maneuver_id:
                    return
                self.current_mode = "position"
                self.get_logger().info("Koniec manewru -> automatyczne przejście w POSITION HOLD.")
                self.send_parameters_to_controller({"use_pos_pid": True})


class HttpServerHandler(http.server.BaseHTTPRequestHandler):
    ros_node = None

    def do_GET(self):
        if self.path == '/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.ros_node.get_state_dict()).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body.decode('utf-8'))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode('utf-8'))
            threading.Thread(target=self.ros_node.execute_maneuver, args=(data,), daemon=True).start()
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def log_message(self, format, *args):
        return


def start_http_server(node):
    HttpServerHandler.ros_node = node
    server = http.server.HTTPServer(('0.0.0.0', 5000), HttpServerHandler)
    server.serve_forever()


def main(args=None):
    rclpy.init(args=args)
    node = MostAINode()
    threading.Thread(target=start_http_server, args=(node,), daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()