import http.server
import json
import threading
import time
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters


class MostAINode(Node):
    """
    Węzeł mostu AI: odbiera przez sieć HTTP JSON-y z komputera stacjonarnego,
    aplikuje je do regulatora drona za pomocą natywnych usług ROS 2 i zarządza
    czasem trwania manewru (duration_sec).
    """
    def __init__(self):
        super().__init__('most_ai_node')
        self.get_logger().info('Most AI uruchomiony i gotowy na komendy z PC!')

        # Klient usługi ROS 2 do zmiany parametrów w węźle regulatora (/regulator_node)
        self.param_client = self.create_client(SetParameters, '/regulator_node/set_parameters')

        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Oczekiwanie na uruchomienie węzła regulatora...')

    def wyslij_parametry_do_regulatora(self, param_dict: dict):
        """Wysyła słownik parametrów do węzła regulatora przez usługę ROS 2."""
        request = SetParameters.Request()

        for name, val in param_dict.items():
            p = Parameter()
            p.name = name
            val_msg = ParameterValue()

            if isinstance(val, float):
                val_msg.type = ParameterType.PARAMETER_DOUBLE
                val_msg.double_value = val
            elif isinstance(val, int):
                val_msg.type = ParameterType.PARAMETER_INTEGER
                val_msg.integer_value = val
            elif isinstance(val, bool):
                val_msg.type = ParameterType.PARAMETER_BOOL
                val_msg.bool_value = val
            elif isinstance(val, str):
                val_msg.type = ParameterType.PARAMETER_STRING
                val_msg.string_value = val
            else:
                continue

            p.value = val_msg
            request.parameters.append(p)

        # Asynchroniczne wywołanie usługi
        self.param_client.call_async(request)

    def wykonaj_manewr(self, dane_json: dict):
        """Obsługuje pojedynczy manewr wraz z odliczaniem czasu trwania (duration_sec)."""
        target_roll = float(dane_json.get("target_roll", 0.0))
        target_pitch = float(dane_json.get("target_pitch", 0.0))
        target_vz = float(dane_json.get("target_vz", 0.0))
        target_yaw_rate = float(dane_json.get("target_yaw_rate", 0.0))
        duration = float(dane_json.get("duration_sec", 2.0))

        self.get_logger().info(
            f"Wykonuję manewr: Roll={target_roll}, Pitch={target_pitch}, Vz={target_vz}, Czas={duration}s"
        )

        # 1. Aplikowanie parametrów lotu
        self.wyslij_parametry_do_regulatora({
            "target_roll": target_roll,
            "target_pitch": target_pitch,
            "target_vz": target_vz,
            "target_yaw_rate": target_yaw_rate
        })

        # 2. Jeśli podano czas trwania > 0, czekamy, a potem bezpiecznie zatrzymujemy drona
        if duration > 0.0:
            time.sleep(duration)
            self.get_logger().info("Czas manewru minął – automatyczny zawis (stop).")
            self.wyslij_parametry_do_regulatora({
                "target_roll": 0.0,
                "target_pitch": 0.0,
                "target_vz": 0.0,
                "target_yaw_rate": 0.0
            })


class HttpServerHandler(http.server.BaseHTTPRequestHandler):
    """Prosty serwer HTTP odbierający żądania POST z komputera stacjonarnego."""
    ros_node = None  # Referencja do węzła ROS 2

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            dane = json.loads(body.decode('utf-8'))
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "sukces"}).encode('utf-8'))

            # Przekazanie danych do wątku ROS 2, żeby nie blokować serwera HTTP
            threading.Thread(target=self.ros_node.wykonaj_manewr, args=(dane,)).start()

        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(str(e).encode('utf-8'))

    def log_message(self, format, *args):
        # Wyciszenie domyślnych logów HTTP, żeby nie śmiecić w terminalu ROS 2
        return


def start_http_server(node):
    HttpServerHandler.ros_node = node
    # Nasłuchiwanie na wszystkich interfejsach sieciowych laptopa na porcie 5000
    server = http.server.HTTPServer(('0.0.0.0', 5000), HttpServerHandler)
    server.serve_forever()


def main(args=None):
    rclpy.init(args=args)
    node = MostAINode()

    # Uruchomienie serwera HTTP w osobnym tle (wątku demona)
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
