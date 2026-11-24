import json
import math
import requests

# ==============================================================================
# 1. KONFIGURACJA POŁĄCZENIA
# ==============================================================================
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
LAPTOP_URL = "http://192.168.1.56:5000"
MODEL_NAME = "qwen3:8b"
MAX_HISTORY_TURNS = 6

# ==============================================================================
# 2. KONFIGURACJA PARAMETRÓW MODELU LLM
# ==============================================================================
MODEL_OPTIONS = {
    "num_ctx": 8192,
    "num_predict": 450,
    "temperature": 0.0,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.05,
    "num_gpu": 99,
    "seed": 42
}
KEEP_ALIVE = "30m"

# ==============================================================================
# 3. SCHEMAT SEMANTYCZNY (ZERO ZNAKÓW +/- I ARYTMETYKI PO STRONIE LLM)
# ==============================================================================
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "odpowiedz": {"type": "string"},
        "action": {
            "type": "string",
            "enum": ["set_altitude", "goto_point", "move_relative", "velocity", "land", "stop"]
        },
        "przod_tyl": {"type": "string", "enum": ["brak", "przod", "tyl"]},
        "lewo_prawo": {"type": "string", "enum": ["brak", "lewo", "prawo"]},
        "gora_dol": {"type": "string", "enum": ["brak", "gora", "dol"]},
        "obrot": {"type": "string", "enum": ["brak", "lewo", "prawo"]},
        "dystans_przod_tyl": {"type": "number"},
        "dystans_lewo_prawo": {"type": "number"},
        "dystans_gora_dol": {"type": "number"},
        "dystans_skos": {"type": "number"},
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
        "predkosc": {"type": "string", "enum": ["powoli", "normalnie", "szybko"]},
        "duration_sec": {"type": "number"}
    },
    "required": [
        "odpowiedz", "action",
        "przod_tyl", "lewo_prawo", "gora_dol", "obrot",
        "dystans_przod_tyl", "dystans_lewo_prawo", "dystans_gora_dol", "dystans_skos",
        "x", "y", "z", "predkosc", "duration_sec"
    ]
}

# Krótka przypominajka doklejana do każdej wiadomości (zapobiega degradacji z czasem)
REMINDER_JSON = {
    "akcje": {
        "set_altitude": "docelowa wysokość nad ziemią (np. 'na 9 m') -> ustaw z",
        "goto_point": "konkretne współrzędne punktu (np. 'punkt 0 0 3') -> ustaw x, y, z",
        "move_relative": "przesunięcie o podaną liczbę metrów -> ustaw kierunki + nieujemne dystanse (dla łącznego dystansu po skosie wpisz dystans_skos)",
        "velocity": "lot/obrót na czas lub bez podanych metrów -> ustaw kierunki, predkosc i duration_sec (domyślnie 2.0)",
        "land": "wyląduj",
        "stop": "zatrzymaj się / utrzymaj pozycję / zawiśnij"
    },
    "wazne": "Kierunki wybieraj słownie (przod/tyl, lewo/prawo, gora/dol). Dystanse ZAWSZE dodatnie. Niewykorzystane kierunki='brak', liczby=0.0."
}

SYSTEM_PROMPT = """Jesteś tłumaczem poleceń operatora drona X3 na strukturę JSON.
NIE przeliczasz współrzędnych ani znaków +/- — wybierz wyłącznie właściwą akcję, słowne kierunki i dodatnie wartości.
W polu "odpowiedz" napisz krótko po polsku, jaki dokładnie manewr wykonujesz (nie powtarzaj szablonu "Zrozumiałem polecenie").

PRZYKŁADY (niewymienione kierunki="brak", liczby=0.0, predkosc="normalnie"):
- "leć do góry na 9 metrów" -> {"odpowiedz":"Wznoszę się na wysokość 9 m.", "action":"set_altitude", "z":9.0}
- "poleć 8 metrów do góry" -> {"odpowiedz":"Lecę 8 m w górę.", "action":"move_relative", "gora_dol":"gora", "dystans_gora_dol":8.0}
- "poleć po skosie w lewo w przód 7 metrów" -> {"odpowiedz":"Lecę 7 m po skosie w przód i w lewo.", "action":"move_relative", "przod_tyl":"przod", "lewo_prawo":"lewo", "dystans_skos":7.0}
- "poleć po skosie w prawo w przód 5 metrów" -> {"odpowiedz":"Lecę 5 m po skosie w przód i w prawo.", "action":"move_relative", "przod_tyl":"przod", "lewo_prawo":"prawo", "dystans_skos":5.0}
- "poleć do przodu 3 metry i w lewo 2 metry" -> {"odpowiedz":"Lecę 3 m do przodu i 2 m w lewo.", "action":"move_relative", "przod_tyl":"przod", "lewo_prawo":"lewo", "dystans_przod_tyl":3.0, "dystans_lewo_prawo":2.0}
- "poleć w prawo 2 metry" -> {"odpowiedz":"Przesuwam się 2 m w prawo.", "action":"move_relative", "lewo_prawo":"prawo", "dystans_lewo_prawo":2.0}
- "poleć po skosie w prawo w przód przez 4 sekundy" -> {"odpowiedz":"Lecę po skosie w przód i w prawo przez 4 s.", "action":"velocity", "przod_tyl":"przod", "lewo_prawo":"prawo", "predkosc":"normalnie", "duration_sec":4.0}
- "poleć do góry przez 5 sekund szybko" -> {"odpowiedz":"Lecę szybko w górę przez 5 s.", "action":"velocity", "gora_dol":"gora", "predkosc":"szybko", "duration_sec":5.0}
- "poleć do punktu 0.0 0.0 3.0" -> {"odpowiedz":"Lecę do punktu (0, 0, 3).", "action":"goto_point", "x":0.0, "y":0.0, "z":3.0}
- "utrzymaj pozycję" -> {"odpowiedz":"Zatrzymuję się i utrzymuję bieżącą pozycję.", "action":"stop"}
- "wyląduj" -> {"odpowiedz":"Rozpoczynam lądowanie.", "action":"land"}"""


def clean_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


def estimate_true_history_tokens(chat_history: list) -> int:
    total_chars = sum(len(msg["content"]) for msg in chat_history)
    return int(total_chars / 3.2)


def get_drone_state() -> dict:
    try:
        res = requests.get(f"{LAPTOP_URL}/state", timeout=1)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException:
        return {"mode": "nieznany", "position": {"x": 0.0, "y": 0.0, "z": 0.0}}


def build_ros_command(llm_json: dict, drone_state: dict) -> dict:
    """Deterministycznie tłumaczy słowną intencję LLM na znaki ROS 2 (FLU: +X przód, +Y lewo, +Z góra)."""
    pos = drone_state.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})
    cx = float(pos.get("x", 0.0))
    cy = float(pos.get("y", 0.0))
    cz = float(pos.get("z", 0.0))

    action = str(llm_json.get("action", "stop")).lower()

    # Mapowanie kierunków słownych na znaki osi FLU (+X przód, +Y lewo, +Z góra)
    fb = str(llm_json.get("przod_tyl", "brak")).lower()
    lr = str(llm_json.get("lewo_prawo", "brak")).lower()
    ud = str(llm_json.get("gora_dol", "brak")).lower()
    rot = str(llm_json.get("obrot", "brak")).lower()

    sx = 1.0 if fb == "przod" else (-1.0 if fb == "tyl" else 0.0)
    sy = 1.0 if lr == "lewo" else (-1.0 if lr == "prawo" else 0.0)
    sz = 1.0 if ud == "gora" else (-1.0 if ud == "dol" else 0.0)
    syaw = 1.0 if rot == "lewo" else (-1.0 if rot == "prawo" else 0.0)

    if action == "set_altitude":
        z_val = float(llm_json.get("z", 0.0))
        if z_val <= 0.0:
            z_val = abs(float(llm_json.get("dystans_gora_dol", cz)))
        return {"mode": "position", "target_x": cx, "target_y": cy, "target_z": round(max(0.0, z_val), 2)}

    if action == "goto_point":
        tx = round(float(llm_json.get("x", cx)), 2)
        ty = round(float(llm_json.get("y", cy)), 2)
        tz = round(max(0.0, float(llm_json.get("z", cz))), 2)
        return {"mode": "position", "target_x": tx, "target_y": ty, "target_z": tz}

    if action == "move_relative":
        d_fb = abs(float(llm_json.get("dystans_przod_tyl", 0.0)))
        d_lr = abs(float(llm_json.get("dystans_lewo_prawo", 0.0)))
        d_ud = abs(float(llm_json.get("dystans_gora_dol", 0.0)))
        d_diag = abs(float(llm_json.get("dystans_skos", 0.0)))

        active_axes = int(sx != 0.0) + int(sy != 0.0) + int(sz != 0.0)
        dx = dy = dz = 0.0

        if d_diag > 0.0 and active_axes > 0:
            norm = math.sqrt(sx * sx + sy * sy + sz * sz)
            dx = (sx / norm) * d_diag
            dy = (sy / norm) * d_diag
            dz = (sz / norm) * d_diag
        elif active_axes >= 2:
            # Zabezpieczenie: jeśli przy skosie model wpisał dystans w tylko jedno pole zamiast dystans_skos
            nonzero_active = [d for s, d in ((sx, d_fb), (sy, d_lr), (sz, d_ud)) if s != 0.0 and d > 0.0]
            if len(nonzero_active) == 1:
                norm = math.sqrt(sx * sx + sy * sy + sz * sz)
                dx = (sx / norm) * nonzero_active[0]
                dy = (sy / norm) * nonzero_active[0]
                dz = (sz / norm) * nonzero_active[0]
            else:
                dx = sx * d_fb
                dy = sy * d_lr
                dz = sz * d_ud
        elif active_axes == 1:
            d_single = d_fb or d_lr or d_ud or d_diag
            dx = sx * d_single
            dy = sy * d_single
            dz = sz * d_single

        tx = round(cx + dx, 2)
        ty = round(cy + dy, 2)
        min_safe_z = 0.5 if cz >= 0.5 else cz
        tz = round(max(min_safe_z, cz + dz), 2) if dz < 0.0 else round(max(0.0, cz + dz), 2)
        return {"mode": "position", "target_x": tx, "target_y": ty, "target_z": tz}

    if action == "land":
        return {"mode": "position", "target_x": cx, "target_y": cy, "target_z": 0.0}

    if action == "stop":
        return {"mode": "position", "target_x": cx, "target_y": cy, "target_z": cz}

    # Domyślnie: action == "velocity"
    speed = str(llm_json.get("predkosc", "normalnie")).lower()
    tilt_val = {"powoli": 0.05, "normalnie": 0.15, "szybko": 0.25}.get(speed, 0.15)
    vz_val = {"powoli": 0.2, "normalnie": 0.5, "szybko": 1.0}.get(speed, 0.5)
    yaw_val = {"powoli": 0.3, "normalnie": 0.6, "szybko": 1.0}.get(speed, 0.6)

    # Uwaga na konwencję Roll: lewo (sy=+1) to ujemny Roll (-tilt), prawo (sy=-1) to dodatni Roll (+tilt)
    pitch = sx * tilt_val
    roll = -sy * tilt_val
    vz = sz * vz_val
    yaw_rate = syaw * yaw_val

    duration = float(llm_json.get("duration_sec", 2.0))
    if duration <= 0.0:
        duration = 2.0

    return {
        "mode": "velocity",
        "target_roll": round(roll, 3),
        "target_pitch": round(pitch, 3),
        "target_vz": round(vz, 3),
        "target_yaw_rate": round(yaw_rate, 3),
        "duration_sec": round(duration, 2)
    }


def send_message_to_llm(chat_history: list, current_user_input: str, drone_state: dict):
    wrapped_user_msg = json.dumps({
        "stan_drona": drone_state,
        "przypomnienie": REMINDER_JSON,
        "polecenie": current_user_input
    }, ensure_ascii=False)

    messages_for_api = chat_history + [{"role": "user", "content": wrapped_user_msg}]

    payload = {
        "model": MODEL_NAME,
        "messages": messages_for_api,
        "stream": False,
        "think": False,
        "options": MODEL_OPTIONS,
        "keep_alive": KEEP_ALIVE,
        "format": JSON_SCHEMA
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    raw_content = data.get("message", {}).get("content", "")
    return clean_json_response(raw_content), data


def send_command_to_laptop(command_json: dict):
    try:
        requests.post(LAPTOP_URL, json=command_json, timeout=2).raise_for_status()
        print(" -> [ROS 2]: OK.")
    except requests.exceptions.RequestException as e:
        print(f" -> [ROS 2 BŁĄD]: {e}")


def main():
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"--- AI DRONE CHAT ({MODEL_NAME}) | Komendy: /reset, /wyjdz ---")

    while True:
        try:
            user_input = input("\nKomenda: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ["/wyjdz", "exit", "quit"]:
            break
        if user_input.lower() == "/reset":
            chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(" -> [INFO]: Pamięć wyczyszczona.")
            continue

        drone_state = get_drone_state()
        diag_data = {}

        try:
            llm_json, diag_data = send_message_to_llm(chat_history, user_input, drone_state)
            ros_cmd = build_ros_command(llm_json, drone_state)

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": json.dumps(llm_json, ensure_ascii=False)})
            if len(chat_history) > 1 + MAX_HISTORY_TURNS * 2:
                chat_history = [chat_history[0]] + chat_history[-(MAX_HISTORY_TURNS * 2):]

            print(f"AI: {llm_json.get('odpowiedz')}")
            print(
                f" -> [INTENCJA]: action={llm_json.get('action')} | "
                f"FB={llm_json.get('przod_tyl')} LR={llm_json.get('lewo_prawo')} "
                f"UD={llm_json.get('gora_dol')} ROT={llm_json.get('obrot')}"
            )

            if ros_cmd["mode"] == "position":
                print(
                    f" -> [POS]: X={ros_cmd['target_x']} "
                    f"Y={ros_cmd['target_y']} Z={ros_cmd['target_z']}"
                )
            else:
                print(
                    f" -> [VEL]: R={ros_cmd['target_roll']} P={ros_cmd['target_pitch']} "
                    f"Vz={ros_cmd['target_vz']} Yaw={ros_cmd['target_yaw_rate']} "
                    f"t={ros_cmd['duration_sec']}s"
                )

            send_command_to_laptop(ros_cmd)

            tokens_per_sec = diag_data.get("eval_count", 0) / (diag_data.get("eval_duration", 1) / 1e9)
            true_tok = estimate_true_history_tokens(chat_history)
            ctx_pct = (true_tok / MODEL_OPTIONS["num_ctx"]) * 100
            print(
                f" -> [DIAG]: {len(chat_history)} wiad. | "
                f"Pamięć: ~{true_tok}/{MODEL_OPTIONS['num_ctx']} ({ctx_pct:.1f}%) | "
                f"{tokens_per_sec:.1f} tok/s"
            )

        except json.JSONDecodeError as e:
            raw = diag_data.get("message", {}).get("content", "")
            reason = diag_data.get("done_reason", "nieznany")
            print(f" -> [BŁĄD]: Niepoprawny JSON ({e}) | done_reason={reason} | raw={raw[:180]!r}")
        except requests.exceptions.RequestException as e:
            print(f" -> [BŁĄD OLLAMA]: {e}")


if __name__ == "__main__":
    main()