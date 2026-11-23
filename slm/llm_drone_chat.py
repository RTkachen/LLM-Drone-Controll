import json
import os
import requests

# ==============================================================================
# 1. KONFIGURACJA POŁĄCZENIA (LOKALNIE VS SIEĆ LAN Z LAPTOPEM ROS 2)
# ==============================================================================

# AKTYWNE: Połączenie lokalne (gdy skrypt i Ollama działają na tym samym PC z Windowsem)
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
LAPTOP_URL = "http://192.168.1.56:5000"
MODEL_NAME = "qwen3:8b"
CONTEXT_FILE = "kontekst_lotu.json"


# ==============================================================================
# 2. JAWNA KONFIGURACJA PARAMETRÓW MODELU LLM (WPŁYW NA DZIAŁANIE I PERFORMANCE)
# ==============================================================================
MODEL_OPTIONS = {
    # num_ctx: Rozmiar okna kontekstowego (pamięci krótkotrwałej konwersacji) w tokenach.
    # - Działanie: Określa, ile maksymalnie tokenów (prompt systemowy + cała historia czatu + nowa odpowiedź)
    #   model pamięta jednocześnie. Po przekroczeniu tego limitu najstarsze wiadomości są ucinane.
    # - Performance (VRAM): Kluczowy parametr dla karty 8 GB! Zwiększanie num_ctx liniowo pożera pamięć VRAM
    #   na tzw. bufor KV Cache. Wartość 8192 (8k tokenów) zużywa ok. 1.2 GB VRAM (cały model + kontekst ~ 6.2 GB),
    #   co pozwala utrzymać 100% obliczeń na GPU bez spadku prędkości. Ustawienie np. 32768 przepełni 8 GB VRAM.
    "num_ctx": 8192,

    # num_predict: Maksymalna liczba tokenów, jaką model może wygenerować w JEDNEJ odpowiedzi.
    # - Działanie: Chroni przed zbyt długimi wywodami. Wartość -1 oznacza brak limitu (aż do końca num_ctx).
    # - Performance (Czas): Generowanie każdego kolejnego tokenu trwa ok. 15-25 ms. Ustawienie 1024 gwarantuje,
    #   że odpowiedź nigdy nie będzie trwała dłużej niż kilkanaście sekund.
    "num_predict": 1024,

    # temperature: Poziom losowości / kreatywności modelu (zakres zazwyczaj od 0.0 do 2.0).
    # - Działanie: Rozpłaszcza lub wyostrza rozkład prawdopodobieństwa kolejnych słów.
    #   0.0 = pełny determinizm (model zawsze wybierze najbardziej prawdopodobny token, świetne do JSON/matematyki).
    #   0.2 - 0.4 = odpowiedzi bardzo rzeczowe, techniczne, ale z niewielką elastycznością przy szukaniu nastaw PID.
    #   0.8+ = wysoka kreatywność, większe ryzyko zmyślania (halucynacji) wzorów.
    # - Performance: Brak wpływu na zużycie VRAM i prędkość generowania.
    "temperature": 0.15,

    # top_p (Nucleus Sampling): Próg skumulowanego prawdopodobieństwa (od 0.0 do 1.0).
    # - Działanie: Model bierze pod uwagę tylko te słowa, których suma prawdopodobieństw wynosi top_p (np. 0.9 = 90%).
    #   Odcina najbardziej nieprawdopodobne, bezsensowne słowa z "ogona" rozkładu.
    # - Performance: Brak zauważalnego wpływu na szybkość.
    "top_p": 0.9,

    # top_k: Ograniczenie puli wyboru do K najbardziej prawdopodobnych tokenów w każdym kroku.
    # - Działanie: Przy top_k = 40 model w każdym kroku odrzuca wszystkie słowa poza 40 najlepszymi kandydatami.
    #   Zmniejsza szansę na wygenerowanie błędów składniowych.
    # - Performance: Minimalnie przyspiesza próbkowanie na CPU/GPU.
    "top_k": 40,

    # repeat_penalty: Kara za powtarzanie tych samych słów i fraz (1.0 = brak kary).
    # - Działanie: Zapobiega zapętleniu się modelu (tzw. repetition loop, gdzie model w kółko powtarza jedno zdanie).
    #   Uwaga: przy generowaniu kodu lub JSON z powtarzającymi się nazwami zmiennych (np. kp, ki, kd) nie należy
    #   ustawiać zbyt wysoko (1.1 jest bezpieczne; powyżej 1.3 model zacznie unikać używania nazw zmiennych!).
    "repeat_penalty": 1.1,

    # num_gpu: Liczba warstw sieci neuronowej ładowanych na kartę graficzną.
    # - Działanie / Performance: Wartość 99 (lub -1) wymusza załadowanie WSZYSTKICH warstw modelu na GPU.
    #   Jeśli ustawisz 0, model będzie liczył wyłącznie na procesorze CPU (10x wolniej).
    "num_gpu": 99,

    # seed: Ziarno generatora liczb losowych.
    # - Działanie: Ustawienie konkretnej liczby (np. 42) sprawia, że przy tej samej temperaturze model na to samo
    #   pytanie odpowie zawsze identycznie (powtarzalność eksperymentów naukowych). -1 = losowe ziarno.
    "seed": 42
}

# Czas utrzymania modelu w pamięci VRAM po zakończeniu odpowiedzi:
# "30m" = 30 minut, "-1" = trzymaj w VRAM bezterminowo (brak opóźnienia przy kolejnych pytaniach).
KEEP_ALIVE = "30m"

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "odpowiedz_dla_uzytkownika": {"type": "string"},
        "mode": {"type": "string", "enum": ["velocity", "position"]},
        "hold_current": {"type": "boolean"},
        "target_x": {"type": "number"},
        "target_y": {"type": "number"},
        "target_z": {"type": "number"},
        "target_roll": {"type": "number"},
        "target_pitch": {"type": "number"},
        "target_vz": {"type": "number"},
        "target_yaw_rate": {"type": "number"},
        "duration_sec": {"type": "number"}
    },
    "required": [
        "odpowiedz_dla_uzytkownika", "mode", "hold_current",
        "target_x", "target_y", "target_z",
        "target_roll", "target_pitch", "target_vz", "target_yaw_rate", "duration_sec"
    ]
}

COMMAND_LEGEND_JSON = {
    "mode": "'position' (utrzymanie punktu lub lot na współrzędne X,Y,Z) lub 'velocity' (lot kierunkowy przez czas)",
    "hold_current": "true = zatrzymaj się i trzymaj bieżący punkt (zatrzymaj się/zawiśnij/włącz pozycję); false = leć do target_x,y,z",
    "osie_velocity": "lewo=-roll, prawo=+roll, przód=+pitch, tył=-pitch, góra=+vz, dół=-vz",
    "predkosci": "powoli=0.05 (vz 0.2), zwykle=0.15 (vz 0.5), szybko=0.3 (vz 1.0); brak czasu -> duration_sec=2.0",
    "regula_zerowania": "niewymienione pola ruchu ustaw na 0.0; przy locie w trybie position ustaw pola velocity na 0.0"
}

DEFAULT_SYSTEM_PROMPT = """Jesteś asystentem sterowania dronem X3 w ROS 2.
W każdej wiadomości otrzymujesz obiekt JSON ze stanem drona, legendą komend oraz poleceniem operatora.
Zwracaj WYŁĄCZNIE poprawny JSON zgodny ze schematem:
- Jeśli użytkownik prosi o zatrzymanie, zawis w punkcie, włączenie pozycjonowania lub lot do konkretnych współrzędnych [X, Y, Z], wybierz mode='position'.
- Jeśli użytkownik prosi o ruch kierunkowy (np. w przód, w lewo, na skos, w górę przez N sekund), wybierz mode='velocity' (po czasie dron sam zakotwiczy pozycję)."""


def load_or_create_system_prompt(filepath: str) -> str:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f).get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"system_prompt": DEFAULT_SYSTEM_PROMPT}, f, ensure_ascii=False, indent=2)
    return DEFAULT_SYSTEM_PROMPT


def clean_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return json.loads(text.strip())


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


def send_message_to_llm(chat_history: list, current_user_input: str, drone_state: dict):
    wrapped_user_msg = json.dumps({
        "stan_drona": drone_state,
        "legenda_komend": COMMAND_LEGEND_JSON,
        "polecenie_uzytkownika": current_user_input
    }, ensure_ascii=False)

    messages_for_api = chat_history + [{"role": "user", "content": wrapped_user_msg}]

    payload = {
        "model": MODEL_NAME,
        "messages": messages_for_api,
        "stream": False,
        "options": MODEL_OPTIONS,
        "keep_alive": KEEP_ALIVE,
        "format": JSON_SCHEMA
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return clean_json_response(response.json()["message"]["content"]), response.json()


def send_command_to_laptop(command_json: dict):
    try:
        requests.post(LAPTOP_URL, json=command_json, timeout=2).raise_for_status()
        print(" -> [ROS 2]: OK.")
    except requests.exceptions.RequestException as e:
        print(f" -> [ROS 2 BŁĄD]: {e}")


def main():
    system_prompt = load_or_create_system_prompt(CONTEXT_FILE)
    chat_history = [{"role": "system", "content": system_prompt}]

    print(f"--- AI DRONE CHAT ({MODEL_NAME}) | Komendy: /pos on, /pos off, /reset, /wyjdz ---")

    while True:
        try:
            user_input = input("\nKomenda: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input: continue
        if user_input.lower() in ["/wyjdz", "exit", "quit"]: break
        if user_input.lower() == "/reset":
            chat_history = [{"role": "system", "content": system_prompt}]
            print(" -> [INFO]: Pamięć wyczyszczona.")
            continue

        if user_input.lower() == "/pos on":
            send_command_to_laptop({"mode": "position", "hold_current": True})
            continue
        if user_input.lower() == "/pos off":
            send_command_to_laptop({
                "mode": "velocity", "target_roll": 0.0, "target_pitch": 0.0,
                "target_vz": 0.0, "target_yaw_rate": 0.0, "duration_sec": 0.0, "auto_hold_after": False
            })
            continue

        drone_state = get_drone_state()

        try:
            command_json, diag_data = send_message_to_llm(chat_history, user_input, drone_state)

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": json.dumps(command_json, ensure_ascii=False)})

            print(f"AI: {command_json.get('odpowiedz_dla_uzytkownika')}")
            if command_json.get("mode") == "position":
                print(f" -> [POS]: hold={command_json.get('hold_current')} | "
                      f"X={command_json.get('target_x')} Y={command_json.get('target_y')} Z={command_json.get('target_z')}")
            else:
                print(f" -> [VEL]: R={command_json.get('target_roll')} P={command_json.get('target_pitch')} "
                      f"Vz={command_json.get('target_vz')} t={command_json.get('duration_sec')}s")

            send_command_to_laptop(command_json)

            tokens_per_sec = diag_data.get("eval_count", 0) / (diag_data.get("eval_duration", 1) / 1e9)
            true_tok = estimate_true_history_tokens(chat_history)
            ctx_pct = (true_tok / MODEL_OPTIONS["num_ctx"]) * 100
            print(f" -> [DIAG]: {len(chat_history)} wiad. | Pamięć: ~{true_tok}/{MODEL_OPTIONS['num_ctx']} ({ctx_pct:.1f}%) | {tokens_per_sec:.1f} tok/s")

        except json.JSONDecodeError:
            print(" -> [BŁĄD]: Niepoprawny JSON.")
        except requests.exceptions.RequestException as e:
            print(f" -> [BŁĄD OLLAMA]: {e}")


if __name__ == "__main__":
    main()