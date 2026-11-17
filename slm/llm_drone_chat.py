import json
import os
import requests

# ==============================================================================
# 1. KONFIGURACJA POŁĄCZENIA (LOKALNIE VS SIEĆ LAN Z LAPTOPEM ROS 2)
# ==============================================================================

# AKTYWNE: Połączenie lokalne (gdy skrypt i Ollama działają na tym samym PC z Windowsem)
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# Adres serwera HTTP na laptopie z ROS 2 (czyli uruchomionego węzła most_ai.py).
# UWAGA: Musisz podmienić '192.168.1.X' na rzeczywisty adres IP Twojego laptopa w sieci Wi-Fi.
LAPTOP_URL = "http://192.168.1.56:5000"

# Nazwa używanego modelu (musi być wcześniej pobrany komendą 'ollama pull qwen3:8b')
MODEL_NAME = "qwen3:8b"

# Plik zapisu dla promptu systemowego, pozwalający na jego edycję bez zmiany kodu.
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
        "target_roll": {"type": "number"},
        "target_pitch": {"type": "number"},
        "target_vz": {"type": "number"},
        "target_yaw_rate": {"type": "number"},
        "duration_sec": {"type": "number"}
    },
    "required": [
        "odpowiedz_dla_uzytkownika",
        "target_roll",
        "target_pitch",
        "target_vz",
        "target_yaw_rate",
        "duration_sec"
    ]
}

# ==============================================================================
# 3. KONTEKST SYSTEMOWY DRONA X3 (WPISANY NA SZTYWNO + ZAPIS DO PLIKU JSON)
# ==============================================================================

# Nowy, kuloodporny prompt - abstrakcyjny, oparty na intencjach
DEFAULT_SYSTEM_PROMPT = """Jesteś asystentem sterowania dronem X3. Tłumaczysz intencje na surowy format JSON. Użytkownik używa dowolnego szyku (np. "do przodu i w lewo").

WARTOŚCI BAZOWE (Dla aktywnej osi ruchu):
1. Zwykła prędkość: 0.15 (dla Vz 0.5)
2. Powoli / delikatnie: 0.05 (dla Vz 0.2)
3. Szybko / najszybciej: 0.3 (dla Vz 1.0)

KIERUNKI:
- W lewo -> target_roll (Ujemny)
- W prawo -> target_roll (Dodatni)
- Do przodu -> target_pitch (Dodatni)
- Do tyłu -> target_pitch (Ujemny)
- W górę / do góry -> target_vz (Dodatni)
- W dół / do dołu -> target_vz (Ujemny)

BARDZO WAŻNE REGUŁY:
1. NIEWYMIENIONE w komendzie osie ruchu ZAWSZE ustawiaj na 0.0.
2. Czas (duration_sec): Jeśli nie podano w komendzie, ZAWSZE ustawiaj na 2.0.
3. Przy skosach (np. "do przodu w lewo powoli") ustaw oba wymienione parametry na zadaną wartość.
4. "Zatrzymaj się / zawiśnij": Wszystkie targety = 0.0, czas = 0.0.
5. odpowiedz_dla_uzytkownika: Jedno zdanie po polsku."""

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

def send_message_to_llm(chat_history: list) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": chat_history,
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

    print(f"--- AI DRONE CHAT ({MODEL_NAME}) GOTOWY ---")

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

        chat_history.append({"role": "user", "content": user_input})

        try:
            command_json, diag_data = send_message_to_llm(chat_history)
            chat_history.append({"role": "assistant", "content": json.dumps(command_json, ensure_ascii=False)})

            print(f"AI: {command_json.get('odpowiedz_dla_uzytkownika')}")
            print(f" -> [PARAM]: R={command_json.get('target_roll')} P={command_json.get('target_pitch')} Vz={command_json.get('target_vz')} t={command_json.get('duration_sec')}s")
            
            send_command_to_laptop(command_json)

            tokens_per_sec = diag_data.get("eval_count", 0) / (diag_data.get("eval_duration", 1) / 1e9)
            true_tok = estimate_true_history_tokens(chat_history)
            ctx_pct = (true_tok / MODEL_OPTIONS["num_ctx"]) * 100
            
            print(f" -> [DIAG]: {len(chat_history)} wiadomości | Pamięć: ~{true_tok}/{MODEL_OPTIONS['num_ctx']} ({ctx_pct:.1f}%) | Prędkość: {tokens_per_sec:.1f} tok/s")

        except json.JSONDecodeError:
            print(" -> [BŁĄD]: Niepoprawny JSON.")
            chat_history.pop()
        except requests.exceptions.RequestException as e:
            print(f" -> [BŁĄD OLLAMA]: {e}")
            chat_history.pop()

if __name__ == "__main__":
    main()