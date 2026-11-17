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
LAPTOP_URL = "http://192.168.1.X:5000"

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
    "temperature": 0.35,

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

DEFAULT_SYSTEM_PROMPT = """Jesteś inteligentnym systemem sterowania dronem kwadrotorem X3.
Użytkownik wyda Ci komendę w języku polskim (np. "poleć do przodu", "wznieś się na chwilę", "zawiśnij").
Twoim jedynym zadaniem jest przetłumaczenie tej intencji na parametry lotu i zwrócenie ich w wymaganym formacie JSON.

ZASADY MAPOWANIA RUCHU:
- target_roll: przechył boczny w radianach [-0.3, 0.3]. Ujemny = w lewo, Dodatni = w prawo.
- target_pitch: pochylenie wzdłużne w radianach [-0.3, 0.3]. Dodatni = w przód, Ujemny = w tył.
- target_vz: prędkość pionowa w m/s [-1.0, 1.0]. Dodatni = leć w górę, Ujemny = leć w dół.
- target_yaw_rate: prędkość obracania się wokół własnej osi [rad/s].
- duration_sec: czas w sekundach, przez jaki ten manewr ma trwać. Jeśli użytkownik nie podał czasu, ustaw domyślnie 2.0 sekundy.
- odpowiedz_dla_uzytkownika: krótki, naturalny komunikat po polsku potwierdzający zrozumienie komendy.

Komenda "zatrzymaj się", "zawiśnij" lub "stój" oznacza ustawienie target_roll, target_pitch, target_vz oraz target_yaw_rate na 0.0 przy czasie duration_sec na 0.0."""


def load_or_create_system_prompt(filepath: str) -> str:
    """Wczytuje kontekst drona z pliku JSON na dysku lub tworzy nowy przy pierwszym starcie."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"system_prompt": DEFAULT_SYSTEM_PROMPT}, f, ensure_ascii=False, indent=2)
        return DEFAULT_SYSTEM_PROMPT


def send_message_to_llm(chat_history: list) -> dict:
    """
    Wysyła całą historię rozmowy do serwera Ollama.
    Dzięki opcji "format": JSON_SCHEMA wymusza na modelu wygenerowanie poprawnego słownika JSON.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": chat_history,
        "stream": False,
        "options": MODEL_OPTIONS,
        "keep_alive": KEEP_ALIVE,
        "format": JSON_SCHEMA  # Wymuszenie czystego JSON-a, brak zbędnego tekstu na wyjściu
    }

    # timeout=60 pozwala na dłuższą pracę modelu (szczególnie przy pierwszym ładowaniu do VRAM)
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    
    # Wyciągnięcie wygenerowanego tekstu (który jest poprawnym JSON-em) i konwersja na słownik Pythona
    response_text = response.json()["message"]["content"]
    return json.loads(response_text), response.json()


def send_command_to_laptop(command_json: dict):
    """
    Przesyła gotowy słownik parametrów (wygenerowany przez LLM) do serwera HTTP na laptopie.
    Węzeł most_ai.py na laptopie odbierze ten pakiet i przełoży go na komendy dla silników ROS 2.
    """
    try:
        # timeout=2 zabezpiecza przed zawieszeniem czatu w przypadku utraty połączenia Wi-Fi z laptopem
        res = requests.post(LAPTOP_URL, json=command_json, timeout=2)
        res.raise_for_status()
        print("[SIEĆ] Pomyślnie wysłano komendę do drona (ROS 2).")
    except requests.exceptions.RequestException as e:
        print(f"\n[SIEĆ - BŁĄD] Nie udało się połączyć z laptopem pod adresem {LAPTOP_URL}: {e}")
        print("Upewnij się, że węzeł 'most_ai' działa na laptopie i adres IP jest wpisany poprawnie.")


def main():
    system_prompt = load_or_create_system_prompt(CONTEXT_FILE)
    
    # Pamięć konwersacji: przechowuje kontekst (prompt systemowy) oraz historię poleceń
    chat_history = [{"role": "system", "content": system_prompt}]

    print("=" * 75)
    print(f" Most AI (Windows) -> ROS 2 (Laptop) Gotowy | Model: {MODEL_NAME}")
    print(" Wydawaj polecenia (np. 'Poleć w lewo na 3 sekundy', 'Zatrzymaj się')")
    print("=" * 75)

    while True:
        try:
            user_input = input("\nKomenda lotu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nZamykanie czatu...")
            break

        if not user_input:
            continue
        if user_input.lower() in ["/wyjdz", "exit", "quit"]:
            break
        if user_input.lower() == "/reset":
            # Wyczyszczenie historii przy zachowaniu promptu systemowego
            chat_history = [{"role": "system", "content": system_prompt}]
            print("[INFO] Pamięć konwersacji wyczyszczona.")
            continue

        # 1. Zapis intencji użytkownika do pamięci modelu
        chat_history.append({"role": "user", "content": user_input})

        try:
            # 2. Przekazanie zapytania do AI i odbiór sformatowanej odpowiedzi (JSON)
            command_json, diag_data = send_message_to_llm(chat_history)
            
            # 3. Zapis odpowiedzi AI do historii rozmowy. 
            # Wymagany jest zapis tekstowy JSON-a (ensure_ascii=False zapewnia poprawne polskie znaki).
            chat_history.append({"role": "assistant", "content": json.dumps(command_json, ensure_ascii=False)})

            # 4. Wyświetlenie informacji zwrotnej w czytelny sposób dla operatora
            print(f"\nAI: {command_json.get('odpowiedz_dla_uzytkownika', 'Rozumiem.')}")
            print(f"Parametry: Roll={command_json.get('target_roll')}, Pitch={command_json.get('target_pitch')}, "
                  f"Vz={command_json.get('target_vz')}, Czas={command_json.get('duration_sec')}s")

            # 5. Fizyczne przesłanie sterowania przez sieć LAN do laptopa (ROS 2)
            send_command_to_laptop(command_json)

            # 6. Opcjonalna diagnostyka wydajności (szybkość pracy GPU)
            eval_tokens = diag_data.get("eval_count", 0)
            eval_duration_ns = diag_data.get("eval_duration", 1)
            tokens_per_sec = eval_tokens / (eval_duration_ns / 1e9)
            print(f"(Diagnostyka AI: Generowanie {tokens_per_sec:.1f} tok/s)")

        except json.JSONDecodeError:
            print("\n[BŁĄD]: Model nie zwrócił poprawnego formatu JSON.")
            # Usuwamy ostatnie zapytanie użytkownika, by niespójna odpowiedź nie zatruła pamięci
            chat_history.pop()
        except requests.exceptions.RequestException as e:
            print(f"\n[BŁĄD POŁĄCZENIA Z OLLAMĄ]: {e}")
            chat_history.pop()


if __name__ == "__main__":
    main()