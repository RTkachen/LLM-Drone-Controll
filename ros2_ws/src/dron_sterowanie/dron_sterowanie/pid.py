class PID:
    """
    Dyskretny regulator PID z zabezpieczeniami niezbędnymi w sterowaniu dronami:
    - Ograniczenie sygnału wyjściowego (out_limit).
    - Zabezpieczenie przed tzw. "Integral Windup" (i_limit), zapobiegające zbytniemu
      rozpędzaniu się całki, gdy dron nie może od razu osiągnąć celu.
    - Filtr dolnoprzepustowy (LPF) na członie różniczkującym (D), który tłumi
      szpilki pomiarowe i zapobiega wibracjom silników.
    """
    def __init__(self, kp: float, ki: float, kd: float, out_limit: float = 150.0, i_limit: float = 40.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Maksymalny sygnał wyjściowy (zabezpieczenie przed przesterowaniem silników)
        self.out_limit = out_limit

        # Maksymalna wartość samej całki (ogranicza pamięć błędu z przeszłości)
        self.i_limit = i_limit

        # Pamięć stanu regulatora pomiędzy kolejnymi krokami (dt)
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_deriv = 0.0

        # Flaga chroniąca przed gwałtownym uderzeniem pochodnej (Derivative Kick)
        # przy pierwszym uruchomieniu po starcie lub po resecie.
        self.first_run = True

    def reset(self):
        """Zeruje wewnętrzną pamięć regulatora (wywoływane po wstrzymaniu sterowania)."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_deriv = 0.0
        self.first_run = True

    def compute(self, error: float, dt: float) -> float:
        """Oblicza sumaryczny sygnał sterujący PID na podstawie uchybu (error) i kroku czasowego (dt)."""
        if dt <= 0.0:
            return 0.0

        # 1. Człon P (Proporcjonalny) - siła reakcji tu i teraz na obecny uchyb
        p_term = self.kp * error

        # 2. Człon I (Całkujący) - sumuje uchyb w czasie, by kasować stałe znoszenie (np. przez wiatr)
        self.integral += error * dt
        # Zabezpieczenie anti-windup: obcinamy całkę, jeśli przekroczy i_limit
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        i_term = self.ki * self.integral

        # 3. Człon D (Różniczkujący) - działa jak lepki tłumik, reagując na szybkość zmian uchybu
        if self.first_run:
            raw_deriv = 0.0
            self.first_run = False
        else:
            raw_deriv = (error - self.prev_error) / dt
        self.prev_error = error

        # Filtracja zakłóceń: nowa przefiltrowana pochodna to w 35% nowy wynik, a w 65% stara wartość.
        # Sprawia to, że nagły "skok" błędu z żyroskopu nie uderza z pełną siłą w silniki.
        self.filtered_deriv = 0.35 * raw_deriv + 0.65 * self.filtered_deriv
        d_term = self.kd * self.filtered_deriv

        # Sumujemy P + I + D i na koniec docinamy cały wynik do bezpiecznego przedziału out_limit
        output = p_term + i_term + d_term
        return max(-self.out_limit, min(self.out_limit, output))
