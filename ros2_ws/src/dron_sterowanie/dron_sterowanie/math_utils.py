import math

def euler_from_quaternion(x: float, y: float, z: float, w: float):
    """
    Kwaterniony (x, y, z, w) służą do zapisu orientacji 3D bez problemu "Gimbal Lock",
    ale są nieczytelne dla regulatorów PID. Ta funkcja przelicza je na zrozumiałe
    kąty Eulera w radianach:
    - roll (przechył na lewo/prawo w osi X)
    - pitch (pochylenie dziobu w dół/górę w osi Y)
    - yaw (obrót kompasowy wokół osi Z)
    """
    # Obliczanie przechyłu bocznego (Roll)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Obliczanie pochylenia wzdłużnego (Pitch) z zabezpieczeniem matematycznym
    # (jeśli sinp przekroczy 1.0 ze względu na błąd zmiennoprzecinkowy, obcinamy do 90 stopni)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    # Obliczanie obrotu kompasowego (Yaw)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
