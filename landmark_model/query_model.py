#!/usr/bin/env python3
"""
query_model.py — RAG local + Ollama + orientación de cámara
=============================================================
1. Carga los 32K landmarks del JSON
2. Dadas unas coordenadas + azimuth (brújula), busca los más cercanos
3. Filtra solo los que están en el campo de visión de la cámara
4. Inyecta esa lista en el prompt y llama al modelo de Ollama

Uso:
  python query_model.py <lat> <lon>                    # Sin brújula (todos)
  python query_model.py <lat> <lon> <azimuth>          # Con brújula (FOV 70°)
  python query_model.py <lat> <lon> <azimuth> <fov>    # Brújula + FOV custom
"""

import sys, json, os, math, requests

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "landmark-cataluna"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_PATH = os.path.join(SCRIPT_DIR, "data", "landmarks_cataluna.json")
DEFAULT_FOV = 70  # campo de visión horizontal típico de cámara móvil (grados)

# ── utilidades geográficas ──────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing(lat1, lon1, lat2, lon2):
    """Bearing en grados (0=norte, 90=este, 180=sur, 270=oeste)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl)*math.cos(p2)
    y = math.cos(p1)*math.sin(p2) - math.sin(p1)*math.cos(p2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def angle_diff(a, b):
    """Diferencia angular mínima entre dos ángulos (0-180°)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d

def direction(b):
    dirs = ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]
    return dirs[round(b/45) % 8]

def fmt_dist(m):
    if m < 100: return f"{int(m)} m"
    if m < 1000: return f"{int(m/10)*10} m"
    return f"{m/1000:.1f} km"

# ── carga de datos ──────────────────────────────────────────────────

_LANDMARKS = None

def load_landmarks():
    global _LANDMARKS
    if _LANDMARKS is not None:
        return _LANDMARKS
    if not os.path.exists(LANDMARKS_PATH):
        print(f"❌ No se encuentra {LANDMARKS_PATH}")
        print("   Ejecuta primero: python extract_landmarks.py")
        sys.exit(1)
    with open(LANDMARKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _LANDMARKS = [l for l in data["landmarks"] if "lat" in l and "lon" in l]
    print(f"📚 {len(_LANDMARKS)} landmarks cargados en memoria")
    return _LANDMARKS

# ── búsqueda local (RAG) ───────────────────────────────────────────

def find_nearby(lat, lon, azimuth=None, fov=DEFAULT_FOV, max_dist=3000, max_results=3):
    """
    Busca landmarks cercanos. Si azimuth está definido, filtra solo los
    que caen dentro del campo de visión de la cámara (azimuth ± fov/2).
    """
    landmarks = load_landmarks()
    results = []
    for lm in landmarks:
        d = haversine(lat, lon, lm["lat"], lm["lon"])
        if d <= max_dist:
            b = bearing(lat, lon, lm["lat"], lm["lon"])
            lm_result = {**lm, "distance": d, "bearing_deg": round(b, 1),
                         "direction": direction(b)}

            if azimuth is not None and d > 10:
                # Si d <= 10m estás encima, siempre incluir
                diff = angle_diff(azimuth, b)
                if diff > fov / 2:
                    continue  # fuera del campo de visión
                lm_result["angle_from_center"] = round(diff, 1)

            results.append(lm_result)
    results.sort(key=lambda x: x["distance"])
    return results[:max_results]

def build_context(lat, lon, nearby, azimuth=None):
    """Construye el texto de contexto que se inyecta en el prompt."""
    if not nearby:
        return f"Coordenadas: {lat}, {lon}\nNo se han encontrado landmarks en el campo de visión."
    lines = [f"Coordenadas del usuario: {lat}, {lon}"]
    if azimuth is not None:
        lines.append(f"Cámara apuntando a: {azimuth}° (0°=N, 90°=E, 180°=S, 270°=O)")
    lines.append(f"Lugares en el campo de visión ({len(nearby)}):\n")
    for i, lm in enumerate(nearby, 1):
        name = lm["name"]
        dist = fmt_dist(lm["distance"])
        dirn = lm["direction"]
        bearing_deg = lm.get("bearing_deg", "?")
        angle_off = lm.get("angle_from_center")
        cats = ", ".join(lm.get("categories", [])) or "landmark"
        line = f"{i}. {name} — {dist} al {dirn} (bearing {bearing_deg}°)"
        if angle_off is not None:
            line += f" — desviación del centro: {angle_off}°"
        line += f" — Tipo: {cats}"
        extras = []
        if "architect" in lm:  extras.append(f"Arquitecto: {lm['architect']}")
        if "year" in lm:       extras.append(f"Año: {lm['year']}")
        if "style" in lm:      extras.append(f"Estilo: {lm['style']}")
        if "address" in lm:    extras.append(f"Dir: {lm['address']}")
        if "wikipedia" in lm:  extras.append(f"Wiki: {lm['wikipedia']}")
        if extras:
            line += "  (" + "; ".join(extras) + ")"
        lines.append(line)
    return "\n".join(lines)

# ── consulta a Ollama ──────────────────────────────────────────────

def query(lat, lon, azimuth=None, fov=DEFAULT_FOV):
    nearby = find_nearby(lat, lon, azimuth=azimuth, fov=fov)
    context = build_context(lat, lon, nearby, azimuth=azimuth)

    if azimuth is not None:
        prompt = f"""Coordenadas: {lat}, {lon}
La cámara apunta a {azimuth}° con un campo de visión de {fov}°.
Lugares en el campo de visión:
{context}

Responde SOLO con un JSON con estos campos:
- "target": el nombre del edificio/monumento al que la cámara apunta directamente (el más alineado con el centro de la cámara, priorizando cercanía y menor desviación del centro)
- "target_distance": distancia en metros
- "others": array con los otros edificios visibles en la foto (name y distance)
No añadas texto fuera del JSON."""
    else:
        prompt = f"""Coordenadas: {lat}, {lon}
Lugares cercanos:
{context}

Devuelve el JSON con los que aparecerían en una foto."""

    print(f"\n🔍 Buscando landmarks cerca de ({lat}, {lon})...")
    if azimuth is not None:
        print(f"   🧭 Cámara apuntando a {azimuth}° (FOV: {fov}°)")
    print(f"   Encontrados: {len(nearby)} lugares en el campo de visión")
    if nearby:
        print(f"   Más cercano: {nearby[0]['name']} a {fmt_dist(nearby[0]['distance'])}")
    print(f"\n💬 Preguntando al modelo...\n")

    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate",
                          json={"model": MODEL_NAME, "prompt": prompt, "stream": True},
                          stream=True, timeout=180)
        r.raise_for_status()
        full = ""
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                tok = data.get("response", "")
                print(tok, end="", flush=True)
                full += tok
                if data.get("done"): break
        print("\n")
        return full
    except requests.ConnectionError:
        print("❌ Ollama no responde. ¿Está corriendo? (ollama serve)")
    except Exception as e:
        print(f"❌ Error: {e}")
    return None

# ── CLI ─────────────────────────────────────────────────────────────

def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(MODEL_NAME in m for m in models):
            print(f"⚠️  Modelo '{MODEL_NAME}' no encontrado. Disponibles: {models}")
            print("   Ejecuta: python generate_knowledge.py && ollama create landmark-cataluna -f Modelfile")
            return False
        return True
    except requests.ConnectionError:
        print("❌ Ollama no está corriendo. Inícialo: ollama serve")
        return False

def interactive():
    print("=" * 60)
    print("🏛️  LANDMARK FINDER — Cataluña (RAG + brújula)")
    print("=" * 60)
    print("Formatos aceptados:")
    print("  lat, lon                → busca en todas direcciones")
    print("  lat, lon, azimuth       → filtra por dirección cámara")
    print("  lat, lon, azimuth, fov  → dirección + FOV custom")
    print()
    print("Azimuth: 0°=Norte, 90°=Este, 180°=Sur, 270°=Oeste")
    print("FOV: campo de visión en grados (default 70°)")
    print()
    print("Ejemplos:")
    print("  41.4036, 2.1744              → Sagrada Familia (todos)")
    print("  41.4036, 2.1744, 180         → mirando al sur")
    print("  41.4036, 2.1744, 90, 50      → mirando al este, FOV 50°")
    print("Escribe 'q' para salir.")
    print("=" * 60)
    while True:
        inp = input("\n📍 lat, lon [, azimuth [, fov]]: ").strip()
        if inp.lower() in ("q", "quit", "salir", "exit"): break
        if not inp: continue
        try:
            parts = inp.replace(",", " ").split()
            lat, lon = float(parts[0]), float(parts[1])
            azimuth = float(parts[2]) % 360 if len(parts) >= 3 else None
            fov = float(parts[3]) if len(parts) >= 4 else DEFAULT_FOV
        except (ValueError, IndexError):
            print("❌ Formato: lat, lon [, azimuth [, fov]]")
            continue
        query(lat, lon, azimuth=azimuth, fov=fov)

def main():
    if not check_ollama():
        sys.exit(1)
    load_landmarks()  # pre-cargar
    if len(sys.argv) >= 3:
        try:
            lat = float(sys.argv[1])
            lon = float(sys.argv[2])
            azimuth = float(sys.argv[3]) % 360 if len(sys.argv) >= 4 else None
            fov = float(sys.argv[4]) if len(sys.argv) >= 5 else DEFAULT_FOV
            query(lat, lon, azimuth=azimuth, fov=fov)
        except ValueError:
            print("❌ Uso: python query_model.py <lat> <lon> [azimuth] [fov]")
            sys.exit(1)
    else:
        interactive()

if __name__ == "__main__":
    main()
