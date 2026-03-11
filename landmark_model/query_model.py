#!/usr/bin/env python3
"""
query_model.py — RAG local + Ollama + orientación de cámara
=============================================================
1. Carga los landmarks del JSON y los indexa en una rejilla espacial
2. Dadas unas coordenadas + azimuth (brújula), busca los más cercanos en O(1)
3. Filtra solo los que están en el campo de visión de la cámara
4. Inyecta esa lista compacta en el prompt y llama al modelo de Ollama

Uso:
  python query_model.py <lat> <lon>                    # Sin brújula (todos)
  python query_model.py <lat> <lon> <azimuth>          # Con brújula (FOV 70°)
  python query_model.py <lat> <lon> <azimuth> <fov>    # Brújula + FOV custom
"""

import sys
import json
import os
import math
import requests
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "landmark-finder"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LANDMARKS_PATH = os.path.join(SCRIPT_DIR, "data", "landmarks.json")
DEFAULT_FOV = 70   # campo de visión horizontal típico de cámara móvil (grados)
GRID_SIZE = 0.01   # ~1.1 km por celda de la rejilla espacial

# ── utilidades geográficas ──────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Distancia en metros entre dos coordenadas GPS."""
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """Bearing en grados (0=norte, 90=este, 180=sur, 270=oeste)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def angle_diff(a, b):
    """Diferencia angular mínima entre dos ángulos (0-180°)."""
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def direction(b):
    """Convierte bearing a dirección cardinal."""
    dirs = ["norte", "noreste", "este", "sureste", "sur", "suroeste", "oeste", "noroeste"]
    return dirs[round(b / 45) % 8]


def fmt_dist(m):
    """Formatea distancia de forma legible."""
    if m < 100:
        return f"{int(m)} m"
    if m < 1000:
        return f"{int(m / 10) * 10} m"
    return f"{m / 1000:.1f} km"


# ── índice espacial con rejilla ─────────────────────────────────────

class SpatialIndex:
    """
    Índice espacial con rejilla para búsquedas O(1) por zona.
    En lugar de recorrer 32K+ landmarks en cada consulta, divide el espacio
    en celdas de ~1.1km y solo busca en las celdas vecinas.
    """

    def __init__(self, cell_size=GRID_SIZE):
        self.cell_size = cell_size
        self.grid = defaultdict(list)
        self.total = 0

    def _cell(self, lat, lon):
        return (int(lat / self.cell_size), int(lon / self.cell_size))

    def insert(self, landmark):
        cell = self._cell(landmark["lat"], landmark["lon"])
        self.grid[cell].append(landmark)
        self.total += 1

    def query_radius(self, lat, lon, max_dist_m):
        """Devuelve landmarks candidatos dentro de un radio aproximado."""
        # ~111km por grado de latitud → calculamos cuántas celdas cubrir
        delta = int(max_dist_m / 111_000 / self.cell_size) + 1
        center_cell = self._cell(lat, lon)
        candidates = []
        for di in range(-delta, delta + 1):
            for dj in range(-delta, delta + 1):
                cell = (center_cell[0] + di, center_cell[1] + dj)
                if cell in self.grid:
                    candidates.extend(self.grid[cell])
        return candidates


# ── carga de datos ──────────────────────────────────────────────────

_INDEX = None


def load_landmarks():
    """Carga landmarks y construye índice espacial (una sola vez)."""
    global _INDEX
    if _INDEX is not None:
        return _INDEX

    if not os.path.exists(LANDMARKS_PATH):
        print(f"❌ No se encuentra {LANDMARKS_PATH}")
        print("   Ejecuta primero: python extract_landmarks.py")
        sys.exit(1)

    with open(LANDMARKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    index = SpatialIndex()
    skipped = 0
    for lm in data["landmarks"]:
        if "lat" in lm and "lon" in lm:
            index.insert(lm)
        else:
            skipped += 1

    _INDEX = index
    print(f"📚 {index.total} landmarks indexados ({skipped} sin coordenadas omitidos)")
    return _INDEX


# ── búsqueda local (RAG) ───────────────────────────────────────────

def find_nearby(lat, lon, azimuth=None, fov=DEFAULT_FOV, max_dist=500, max_results=8):
    """
    Busca landmarks cercanos usando el índice espacial.
    Si azimuth está definido, filtra solo los que caen dentro del campo
    de visión de la cámara (azimuth ± fov/2).

    Ordena por score combinado: distancia (principal) + fama (desempate).
    """
    index = load_landmarks()
    candidates = index.query_radius(lat, lon, max_dist)

    results = []
    for lm in candidates:
        d = haversine(lat, lon, lm["lat"], lm["lon"])
        if d > max_dist:
            continue

        b = bearing(lat, lon, lm["lat"], lm["lon"])

        # Construir resultado con el NOMBRE COMPLETO (nunca truncar)
        lm_result = {
            "name": lm["name"],
            "lat": lm["lat"],
            "lon": lm["lon"],
            "distance": round(d, 1),
            "bearing_deg": round(b, 1),
            "direction": direction(b),
            "fame_score": lm.get("fame_score", 0),
            "categories": lm.get("categories", []),
        }

        # Copiar campos opcionales relevantes
        for key in ("architect", "year", "style", "address", "wikipedia",
                     "wikidata", "name_es", "name_en", "name_ca", "description"):
            if key in lm:
                lm_result[key] = lm[key]

        # Filtro por campo de visión si hay azimut
        if azimuth is not None and d > 10:
            diff = angle_diff(azimuth, b)
            if diff > fov / 2:
                continue  # fuera del campo de visión
            lm_result["angle_from_center"] = round(diff, 1)

        results.append(lm_result)

    # Ordenar: menor distancia primero, fama como desempate
    results.sort(key=lambda x: x["distance"] - x["fame_score"] * 5)

    return results[:max_results]


def build_context(nearby):
    """
    Construye texto de contexto COMPACTO para inyectar en el prompt.
    Formato diseñado para minimizar tokens sin perder información clave.
    Los nombres NUNCA se truncan.
    """
    if not nearby:
        return "No landmarks found nearby."

    lines = []
    for i, lm in enumerate(nearby, 1):
        # Nombre completo siempre entre comillas para delimitarlo bien
        name = lm["name"]
        dist = fmt_dist(lm["distance"])
        b_deg = lm["bearing_deg"]

        parts = [f'{i}. "{name}" {dist} @{b_deg}\u00b0']

        angle_off = lm.get("angle_from_center")
        if angle_off is not None:
            parts.append(f"off:{angle_off}\u00b0")

        parts.append(f"fame:{lm['fame_score']}")

        cats = lm.get("categories", [])
        if cats:
            # Solo las 2 más relevantes para ahorrar tokens
            parts.append(f"[{', '.join(cats[:2])}]")

        if "architect" in lm:
            parts.append(f"by {lm['architect']}")

        lines.append(" | ".join(parts))

    return "\n".join(lines)


# ── consulta a Ollama ──────────────────────────────────────────────

def query(lat, lon, azimuth=None, fov=DEFAULT_FOV):
    """Realiza la consulta RAG completa: busca + construye prompt + llama a Ollama."""
    nearby = find_nearby(lat, lon, azimuth=azimuth, fov=fov)
    context = build_context(nearby)

    # Prompt compacto: menos tokens = menos probabilidad de truncado
    if azimuth is not None:
        prompt = (
            f"Pos:{lat},{lon} Cam:{azimuth}\u00b0 FOV:{fov}\u00b0\n"
            f"Landmarks:\n{context}\n\n"
            f'Return JSON: {{"target":"FULL landmark name","target_distance":meters,'
            f'"confidence":"high|medium|low",'
            f'"others":[{{"name":"FULL name","distance":meters}}]}}\n'
            f"Use EXACT complete names from the list. Pick closest to camera center."
        )
    else:
        prompt = (
            f"Pos:{lat},{lon}\n"
            f"Landmarks:\n{context}\n\n"
            f"Return a JSON array with ALL {len(nearby)} landmarks listed above.\n"
            f'Format: [{{"name":"FULL landmark name","distance":meters,"confidence":"high|medium|low"}}]\n'
            f"Include EVERY landmark. Use EXACT complete names from the list."
        )

    # Info al usuario
    print(f"\n\U0001f50d Buscando landmarks cerca de ({lat}, {lon})...")
    if azimuth is not None:
        print(f"   \U0001f9ed C\u00e1mara apuntando a {azimuth}\u00b0 (FOV: {fov}\u00b0)")
    print(f"   \U0001f4cd Encontrados: {len(nearby)} lugares")
    if nearby:
        print(f"   \U0001f3db\ufe0f  M\u00e1s cercano: {nearby[0]['name']} a {fmt_dist(nearby[0]['distance'])}")
    print(f"\n\U0001f4ac Preguntando al modelo...\n")

    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": MODEL_NAME, "prompt": prompt, "stream": True},
            stream=True,
            timeout=180,
        )
        r.raise_for_status()

        full = ""
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                tok = data.get("response", "")
                print(tok, end="", flush=True)
                full += tok
                if data.get("done"):
                    break
        print("\n")

        # Validar y mostrar JSON parseado
        _validate_response(full)
        return full

    except requests.ConnectionError:
        print("\u274c Ollama no responde. \u00bfEst\u00e1 corriendo? (ollama serve)")
    except Exception as e:
        print(f"\u274c Error: {e}")
    return None


def _validate_response(raw):
    """Intenta parsear la respuesta como JSON y reporta el resultado."""
    text = raw.strip()

    # Intento directo
    try:
        parsed = json.loads(text)
        print("\u2705 Respuesta JSON v\u00e1lida")
        return parsed
    except json.JSONDecodeError:
        pass

    # Intento extrayendo JSON embebido (buscar array o objeto)
    start = text.find('[')
    if start == -1:
        start = text.find('{')
    end_bracket = text.rfind(']')
    end_brace = text.rfind('}')
    end = max(end_bracket, end_brace)
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            print("\u2705 JSON extra\u00eddo correctamente")
            return parsed
        except json.JSONDecodeError:
            pass

    # Intento: objetos JSON separados por comas sin [] → envolverlos
    if text.startswith('{') and text.endswith('}') and '},{' in text:
        try:
            parsed = json.loads('[' + text + ']')
            print("\u2705 JSON reparado (a\u00f1adidos corchetes)")
            return parsed
        except json.JSONDecodeError:
            pass

    print("\u26a0\ufe0f  La respuesta no es JSON v\u00e1lido")
    return None


# ── CLI ─────────────────────────────────────────────────────────────

def check_ollama():
    """Verifica que Ollama est\u00e9 corriendo y el modelo disponible."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(MODEL_NAME in m for m in models):
            print(f"\u26a0\ufe0f  Modelo '{MODEL_NAME}' no encontrado.")
            print(f"   Modelos disponibles: {models}")
            print(f"   Ejecuta: python generate_knowledge.py && ollama create landmark-finder -f Modelfile")
            return False
        return True
    except requests.ConnectionError:
        print("\u274c Ollama no est\u00e1 corriendo. In\u00edcialo: ollama serve")
        return False


def interactive():
    """Modo interactivo: pide coordenadas al usuario en bucle."""
    print("=" * 60)
    print("\U0001f3db\ufe0f  LANDMARK FINDER \u2014 Catalu\u00f1a (RAG + br\u00fajula)")
    print("=" * 60)
    print("Formatos:")
    print("  lat, lon                \u2192 busca en todas direcciones")
    print("  lat, lon, azimuth       \u2192 filtra por direcci\u00f3n c\u00e1mara")
    print("  lat, lon, azimuth, fov  \u2192 direcci\u00f3n + FOV custom")
    print()
    print("Azimuth: 0\u00b0=N, 90\u00b0=E, 180\u00b0=S, 270\u00b0=O")
    print("FOV: campo de visi\u00f3n en grados (default 70\u00b0)")
    print()
    print("Ejemplos:")
    print("  41.4036, 2.1744              \u2192 Sagrada Familia")
    print("  41.4036, 2.1744, 180         \u2192 mirando al sur")
    print("  41.4036, 2.1744, 90, 50      \u2192 mirando al este, FOV 50\u00b0")
    print("Escribe 'q' para salir.")
    print("=" * 60)

    while True:
        inp = input("\n\U0001f4cd lat, lon [, azimuth [, fov]]: ").strip()
        if inp.lower() in ("q", "quit", "salir", "exit"):
            break
        if not inp:
            continue
        try:
            parts = inp.replace(",", " ").split()
            lat, lon = float(parts[0]), float(parts[1])
            azimuth = float(parts[2]) % 360 if len(parts) >= 3 else None
            fov = float(parts[3]) if len(parts) >= 4 else DEFAULT_FOV
        except (ValueError, IndexError):
            print("\u274c Formato: lat, lon [, azimuth [, fov]]")
            continue
        query(lat, lon, azimuth=azimuth, fov=fov)


def main():
    if not check_ollama():
        sys.exit(1)
    load_landmarks()  # pre-cargar e indexar
    if len(sys.argv) >= 3:
        try:
            lat = float(sys.argv[1])
            lon = float(sys.argv[2])
            azimuth = float(sys.argv[3]) % 360 if len(sys.argv) >= 4 else None
            fov = float(sys.argv[4]) if len(sys.argv) >= 5 else DEFAULT_FOV
            query(lat, lon, azimuth=azimuth, fov=fov)
        except ValueError:
            print("\u274c Uso: python query_model.py <lat> <lon> [azimuth] [fov]")
            sys.exit(1)
    else:
        interactive()


if __name__ == "__main__":
    main()
