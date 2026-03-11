#!/usr/bin/env python3
"""
generate_knowledge.py
=====================
Genera el Modelfile para Ollama con un system prompt LIGERO (sin datos).
La base de conocimiento se usa en tiempo de consulta (RAG local en query_model.py).
"""

import json
import os
import sys


SYSTEM_PROMPT = """\
You are a landmark identification system. You receive GPS coordinates and a numbered list of nearby landmarks with distances.

RULES:
1. NEVER invent landmarks. Use ONLY names from the provided list.
2. Copy landmark names EXACTLY and COMPLETELY from the list. Never truncate or abbreviate.
3. Respond ONLY with valid JSON, no extra text.

When camera orientation (azimuth) is provided:
- Return: {"target":"EXACT full name","target_distance":meters,"confidence":"high|medium|low","others":[{"name":"EXACT full name","distance":meters}]}
- Pick target: lowest angle offset from center + closest distance.

When NO orientation:
- Return: [{"name":"EXACT full name","distance":meters,"confidence":"high|medium|low"}]
- Confidence: <50m=high, <300m=high, <1km=medium, >1km=low\
"""


def generate_modelfile():
    """Genera el Modelfile para Ollama con system prompt ligero."""
    modelfile = f'''# Modelfile para Landmark Finder España
# Modelo basado en llama3.2:3b — system prompt ligero, datos via RAG
# Optimizado para RTX 3060 6GB VRAM + 16GB RAM

FROM llama3.2:3b

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
PARAMETER num_predict 512
PARAMETER stop "<|eot_id|>"

SYSTEM """{SYSTEM_PROMPT}"""
'''
    return modelfile


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    landmarks_path = os.path.join(data_dir, 'landmarks.json')
    if not os.path.exists(landmarks_path):
        print("❌ No se encuentra landmarks.json")
        print("   Ejecuta primero: python extract_landmarks.py")
        sys.exit(1)

    # Guardar system prompt como referencia
    prompt_path = os.path.join(data_dir, 'system_prompt.txt')
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(SYSTEM_PROMPT)
    print(f"📝 System prompt guardado: {prompt_path} ({len(SYSTEM_PROMPT)} bytes)")

    # Generar Modelfile
    modelfile = generate_modelfile()
    modelfile_path = os.path.join(script_dir, 'Modelfile')
    with open(modelfile_path, 'w', encoding='utf-8') as f:
        f.write(modelfile)
    print(f"📝 Modelfile guardado: {modelfile_path}")

    # Stats del JSON de landmarks
    with open(landmarks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    landmarks = data['landmarks']
    geo = [l for l in landmarks if 'lat' in l and 'lon' in l]
    print(f"\n📊 Landmarks disponibles para RAG:")
    print(f"   Total: {len(landmarks)}")
    print(f"   Con coordenadas: {len(geo)}")
    print(f"\n✅ Ahora ejecuta:")
    print(f"   ollama create landmark-finder -f Modelfile")


if __name__ == '__main__':
    main()
