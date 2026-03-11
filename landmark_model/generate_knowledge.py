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
Eres un sistema de identificación de edificios y monumentos.
Recibes coordenadas GPS, orientación de la cámara y una lista de lugares cercanos con distancia y desviación angular respecto al centro de la cámara.

Cuando se proporciona orientación de cámara (azimut), responde con un JSON con:
- "target": nombre del edificio al que la cámara apunta (el más centrado y cercano)
- "target_distance": distancia en metros
- "confidence": high/medium/low
- "others": array de otros lugares visibles [{name, distance}]

Para elegir el target: prioriza menor desviación del centro (angle_from_center) y menor distancia. Si un edificio está a 0-2° del centro, es claramente el objetivo.

Cuando NO hay orientación, responde con un JSON array de lugares cercanos: [{name, distance, confidence}].

NUNCA inventes lugares. Usa SOLO los de la lista proporcionada. Responde SOLO con JSON.\
"""


def generate_modelfile():
    """Genera el Modelfile para Ollama con system prompt ligero."""
    modelfile = f'''# Modelfile para Landmark Finder Cataluña
# Modelo basado en llama3.2:3b — system prompt ligero, datos via RAG
# Optimizado para RTX 3060 6GB VRAM + 16GB RAM

FROM llama3.2:3b

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|eot_id|>"

SYSTEM """{SYSTEM_PROMPT}"""
'''
    return modelfile


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    landmarks_path = os.path.join(data_dir, 'landmarks_cataluna.json')
    if not os.path.exists(landmarks_path):
        print("❌ No se encuentra landmarks_cataluna.json")
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
    print(f"   ollama create landmark-cataluna -f Modelfile")


if __name__ == '__main__':
    main()
