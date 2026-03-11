#!/usr/bin/env python3
"""
setup.py
========
Script de setup completo que:
1. Instala dependencias Python
2. Extrae landmarks del archivo OSM
3. Genera la base de conocimiento
4. Crea y registra el modelo en Ollama

Uso:
  python setup.py
"""

import subprocess
import sys
import os
import shutil


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
MODEL_NAME = "landmark-finder"


def run_command(cmd, description, check=True):
    """Ejecuta un comando mostrando progreso."""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"{'='*60}")
    print(f"   Comando: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), check=check,
                          capture_output=False)
    return result.returncode == 0


def check_prerequisites():
    """Verifica requisitos previos."""
    print("\n🔍 Verificando requisitos...\n")
    errors = []

    # Python
    print(f"  ✅ Python {sys.version.split()[0]}")

    # Archivos PBF
    import glob
    pbf_files = glob.glob(os.path.join(PARENT_DIR, '*.osm.pbf'))
    if pbf_files:
        total_mb = sum(os.path.getsize(f) for f in pbf_files) / (1024 * 1024)
        print(f"  ✅ {len(pbf_files)} archivos OSM encontrados ({total_mb:.0f} MB total)")
        for pf in pbf_files:
            print(f"     • {os.path.basename(pf)}")
    else:
        errors.append(f"  ❌ No se encuentran archivos .osm.pbf en: {PARENT_DIR}")

    # Ollama
    ollama_path = shutil.which('ollama')
    if ollama_path:
        print(f"  ✅ Ollama encontrado: {ollama_path}")
    else:
        errors.append("  ❌ Ollama no está instalado")
        errors.append("     Instálalo desde: https://ollama.ai/download")

    # Verificar si Ollama está corriendo
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            print("  ✅ Ollama está corriendo")
        else:
            errors.append("  ⚠️  Ollama no responde correctamente")
    except Exception:
        errors.append("  ⚠️  Ollama no está corriendo. Inícialo con: ollama serve")

    if errors:
        print("\n⚠️  Problemas encontrados:")
        for e in errors:
            print(e)
        return False

    return True


def step1_install_deps():
    """Paso 1: Instalar dependencias."""
    print(f"\n{'='*60}")
    print("📦 PASO 1: Instalando dependencias Python...")
    print(f"{'='*60}")

    deps = ['osmium', 'requests', 'geopy']
    for dep in deps:
        subprocess.run([sys.executable, '-m', 'pip', 'install', dep],
                      capture_output=True)
        print(f"  ✅ {dep}")
    
    print("  ✅ Dependencias instaladas")
    return True


def step2_extract_landmarks():
    """Paso 2: Extraer landmarks del archivo OSM."""
    landmarks_file = os.path.join(DATA_DIR, 'landmarks.json')
    
    if os.path.exists(landmarks_file):
        size = os.path.getsize(landmarks_file)
        if size > 1000:  # Más de 1KB = probablemente válido
            print(f"\n  ℹ️  landmarks.json ya existe ({size/1024:.0f} KB)")
            resp = input("  ¿Regenerar? (s/n): ").strip().lower()
            if resp != 's':
                print("  ⏭️  Saltando extracción")
                return True

    return run_command(
        [sys.executable, os.path.join(SCRIPT_DIR, 'extract_landmarks.py')],
        "PASO 2: Extrayendo landmarks del archivo OSM (puede tardar varios minutos)..."
    )


def step3_generate_knowledge():
    """Paso 3: Generar base de conocimiento."""
    return run_command(
        [sys.executable, os.path.join(SCRIPT_DIR, 'generate_knowledge.py')],
        "PASO 3: Generando base de conocimiento y Modelfile..."
    )


def step4_pull_base_model():
    """Paso 4: Descargar modelo base si no existe."""
    print(f"\n{'='*60}")
    print("📥 PASO 4: Verificando modelo base llama3.2:3b...")
    print(f"{'='*60}")

    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        
        if any('llama3.2:3b' in m for m in models):
            print("  ✅ llama3.2:3b ya está descargado")
            return True
    except Exception:
        pass

    print("  📥 Descargando llama3.2:3b...")
    return run_command('ollama pull llama3.2:3b',
                      "Descargando modelo base llama3.2:3b...")


def step5_create_model():
    """Paso 5: Crear el modelo personalizado en Ollama."""
    modelfile_path = os.path.join(SCRIPT_DIR, 'Modelfile')
    
    if not os.path.exists(modelfile_path):
        print("  ❌ No se encuentra Modelfile. Ejecuta paso 3 primero.")
        return False

    return run_command(
        f'cd "{SCRIPT_DIR}" && ollama create {MODEL_NAME} -f Modelfile',
        f"PASO 5: Creando modelo '{MODEL_NAME}' en Ollama..."
    )


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║         🏛️  LANDMARK FINDER - SETUP ESPAÑA  🏛️         ║
║                                                          ║
║  Este script configurará un modelo de Ollama capaz de    ║
║  identificar edificios famosos a partir de coordenadas   ║
╚══════════════════════════════════════════════════════════╝
    """)

    # Verificar requisitos
    if not check_prerequisites():
        print("\n❌ Corrige los problemas anteriores y vuelve a ejecutar.")
        print("   Si Ollama no está corriendo, ábrelo en otra terminal: ollama serve")
        sys.exit(1)

    # Ejecutar pasos
    steps = [
        ("Instalar dependencias", step1_install_deps),
        ("Extraer landmarks", step2_extract_landmarks),
        ("Generar conocimiento", step3_generate_knowledge),
        ("Descargar modelo base", step4_pull_base_model),
        ("Crear modelo Ollama", step5_create_model),
    ]

    for name, func in steps:
        try:
            success = func()
            if not success:
                print(f"\n❌ Falló el paso: {name}")
                print("   Corrige el error y vuelve a ejecutar setup.py")
                sys.exit(1)
        except Exception as e:
            print(f"\n❌ Error en '{name}': {e}")
            sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                  ✅ SETUP COMPLETADO                     ║
╚══════════════════════════════════════════════════════════╝

🎉 El modelo '{MODEL_NAME}' está listo.

Para usarlo:

  1. Modo interactivo:
     python query_model.py

  2. Consulta directa (ejemplo: Sagrada Familia):
     python query_model.py 41.4036 2.1744

  3. Desde la línea de comandos de Ollama:
     ollama run {MODEL_NAME}
     > Estoy en las coordenadas 41.4036, 2.1744. ¿Qué edificios famosos hay cerca?

¡Disfruta! 🏛️
    """)


if __name__ == '__main__':
    main()
