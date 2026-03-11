#!/usr/bin/env python3
"""
extract_landmarks.py
====================
Extrae edificios famosos, monumentos, lugares de interés turístico y 
puntos de referencia del archivo OSM PBF de Cataluña.

Genera un archivo JSON con la base de conocimiento de landmarks.
"""

import json
import sys
import os
import math

try:
    import osmium
except ImportError:
    print("Instalando pyosmium...")
    os.system(f"{sys.executable} -m pip install osmium")
    import osmium


class LandmarkHandler(osmium.SimpleHandler):
    """Handler de OSM que extrae landmarks relevantes."""

    # Categorías de elementos que nos interesan
    LANDMARK_TAGS = {
        # Edificios históricos y famosos
        'historic': [
            'monument', 'memorial', 'castle', 'ruins', 'archaeological_site',
            'church', 'cathedral', 'monastery', 'palace', 'fort', 'tower',
            'city_gate', 'building', 'heritage', 'manor', 'wayside_shrine'
        ],
        # Turismo
        'tourism': [
            'attraction', 'museum', 'artwork', 'gallery', 'viewpoint',
            'theme_park', 'zoo'
        ],
        # Edificios religiosos
        'amenity': [
            'place_of_worship', 'theatre', 'arts_centre', 'library',
            'university', 'fountain'
        ],
        # Edificios especiales
        'building': [
            'cathedral', 'church', 'chapel', 'mosque', 'synagogue',
            'temple', 'stadium', 'palace', 'castle', 'civic',
            'government', 'hospital', 'university', 'train_station'
        ],
        # Ocio
        'leisure': [
            'stadium', 'sports_centre', 'park', 'garden'
        ],
        # Man made
        'man_made': [
            'tower', 'lighthouse', 'bridge', 'observatory'
        ],
    }

    # Tags que indican que un lugar es "famoso" o notable
    FAME_INDICATORS = [
        'wikidata', 'wikipedia', 'heritage', 'heritage:operator',
        'historic:civilization', 'architect', 'name:en',
        'tourism', 'image', 'wikimedia_commons'
    ]

    def __init__(self):
        super().__init__()
        self.landmarks = []
        self.seen_names = set()
        self.count = 0

    def _is_landmark(self, tags):
        """Comprueba si el elemento es un landmark relevante."""
        for category, values in self.LANDMARK_TAGS.items():
            if category in tags:
                tag_value = tags.get(category)
                if tag_value in values or tag_value == 'yes':
                    return True
        return False

    def _calculate_fame_score(self, tags):
        """Calcula un score de 'fama' basado en los tags disponibles."""
        score = 0
        for indicator in self.FAME_INDICATORS:
            if indicator in tags:
                score += 1
        # Bonus por tener nombre en varios idiomas
        name_langs = sum(1 for k in tags if k.startswith('name:'))
        score += min(name_langs, 5)
        return score

    def _extract_info(self, tags, lat, lon):
        """Extrae información relevante de los tags."""
        name = tags.get('name', tags.get('name:es', tags.get('name:ca', '')))
        if not name:
            return None

        # Evitar duplicados
        name_key = f"{name}_{round(lat, 3)}_{round(lon, 3)}"
        if name_key in self.seen_names:
            return None
        self.seen_names.add(name_key)

        fame_score = self._calculate_fame_score(tags)

        # Solo incluir landmarks con cierta notabilidad
        if fame_score < 1 and 'tourism' not in tags and 'historic' not in tags:
            return None

        info = {
            'name': name,
            'lat': round(lat, 6),
            'lon': round(lon, 6),
            'fame_score': fame_score,
        }

        # Información adicional
        if 'name:es' in tags:
            info['name_es'] = tags['name:es']
        if 'name:ca' in tags:
            info['name_ca'] = tags['name:ca']
        if 'name:en' in tags:
            info['name_en'] = tags['name:en']

        # Tipo de landmark
        categories = []
        for cat in ['historic', 'tourism', 'amenity', 'building', 'leisure', 'man_made']:
            if cat in tags:
                categories.append(f"{cat}={tags[cat]}")
        info['categories'] = categories

        # Descripción
        if 'description' in tags:
            info['description'] = tags['description']
        if 'description:es' in tags:
            info['description'] = tags['description:es']

        # Wikipedia/Wikidata
        if 'wikipedia' in tags:
            info['wikipedia'] = tags['wikipedia']
        if 'wikidata' in tags:
            info['wikidata'] = tags['wikidata']

        # Arquitecto
        if 'architect' in tags:
            info['architect'] = tags['architect']

        # Año/época
        if 'start_date' in tags:
            info['year'] = tags['start_date']
        elif 'building:year' in tags:
            info['year'] = tags['building:year']

        # Estilo arquitectónico
        if 'building:architecture' in tags:
            info['style'] = tags['building:architecture']
        elif 'architect:style' in tags:
            info['style'] = tags['architect:style']

        # Dirección
        addr_parts = []
        if 'addr:street' in tags:
            addr_parts.append(tags['addr:street'])
        if 'addr:housenumber' in tags:
            addr_parts.append(tags['addr:housenumber'])
        if 'addr:city' in tags:
            addr_parts.append(tags['addr:city'])
        if addr_parts:
            info['address'] = ', '.join(addr_parts)

        # Religión (para edificios religiosos)
        if 'religion' in tags:
            info['religion'] = tags['religion']
        if 'denomination' in tags:
            info['denomination'] = tags['denomination']

        return info

    def node(self, n):
        """Procesa nodos OSM."""
        tags = {tag.k: tag.v for tag in n.tags}
        if self._is_landmark(tags):
            info = self._extract_info(tags, n.location.lat, n.location.lon)
            if info:
                info['osm_type'] = 'node'
                info['osm_id'] = n.id
                self.landmarks.append(info)
                self.count += 1
                if self.count % 100 == 0:
                    print(f"  Encontrados {self.count} landmarks...")

    def way(self, w):
        """Procesa ways OSM (edificios, etc.)."""
        tags = {tag.k: tag.v for tag in w.tags}
        if self._is_landmark(tags):
            # Calcular centroide
            try:
                nodes = list(w.nodes)
                if nodes:
                    lats = [n.lat for n in nodes if n.location.valid()]
                    lons = [n.lon for n in nodes if n.location.valid()]
                    if lats and lons:
                        lat = sum(lats) / len(lats)
                        lon = sum(lons) / len(lons)
                        info = self._extract_info(tags, lat, lon)
                        if info:
                            info['osm_type'] = 'way'
                            info['osm_id'] = w.id
                            self.landmarks.append(info)
                            self.count += 1
                            if self.count % 100 == 0:
                                print(f"  Encontrados {self.count} landmarks...")
            except Exception:
                pass

    def relation(self, r):
        """Procesa relaciones OSM."""
        tags = {tag.k: tag.v for tag in r.tags}
        if self._is_landmark(tags):
            # Para relaciones, usamos el primer miembro con ubicación
            # o simplemente registramos sin coordenadas exactas
            name = tags.get('name', '')
            if name and name not in self.seen_names:
                info = {
                    'name': name,
                    'osm_type': 'relation',
                    'osm_id': r.id,
                    'categories': [],
                    'fame_score': self._calculate_fame_score(tags),
                }
                for cat in ['historic', 'tourism', 'amenity', 'building']:
                    if cat in tags:
                        info['categories'].append(f"{cat}={tags[cat]}")
                if 'wikipedia' in tags:
                    info['wikipedia'] = tags['wikipedia']
                if 'wikidata' in tags:
                    info['wikidata'] = tags['wikidata']
                if 'architect' in tags:
                    info['architect'] = tags['architect']
                # Relaciones sin coordenadas se marcan para resolver después
                info['needs_geocoding'] = True
                self.landmarks.append(info)
                self.seen_names.add(name)


def extract_landmarks(pbf_path, output_path, region_name='España'):
    """Extrae landmarks del archivo PBF."""
    print(f"📍 Extrayendo landmarks de: {pbf_path}")
    print(f"   Esto puede tardar varios minutos...\n")

    handler = LandmarkHandler()

    # Primera pasada: necesitamos locations para ways
    handler.apply_file(pbf_path, locations=True)

    # Filtrar landmarks sin coordenadas válidas
    valid_landmarks = []
    for lm in handler.landmarks:
        if 'lat' in lm and 'lon' in lm:
            valid_landmarks.append(lm)
        elif lm.get('needs_geocoding'):
            # Mantener para referencia pero sin coordenadas
            valid_landmarks.append(lm)

    # Ordenar por score de fama (más famosos primero)
    valid_landmarks.sort(key=lambda x: x.get('fame_score', 0), reverse=True)

    print(f"\n✅ Extraídos {len(valid_landmarks)} landmarks")
    print(f"   Top 10 más notables:")
    for i, lm in enumerate(valid_landmarks[:10]):
        coords = f"({lm.get('lat', '?')}, {lm.get('lon', '?')})"
        print(f"   {i+1}. {lm['name']} {coords} [score: {lm.get('fame_score', 0)}]")

    # Guardar JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'source': os.path.basename(pbf_path),
                'total_landmarks': len(valid_landmarks),
                'region': region_name,
            },
            'landmarks': valid_landmarks
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Guardado en: {output_path}")
    return valid_landmarks


def merge_landmarks(data_dir, output_path):
    """Fusiona todos los JSON regionales en un solo landmarks.json."""
    import glob
    all_landmarks = []
    sources = []
    seen_keys = set()

    json_files = sorted(glob.glob(os.path.join(data_dir, 'landmarks_*.json')))
    if not json_files:
        print("❌ No se encontraron archivos landmarks_*.json")
        return []

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        region = data.get('metadata', {}).get('region', os.path.basename(jf))
        lms = data.get('landmarks', [])
        added = 0
        for lm in lms:
            # Deduplicar por nombre + coordenadas aproximadas
            key = f"{lm.get('name', '')}_{round(lm.get('lat', 0), 3)}_{round(lm.get('lon', 0), 3)}"
            if key not in seen_keys:
                seen_keys.add(key)
                lm['region'] = region
                all_landmarks.append(lm)
                added += 1
        sources.append(region)
        print(f"  📂 {os.path.basename(jf)}: {added} landmarks ({len(lms) - added} duplicados)")

    # Ordenar por score de fama
    all_landmarks.sort(key=lambda x: x.get('fame_score', 0), reverse=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'sources': sources,
                'total_landmarks': len(all_landmarks),
                'regions': sources,
            },
            'landmarks': all_landmarks
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Total combinado: {len(all_landmarks)} landmarks de {len(sources)} regiones")
    print(f"💾 Guardado en: {output_path}")
    return all_landmarks


def region_from_filename(filename):
    """Extrae nombre de región legible del nombre del archivo PBF."""
    base = os.path.basename(filename).replace('.osm.pbf', '')
    # Quitar fecha (ej: cataluna-260310 → cataluna)
    parts = base.rsplit('-', 1)
    if len(parts) == 2 and parts[1].isdigit():
        base = parts[0]
    # Nombres legibles
    names = {
        'cataluna': 'Cataluña',
        'madrid': 'Madrid',
        'valencia': 'Valencia',
        'pais-vasco': 'País Vasco',
        'andalucia': 'Andalucía',
        'galicia': 'Galicia',
        'aragon': 'Aragón',
        'castilla-y-leon': 'Castilla y León',
        'castilla-la-mancha': 'Castilla-La Mancha',
        'spain': 'España',
    }
    return names.get(base, base.replace('-', ' ').title())


if __name__ == '__main__':
    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Buscar TODOS los archivos .osm.pbf en el directorio padre
    pbf_files = sorted(glob.glob(os.path.join(parent_dir, '*.osm.pbf')))

    if not pbf_files:
        print(f"❌ No se encontraron archivos .osm.pbf en: {parent_dir}")
        sys.exit(1)

    print(f"📍 Encontrados {len(pbf_files)} archivos PBF:")
    for pf in pbf_files:
        size_mb = os.path.getsize(pf) / (1024 * 1024)
        print(f"   • {os.path.basename(pf)} ({size_mb:.0f} MB)")
    print()

    # Procesar cada PBF
    for pbf_path in pbf_files:
        region = region_from_filename(pbf_path)
        base_name = os.path.basename(pbf_path).replace('.osm.pbf', '')
        # Quitar fecha del nombre para el JSON
        parts = base_name.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            base_name = parts[0]
        output_path = os.path.join(data_dir, f'landmarks_{base_name}.json')

        # Si ya existe y es reciente, preguntar
        if os.path.exists(output_path):
            size_kb = os.path.getsize(output_path) / 1024
            print(f"\nℹ️  landmarks_{base_name}.json ya existe ({size_kb:.0f} KB)")
            resp = input(f"   ¿Regenerar {region}? (s/n): ").strip().lower()
            if resp != 's':
                print(f"   ⏭️  Saltando {region}")
                continue

        print(f"\n{'='*60}")
        print(f"🗺️  Procesando: {region}")
        print(f"{'='*60}")
        extract_landmarks(pbf_path, output_path, region_name=region)

    # Fusionar todos en un solo archivo
    print(f"\n{'='*60}")
    print(f"🔗 Fusionando todos los landmarks...")
    print(f"{'='*60}")
    merged_path = os.path.join(data_dir, 'landmarks.json')
    merge_landmarks(data_dir, merged_path)
