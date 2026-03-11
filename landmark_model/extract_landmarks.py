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


def extract_landmarks(pbf_path, output_path):
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
                'region': 'Cataluña, España',
            },
            'landmarks': valid_landmarks
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Guardado en: {output_path}")
    return valid_landmarks


if __name__ == '__main__':
    # Rutas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    pbf_path = os.path.join(parent_dir, 'cataluna-260222.osm.pbf')
    output_path = os.path.join(script_dir, 'data', 'landmarks_cataluna.json')

    if not os.path.exists(pbf_path):
        print(f"❌ No se encuentra el archivo: {pbf_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extract_landmarks(pbf_path, output_path)
