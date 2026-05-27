import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import glob
from matplotlib.colors import to_rgba
try:
    import contextily as ctx
except ImportError:
    ctx = None
    print("---")
    print("INFO: 'contextily' library not found. Basemap tiles will be disabled.")
    print("      To enable map tiles, install it by running this in your terminal:")
    print("      pip install contextily")
    print("---")
import colorsys
from matplotlib.patches import Rectangle
import threading # Lo dejo aunque ahora no lo use directamente, por si alguna parte del programa depende de esto.

import math
import json
import hashlib
import tempfile
from io import BytesIO
import xml.etree.ElementTree as ET

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except ImportError:
    Transformer = None
    PYPROJ_AVAILABLE = False
    print("---")
    print("INFO: 'requests' library not found. OSM building download will be disabled.")
    print("      To enable OSM building download, install it by running this in your terminal:")
    print("      pip install requests")
    print("---")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.neighbors import KNeighborsRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestRegressor = None
    train_test_split = None
    mean_absolute_error = None
    r2_score = None
    KNeighborsRegressor = None
    SKLEARN_AVAILABLE = False
    print("---")
    print("INFO: 'scikit-learn' library not found. AI coverage prediction will be disabled.")
    print("      To enable AI prediction, install it by running this in your terminal:")
    print("      pip install scikit-learn")
    print("---")

# Intento cargar adjustText; si no está instalado, solo aviso porque no es imprescindible.
try:
    from adjustText import adjustText
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("---")
    print("INFO: 'adjustText' library not found. Labels may overlap.")
    print("      For automatic label adjustment, install it by running this in your terminal:")
    print("      pip install adjustText")
    print("---")

# Intento cargar Pillow; hace falta para exportar GIF.
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("---")
    print("INFO: 'Pillow' (PIL) library not found. GIF export will be disabled.")
    print("      To enable GIF export, install it by running this in your terminal:")
    print("      pip install Pillow")
    print("---")


# --- Constantes generales y detección de la estructura de archivos ---

FILE_PATTERN = r"(?P<tester>[a-zA-Z0-9]+)_(?P<operator>[a-zA-Z0-9]+)_(?P<date>\d{4}-\d{2}-\d{2}|\d{2}-\d{2})(?:_(?P<technology>[2345]G))?_.*\.(?:csv|txt|kml|clf)"

DATE_TOKEN_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    re.compile(r"^\d{2}-\d{2}$"),
]
TECH_TOKEN_RE = re.compile(r"^[2345]G$", re.IGNORECASE)
DEFAULT_MERGE_TOLERANCE_MINUTES = 5
IGNORED_PREFIX_TOKENS = {'gmonpro', 'gmon', 'g', 'gm', 'cell', 'log', 'celllog', 'celllogs', 'netmonitor', 'net', 'monitor'}


def normalize_token(value, default="Unknown"):
    value = "" if value is None else str(value).strip()
    return value if value else default


def parse_campaign_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = [p for p in re.split(r"[_\s]+", stem) if p]
    if len(parts) < 2:
        return None

    date_idx = None
    for i, part in enumerate(parts):
        if any(pattern.fullmatch(part) for pattern in DATE_TOKEN_PATTERNS):
            date_idx = i
            break
    if date_idx is None:
        return None

    pre = parts[:date_idx]
    while pre and pre[0].lower() in IGNORED_PREFIX_TOKENS:
        pre.pop(0)
    if len(pre) < 2:
        return None

    tester = normalize_token(pre[0])
    operator = normalize_token(pre[1])
    date_value = parts[date_idx]

    technology = "Unknown"
    for token in parts[date_idx + 1:]:
        if TECH_TOKEN_RE.fullmatch(token.upper()):
            technology = token.upper()
            break

    return {
        "tester": tester,
        "operator": operator,
        "date": date_value,
        "technology": technology,
        "stem": stem,
        "filename": os.path.basename(filename),
    }


def campaign_matches(meta, params):
    if not meta:
        return False
    for key in ("tester", "operator", "date", "technology"):
        if normalize_token(meta.get(key)).lower() != normalize_token(params.get(key)).lower():
            return False
    return True

CSV_COLUMNS = {
    'RSRP': 'RSRP/RSCP', 
    'RSRQ': 'RSRQ/ECIO',
    'SINR': 'SINR',
    'SNR': 'SNR',
    'LAT': 'LAT', 
    'LON': 'LON', 
    'SYSTEM': 'SYSTEM',
    'XCI': 'XCI',
    'TIME': 'TIME',
    'DATE': 'DATE',
    'PCI': 'PCI/PSC/BSIC',
    'CELL_ID': 'CELL_ID',
    'LOCAL_CID': 'LOCAL_CID',
    'LAC_TAC': 'LAC/TAC',
    'OPERATOR': 'OPERATOR',
    'DOWNLOAD_SPEED': 'DOWNLOAD_SPEED',
    'UPLOAD_SPEED': 'UPLOAD_SPEED', 
    'LATENCY': 'LATENCY'
}

# Parámetros que se pueden animar en las gráficas
ANIMATION_PARAMS = ['RSRP', 'RSRQ', 'SINR', 'SNR', 'TECHNOLOGY', 'DOWNLOAD_SPEED', 'UPLOAD_SPEED', 'LATENCY']

# Unidades que quiero mostrar junto a cada parámetro


# Función auxiliar para que el mapa base se vea más nítido en el análisis
try:
    from rasterio.enums import Resampling as RasterioResampling
except Exception:
    RasterioResampling = None

def add_sharp_basemap(ax, zoom):
    """Añado el mapa base de OSM con más nitidez cuando contextily está disponible."""
    if ctx is None:
        ax.grid(True, linestyle='--', alpha=0.25)
        return
    try:
        kwargs = dict(
            crs='EPSG:4326',
            source=ctx.providers.OpenStreetMap.Mapnik,
            zoom=zoom,
            interpolation='nearest',
            attribution_size=6,
            reset_extent=False,
        )
        if RasterioResampling is not None:
            kwargs['resampling'] = RasterioResampling.nearest
        ctx.add_basemap(ax, **kwargs)
    except Exception:
        try:
            ctx.add_basemap(
                ax,
                crs='EPSG:4326',
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=zoom,
                interpolation='nearest',
                reset_extent=False,
            )
        except Exception:
            ax.grid(True, linestyle='--', alpha=0.25)

PARAMETER_UNITS = {
    'RSRP': 'dBm',
    'RSRQ': 'dB',
    'SINR': 'dB',
    'SNR': 'dB',
    'DOWNLOAD_SPEED': 'Mbps',
    'UPLOAD_SPEED': 'Mbps',
    'LATENCY': 'ms',
    'JITTER': 'ms', # Añadido para este caso
    'TECHNOLOGY': '' # Sin unidad
}

# Función auxiliar para mostrar el nombre del parámetro con su unidad
def get_param_with_unit(param):
    """Devuelve el nombre del parámetro con su unidad para mostrarlo en pantalla."""
    unit = PARAMETER_UNITS.get(param, '')
    if unit:
        # Caso especial para los nombres de velocidad
        if param == 'DOWNLOAD_SPEED':
            return f"Download ({unit})"
        if param == 'UPLOAD_SPEED':
            return f"Upload ({unit})"
        if param == 'JITTER': # Añadido para este caso
            return f"Jitter ({unit})" # Añadido para este caso
        return f"{param} ({unit})"
    return param


# Paleta de colores para comparar varios sets
COLOR_PALETTE = [
    '#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF',
    '#FFA500', '#800080', '#008000', '#FFC0CB', '#A52A2A', '#000080'
]

# Colores para las estaciones base
CELL_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#5F27CD']

# Equivalencias de nombres de operadores
OPERATOR_MAPPING = {
    'VODA': 'Vodafone',
    'VF': 'Vodafone',
    'ORAN': 'Orange',
    'ORA': 'Orange',
    'TMO': 'T-Mobile',
    'TMUS': 'T-Mobile',
    'ATT': 'AT&T',
    'VER': 'Verizon',
    'VERI': 'Verizon',
    'SPR': 'Sprint',
    'TEL': 'Telefonica',
    'D-T': 'Deutsche Telekom',
    'BT': 'BT',
    'EE': 'EE',
    'DIGI': 'DIGIMOVIL',
    'DIGIM': 'DIGIMOVIL',
    'DIGIMO': 'DIGIMOVIL'
}

# Mapeo de tecnologías usando varias formas de detección
TECHNOLOGY_MAPPING = {
    '2G': ['GSM', 'GPRS', 'EDGE', 'GERAN'],
    '3G': ['WCDMA', 'UMTS', 'HSPA', 'HSPA+', 'UTRAN'],
    '4G': ['LTE', 'LTE-A', 'LTE+', 'E-UTRAN'],
    '5G': ['NR', '5G', '5G NR', 'NG-RAN']
}

# Códigos RAT que aparecen en los archivos CLF
RAT_CODE_MAPPING = {
    '01': '2G',  # GSM
    '02': '3G',  # UMTS
    '03': '2G',  # GSM compacto
    '04': '3G',  # UTRAN
    '05': '3G',  # EGPRS
    '06': '4G',  # LTE
    '07': '4G',  # LTE
    '08': '3G',  # HSDPA
    '09': '3G',  # HSUPA
    '10': '3G',  # HSPA
    '11': '3G',  # HSPA+
    '12': '3G',  # DC-HSPA+
    '13': '5G',  # NR
    '14': '5G',  # NR
    '15': '5G',  # NR
}

# Códigos SYSTEM que vienen en los CSV
SYSTEM_CODE_MAPPING = {
    '0': 'UNKNOWN',  # Sin servicio
    '1': '2G',       # GSM
    '2': '3G',       # UMTS
    '3': '4G',       # LTE
    '4': '4G',       # LTE
    '5': '5G',       # NR
    '6': '5G',       # NR
}

# Colores para representar cada tecnología en el heatmap
TECHNOLOGY_COLORS = {
    '2G': '#FF6B6B',  # Rojo
    '3G': '#4ECDC4',  # Azul verdoso
    '4G': '#45B7D1',  # Azul
    '5G': '#96CEB4',  # Verde
    'UNKNOWN': '#FFA500'  # Naranja
}

# Orden de tecnologías para el eje Y cuando se dibujan como serie
TECHNOLOGY_ORDER = {'2G': 1, '3G': 2, '4G': 3, '5G': 4}

# Criterios de cobertura según la tabla que estoy usando, sin rangos solapados
COVERAGE_CRITERIA = {
    'Excellent': { # Desde X hacia arriba
        'RSRP': {'min': -80, 'max': float('inf')},
        'RSRQ': {'min': -10, 'max': float('inf')},
        'SNR': {'min': 20, 'max': float('inf')}
    },
    'Good': { # Desde X hasta antes de Y
        'RSRP': {'min': -90, 'max': -80},
        'RSRQ': {'min': -15, 'max': -10},
        'SNR': {'min': 13, 'max': 20}
    },
    'Mid Cell': { # Desde X hasta antes de Y
        'RSRP': {'min': -100, 'max': -90},
        'RSRQ': {'min': -20, 'max': -15},
        'SNR': {'min': 0, 'max': 13}
    },
    'Cell Edge': { # Por debajo de X
        'RSRP': {'min': float('-inf'), 'max': -100},
        'RSRQ': {'min': float('-inf'), 'max': -20},
        'SNR': {'min': float('-inf'), 'max': 0}
    }
}

# Colores ajustados para que coincidan con las capturas de referencia
COVERAGE_COLORS = {
    'Excellent': '#00FF00',  # Verde
    'Good': '#FFFF00',       # Amarillo
    'Mid Cell': '#FFA500',   # Naranja
    'Cell Edge': '#FF0000', # Rojo
    'No Coverage': '#808080' # Gris
}

# Colores del gráfico de rendimiento, siguiendo la misma idea que cobertura
PERFORMANCE_COLORS = {
    'Excellent': COVERAGE_COLORS['Excellent'],  # Verde
    'Good': COVERAGE_COLORS['Good'],       # Amarillo
    'Fair': COVERAGE_COLORS['Mid Cell'],   # Naranja
    'Poor': COVERAGE_COLORS['Cell Edge'], # Rojo
    'No Data': COVERAGE_COLORS['No Coverage'] # Gris
}

# --- Procesado principal de datos y lógica de análisis ---

class DriveTestAnalysis:
    def __init__(self, folder_path, sets_params, merge_tolerance_minutes=5):
        self.folder_path = folder_path
        self.sets_params = sets_params
        self.data = {}
        self.combined_data = pd.DataFrame()
        self.is_ready = False
        self.color_map = {}
        self.cell_towers = {}
        self.operator_names = {}
        self.stats_cache = {}
        self.coverage_stats = {}
        self.merge_tolerance_minutes = int(merge_tolerance_minutes)

    def _find_file(self, set_params, file_kind):
        """Busca el archivo que corresponde a la campaña y al tipo indicado."""
        file_kind = file_kind.upper()
        candidates = []
        for filename in os.listdir(self.folder_path):
            path = os.path.join(self.folder_path, filename)
            if not os.path.isfile(path):
                continue
            meta = parse_campaign_filename(filename)
            if not campaign_matches(meta, set_params):
                continue
            upper = filename.upper()
            if file_kind == 'GMON' and upper.endswith('.CSV') and ('G_MON' in upper or 'GMON' in upper):
                candidates.append(path)
            elif file_kind == 'GMON_KML' and upper.endswith('.KML') and ('G_MON' in upper or 'GMON' in upper):
                candidates.append(path)
            elif file_kind == 'CELLLOG_KML' and upper.endswith('.KML') and ('CELLLOG' in upper or 'CELL_LOG' in upper or 'NETMONITOR' in upper):
                candidates.append(path)
            elif file_kind == 'CLF' and upper.endswith('.CLF'):
                candidates.append(path)
            elif file_kind == 'SPEEDTEST' and upper.endswith('.CSV') and 'SPEEDTEST' in upper:
                candidates.append(path)
        if not candidates:
            return None, f"Could not find a {file_kind} file for {set_params['tester']} | {set_params['operator']} | {set_params['date']} | {set_params['technology']}"
        return sorted(candidates)[0], None

    def _build_radio_datetime(self, df):
        dt = pd.Series(pd.NaT, index=df.index)
        if 'DATE' in df.columns and 'TIME' in df.columns:
            combined = (df['DATE'].astype(str).str.strip() + ' ' + df['TIME'].astype(str).str.strip()).str.strip()
            for dayfirst in (False, True):
                parsed = pd.to_datetime(combined, errors='coerce', dayfirst=dayfirst)
                if parsed.notna().sum() > 0:
                    dt = parsed
                    break
        elif 'TIME' in df.columns:
            dt = pd.to_datetime(df['TIME'].astype(str).str.strip(), errors='coerce', dayfirst=True)
        return dt

    def _load_speedtest_file(self, set_params):
        filepath, error = self._find_file(set_params, 'SPEEDTEST')
        if error:
            return None
        try:
            try:
                sdf = pd.read_csv(filepath)
            except Exception:
                sdf = pd.read_csv(filepath, sep=';', encoding='latin-1')
            if len(sdf.columns) == 1:
                try:
                    sdf = pd.read_csv(filepath, sep=';', encoding='latin-1')
                except Exception:
                    pass
            sdf = sdf.rename(columns={
                'Fecha': 'DATE_STR', 'Date': 'DATE_STR',
                'Velocidad de bajada': 'DOWNLOAD_SPEED', 'Download Speed': 'DOWNLOAD_SPEED',
                'Velocidad de subida': 'UPLOAD_SPEED', 'Upload Speed': 'UPLOAD_SPEED',
                'Latencia': 'LATENCY', 'Latency': 'LATENCY',
                'TipCon': 'ACCESS_TYPE', 'Lat': 'ST_LAT', 'Lon': 'ST_LON'
            })
            if 'DATE_STR' not in sdf.columns:
                return None
            out = pd.DataFrame()
            out['DATETIME'] = pd.to_datetime(sdf['DATE_STR'].astype(str).str.strip(), errors='coerce', dayfirst=True)
            out['DOWNLOAD_SPEED'] = pd.to_numeric(sdf.get('DOWNLOAD_SPEED'), errors='coerce')
            out['UPLOAD_SPEED'] = pd.to_numeric(sdf.get('UPLOAD_SPEED'), errors='coerce')
            out['LATENCY'] = pd.to_numeric(sdf.get('LATENCY'), errors='coerce')
            if 'ACCESS_TYPE' in sdf.columns:
                out['ACCESS_TYPE'] = sdf['ACCESS_TYPE'].astype(str)
            out = out.dropna(subset=['DATETIME']).sort_values('DATETIME').reset_index(drop=True)
            return out if not out.empty else None
        except Exception as exc:
            print(f"Failed to load speedtest file {os.path.basename(filepath)}: {exc}")
            return None

    def _merge_speedtest_data(self, df, speedtest_df):
        if speedtest_df is None or speedtest_df.empty:
            return df
        merged_df = df.copy()
        for col in ['DOWNLOAD_SPEED', 'UPLOAD_SPEED', 'LATENCY']:
            if col in merged_df.columns:
                merged_df = merged_df.drop(columns=[col])
        merged_df['DATETIME'] = self._build_radio_datetime(merged_df)
        if merged_df['DATETIME'].notna().sum() == 0:
            return merged_df
        radio_sorted = merged_df.reset_index().sort_values('DATETIME')
        speed_sorted = speedtest_df[['DATETIME', 'DOWNLOAD_SPEED', 'UPLOAD_SPEED', 'LATENCY']].sort_values('DATETIME')
        merged_sorted = pd.merge_asof(radio_sorted, speed_sorted, on='DATETIME', direction='nearest', tolerance=pd.Timedelta(minutes=self.merge_tolerance_minutes))
        return merged_sorted.sort_values('index').drop(columns=['index']).reset_index(drop=True)
    def _safe_convert_to_numeric(self, data):
        """Convierte datos a número intentando cubrir formatos raros o con comas."""
        try:
            if isinstance(data, pd.Series) and pd.api.types.is_numeric_dtype(data):
                return data
            if isinstance(data, pd.Series):
                series_str = data.astype(str)
                series_clean = series_str.str.replace(',', '.', regex=False)
                return pd.to_numeric(series_clean, errors='coerce')
            if isinstance(data, pd.DataFrame):
                series = data.iloc[:, 0]
                series_str = series.astype(str)
                series_clean = series_str.str.replace(',', '.', regex=False)
                return pd.to_numeric(series_clean, errors='coerce')
            return pd.to_numeric(data, errors='coerce')
        except Exception as e:
            print(f"Error converting to numeric: {e}")
            return pd.to_numeric(data, errors='coerce')

    # Detección del operador usando el archivo concreto que se está procesando.
    def _detect_operator(self, df, filepath):
        """
        Detecto primero el operador desde los datos y, si no se puede, desde el nombre del archivo.
        """
        # 1. Primero intento sacarlo de la columna OPERATOR del dataframe
        if 'OPERATOR' in df.columns and not df['OPERATOR'].isnull().all():
            operator_code = df['OPERATOR'].iloc[0]
            if isinstance(operator_code, str):
                operator_code = operator_code.upper()
                for code, name in OPERATOR_MAPPING.items():
                    if code in operator_code:
                        return name
                return operator_code # Si no lo tengo mapeado, devuelvo el código tal cual

        # 2. Si no aparece en datos, miro el nombre del archivo
        if filepath:
            filename = os.path.basename(filepath).upper()
            for code, name in OPERATOR_MAPPING.items():
                if code in filename:
                    return name

        # 3. Si no encuentro nada, lo dejo como Unknown
        return "Unknown"

    def _detect_technology_from_clf(self, row_data):
        """Detecta la tecnología en archivos CLF separados por tabuladores."""
        if isinstance(row_data, str):
            # Separación de la fila cuando viene tabulada
            columns = row_data.split('\t')
            if len(columns) > 6:
                rat_code = columns[6].strip()  # Séptima columna, teniendo en cuenta que Python empieza en 0
                return RAT_CODE_MAPPING.get(rat_code, 'UNKNOWN')
        return 'UNKNOWN'

    def _detect_technology_from_csv(self, system_value, band_value=None):
        """Detecta la tecnología cuando los datos vienen de un CSV."""
        if pd.isna(system_value):
            return 'UNKNOWN'
        
        # Lo paso a texto para tratar bien los códigos numéricos de sistema
        system_str = str(system_value).strip()
        
        # Primero pruebo con el mapeo de códigos SYSTEM
        if system_str in SYSTEM_CODE_MAPPING:
            tech = SYSTEM_CODE_MAPPING[system_str]
            if tech != 'UNKNOWN':
                return tech
        
        # Si no encaja, intento detectarlo por texto
        system_upper = system_str.upper()
        for tech, keywords in TECHNOLOGY_MAPPING.items():
            for keyword in keywords:
                if keyword in system_upper:
                    return tech
        
        # Como último apoyo, miro la banda por si da alguna pista
        if band_value and not pd.isna(band_value):
            band_str = str(band_value).upper()
            if any(x in band_str for x in ['GSM', '2G']):
                return '2G'
            elif any(x in band_str for x in ['UMTS', 'WCDMA', '3G']):
                return '3G'
            elif any(x in band_str for x in ['LTE', '4G']):
                return '4G'
            elif any(x in band_str for x in ['NR', '5G']):
                return '5G'
        
        return 'UNKNOWN'

    def _detect_technology(self, system_value, band_value=None, row_data=None, file_type='csv'):
        """Detección de tecnología pensada para funcionar con varios formatos de archivo."""
        if file_type == 'clf' and row_data is not None:
            return self._detect_technology_from_clf(row_data)
        else:
            return self._detect_technology_from_csv(system_value, band_value)

    def _extract_cell_towers(self, df, label):
        """Extrae las estaciones/celdas únicas usando XCI cuando está disponible."""
        cell_towers = {}
        has_xci = 'XCI' in df.columns and not df['XCI'].isnull().all()
        has_pci = 'PCI' in df.columns and not df['PCI'].isnull().all()
        
        if has_xci:
            cell_identifier = 'XCI'
        elif has_pci:
            print(f"Warning: No XCI column found for {label}. Falling back to PCI. (Note: PCI is not unique)")
            cell_identifier = 'PCI'
        else:
            print(f"Warning: No XCI or PCI column found for {label}. Cannot extract cell towers.")
            return cell_towers
        
        for cell_value in df[cell_identifier].unique():
            if pd.isna(cell_value):
                continue
                
            cell_data = df[df[cell_identifier] == cell_value]
            cell_data_first = cell_data.iloc[0]
            
            tower_lat = cell_data['LAT'].mean()
            tower_lon = cell_data['LON'].mean()
            operator = self.operator_names.get(label, "Unknown")
            
            try: xci_val = int(cell_data_first.get('XCI', 0))
            except (ValueError, TypeError): xci_val = cell_data_first.get('XCI', 'N/A')
            try: pci_val = int(cell_data_first.get('PCI', -1))
            except (ValueError, TypeError): pci_val = -1
            try: lac_val = int(cell_data_first.get('LAC_TAC', -1))
            except (ValueError, TypeError): lac_val = -1
            try: cid_val = int(cell_data_first.get('LOCAL_CID', -1))
            except (ValueError, TypeError): cid_val = -1

            cell_name = f"XCI: {xci_val}"
            pci_str = f"PCI: {pci_val}" if pci_val != -1 else ""
            lac_str = f"TAC: {lac_val}" if lac_val != -1 else ""
            cid_str = f"CID: {cid_val}" if cid_val != -1 else ""
            
            details = " | ".join(filter(None, [pci_str, lac_str, cid_str]))
            display_name = f"{cell_name}\n{details}\n({operator})"
            
            cell_towers[cell_value] = {
                'lat': tower_lat,
                'lon': tower_lon,
                'name': cell_name,
                'display_name': display_name,
                'operator': operator,
                'color': CELL_COLORS[len(cell_towers) % len(CELL_COLORS)],
                'samples': len(cell_data)
            }
        
        return cell_towers

    # Devuelvo tanto el dataframe como la ruta del archivo para usarlo después
    def _load_and_clean_data(self, set_params, label):
        """
        Carga una campaña y, si el CSV de G-MoN viene vacío o incompleto,
        prueba automáticamente con el CLF. Esto lo hago porque a veces G-MoN
        genera un CSV con una sola fila sin celda, pero NetMonitor sí guarda la ruta.
        """
        gmon_path, gmon_error = self._find_file(set_params, 'GMON')
        clf_path, clf_error = self._find_file(set_params, 'CLF')

        candidates = []
        if gmon_path:
            candidates.append(('GMON', gmon_path))
        if clf_path:
            candidates.append(('CLF', clf_path))

        if not candidates:
            messagebox.showerror(
                "File Error",
                f"No he encontrado GMON ni CLF para {set_params['tester']} | {set_params['operator']} | {set_params['date']} | {set_params['technology']}"
            )
            return None, None

        last_error = None

        def _read_gmon_csv(filepath):
            df = None
            for encoding in ['latin-1', 'utf-8', 'iso-8859-1']:
                try:
                    df = pd.read_csv(filepath, delimiter=';', encoding=encoding, on_bad_lines='skip')
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise ValueError("No se ha podido leer el CSV de G-MoN con las codificaciones habituales")

            actual_col_map, used_targets = {}, set()
            for col_name in df.columns:
                for target_key, target_value in CSV_COLUMNS.items():
                    if target_value.lower() in str(col_name).lower() and target_key not in used_targets:
                        actual_col_map[col_name] = target_key
                        used_targets.add(target_key)
                        break
            if actual_col_map:
                df.rename(columns=actual_col_map, inplace=True)

            df = df.loc[:, ~df.columns.duplicated()]

            if 'SYSTEM' in df.columns:
                band_col = df['BAND'] if 'BAND' in df.columns else None
                df['TECHNOLOGY'] = df.apply(
                    lambda row: self._detect_technology(
                        row['SYSTEM'],
                        band_col.iloc[row.name] if band_col is not None and row.name in band_col.index else None,
                        file_type='csv'
                    ),
                    axis=1
                )
            else:
                df['TECHNOLOGY'] = normalize_token(set_params.get('technology'))

            return df

        def _read_clf_file(filepath):
            # Los CLF de NetMonitor no tienen cabecera real, por eso uso header=None.
            # Estructura habitual:
            # 0 fecha, 1 hora, 2 XCI/CID, 3 TAC/LAC, 6 RAT, 7 LAT, 8 LON, 10 RSRP.
            df = None
            for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                try:
                    df = pd.read_csv(filepath, delimiter='\t', encoding=encoding, header=None, on_bad_lines='skip')
                    break
                except UnicodeDecodeError:
                    continue
            if df is None or df.empty:
                raise ValueError("No se ha podido leer el archivo CLF")

            rename_map = {}
            if df.shape[1] > 0: rename_map[0] = 'DATE_RAW'
            if df.shape[1] > 1: rename_map[1] = 'TIME_RAW'
            if df.shape[1] > 2: rename_map[2] = 'XCI'
            if df.shape[1] > 3: rename_map[3] = 'LAC_TAC'
            if df.shape[1] > 6: rename_map[6] = 'TECH_CODE'
            if df.shape[1] > 7: rename_map[7] = 'LAT'
            if df.shape[1] > 8: rename_map[8] = 'LON'
            if df.shape[1] > 10: rename_map[10] = 'RSRP'
            df.rename(columns=rename_map, inplace=True)

            if 'DATE_RAW' in df.columns:
                raw_date = df['DATE_RAW'].astype(str).str.strip()
                parsed_date = pd.to_datetime(raw_date, format='%Y%m%d', errors='coerce')
                df['DATE'] = parsed_date.dt.strftime('%Y/%m/%d')

            if 'TIME_RAW' in df.columns:
                raw_time = df['TIME_RAW'].astype(str).str.extract(r'(\d{6})', expand=False)
                df['TIME'] = raw_time.str.replace(r'(\d{2})(\d{2})(\d{2})', r'\1:\2:\3', regex=True)

            if 'TECH_CODE' in df.columns:
                df['TECHNOLOGY'] = df['TECH_CODE'].astype(str).str.strip().map(RAT_CODE_MAPPING).fillna(normalize_token(set_params.get('technology')))
            else:
                df['TECHNOLOGY'] = normalize_token(set_params.get('technology'))

            # En CLF no siempre vienen RSRQ/SINR/SNR, así que solo se pintará lo que exista.
            return df

        def _normalizar_y_validar(df, filepath, source_kind):
            missing_required = [k for k in ['LAT', 'LON'] if k not in df.columns]
            if missing_required:
                raise ValueError(f"Faltan columnas esenciales: {missing_required}. Columnas disponibles: {df.columns.tolist()}")

            for col in ['LAT', 'LON']:
                df[col] = self._safe_convert_to_numeric(df[col])

            for col in ['RSRP', 'RSRQ', 'SINR', 'SNR', 'PCI', 'CELL_ID', 'XCI', 'LOCAL_CID', 'LAC_TAC', 'DOWNLOAD_SPEED', 'UPLOAD_SPEED', 'LATENCY']:
                if col in df.columns:
                    df[col] = self._safe_convert_to_numeric(df[col])

            if 'OPERATOR' in df.columns:
                df['OPERATOR'] = df['OPERATOR'].astype(str)

            # Quito coordenadas inválidas. Esto evita que un punto 0,0 del CLF rompa la escala del mapa.
            df.dropna(subset=['LAT', 'LON'], inplace=True)
            df = df[(df['LAT'].abs() > 0.001) & (df['LON'].abs() > 0.001)].copy()

            if 'RSRP' not in df.columns and 'RSRP/RSCP' in df.columns:
                df['RSRP'] = df['RSRP/RSCP']
            if 'RSRQ' not in df.columns and 'RSRQ/ECIO' in df.columns:
                df['RSRQ'] = df['RSRQ/ECIO']

            # Algunos CLF meten filas de "sin servicio" con XCI=0 y RSRP=99.
            # Eso no es una medida real, así que lo quito para que no rompa mapas ni handovers.
            if 'XCI' in df.columns:
                df.loc[pd.to_numeric(df['XCI'], errors='coerce') == 0, 'XCI'] = np.nan
            if 'RSRP' in df.columns:
                df.loc[pd.to_numeric(df['RSRP'], errors='coerce') > 0, 'RSRP'] = np.nan
                if df['RSRP'].notna().sum() > 0:
                    df = df[df['RSRP'].notna()].copy()

            if df.empty:
                raise ValueError("Después de limpiar coordenadas no quedan muestras válidas")

            return df

        def _gmon_esta_incompleto(df):
            # Si G-MoN solo ha guardado una fila sin celda/señal, no lo uso como fuente principal.
            n_points = len(df)
            valid_rsrp = df['RSRP'].notna().sum() if 'RSRP' in df.columns else 0
            valid_xci = df['XCI'].notna().sum() if 'XCI' in df.columns else 0
            valid_pci = df['PCI'].notna().sum() if 'PCI' in df.columns else 0
            if n_points < 3:
                return True
            if valid_rsrp < 3 and (valid_xci + valid_pci) < 3:
                return True
            return False

        for source_kind, filepath in candidates:
            try:
                if source_kind == 'GMON':
                    df = _read_gmon_csv(filepath)
                else:
                    df = _read_clf_file(filepath)

                df = _normalizar_y_validar(df, filepath, source_kind)

                # Caso concreto que te estaba fallando: el 4G tenía un CSV de G-MoN de 1 fila.
                # Si existe CLF, sigo con CLF antes de intentar dibujar algo degenerado.
                if source_kind == 'GMON' and clf_path and _gmon_esta_incompleto(df):
                    last_error = f"{os.path.basename(filepath)} está incompleto; se usará CLF como respaldo."
                    print(f"INFO: {last_error}")
                    continue

                speedtest_df = self._load_speedtest_file(set_params)
                if speedtest_df is not None:
                    df = self._merge_speedtest_data(df, speedtest_df)

                df['Test_Label'] = label
                df['SCENARIO_TECH'] = normalize_token(set_params.get('technology'))
                df.reset_index(drop=True, inplace=True)

                if 'XCI' in df.columns:
                    df['XCI_numeric'] = pd.to_numeric(df['XCI'], errors='coerce')
                    df['HANDOVER_FLAG'] = (df['XCI_numeric'] != df['XCI_numeric'].shift(1)).fillna(False)
                else:
                    df['HANDOVER_FLAG'] = False

                # Si el archivo viene de CLF y no trae algunas columnas, dejo el dataframe igualmente válido.
                return df, filepath

            except Exception as e:
                last_error = f"{os.path.basename(filepath)}: {str(e)}"
                print(f"WARNING: no he podido procesar {last_error}")
                continue

        messagebox.showerror(
            "Processing Error",
            f"No he podido procesar la campaña.\nÚltimo error: {last_error}"
        )
        return None, None

    def run_analysis(self):
        """Método principal para cargar datos y dejarlos listos para el análisis."""
        self.data = {}
        all_dataframes = []
        for i, set_params in enumerate(self.sets_params):
            label = f"Set {i+1}: {set_params['tester']} | {set_params['operator']} | {set_params['date']} | {set_params['technology']}"
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            self.color_map[label] = color
            df, filepath = self._load_and_clean_data(set_params, label)
            if df is None or df.empty:
                return False
            self.data[label] = df
            detected_operator = self._detect_operator(df, filepath)
            self.operator_names[label] = normalize_token(set_params.get('operator'), detected_operator)
            self.cell_towers[label] = self._extract_cell_towers(df, label)
            all_dataframes.append(df)
        if all_dataframes:
            self.combined_data = pd.concat(all_dataframes, ignore_index=True)
            self.is_ready = True
            return True
        return False
    def get_available_parameters(self):
        """Devuelve los parámetros que realmente existen en los datos cargados."""
        if not self.is_ready:
            return []
        available = []
        for param in ANIMATION_PARAMS:
            if param in self.combined_data.columns and not self.combined_data[param].isnull().all():
                available.append(param)
        return available

    def get_handover_data(self):
        """Extrae los eventos de handover para representarlos en el mapa."""
        if not self.is_ready or 'HANDOVER_FLAG' not in self.combined_data.columns: 
            return pd.DataFrame()
        handover_events = self.combined_data[self.combined_data['HANDOVER_FLAG'] == True].copy()
        return handover_events.iloc[1:] if not handover_events.empty else pd.DataFrame()

    def get_current_cell(self, label, frame_index):
        """Obtiene la celda actual para un frame concreto de la animación."""
        if label not in self.data or frame_index >= len(self.data[label]):
            return None
            
        current_row = self.data[label].iloc[frame_index]
        cell_id = None
        
        if 'XCI' in self.data[label].columns and not pd.isna(current_row.get('XCI')):
            cell_id = current_row['XCI']
        elif 'PCI' in self.data[label].columns and not pd.isna(current_row.get('PCI')):
            cell_id = current_row['PCI']
        elif 'CELL_ID' in self.data[label].columns and not pd.isna(current_row.get('CELL_ID')):
            cell_id = current_row['CELL_ID']
        else:
            return None
            
        cell_towers = self.cell_towers.get(label, {})
        return cell_towers.get(cell_id, None)

    def get_statistics(self, label, param):
        """Calcula mínimo, máximo y media para un set y un parámetro."""
        if label not in self.data or param not in self.data[label].columns:
            return None, None, None
            
        valid_data = self.data[label][param].dropna()
        
        if len(valid_data) == 0:
            return None, None, None
            
        return valid_data.min(), valid_data.max(), valid_data.mean()

    def calculate_coverage_statistics(self):
        """
        Calcula estadísticas de cobertura y ajusta el redondeo para que
        los porcentajes mostrados sumen siempre 100%.
        """
        if not self.is_ready:
            return {}
            
        coverage_stats = {}
        
        for label, df in self.data.items():
            coverage_stats[label] = {}
            total_samples = len(df)
            
            for param in ['RSRP', 'RSRQ', 'SNR']:
                if param not in df.columns or total_samples == 0:
                    continue
                    
                param_floats = []
                valid_data = df[param].dropna()
                valid_count = len(valid_data)
                
                # --- A. Calculo las categorías de cobertura con porcentajes reales ---
                for category, criteria in COVERAGE_CRITERIA.items():
                    if param in criteria:
                        min_val = criteria[param]['min']
                        max_val = criteria[param]['max']
                        
                        # Importante: los rangos no se pueden solapar
                        if min_val == float('-inf'):
                            mask = valid_data < max_val 
                        elif max_val == float('inf'):
                            mask = valid_data >= min_val 
                        else:
                            mask = (valid_data >= min_val) & (valid_data < max_val)
                        
                        count = mask.sum()
                        percentage = (count / total_samples) * 100
                        param_floats.append((category, percentage))
                
                # --- B. Calculo el porcentaje de datos sin cobertura o inválidos ---
                no_coverage_count = total_samples - valid_count
                no_coverage_percentage = (no_coverage_count / total_samples) * 100
                param_floats.append(('No Coverage', no_coverage_percentage))
                
                
                # --- C. Ajusto el redondeo con el método del resto mayor ---
                param_stats_final = {}
                initial_sum = 0
                remainders = [] # (resto, nombre de categoría)
                
                for category, float_perc in param_floats:
                    # Redondeo hacia abajo el porcentaje inicial
                    rounded_down = int(float_perc) 
                    
                    # Guardo el resto para corregir luego el total
                    remainder = float_perc - rounded_down
                    remainders.append((remainder, category))
                    
                    # Guardo el valor entero inicial
                    param_stats_final[category] = rounded_down
                    initial_sum += rounded_down
                
                # Calculo cuántos puntos faltan para llegar exactamente al 100%
                difference = 100 - initial_sum
                
                # Ordeno los restos de mayor a menor
                remainders.sort(key=lambda x: x[0], reverse=True)
                
                # Reparto la diferencia empezando por las categorías con mayor resto
                # Uso round para evitar errores típicos de coma flotante como 0.999999999 -> 1
                for i in range(int(round(difference))):
                    if i < len(remainders):
                        category_to_adjust = remainders[i][1]
                        param_stats_final[category_to_adjust] += 1
                
                # 4. Comprobación final y guardado
                final_sum = sum(param_stats_final.values())
                if final_sum != 100 and total_samples > 0:
                    # Esto puede pasar con pocas muestras por el redondeo
                    # Ejemplo típico: 100 - 99.999999... puede dar un ajuste raro
                    # Como seguridad final, corrijo sobre la categoría con mayor resto
                    if final_sum != 100 and remainders:
                        param_stats_final[remainders[0][1]] += (100 - final_sum)
                        final_sum = sum(param_stats_final.values())

                    if final_sum != 100:
                        print(f"WARNING: Final adjusted sum for {label}, {param} is {final_sum} instead of 100. Check calculation logic.")
                
                coverage_stats[label][param] = param_stats_final
        
        self.coverage_stats = coverage_stats
        return coverage_stats

    def get_all_jitter_stats(self):
        """Calcula el jitter medio de cada set a partir de las muestras de latencia."""
        if not self.is_ready:
            return {}
        
        results = {}
        for label, df in self.data.items():
            # Compruebo si existe la columna de latencia
            if 'LATENCY' not in df.columns:
                continue
            
            # Cojo solo las muestras válidas de latencia
            valid_latency = df['LATENCY'].dropna()
            
            # Limito el cálculo a las primeras 150 muestras
            if len(valid_latency) > 150:
                valid_latency = valid_latency.iloc[:150]
            # Fin de este ajuste
            
            # Hace falta al menos dos muestras para calcular diferencias
            if len(valid_latency) < 2:
                continue 
            
            # 1. Calculo la diferencia entre muestras consecutivas
            # 2. Uso el valor absoluto de esa diferencia
            # 3. Calculo la media de esas diferencias
            #    mean() ya gestiona bien el número de muestras válidas
            #    diff() genera un NaN inicial y mean() lo ignora
            jitter = valid_latency.diff().abs().mean()
            
            # Guardo el jitter solo si sale un valor válido
            if pd.notna(jitter):
                results[label] = jitter
        
        return results

# --- Interfaz gráfica con Tkinter ---

# --- Funciones auxiliares para la predicción IA de cobertura ---

EARTH_M_PER_DEG_LAT = 111320.0
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
OSM_CACHE_DIR = os.path.join(tempfile.gettempdir(), "drivetest_studio_osm_cache")

CAT_WFS_URLS = [
    "https://ovc.catastro.meh.es/INSPIRE/wfsBU.aspx",
    "http://ovc.catastro.meh.es/INSPIRE/wfsBU.aspx",
    "https://www.catastro.hacienda.gob.es/INSPIRE/wfsBU.aspx",
    "http://www.catastro.hacienda.gob.es/INSPIRE/wfsBU.aspx",
]
CAT_CACHE_DIR = os.path.join(tempfile.gettempdir(), "drivetest_studio_catastro_cache")


def _catastro_cache_path(south, west, north, east):
    os.makedirs(CAT_CACHE_DIR, exist_ok=True)
    key = f"{south:.6f}_{west:.6f}_{north:.6f}_{east:.6f}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(CAT_CACHE_DIR, f"{digest}.json")


def _localname(tag):
    return str(tag).split('}')[-1] if tag else ''


def _normalize_coord_pair(a, b, origin_lat, origin_lon):
    score_latlon = abs(a - origin_lat) + abs(b - origin_lon)
    score_lonlat = abs(b - origin_lat) + abs(a - origin_lon)
    if score_latlon <= score_lonlat:
        return a, b
    return b, a


def _parse_poslist_bbox(pos_text, origin_lat, origin_lon):
    coords = []
    for part in str(pos_text).replace('\n', ' ').split():
        try:
            coords.append(float(part))
        except Exception:
            pass
    if len(coords) < 4:
        return None
    pts = []
    for i in range(0, len(coords) - 1, 2):
        a = coords[i]
        b = coords[i + 1]
        lat, lon = _normalize_coord_pair(a, b, origin_lat, origin_lon)
        pts.append((lat, lon))
    if len(pts) < 3:
        return None
    lats = [lat for lat, _ in pts]
    lons = [lon for _, lon in pts]
    return (min(lats), min(lons), max(lats), max(lons))


def _extract_numeric_descendant(feature_elem, names):
    wanted = {n.lower() for n in names}
    for child in feature_elem.iter():
        lname = _localname(child.tag).lower()
        if lname in wanted:
            try:
                return float(str(child.text).replace(',', '.').replace('m', '').strip())
            except Exception:
                continue
    return None


def _dedupe_buildings(buildings):
    seen = set()
    out = []
    for b in buildings:
        rect = b.get('rect')
        key = tuple(round(v, 1) for v in rect) if rect else None
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(b)
    return out


def _parse_catastro_gml_buildings(xml_text, origin_lat, origin_lon):
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    buildings = []
    feature_tags = {'BUILDING', 'BUILDINGPART', 'OTHERCONSTRUCTION'}
    for elem in root.iter():
        lname = _localname(elem.tag).upper()
        if lname not in feature_tags:
            continue
        bboxes = []
        for child in elem.iter():
            if _localname(child.tag).lower() == 'poslist' and child.text:
                bbox = _parse_poslist_bbox(child.text, origin_lat, origin_lon)
                if bbox:
                    bboxes.append(bbox)
        if not bboxes:
            continue
        min_lat = min(b[0] for b in bboxes)
        min_lon = min(b[1] for b in bboxes)
        max_lat = max(b[2] for b in bboxes)
        max_lon = max(b[3] for b in bboxes)
        x1, y1 = _latlon_to_local_xy(min_lat, min_lon, origin_lat, origin_lon)
        x2, y2 = _latlon_to_local_xy(max_lat, max_lon, origin_lat, origin_lon)
        rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        height_val = _extract_numeric_descendant(elem, ['measuredHeight', 'heightAboveGround', 'height']) 
        floors = _extract_numeric_descendant(elem, ['numberOfFloorsAboveGround', 'floorsAboveGround'])
        if height_val is None and floors is not None:
            height_val = max(3.0, floors * 3.0)
        if height_val is None:
            height_val = 12.0 if lname == 'BUILDINGPART' else 9.0
        buildings.append({
            'rect': rect,
            'centroid_x': float((rect[0] + rect[2]) / 2.0),
            'centroid_y': float((rect[1] + rect[3]) / 2.0),
            'area': _bbox_area(rect),
            'height': float(height_val),
            'latlon_bbox': (min_lat, min_lon, max_lat, max_lon),
        })
    return _dedupe_buildings(buildings)


def _latlon_bbox_to_utm30_bbox(south, west, north, east):
    if not PYPROJ_AVAILABLE:
        return None
    try:
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:25830', always_xy=True)
        xs = []
        ys = []
        for lon, lat in [(west, south), (west, north), (east, south), (east, north)]:
            x, y = transformer.transform(lon, lat)
            xs.append(x)
            ys.append(y)
        return (min(xs), min(ys), max(xs), max(ys))
    except Exception:
        return None


def _fetch_catastro_wfs_buildings(min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon):
    if not REQUESTS_AVAILABLE:
        return [], 'requests no disponible'
    pad_lat = max((max_lat - min_lat) * 0.05, 0.0004)
    pad_lon = max((max_lon - min_lon) * 0.05, 0.0004)
    south = min_lat - pad_lat
    north = max_lat + pad_lat
    west = min_lon - pad_lon
    east = max_lon + pad_lon

    cache_path = _catastro_cache_path(south, west, north, east)
    cached_buildings, cached_note = _load_buildings_from_cache(cache_path)

    session = requests.Session()
    session.headers.update({
        'User-Agent': 'DriveTestStudio/1.0 (Catastro WFS BU; educational use)'
    })
    errors = []

    bbox_candidates = []
    utm30 = _latlon_bbox_to_utm30_bbox(south, west, north, east)
    if utm30 is not None:
        x1, y1, x2, y2 = utm30
        bbox_candidates.append((f'{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}', 'EPSG::25830', 'utm30'))
    # En WFS 2.0 el orden de ejes en EPSG:4326 puede dar guerra, así que pruebo ambos órdenes
    bbox_candidates.append((f'{south:.8f},{west:.8f},{north:.8f},{east:.8f}', 'EPSG:4326', 'latlon'))
    bbox_candidates.append((f'{west:.8f},{south:.8f},{east:.8f},{north:.8f}', 'EPSG:4326', 'lonlat'))

    for endpoint in CAT_WFS_URLS:
        for typename in ['BU.BUILDINGPART', 'BU:BUILDINGPART', 'BUILDINGPART', 'BU.BUILDING', 'BU:BUILDING', 'BUILDING']:
            for bbox_value, srsname, bbox_mode in bbox_candidates:
                params = {
                    'service': 'wfs',
                    'version': '2.0.0',
                    'request': 'GetFeature',
                    'typenames': typename,
                    'bbox': bbox_value,
                    'srsname': srsname,
                }
                try:
                    response = session.get(endpoint, params=params, timeout=40)
                    response.raise_for_status()
                    content = response.content
                    text_lower = content[:5000].decode('utf-8', errors='ignore').lower()
                    if '<html' in text_lower and ('página no encontrada' in text_lower or 'page not found' in text_lower):
                        raise RuntimeError('respuesta HTML no válida del Catastro')
                    buildings = _parse_catastro_gml_buildings(content, origin_lat, origin_lon)
                    if buildings:
                        _save_buildings_to_cache(cache_path, buildings, f'Catastro WFS {typename} {endpoint} {srsname} {bbox_mode}')
                        return buildings, None
                    errors.append(f'{typename} {srsname} {bbox_mode}: sin resultados')
                except Exception as exc:
                    errors.append(f'{typename} {srsname} {bbox_mode}: {exc}')
    if cached_buildings:
        return cached_buildings, f'Catastro WFS no disponible, usando caché local ({cached_note})'
    short_error = ' | '.join(errors[:4]) if errors else 'sin respuesta Catastro WFS'
    return [], f'sin edificios Catastro WFS ({short_error})'


def _fetch_buildings(min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon):
    buildings, note = _fetch_catastro_wfs_buildings(min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon)
    if buildings:
        return buildings, note, 'Catastro WFS BU'
    buildings, note = _fetch_osm_buildings(min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon)
    if buildings:
        return buildings, note, 'OSM'
    return [], note or 'sin edificios', 'Ninguno'



def _meters_per_deg_lon(lat_deg):
    return max(1.0, EARTH_M_PER_DEG_LAT * math.cos(math.radians(lat_deg)))


def _latlon_to_local_xy(lat, lon, origin_lat, origin_lon):
    mx = (lon - origin_lon) * _meters_per_deg_lon(origin_lat)
    my = (lat - origin_lat) * EARTH_M_PER_DEG_LAT
    return mx, my


def _local_xy_to_latlon(x, y, origin_lat, origin_lon):
    lat = origin_lat + (y / EARTH_M_PER_DEG_LAT)
    lon = origin_lon + (x / _meters_per_deg_lon(origin_lat))
    return lat, lon


def _segment_intersects_rect(x1, y1, x2, y2, rect):
    xmin, ymin, xmax, ymax = rect
    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - xmin, xmax - x1, y1 - ymin, ymax - y1]
    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return True


def _point_to_rect_distance(x, y, rect):
    xmin, ymin, xmax, ymax = rect
    dx = max(xmin - x, 0, x - xmax)
    dy = max(ymin - y, 0, y - ymax)
    return math.hypot(dx, dy)


def _bbox_area(rect):
    xmin, ymin, xmax, ymax = rect
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)



def _osm_cache_path(south, west, north, east):
    os.makedirs(OSM_CACHE_DIR, exist_ok=True)
    key = f"{south:.6f}_{west:.6f}_{north:.6f}_{east:.6f}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(OSM_CACHE_DIR, f"{digest}.json")


def _load_buildings_from_cache(cache_path):
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload.get("buildings", []), payload.get("note", "cache local")
    except Exception:
        pass
    return None, None


def _save_buildings_to_cache(cache_path, buildings, note="descarga OSM correcta"):
    try:
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({"buildings": buildings, "note": note}, fh)
    except Exception:
        pass


def _build_overpass_queries(south, west, north, east):
    bbox = f"({south},{west},{north},{east})"
    queries = []
    # Primero pido solo ways, que suele responder antes y normalmente basta en zonas urbanas.
    queries.append(
        '[out:json][timeout:20];'
        f'(way["building"]{bbox};);'
        'out geom;'
    )
    # Si con eso no llega, hago un segundo intento con ways y relations.
    queries.append(
        '[out:json][timeout:30];'
        f'(way["building"]{bbox};relation["building"]{bbox};);'
        'out geom;'
    )
    return queries


def _parse_overpass_buildings(payload, origin_lat, origin_lon):
    buildings = []
    for element in payload.get('elements', []):
        geometry = element.get('geometry') or []
        if len(geometry) < 3:
            continue
        xs, ys = [], []
        valid_points = []
        for point in geometry:
            lat = point.get('lat')
            lon = point.get('lon')
            if lat is None or lon is None:
                continue
            x, y = _latlon_to_local_xy(lat, lon, origin_lat, origin_lon)
            xs.append(x)
            ys.append(y)
            valid_points.append((lat, lon))
        if len(xs) < 3:
            continue
        rect = (min(xs), min(ys), max(xs), max(ys))
        tags = element.get('tags', {}) or {}
        height_raw = tags.get('height') or tags.get('building:levels')
        try:
            height_val = float(str(height_raw).replace('m', '').strip())
            if 'building:levels' in tags and 'height' not in tags:
                height_val *= 3.0
        except Exception:
            height_val = 9.0
        buildings.append({
            'rect': rect,
            'centroid_x': float(sum(xs) / len(xs)),
            'centroid_y': float(sum(ys) / len(ys)),
            'area': _bbox_area(rect),
            'height': height_val,
            'latlon_bbox': (
                min(lat for lat, _ in valid_points),
                min(lon for _, lon in valid_points),
                max(lat for lat, _ in valid_points),
                max(lon for _, lon in valid_points),
            ),
        })
    return buildings


def _fetch_osm_buildings(min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon):
    if not REQUESTS_AVAILABLE:
        return [], 'requests no disponible'
    pad_lat = max((max_lat - min_lat) * 0.08, 0.0008)
    pad_lon = max((max_lon - min_lon) * 0.08, 0.0008)
    south = min_lat - pad_lat
    north = max_lat + pad_lat
    west = min_lon - pad_lon
    east = max_lon + pad_lon

    cache_path = _osm_cache_path(south, west, north, east)
    cached_buildings, cached_note = _load_buildings_from_cache(cache_path)

    session = requests.Session()
    session.headers.update({
        "User-Agent": "DriveTestStudio/1.0 (coverage prediction; educational use)"
    })

    errors = []
    for endpoint in OVERPASS_URLS:
        for query in _build_overpass_queries(south, west, north, east):
            try:
                response = session.post(endpoint, data={'data': query}, timeout=25)
                response.raise_for_status()
                payload = response.json()
                buildings = _parse_overpass_buildings(payload, origin_lat, origin_lon)
                # Si devuelve algo útil, lo guardo en caché y sigo.
                if buildings:
                    _save_buildings_to_cache(cache_path, buildings, f"OSM live: {endpoint}")
                    return buildings, None
                errors.append(f"{endpoint}: respuesta vacía")
            except Exception as exc:
                errors.append(f"{endpoint}: {exc}")

    # Si falla la descarga, tiro de caché si existe.
    if cached_buildings:
        return cached_buildings, f"OSM live no disponible, usando caché local ({cached_note})"

    # Último recurso: seguir sin edificios para no bloquear el análisis.
    short_error = " | ".join(errors[:3]) if errors else "sin respuesta OSM"
    return [], f"sin edificios OSM ({short_error})"


def _flatten_towers_from_processor(processor, tech_filter=None):
    towers = []
    combined = getattr(processor, 'combined_data', pd.DataFrame()).copy()

    # Estrategia principal: estimar torres a partir de las muestras reales,
    # usando las muestras con mejor señal de cada celda para aproximarme mejor a la estación base.
    if not combined.empty:
        lat_col = 'LAT' if 'LAT' in combined.columns else ('lat' if 'lat' in combined.columns else None)
        lon_col = 'LON' if 'LON' in combined.columns else ('lon' if 'lon' in combined.columns else None)

        if lat_col and lon_col:
            if tech_filter and 'SCENARIO_TECH' in combined.columns:
                combined = combined[combined['SCENARIO_TECH'].astype(str).str.upper().str.strip() == tech_filter.upper()].copy()
            elif tech_filter and 'TECHNOLOGY' in combined.columns:
                combined = combined[combined['TECHNOLOGY'].astype(str).str.upper().str.strip() == tech_filter.upper()].copy()

            id_col = None
            for candidate in ['XCI', 'CELL_ID', 'PCI']:
                if candidate in combined.columns and combined[candidate].notna().sum() >= 3:
                    id_col = candidate
                    break

            signal_col = None
            for candidate in ['RSRP', 'RSRP/RSCP', 'signal_primary_dbm', 'NR_SS_RSRP']:
                if candidate in combined.columns and combined[candidate].notna().any():
                    signal_col = candidate
                    break

            if id_col is not None:
                work = combined.dropna(subset=[lat_col, lon_col, id_col]).copy()
                if not work.empty:
                    for cell_id, chunk in work.groupby(id_col):
                        top = chunk.copy()
                        if signal_col is not None:
                            sig = pd.to_numeric(top[signal_col], errors='coerce')
                            top = top.loc[sig.notna()].copy()
                            if not top.empty:
                                q = sig.loc[top.index].quantile(0.75)
                                top_q = top.loc[sig.loc[top.index] >= q].copy()
                                if len(top_q) >= 3:
                                    top = top_q
                                else:
                                    top = top.nlargest(min(len(top), 6), signal_col)

                        if top.empty:
                            continue

                        lat = float(pd.to_numeric(top[lat_col], errors='coerce').median())
                        lon = float(pd.to_numeric(top[lon_col], errors='coerce').median())
                        if pd.isna(lat) or pd.isna(lon):
                            continue

                        towers.append({
                            'id': str(cell_id),
                            'lat': lat,
                            'lon': lon,
                            'label': 'derived_strong_samples',
                            'samples': int(len(chunk)),
                        })

                if towers:
                    # Quito duplicados cercanos con el mismo identificador.
                    dedup = []
                    seen = set()
                    for tower in towers:
                        uniq = (str(tower['id']), round(float(tower['lat']), 6), round(float(tower['lon']), 6))
                        if uniq in seen:
                            continue
                        seen.add(uniq)
                        dedup.append(tower)
                    return dedup

    # Si no puedo estimarlas aquí, uso las torres que ya trae el procesador.
    seen = set()
    if getattr(processor, 'cell_towers', None):
        for label, tower_dict in processor.cell_towers.items():
            label_df = processor.data.get(label)
            if label_df is not None and tech_filter:
                if 'SCENARIO_TECH' in label_df.columns:
                    techs = set(label_df['SCENARIO_TECH'].astype(str).str.upper().str.strip().dropna())
                    if tech_filter.upper() not in techs:
                        continue
                elif 'TECHNOLOGY' in label_df.columns:
                    techs = set(label_df['TECHNOLOGY'].astype(str).str.upper().str.strip().dropna())
                    if tech_filter.upper() not in techs:
                        continue
            for key, tower in tower_dict.items():
                lat = tower.get('lat')
                lon = tower.get('lon')
                if pd.isna(lat) or pd.isna(lon):
                    continue
                uniq = (str(key), round(float(lat), 6), round(float(lon), 6))
                if uniq in seen:
                    continue
                seen.add(uniq)
                towers.append({'id': str(key), 'lat': float(lat), 'lon': float(lon), 'label': label})
    return towers
def _build_prediction_feature_frame(df, towers, buildings, target_col):
    feature_cols = [
        'x_m', 'y_m', 'nearest_tower_dist_m', 'serving_tower_dist_m', 'serving_dx_m', 'serving_dy_m',
        'tower_count_250m', 'tower_count_500m', 'nearest_building_dist_m', 'building_count_40m',
        'building_count_80m', 'building_area_40m', 'building_area_80m', 'los_block_count',
        'los_block_area', 'los_block_height_sum'
    ]
    if df.empty:
        return pd.DataFrame(columns=feature_cols), None, None

    origin_lat = float(df['LAT'].mean())
    origin_lon = float(df['LON'].mean())
    tower_xy = []
    for tower in towers:
        tx, ty = _latlon_to_local_xy(tower['lat'], tower['lon'], origin_lat, origin_lon)
        tower_xy.append({**tower, 'x_m': tx, 'y_m': ty})

    rows = []
    for _, row in df.iterrows():
        lat = float(row['LAT'])
        lon = float(row['LON'])
        x, y = _latlon_to_local_xy(lat, lon, origin_lat, origin_lon)

        dists = []
        serving_tower = None
        serving_id = str(row['XCI']) if 'XCI' in row.index and pd.notna(row['XCI']) else None
        for tower in tower_xy:
            dist = math.hypot(x - tower['x_m'], y - tower['y_m'])
            dists.append((dist, tower))
            if serving_id is not None and tower['id'] == serving_id:
                serving_tower = tower
        dists.sort(key=lambda item: item[0])
        nearest_tower_dist = dists[0][0] if dists else np.nan
        if serving_tower is None and dists:
            serving_tower = dists[0][1]
        if serving_tower is not None:
            serving_dist = math.hypot(x - serving_tower['x_m'], y - serving_tower['y_m'])
            serving_dx = x - serving_tower['x_m']
            serving_dy = y - serving_tower['y_m']
        else:
            serving_dist = serving_dx = serving_dy = np.nan

        tower_count_250 = sum(1 for dist, _ in dists if dist <= 250.0)
        tower_count_500 = sum(1 for dist, _ in dists if dist <= 500.0)

        b40 = b80 = 0
        a40 = a80 = 0.0
        nearest_bld = np.nan
        los_blocks = 0
        los_area = 0.0
        los_hsum = 0.0
        for building in buildings:
            cx = building['centroid_x']
            cy = building['centroid_y']
            center_dist = math.hypot(x - cx, y - cy)
            if center_dist <= 40.0:
                b40 += 1
                a40 += building['area']
            if center_dist <= 80.0:
                b80 += 1
                a80 += building['area']
            rect_dist = _point_to_rect_distance(x, y, building['rect'])
            if pd.isna(nearest_bld) or rect_dist < nearest_bld:
                nearest_bld = rect_dist
            if serving_tower is not None and _segment_intersects_rect(x, y, serving_tower['x_m'], serving_tower['y_m'], building['rect']):
                los_blocks += 1
                los_area += building['area']
                los_hsum += building['height']

        rows.append({
            'LAT': lat, 'LON': lon, target_col: row[target_col],
            'x_m': x, 'y_m': y,
            'nearest_tower_dist_m': nearest_tower_dist,
            'serving_tower_dist_m': serving_dist,
            'serving_dx_m': serving_dx,
            'serving_dy_m': serving_dy,
            'tower_count_250m': tower_count_250,
            'tower_count_500m': tower_count_500,
            'nearest_building_dist_m': nearest_bld,
            'building_count_40m': b40,
            'building_count_80m': b80,
            'building_area_40m': a40,
            'building_area_80m': a80,
            'los_block_count': los_blocks,
            'los_block_area': los_area,
            'los_block_height_sum': los_hsum,
        })

    feature_df = pd.DataFrame(rows)
    return feature_df, (origin_lat, origin_lon), tower_xy


def _augment_prediction_features(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    for src, dst in [
        ('nearest_tower_dist_m', 'nearest_tower_log_m'),
        ('serving_tower_dist_m', 'serving_tower_log_m'),
        ('nearest_building_dist_m', 'nearest_building_log_m'),
    ]:
        if src in out.columns:
            vals = pd.to_numeric(out[src], errors='coerce').fillna(1e6).clip(lower=0.0)
            out[dst] = np.log1p(vals)

    if 'serving_dx_m' in out.columns and 'serving_dy_m' in out.columns:
        dx = pd.to_numeric(out['serving_dx_m'], errors='coerce').fillna(0.0)
        dy = pd.to_numeric(out['serving_dy_m'], errors='coerce').fillna(0.0)
        norm = np.hypot(dx, dy)
        norm = np.where(norm < 1e-6, 1.0, norm)
        out['serving_dir_cos'] = dx / norm
        out['serving_dir_sin'] = dy / norm
    else:
        out['serving_dir_cos'] = 0.0
        out['serving_dir_sin'] = 0.0

    if 'building_area_40m' in out.columns and 'building_area_80m' in out.columns:
        out['building_area_ratio_40_80'] = pd.to_numeric(out['building_area_40m'], errors='coerce').fillna(0.0) / (pd.to_numeric(out['building_area_80m'], errors='coerce').fillna(0.0) + 1.0)
    if 'los_block_count' in out.columns and 'serving_tower_dist_m' in out.columns:
        out['los_per_100m'] = 100.0 * pd.to_numeric(out['los_block_count'], errors='coerce').fillna(0.0) / (pd.to_numeric(out['serving_tower_dist_m'], errors='coerce').fillna(0.0) + 100.0)
    return out


def _predict_with_bundle(bundle, X_df):
    model = bundle['model']
    pred = model.predict(X_df[bundle['feature_cols']])
    if bundle.get('residual_knn') is not None and {'x_m', 'y_m'}.issubset(X_df.columns):
        coords = X_df[['x_m', 'y_m']].copy()
        pred = pred + bundle['residual_knn'].predict(coords)
    lo = bundle.get('pred_clip_lo')
    hi = bundle.get('pred_clip_hi')
    if lo is not None or hi is not None:
        pred = np.clip(pred, lo if lo is not None else np.nanmin(pred), hi if hi is not None else np.nanmax(pred))
    return pred




def _smooth_prediction_matrix(matrix, passes=1):
    """
    Suavizado visual nan-aware 3x3 para que el mapa IA no se vea excesivamente cuadriculado.
    No cambia los valores del modelo ni las métricas; solo afecta a la matriz que se pinta.
    """
    try:
        arr = np.array(matrix, dtype=float)
        passes = max(0, int(passes))
        if passes == 0 or arr.size == 0:
            return arr
        for _ in range(passes):
            out = arr.copy()
            nrows, ncols = arr.shape
            for r in range(nrows):
                for c in range(ncols):
                    if np.isnan(arr[r, c]):
                        continue
                    r0, r1 = max(0, r - 1), min(nrows, r + 2)
                    c0, c1 = max(0, c - 1), min(ncols, c + 2)
                    window = arr[r0:r1, c0:c1]
                    vals = window[~np.isnan(window)]
                    if len(vals) >= 3:
                        out[r, c] = float(np.mean(vals))
            arr = out
        return arr
    except Exception:
        return matrix


def _centers_to_edges(values):
    """Devuelve bordes a partir de centros para que pcolormesh pinte celdas reales."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if arr.size == 1:
        step = 1.0
        return np.array([arr[0] - step / 2.0, arr[0] + step / 2.0])
    mids = (arr[:-1] + arr[1:]) / 2.0
    first = arr[0] - (mids[0] - arr[0])
    last = arr[-1] + (arr[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])

def _build_prediction_grid(feature_df, origin, towers_xy, buildings, resolution_m):
    origin_lat, origin_lon = origin
    min_x = float(feature_df['x_m'].min())
    max_x = float(feature_df['x_m'].max())
    min_y = float(feature_df['y_m'].min())
    max_y = float(feature_df['y_m'].max())
    pad_x = max(20.0, (max_x - min_x) * 0.04)
    pad_y = max(20.0, (max_y - min_y) * 0.04)
    xs = np.arange(min_x - pad_x, max_x + pad_x + resolution_m, resolution_m)
    ys = np.arange(min_y - pad_y, max_y + pad_y + resolution_m, resolution_m)

    route_points = feature_df[['x_m', 'y_m']].to_numpy(dtype=float)

    # El grid tiene que controlar solo el tamaño de celda del mapa.
    # Antes el buffer alrededor de la ruta dependía de la resolución,
    # por eso 20/30/50/75 m podían acabar pareciéndose demasiado.
    # Al subir el tamaño de celda también crecía la zona visible.
    # Ahora dejo fija la zona de predicción y el grid sí cambia la granularidad.
    # Así los bloques se ven realmente más grandes o más pequeños.
    route_buffer_m = 65.0

    records = []
    for yy in ys:
        for xx in xs:
            if len(route_points):
                route_dists = np.hypot(route_points[:, 0] - xx, route_points[:, 1] - yy)
                nearest_route_dist = float(np.min(route_dists))
            else:
                nearest_route_dist = np.nan
            inside_route_buffer = pd.notna(nearest_route_dist) and nearest_route_dist <= route_buffer_m

            dists = []
            serving_tower = None
            for tower in towers_xy:
                dist = math.hypot(xx - tower['x_m'], yy - tower['y_m'])
                dists.append((dist, tower))
            dists.sort(key=lambda item: item[0])
            nearest_tower_dist = dists[0][0] if dists else np.nan
            if dists:
                serving_tower = dists[0][1]
                serving_dist = dists[0][0]
                serving_dx = xx - serving_tower['x_m']
                serving_dy = yy - serving_tower['y_m']
            else:
                serving_dist = serving_dx = serving_dy = np.nan

            tower_count_250 = sum(1 for dist, _ in dists if dist <= 250.0)
            tower_count_500 = sum(1 for dist, _ in dists if dist <= 500.0)

            b40 = b80 = 0
            a40 = a80 = 0.0
            nearest_bld = np.nan
            los_blocks = 0
            los_area = 0.0
            los_hsum = 0.0
            for building in buildings:
                cx = building['centroid_x']
                cy = building['centroid_y']
                center_dist = math.hypot(xx - cx, yy - cy)
                if center_dist <= 40.0:
                    b40 += 1
                    a40 += building['area']
                if center_dist <= 80.0:
                    b80 += 1
                    a80 += building['area']
                rect_dist = _point_to_rect_distance(xx, yy, building['rect'])
                if pd.isna(nearest_bld) or rect_dist < nearest_bld:
                    nearest_bld = rect_dist
                if serving_tower is not None and _segment_intersects_rect(xx, yy, serving_tower['x_m'], serving_tower['y_m'], building['rect']):
                    los_blocks += 1
                    los_area += building['area']
                    los_hsum += building['height']
            lat, lon = _local_xy_to_latlon(xx, yy, origin_lat, origin_lon)
            records.append({
                'LAT': lat, 'LON': lon, 'x_m': xx, 'y_m': yy,
                'nearest_route_dist_m': nearest_route_dist,
                'inside_route_buffer': inside_route_buffer,
                'nearest_tower_dist_m': nearest_tower_dist,
                'serving_tower_dist_m': serving_dist,
                'serving_dx_m': serving_dx,
                'serving_dy_m': serving_dy,
                'tower_count_250m': tower_count_250,
                'tower_count_500m': tower_count_500,
                'nearest_building_dist_m': nearest_bld,
                'building_count_40m': b40,
                'building_count_80m': b80,
                'building_area_40m': a40,
                'building_area_80m': a80,
                'los_block_count': los_blocks,
                'los_block_area': los_area,
                'los_block_height_sum': los_hsum,
            })
    return pd.DataFrame(records), xs, ys



class DriveTestGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DriveTest Studio")
        self.geometry("1560x940")
        self.minsize(1380, 820)
        self.configure(bg="#12161c")
        self.folder_path = ""
        self.available_tests = set()
        self.analysis_processor = None
        
        self.num_sets = 0
        self.set_frames = []

        self.tab_control = None
        self.analysis_tab = None
        self.statistics_tab = None

        self.main_paned_window = None
        self.top_frame = None
        self.bottom_frame = None
        self.top_left_frame = None
        self.top_right_frame = None
        
        self.ho_map_fig = None
        self.ho_map_ax = None
        self.ho_map_canvas = None
        
        self.heatmap_fig = None
        self.heatmap_ax = None
        self.heatmap_canvas = None
        
        self.line_plot_fig = None
        self.line_plot_ax = None
        self.line_plot_canvas = None
        
        self.stats_fig = None
        self.stats_ax = None
        self.stats_canvas = None
        
        self.set_dropdown = None
        self.param_dropdown = None
        self.animate_button = None
        self.pause_button = None
        self.stop_button = None
        
        self.animation = None
        self.animation_paused = False
        self.animation_max_frames = 0
        self.animation_data_cache = {}
        
        self.red_line = None
        self.ho_map_markers = {}
        self.heatmap_marker = None
        self.cell_connection_line = None
        self.cell_symbols = {}
        self.cell_labels = {}

        self.stat_lines = {}

        self.ho_original_limits = None
        self.heatmap_original_limits = None
        
        self.heatmap_zoom_button = None
        self.heatmap_zoom_mode_active = False
        self.heatmap_click_cid = None 
        self.ho_map_click_cid = None

        self.stats_type_dropdown = None
        self.generate_stats_button = None
        self.stats_main_frame = None
        self.raw_param_list = []

        self.export_ho_map_button = None
        self.export_heatmap_button = None
        self.export_line_plot_button = None
        self.capture_button = None
        self.is_capturing = False
        self.capture_buffers = {'ho_map': [], 'heatmap': [], 'line_plot': []}
        self.captured_gifs = {'ho_map': [], 'heatmap': [], 'line_plot': []}

        self.current_campaign_var = tk.StringVar(value="Sin análisis cargado")
        self.metric_vars = {}
        self.detected_listbox = None


        self.ai_tab = None
        self.ai_plot_frame = None
        self.ai_info_text = None
        self.ai_canvas = None
        self.ai_fig = None
        self.ai_model_bundle = None
        self.ai_prediction_df = None
        self.ai_export_button = None
        self.ai_train_button = None
        self.ai_predict_button = None
        self.ai_tech_var = tk.StringVar(value='4G')
        self.ai_target_var = tk.StringVar(value='RSRP')
        self.ai_resolution_var = tk.StringVar(value='35')
        self.ai_status_var = tk.StringVar(value='IA lista para entrenar.')
        self.ai_status_label = None
        self.ai_progress = None

        self._configure_theme()
        self._create_main_layout()

    def _configure_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use('clam')
        except Exception:
            pass

        bg = '#12161c'
        panel = '#1c2230'
        panel2 = '#20293a'
        fg = '#edf2f7'
        accent = '#2f81f7'
        muted = '#95a1b2'
        border = '#2f3a4f'

        style.configure('TFrame', background=bg)
        style.configure('Panel.TFrame', background=panel)
        style.configure('Inner.TFrame', background=panel2)
        style.configure('TLabel', background=bg, foreground=fg, font=('Segoe UI', 9))
        style.configure('Panel.TLabel', background=panel, foreground=fg, font=('Segoe UI', 9))
        style.configure('Inner.TLabel', background=panel2, foreground=fg, font=('Segoe UI', 9))
        style.configure('Muted.TLabel', background=panel, foreground=muted, font=('Segoe UI', 9))
        style.configure('Success.TLabel', background=panel, foreground='#7CFC8A', font=('Segoe UI', 9, 'bold'))
        style.configure('Warning.TLabel', background=panel, foreground='#ffb347', font=('Segoe UI', 9, 'bold'))
        style.configure('Heading.TLabel', background=panel, foreground=fg, font=('Segoe UI', 12, 'bold'))
        style.configure('SubHeading.TLabel', background=panel, foreground=fg, font=('Segoe UI', 11, 'bold'))
        style.configure('TButton', font=('Segoe UI', 9), padding=7)
        style.configure('Accent.TButton', background=accent, foreground='white')
        style.map('Accent.TButton', background=[('active', '#3b8cff'), ('!disabled', accent)], foreground=[('!disabled', 'white')])
        style.configure('TLabelframe', background=panel, bordercolor=border, relief='solid')
        style.configure('TLabelframe.Label', background=panel, foreground=fg, font=('Segoe UI', 9, 'bold'))
        style.configure('TNotebook', background=bg, borderwidth=0)
        style.configure('TNotebook.Tab', padding=(14, 8))
        style.map('TNotebook.Tab', background=[('selected', panel2), ('!selected', panel)], foreground=[('selected', fg), ('!selected', muted)])
        style.configure('Treeview', background=panel2, fieldbackground=panel2, foreground=fg, rowheight=26)
        style.configure('Treeview.Heading', background=panel, foreground=fg, font=('Segoe UI', 9, 'bold'))
        style.configure('TCombobox', padding=4)
        style.configure('TSpinbox', padding=4)

    def _create_main_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        outer = ttk.Frame(self)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)

        left_wrapper = ttk.Frame(outer, style='Panel.TFrame', padding=14)
        left_wrapper.grid(row=0, column=0, sticky='ns', padx=(10, 6), pady=10)
        left_wrapper.rowconfigure(2, weight=1)
        left_wrapper.columnconfigure(0, weight=1)

        ttk.Label(left_wrapper, text='DriveTest Studio', style='Heading.TLabel').grid(row=0, column=0, sticky='w')
        ttk.Label(
            left_wrapper,
            text='Programa unificado para ingestión, fusión y análisis de Drive Test.',
            style='Muted.TLabel',
            wraplength=320,
        ).grid(row=1, column=0, sticky='w', pady=(6, 14))

        self.left_panel = ttk.Frame(left_wrapper, style='Panel.TFrame')
        self.left_panel.grid(row=2, column=0, sticky='nsew')
        self.left_panel.rowconfigure(0, weight=1)
        self.left_panel.columnconfigure(0, weight=1)

        right = ttk.Frame(outer, style='Panel.TFrame', padding=10)
        right.grid(row=0, column=1, sticky='nsew', padx=(6, 10), pady=10)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        header = tk.Frame(right, bg='#1c2230')
        header.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text='Vista de resultados',
            bg='#1c2230',
            fg='#edf2f7',
            font=('Segoe UI', 16, 'bold'),
            anchor='w'
        ).grid(row=0, column=0, sticky='w')

        self.metric_vars = {key: tk.StringVar(value='-') for key in ['samples', 'speedtests', 'handovers', 'distance', 'signal']}

        self.tab_control = ttk.Notebook(right)
        self.tab_control.grid(row=1, column=0, sticky='nsew')

        self.analysis_tab = ttk.Frame(self.tab_control, style='Panel.TFrame')
        self.statistics_tab = ttk.Frame(self.tab_control, style='Panel.TFrame')
        self.ai_tab = ttk.Frame(self.tab_control, style='Panel.TFrame')
        self.tab_control.add(self.analysis_tab, text='Análisis')
        self.tab_control.add(self.statistics_tab, text='Estadísticas')
        self.tab_control.add(self.ai_tab, text='Predicción IA')
        self.analysis_tab.grid_rowconfigure(0, weight=1)
        self.analysis_tab.grid_columnconfigure(0, weight=1)
        self.statistics_tab.grid_rowconfigure(0, weight=1)
        self.statistics_tab.grid_columnconfigure(0, weight=1)
        self.ai_tab.grid_rowconfigure(0, weight=1)
        self.ai_tab.grid_columnconfigure(0, weight=1)

        self._create_control_panel()
        self._create_enhanced_dashboard()
        self._create_statistics_tab()
        self._create_ai_tab()

    def _create_statistics_tab(self):
        """Crea la pestaña de estadísticas con el desplegable de tipos."""
        # Limpio widgets anteriores por si se vuelve a construir la pestaña
        for widget in self.statistics_tab.winfo_children():
            widget.destroy()
            
        # Contenedor principal de la pestaña de estadísticas
        main_container = ttk.Frame(self.statistics_tab)
        main_container.grid(row=0, column=0, sticky="nsew")
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Zona de controles: desplegable y botón
        control_frame = ttk.LabelFrame(main_container, text="Opciones de estadísticas", padding="10")
        control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        control_frame.grid_columnconfigure(1, weight=1)
        
        # Desplegable del tipo de estadística
        ttk.Label(control_frame, text="Tipo de estadística:").grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w")
        
        self.stats_type_dropdown = ttk.Combobox(control_frame, state="readonly", width=20)
        # Añado también la opción de jitter
        self.stats_type_dropdown['values'] = ['Signal Quality', 'Performance', 'Jitter']
        self.stats_type_dropdown.current(0)  # Dejo seleccionada la primera opción por defecto
        self.stats_type_dropdown.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="ew")
        
        # Botón para generar las estadísticas
        self.generate_stats_button = ttk.Button(control_frame, text="Generar estadísticas", 
                                              command=self._generate_statistics, state=tk.DISABLED)
        self.generate_stats_button.grid(row=0, column=2, padx=5, pady=5)
        
        # Zona donde se dibujan las estadísticas
        self.stats_main_frame = ttk.Frame(main_container)
        self.stats_main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.stats_main_frame.grid_rowconfigure(0, weight=1)
        self.stats_main_frame.grid_columnconfigure(0, weight=1)
        
        # Texto inicial antes de generar nada
        self._add_placeholder_stats(self.stats_main_frame, "Select statistics type and click 'Generar estadísticas'")

    def _add_placeholder_stats(self, parent, text):
        """Añade un texto provisional en la zona de estadísticas."""
        for widget in parent.winfo_children():
            widget.destroy()
            
        ph = ttk.Label(parent, text=text, font=('Arial', 12), foreground="gray")
        ph.grid(row=0, column=0, sticky="")

    def _generate_statistics(self):
        """Genera las estadísticas según la opción elegida en el desplegable."""
        if not self.analysis_processor or not self.analysis_processor.is_ready:
            self._add_placeholder_stats(self.stats_main_frame, "No hay datos. Genera antes el análisis.")
            return
            
        stats_type = self.stats_type_dropdown.get()
        
        if stats_type == 'Signal Quality':
            self._plot_coverage_statistics()
        elif stats_type == 'Performance':
            self._plot_performance_statistics()
        elif stats_type == 'Jitter': # Añadido para este caso
            self._plot_jitter_statistics() # Añadido para este caso
        else:
            self._add_placeholder_stats(self.stats_main_frame, f"Tipo de estadística '{stats_type}' todavía no está implementado.")

    def _create_coverage_criteria_legend(self):
        """Crea una figura aparte con los criterios de cobertura."""
        criteria_fig, criteria_ax = plt.subplots(figsize=(10, 6), dpi=100)
        criteria_ax.axis('off')
        
        # Añado unidades en los títulos
        criteria_text = [
            "COVERAGE CRITERIA LEGEND:",
            "",
            "RSRP (dBm):",
            "  Excellent (≥ -80 dBm)",
            "  Good (≥ -90 to < -80 dBm)", 
            "  Mid Cell (≥ -100 to < -90 dBm)",
            "  Cell Edge (< -100 dBm)",
            "",
            "RSRQ (dB):",
            "  Excellent (≥ -10 dB)",
            "  Good (≥ -15 to < -10 dB)",
            "  Mid Cell (≥ -20 to < -15 dB)",
            "  Cell Edge (< -20 dB)",
            "",
            "SNR (dB):",
            "  Excellent (≥ 20 dB)",
            "  Good (≥ 13 to < 20 dB)",
            "  Mid Cell (≥ 0 to < 13 dB)",
            "  Cell Edge (< 0 dB)",
            "",
            "No Coverage: Missing or invalid data"
        ]
        
        y_pos = 0.95
        line_height = 0.045
        
        for i, line in enumerate(criteria_text):
            if line.startswith("COVERAGE CRITERIA LEGEND:"):
                criteria_ax.text(0.1, y_pos, line, fontsize=14, fontweight='bold', 
                               transform=criteria_ax.transAxes, verticalalignment='top')
            elif line.startswith("  "):
                category = line.split('(')[0].strip()
                if category in COVERAGE_COLORS:
                    rect = Rectangle((0.08, y_pos - 0.02), 0.02, 0.02, 
                                   facecolor=COVERAGE_COLORS[category],
                                   edgecolor='black', transform=criteria_ax.transAxes)
                    criteria_ax.add_patch(rect)
                
                criteria_ax.text(0.12, y_pos, line, fontsize=10, 
                               transform=criteria_ax.transAxes, verticalalignment='top')
            # Compruebo títulos que llevan unidades
            elif line.endswith("):"): 
                criteria_ax.text(0.1, y_pos, line, fontsize=12, fontweight='bold',
                               transform=criteria_ax.transAxes, verticalalignment='top')
            else:
                criteria_ax.text(0.1, y_pos, line, fontsize=10,
                               transform=criteria_ax.transAxes, verticalalignment='top')
            
            y_pos -= line_height
        
        criteria_fig.tight_layout()
        return criteria_fig

    
    def _plot_coverage_statistics(self):
        """
        Dibuja las estadísticas de cobertura en barras agrupadas.
        A la izquierda queda la gráfica y a la derecha la leyenda.
        """
        for widget in self.stats_main_frame.winfo_children():
            widget.destroy()

        if not self.analysis_processor or not self.analysis_processor.is_ready:
            self._add_placeholder_stats(self.stats_main_frame, "No hay datos para analizar cobertura.")
            return

        coverage_stats = self.analysis_processor.calculate_coverage_statistics()
        if not coverage_stats:
            self._add_placeholder_stats(self.stats_main_frame, "No hay datos de cobertura disponibles.")
            return

        stats_main_frame = ttk.Frame(self.stats_main_frame)
        stats_main_frame.grid(row=0, column=0, sticky="nsew")
        stats_main_frame.grid_rowconfigure(0, weight=1)
        stats_main_frame.grid_columnconfigure(0, weight=2)
        stats_main_frame.grid_columnconfigure(1, weight=1)

        self.stats_fig, self.stats_ax = plt.subplots(figsize=(10, 5), dpi=100)
        parameters = ['RSRP', 'RSRQ', 'SNR']
        categories = ['Excellent', 'Good', 'Mid Cell', 'Cell Edge', 'No Coverage']
        sets = list(coverage_stats.keys())

        bar_width = 0.15
        num_cats = len(categories)
        cluster_width = bar_width * num_cats
        group_padding = 0.4
        set_padding = 0.8
        offsets = (np.arange(num_cats) - (num_cats - 1) / 2) * bar_width

        x_ticks, x_tick_labels = [], []
        current_x = 0

        for set_idx, set_label in enumerate(sets):
            set_data = coverage_stats[set_label]
            set_x_start = current_x

            for param in parameters:
                param_data = set_data.get(param, {})
                cluster_center_x = current_x + (cluster_width / 2)
                x_ticks.append(cluster_center_x)
                # Uso la función auxiliar para mostrar parámetro y unidad
                x_tick_labels.append(get_param_with_unit(param))
                for cat_idx, category in enumerate(categories):
                    percentage = param_data.get(category, 0)
                    x_pos = cluster_center_x + offsets[cat_idx]
                    color = COVERAGE_COLORS.get(category, '#808080')
                    self.stats_ax.bar(x_pos, percentage, bar_width, color=color, edgecolor='black', linewidth=0.4)
                    if percentage >= 1:
                        self.stats_ax.text(x_pos, percentage + 0.6, f'{percentage:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
                current_x += cluster_width + group_padding

            set_x_center = (set_x_start + current_x - group_padding) / 2
            self.stats_ax.text(set_x_center, -0.28, set_label.split(':')[0], transform=self.stats_ax.get_xaxis_transform(), ha='center', va='top', fontsize=10, fontweight='bold')
            current_x += set_padding

        self.stats_ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
        self.stats_ax.set_title('Coverage Statistics by Set and Parameter', fontsize=14, fontweight='bold')
        self.stats_ax.set_xticks(x_ticks)
        # Giro las etiquetas para que entren mejor
        self.stats_ax.set_xticklabels(x_tick_labels, fontsize=11, fontweight='bold', rotation=90)
        self.stats_ax.set_ylim(0, 110)
        self.stats_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        self.stats_ax.grid(True, axis='y', alpha=0.5, linestyle='--')
        self.stats_ax.set_xlim(-set_padding, current_x - group_padding - set_padding)
        self.stats_fig.tight_layout()
        # Aumento margen inferior para que no se corten las etiquetas giradas
        plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.30 if len(sets) > 1 else 0.24)

        stats_plot_frame = ttk.Frame(stats_main_frame)
        stats_plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        stats_plot_frame.grid_rowconfigure(0, weight=1)
        stats_plot_frame.grid_columnconfigure(0, weight=1)
        self.stats_canvas = self._embed_plot(self.stats_fig, stats_plot_frame)

        legend_frame = ttk.Frame(stats_main_frame, padding=(5, 5))
        legend_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(legend_frame, text="Legend", font=('Arial', 12, 'bold')).pack(anchor="w", pady=(0, 8))
        for category in categories:
            rowf = ttk.Frame(legend_frame)
            rowf.pack(anchor="w", pady=3, fill='x')
            color_box = tk.Canvas(rowf, width=16, height=12, highlightthickness=1)
            color_box.create_rectangle(0, 0, 16, 12, fill=COVERAGE_COLORS.get(category, '#808080'), outline='black')
            color_box.pack(side="left", padx=(0, 8))
            ttk.Label(rowf, text=category, font=('Arial', 10)).pack(side="left")


    
    def _plot_performance_statistics(self):
        """
        Dibuja las estadísticas de rendimiento con el mismo estilo que Signal Quality.
        Mantengo la gráfica a la izquierda y la leyenda a la derecha.
        """
        for widget in self.stats_main_frame.winfo_children():
            widget.destroy()

        if not self.analysis_processor or not self.analysis_processor.is_ready:
            self._add_placeholder_stats(self.stats_main_frame, "No hay datos para analizar rendimiento.")
            return

        performance_stats = {}
        for label, df in self.analysis_processor.data.items():
            performance_stats[label] = {}
            
            # Inicio del ajuste
            # Para descarga/subida uso el total de filas como referencia
            total_samples_all_rows = len(df) 
            
            for param in ['DOWNLOAD_SPEED', 'UPLOAD_SPEED', 'LATENCY']:
                if param not in df.columns:
                    continue
                
                # Cojo las muestras válidas del parámetro
                valid_samples_for_param = df[param].dropna()
                
                # Categorías por defecto
                categories = ['Excellent', 'Good', 'Fair', 'Poor']
                
                if param == 'LATENCY':
                    # Lógica específica para latencia
                    
                    # 1. Me quedo como máximo con 150 muestras válidas
                    if len(valid_samples_for_param) > 150:
                        valid_samples_for_param = valid_samples_for_param.iloc[:150]
                    
                    # 2. El 100% se calcula sobre las muestras usadas
                    total_samples_for_perc = len(valid_samples_for_param)
                    if total_samples_for_perc == 0:
                        continue
                    
                    # 3. En latencia no uso categoría 'No Data'
                    categories_with_no = categories
                    no_data = 0
                    
                    # 4. Calculo los conteos por categoría
                    valid = valid_samples_for_param
                    category_counts = {cat: 0 for cat in categories}
                    category_counts['Excellent'] = (valid <= 30).sum()
                    category_counts['Good'] = ((valid > 30) & (valid <= 70)).sum()
                    category_counts['Fair'] = ((valid > 70) & (valid <= 150)).sum()
                    category_counts['Poor'] = (valid > 150).sum()
                    
                else:
                    # Lógica para descarga y subida
                    if total_samples_all_rows == 0:
                        continue
                        
                    # 1. El 100% es el total de filas del dataframe
                    total_samples_for_perc = total_samples_all_rows
                    
                    # 2. Aquí sí incluyo la categoría 'No Data'
                    categories_with_no = categories + ['No Data']
                    valid = valid_samples_for_param
                    valid_count = len(valid)
                    no_data = total_samples_for_perc - valid_count
                    
                    # 3. Calculo los conteos
                    category_counts = {cat: 0 for cat in categories}
                    if param == 'DOWNLOAD_SPEED':
                        category_counts['Excellent'] = (valid >= 50).sum()
                        category_counts['Good'] = ((valid >= 20) & (valid < 50)).sum()
                        category_counts['Fair'] = ((valid >= 5) & (valid < 20)).sum()
                        category_counts['Poor'] = (valid < 5).sum()
                    elif param == 'UPLOAD_SPEED':
                        category_counts['Excellent'] = (valid >= 10).sum()
                        category_counts['Good'] = ((valid >= 5) & (valid < 10)).sum()
                        category_counts['Fair'] = ((valid >= 1) & (valid < 5)).sum()
                        category_counts['Poor'] = (valid < 1).sum()

                # Cálculo común de porcentajes y redondeo
                param_floats = []
                # Calculo el porcentaje de cada categoría
                for cat in categories:
                    perc = (category_counts[cat] / total_samples_for_perc) * 100 if total_samples_for_perc > 0 else 0
                    param_floats.append((cat, perc))
                
                # Añado 'No Data' solo para descarga/subida
                if param != 'LATENCY':
                    no_data_perc = (no_data / total_samples_for_perc) * 100 if total_samples_for_perc > 0 else 0
                    param_floats.append(('No Data', no_data_perc))
                else:
                    # En latencia también controlo que el total sea 100%
                    # Simplemente no añado la categoría 'No Data'
                    categories_with_no = categories # Sobrescribo la lista para este caso

                # Redondeo con el método del resto mayor
                param_stats_final = {}
                initial_sum = 0
                remainders = []
                for cat, fperc in param_floats:
                    floored = int(fperc)
                    param_stats_final[cat] = floored
                    initial_sum += floored
                    remainders.append((fperc - floored, cat))
                
                difference = 100 - initial_sum
                remainders.sort(key=lambda x: x[0], reverse=True)
                
                for i in range(int(round(difference))):
                    if i < len(remainders):
                        cat_to_inc = remainders[i][1]
                        param_stats_final[cat_to_inc] = param_stats_final.get(cat_to_inc, 0) + 1
                
                # Última corrección por si el redondeo no suma 100%
                final_sum = sum(param_stats_final.values())
                if final_sum != 100 and remainders:
                    param_stats_final[remainders[0][1]] += (100 - final_sum)

                performance_stats[label][param] = param_stats_final
            # Fin de este ajuste

        if not performance_stats:
            self._add_placeholder_stats(self.stats_main_frame, "No hay datos de rendimiento disponibles.")
            return

        stats_main_frame = ttk.Frame(self.stats_main_frame)
        stats_main_frame.grid(row=0, column=0, sticky="nsew")
        stats_main_frame.grid_rowconfigure(0, weight=1)
        stats_main_frame.grid_columnconfigure(0, weight=2)
        stats_main_frame.grid_columnconfigure(1, weight=1)

        self.stats_fig, self.stats_ax = plt.subplots(figsize=(10, 5), dpi=100)
        parameters = ['DOWNLOAD_SPEED', 'UPLOAD_SPEED', 'LATENCY']
        
        # Las categorías de la leyenda tienen que ser dinámicas
        # Recojo todas las categorías usadas en los parámetros
        all_categories = set()
        for set_label in performance_stats:
            for param in parameters:
                if param in performance_stats[set_label]:
                    all_categories.update(performance_stats[set_label][param].keys())
        
        # Defino el orden en el que quiero mostrarlas
        category_order_map = {'Excellent': 1, 'Good': 2, 'Fair': 3, 'Poor': 4, 'No Data': 5}
        # Ordeno las categorías según ese orden
        categories_to_plot = sorted(list(all_categories), key=lambda x: category_order_map.get(x, 99))
        # Fin de este ajuste
        
        sets = list(performance_stats.keys())

        bar_width = 0.15
        num_cats = len(categories_to_plot) # Uso la lista dinámica de categorías
        cluster_width = bar_width * num_cats
        group_padding = 0.4
        set_padding = 0.8
        offsets = (np.arange(num_cats) - (num_cats - 1) / 2) * bar_width

        x_ticks, x_tick_labels = [], []
        current_x = 0

        for set_idx, set_label in enumerate(sets):
            set_data = performance_stats[set_label]
            set_x_start = current_x
            for param in parameters:
                param_data = set_data.get(param, {})
                cluster_center_x = current_x + (cluster_width / 2)
                x_ticks.append(cluster_center_x)
                # Uso la función auxiliar para mostrar parámetro y unidad
                x_tick_labels.append(get_param_with_unit(param))

                # Uso categorías dinámicas al dibujar
                for cat_idx, category in enumerate(categories_to_plot):
                    # Solo dibujo si ese parámetro tiene esa categoría
                    # Por ejemplo, latencia no lleva 'No Data'
                    percentage = param_data.get(category, 0)
                    
                    x_pos = cluster_center_x + offsets[cat_idx]
                    # Uso los colores propios de rendimiento
                    color = PERFORMANCE_COLORS.get(category, '#808080')
                    
                    # Dibujo la barra solo si tiene porcentaje o si toca mostrar 'No Data'
                    # Así evito que 'No Data' aparezca en latencia
                    if percentage > 0 or (param != 'LATENCY' and category == 'No Data'):
                        self.stats_ax.bar(x_pos, percentage, bar_width, color=color, edgecolor='black', linewidth=0.4)
                    
                    if percentage >= 1:
                        self.stats_ax.text(x_pos, percentage + 0.6, f'{percentage:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
                # Fin de este ajuste
                
                current_x += cluster_width + group_padding

            set_x_center = (set_x_start + current_x - group_padding) / 2
            self.stats_ax.text(set_x_center, -0.40, set_label.split(':')[0], transform=self.stats_ax.get_xaxis_transform(), ha='center', va='top', fontsize=10, fontweight='bold')
            current_x += set_padding

        self.stats_ax.set_ylabel('Percentage', fontsize=12, fontweight='bold')
        self.stats_ax.set_title('Performance Statistics by Set and Parameter', fontsize=14, fontweight='bold')
        self.stats_ax.set_xticks(x_ticks)
        # Giro las etiquetas para que entren mejor
        self.stats_ax.set_xticklabels(x_tick_labels, fontsize=11, fontweight='bold', rotation=90)
        self.stats_ax.set_ylim(0, 110)
        self.stats_ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        self.stats_ax.grid(True, axis='y', alpha=0.5, linestyle='--')
        self.stats_ax.set_xlim(-set_padding, current_x - group_padding - set_padding)
        self.stats_fig.tight_layout()
        # Aumento margen inferior para que no se corten las etiquetas giradas
        plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.40 if len(sets) > 1 else 0.36)

        stats_plot_frame = ttk.Frame(stats_main_frame)
        stats_plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        stats_plot_frame.grid_rowconfigure(0, weight=1)
        stats_plot_frame.grid_columnconfigure(0, weight=1)
        self.stats_canvas = self._embed_plot(self.stats_fig, stats_plot_frame)

        legend_frame = ttk.Frame(stats_main_frame, padding=(5, 5))
        legend_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(legend_frame, text="Legend", font=('Arial', 12, 'bold')).pack(anchor="w", pady=(0, 8))
        
        # La leyenda también usa categorías dinámicas
        for category in categories_to_plot:
            rowf = ttk.Frame(legend_frame)
            rowf.pack(anchor="w", pady=3, fill='x')
            color_box = tk.Canvas(rowf, width=16, height=12, highlightthickness=1)
            # Uso los colores propios de rendimiento
            color_box.create_rectangle(0, 0, 16, 12, fill=PERFORMANCE_COLORS.get(category, '#808080'), outline='black')
            color_box.pack(side="left", padx=(0, 8))
            ttk.Label(rowf, text=category, font=('Arial', 10)).pack(side="left")
        # Fin de este ajuste

    def _plot_jitter_statistics(self):
        """
        Dibuja el jitter medio de cada set en una gráfica de barras.
        """
        for widget in self.stats_main_frame.winfo_children():
            widget.destroy()

        if not self.analysis_processor or not self.analysis_processor.is_ready:
            self._add_placeholder_stats(self.stats_main_frame, "No hay datos para analizar jitter.")
            return

        # Llamo al método que calcula el jitter en el procesador
        jitter_stats = self.analysis_processor.get_all_jitter_stats() 

        if not jitter_stats:
            self._add_placeholder_stats(self.stats_main_frame, "No Jitter data found.\n(Requires 'LATENCY' column with at least 2 samples per set)")
            return

        stats_main_frame = ttk.Frame(self.stats_main_frame)
        stats_main_frame.grid(row=0, column=0, sticky="nsew")
        stats_main_frame.grid_rowconfigure(0, weight=1)
        stats_main_frame.grid_columnconfigure(0, weight=1) # Una sola columna

        self.stats_fig, self.stats_ax = plt.subplots(figsize=(10, 5), dpi=100)
        
        labels = list(jitter_stats.keys())
        values = list(jitter_stats.values())
        # Cojo los colores que ya tiene asignados cada set
        colors = [self.analysis_processor.color_map.get(label, '#808080') for label in labels]
        
        bars = self.stats_ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.4)
        
        # Añado el valor encima de cada barra
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                self.stats_ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                                 f'{height:.2f} ms',
                                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        self.stats_ax.set_ylabel('Average Jitter (ms)', fontsize=12, fontweight='bold')
        self.stats_ax.set_title('Average Jitter by Test Set', fontsize=14, fontweight='bold')
        
        # Formateo las etiquetas del eje X
        short_labels = [label.split(':')[0] for label in labels] # Por ejemplo: 'Set 1'
        self.stats_ax.set_xticks(range(len(short_labels)))
        self.stats_ax.set_xticklabels(short_labels, rotation=0, ha='center', fontsize=10)

        self.stats_ax.grid(True, axis='y', alpha=0.5, linestyle='--')
        
        # Ajusto el límite Y para que el texto se vea bien
        max_val = max(values) if values else 0 # Contemplo el caso de valores vacíos
        self.stats_ax.set_ylim(0, max_val * 1.15 if max_val > 0 else 10)

        self.stats_fig.tight_layout()

        self.stats_canvas = self._embed_plot(self.stats_fig, stats_main_frame)



    def _create_ai_tab(self):
        container = ttk.Frame(self.ai_tab, style='Panel.TFrame', padding=10)
        container.grid(row=0, column=0, sticky='nsew')
        container.rowconfigure(1, weight=1)
        container.columnconfigure(1, weight=1)

        controls = ttk.LabelFrame(container, text='Modelo predictivo de cobertura', padding=10)
        controls.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 10))
        for c in range(8):
            controls.columnconfigure(c, weight=1 if c in (1, 3, 5) else 0)

        ttk.Label(controls, text='Tecnología:', style='Panel.TLabel').grid(row=0, column=0, sticky='w', padx=(0, 6), pady=4)
        self.ai_tech_combo = ttk.Combobox(controls, state='readonly', textvariable=self.ai_tech_var, values=['3G', '4G', '5G'])
        self.ai_tech_combo.grid(row=0, column=1, sticky='ew', pady=4)

        ttk.Label(controls, text='Objetivo:', style='Panel.TLabel').grid(row=0, column=2, sticky='w', padx=(10, 6), pady=4)
        self.ai_target_combo = ttk.Combobox(controls, state='readonly', textvariable=self.ai_target_var, values=['RSRP'])
        self.ai_target_combo.grid(row=0, column=3, sticky='ew', pady=4)

        ttk.Label(controls, text='Grid (m):', style='Panel.TLabel').grid(row=0, column=4, sticky='w', padx=(10, 6), pady=4)
        self.ai_resolution_combo = ttk.Combobox(controls, state='readonly', textvariable=self.ai_resolution_var, values=['20', '30', '35', '50', '75'])
        self.ai_resolution_combo.grid(row=0, column=5, sticky='ew', pady=4)
        self.ai_resolution_combo.bind('<<ComboboxSelected>>', self._on_ai_resolution_changed)

        self.ai_train_button = ttk.Button(controls, text='Entrenar modelo', width=20, style='Accent.TButton', command=self._train_ai_model)
        self.ai_train_button.grid(row=0, column=6, padx=(10, 4), pady=4, sticky='ew')
        self.ai_predict_button = ttk.Button(controls, text='Generar mapa IA', width=20, command=self._generate_ai_prediction, state=tk.DISABLED)
        self.ai_predict_button.grid(row=0, column=7, padx=(4, 4), pady=4, sticky='ew')

        self.ai_export_button = ttk.Button(controls, text='Exportar PNG', width=16, command=self._export_ai_prediction, state=tk.DISABLED)
        self.ai_export_button.grid(row=1, column=7, padx=(4, 4), pady=(6, 0), sticky='ew')
        self.ai_status_label = tk.Label(
            controls,
            textvariable=self.ai_status_var,
            bg='#1c2230',
            fg='#ffb347',
            anchor='w',
            justify='left',
            font=('Segoe UI', 10, 'bold'),
            padx=4,
            pady=2,
            wraplength=1100
        )
        self.ai_status_label.grid(row=1, column=0, columnspan=7, sticky='ew', pady=(6, 0))
        self.ai_progress = None

        info_frame = ttk.LabelFrame(container, text='Resumen del modelo', padding=8)
        info_frame.grid(row=1, column=0, sticky='nsew', padx=(0, 8))
        info_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)
        self.ai_info_text = tk.Text(info_frame, wrap='word', bg='#20293a', fg='#edf2f7', insertbackground='white', width=38)
        self.ai_info_text.grid(row=0, column=0, sticky='nsew')

        plot_frame = ttk.LabelFrame(container, text='Mapa predictivo', padding=6)
        plot_frame.grid(row=1, column=1, sticky='nsew')
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self.ai_plot_frame = plot_frame

        self._write_ai_info('Carga un análisis y entrena el modelo para generar la predicción.')

    def _write_ai_info(self, text):
        if self.ai_info_text is None:
            return
        self.ai_info_text.delete('1.0', tk.END)
        self.ai_info_text.insert('1.0', text)

    def _update_ai_controls(self):
        if not self.analysis_processor or not self.analysis_processor.is_ready:
            return

        # Para la IA uso la tecnología del escenario seleccionado,
        # es decir, el 3G/4G/5G que viene en el nombre del archivo, no siempre la tecnología radio
        # instantánea que detecta el móvil. Esto es importante en 5G NSA, porque el teléfono
        # puede registrar muestras LTE/4G aunque el escenario sea 5G.
        tech_values = []
        combined = self.analysis_processor.combined_data
        if 'SCENARIO_TECH' in combined.columns:
            tech_values = sorted({
                str(v).upper().strip()
                for v in combined['SCENARIO_TECH'].dropna()
                if str(v).upper().strip() in {'3G', '4G', '5G'}
            })

        # Si el archivo antiguo no trae escenario, uso TECHNOLOGY como respaldo.
        if not tech_values and 'TECHNOLOGY' in combined.columns:
            tech_values = sorted({
                str(v).upper().strip()
                for v in combined['TECHNOLOGY'].dropna()
                if str(v).upper().strip() in {'3G', '4G', '5G'}
            })

        if not tech_values:
            tech_values = ['3G', '4G', '5G']

        self.ai_tech_combo['values'] = tech_values
        if self.ai_tech_var.get().upper().strip() not in tech_values:
            self.ai_tech_var.set(tech_values[0])

    def _on_ai_resolution_changed(self, _event=None):
        # Cambiar el grid no obliga a reentrenar, pero sí a regenerar el mapa.
        # Invalido solo la parte visual para que no parezca que sigue usando
        # el grid anterior.
        try:
            grid_txt = self.ai_resolution_combo.get().strip() or self.ai_resolution_var.get().strip()
        except Exception:
            grid_txt = self.ai_resolution_var.get().strip()
        self.ai_prediction_df = None
        self.ai_fig = None
        if self.ai_model_bundle:
            self._set_ai_status(f'IA: grid cambiado a {grid_txt} m. Pulsa "Generar mapa IA" para recalcular el mapa.', 'busy')
            try:
                self.ai_predict_button.configure(state=tk.NORMAL)
                self.ai_export_button.configure(state=tk.DISABLED)
            except Exception:
                pass
        else:
            self._set_ai_status(f'IA: grid seleccionado: {grid_txt} m. Entrena el modelo antes de generar el mapa.', 'muted')

    def _set_ai_status(self, text, mode='muted'):
        self.ai_status_var.set(text)
        if self.ai_status_label is not None:
            color_map = {'muted': '#c9d1d9', 'busy': '#ffb347', 'success': '#7CFC8A', 'error': '#ff7b72'}
            self.ai_status_label.configure(fg=color_map.get(mode, '#c9d1d9'))
        self.update_idletasks()

    def _set_ai_busy(self, busy=True, message=None, success=False, enable_predict=None, enable_export=None, action=None):
        if message is not None:
            mode = 'success' if success else ('busy' if busy else 'muted')
            self._set_ai_status(message, mode)
        if self.ai_train_button is not None:
            train_text = 'Entrenando modelo...' if (busy and action == 'train') else 'Entrenar modelo'
            self.ai_train_button.configure(text=train_text, state=tk.DISABLED if busy else tk.NORMAL)
        if self.ai_predict_button is not None:
            predict_text = 'Generando mapa IA...' if (busy and action == 'predict') else 'Generar mapa IA'
            if busy:
                self.ai_predict_button.configure(text=predict_text, state=tk.DISABLED)
            elif enable_predict is not None:
                self.ai_predict_button.configure(text=predict_text, state=tk.NORMAL if enable_predict else tk.DISABLED)
            else:
                self.ai_predict_button.configure(text=predict_text)
        if self.ai_export_button is not None and enable_export is not None:
            self.ai_export_button.configure(state=tk.NORMAL if enable_export else tk.DISABLED)
        if self.ai_train_button is not None:
            self.ai_train_button.update_idletasks()
        if self.ai_predict_button is not None:
            self.ai_predict_button.update_idletasks()
        self.update_idletasks()

    def _train_ai_model(self):
        if not SKLEARN_AVAILABLE:
            messagebox.showwarning('Predicción IA', 'Falta scikit-learn. Instálalo con: pip install scikit-learn')
            return
        if not self.analysis_processor or not self.analysis_processor.is_ready:
            messagebox.showwarning('Predicción IA', 'Primero genera un análisis.')
            return

        self._set_ai_busy(True, 'IA: entrenando modelo y preparando variables...', enable_predict=False, enable_export=False, action='train')
        try:
            tech = self.ai_tech_var.get().strip().upper()
            target = self.ai_target_var.get().strip()
            combined = self.analysis_processor.combined_data.copy()
            target_candidates = [target, CSV_COLUMNS.get(target, ''), 'RSRP/RSCP' if target == 'RSRP' else target]
            target_col = None
            for candidate in target_candidates:
                if candidate and candidate in combined.columns:
                    target_col = candidate
                    break
            if target_col is None:
                messagebox.showwarning('Predicción IA', f'No se encontró la columna objetivo para {target}.')
                self._set_ai_busy(False, 'IA: no se encontró la columna objetivo.', enable_predict=False, enable_export=False)
                return

            df = combined.copy()
            # Filtro por escenario de campaña. En 5G NSA el sistema puede aparecer como LTE,
            # por eso doy prioridad a SCENARIO_TECH, que viene del nombre del archivo o del selector.
            if 'SCENARIO_TECH' in df.columns:
                df = df[df['SCENARIO_TECH'].astype(str).str.upper().str.strip() == tech].copy()
            elif 'TECHNOLOGY' in df.columns:
                df = df[df['TECHNOLOGY'].astype(str).str.upper().str.strip() == tech].copy()
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
            df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
            df = df.dropna(subset=['LAT', 'LON', target_col]).copy()

            if len(df) < 40:
                messagebox.showwarning('Predicción IA', 'Necesitas al menos 40 muestras válidas para entrenar el modelo.')
                self._set_ai_busy(False, 'IA: muestras insuficientes para entrenar.', enable_predict=False, enable_export=False)
                return

            self._set_ai_status('IA: consultando Catastro WFS BU, estimando estaciones base y construyendo variables...', 'busy')
            towers = _flatten_towers_from_processor(self.analysis_processor, tech_filter=tech)
            if not towers:
                messagebox.showwarning('Predicción IA', 'No se pudieron estimar estaciones base para esta tecnología.')
                self._set_ai_busy(False, 'IA: no se pudieron estimar estaciones base.', enable_predict=False, enable_export=False)
                return

            min_lat, max_lat = float(df['LAT'].min()), float(df['LAT'].max())
            min_lon, max_lon = float(df['LON'].min()), float(df['LON'].max())
            origin_lat = float(df['LAT'].mean())
            origin_lon = float(df['LON'].mean())
            buildings, building_error, building_source = _fetch_buildings(min_lat, max_lat, min_lon, max_lon, origin_lat, origin_lon)

            feature_df, origin, towers_xy = _build_prediction_feature_frame(df, towers, buildings, target_col)
            feature_df = _augment_prediction_features(feature_df)
            feature_cols = [c for c in feature_df.columns if c not in ['LAT', 'LON', target_col]]
            X = feature_df[feature_cols].copy()
            y = pd.to_numeric(feature_df[target_col], errors='coerce').copy()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            model = RandomForestRegressor(
                n_estimators=420,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                oob_score=True,
            )
            model.fit(X_train, y_train)

            residual_knn = None
            pred_train = model.predict(X_train)
            if KNeighborsRegressor is not None and len(X_train) >= 24 and {'x_m', 'y_m'}.issubset(X_train.columns):
                residual_knn = KNeighborsRegressor(
                    n_neighbors=max(4, min(12, len(X_train) // 20)),
                    weights='distance'
                )
                residual_knn.fit(X_train[['x_m', 'y_m']], y_train - pred_train)

            pred = model.predict(X_test)
            if residual_knn is not None and {'x_m', 'y_m'}.issubset(X_test.columns):
                pred = pred + residual_knn.predict(X_test[['x_m', 'y_m']])

            clip_lo = float(np.nanquantile(y, 0.02) - 2.0)
            clip_hi = float(np.nanquantile(y, 0.98) + 2.0)
            pred = np.clip(pred, clip_lo, clip_hi)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)

            importance_pairs = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: x[1], reverse=True)
            top_lines = ['Importancia de variables:']
            for name, value in importance_pairs[:10]:
                top_lines.append(f'  - {name}: {value:.3f}')

            model_desc = 'RandomForest + corrección local de residuales + suavizado visual' if residual_knn is not None else 'RandomForest'
            info = [
                f'Tecnología: {tech}',
                f'Objetivo: {target}',
                f'Muestras usadas: {len(feature_df)}',
                f'Estaciones base estimadas: {len(towers_xy)}',
                f'Edificios {building_source}: {len(buildings)}',
                f'Modelo: {model_desc}',
                f'MAE hold-out: {mae:.2f} dB',
                f'R² hold-out: {r2:.3f}',
                '',
            ] + top_lines
            if towers:
                info += ['', f'Nota torres: estimación robusta basada en muestras de mejor señal por celda.']
            info += ['', f'Fuente edificios: {building_source}']
            info += [f'Mapa IA: recorte a entorno de ruta y suavizado visual 3x3 al renderizar.']
            if building_error:
                info += [f'Nota edificios: {building_error}']

            self.ai_model_bundle = {
                'tech': tech,
                'target': target,
                'target_col': target_col,
                'model': model,
                'feature_cols': feature_cols,
                'feature_df': feature_df,
                'origin': origin,
                'towers_xy': towers_xy,
                'towers': towers,
                'buildings': buildings,
                'building_source': building_source,
                'mae': mae,
                'r2': r2,
                'residual_knn': residual_knn,
                'pred_clip_lo': clip_lo,
                'pred_clip_hi': clip_hi,
                'info_text': chr(10).join(info),
            }
            self._write_ai_info(self.ai_model_bundle['info_text'])
            self._set_ai_busy(False, 'IA: modelo entrenado. Ahora pulsa "Generar mapa IA".', success=True, enable_predict=True, enable_export=False)
        except Exception as exc:
            self._set_ai_busy(False, f'IA: error al entrenar el modelo: {exc}', enable_predict=False, enable_export=False)
            self._write_ai_info(f'Error IA al entrenar:\n{exc}')
            raise

    def _generate_ai_prediction(self):
        if not self.ai_model_bundle:
            self._train_ai_model()
            if not self.ai_model_bundle:
                return

        try:
            resolution_text = self.ai_resolution_combo.get().strip() or self.ai_resolution_var.get().strip()
            resolution_m = max(10, int(float(resolution_text)))
            self.ai_resolution_var.set(str(resolution_m))
        except Exception:
            resolution_m = 35
            self.ai_resolution_var.set('35')

        bundle = self.ai_model_bundle
        self._set_ai_busy(True, f'IA: generando mapa predictivo con grid de {resolution_m} m...', enable_predict=False, enable_export=False)
        try:
            grid_df, xs, ys = _build_prediction_grid(
                bundle['feature_df'],
                bundle['origin'],
                bundle['towers_xy'],
                bundle['buildings'],
                resolution_m,
            )
            grid_df = _augment_prediction_features(grid_df)
            grid_df['grid_resolution_m'] = resolution_m
            grid_df['grid_nx'] = len(xs)
            grid_df['grid_ny'] = len(ys)
            X_grid = grid_df[bundle['feature_cols']].copy()
            grid_df['prediction'] = _predict_with_bundle(bundle, grid_df)
            self.ai_prediction_df = grid_df

            pred_matrix = grid_df['prediction'].to_numpy().reshape(len(ys), len(xs))
            lon_matrix = grid_df['LON'].to_numpy().reshape(len(ys), len(xs))
            lat_matrix = grid_df['LAT'].to_numpy().reshape(len(ys), len(xs))
            if 'inside_route_buffer' in grid_df.columns:
                mask_matrix = grid_df['inside_route_buffer'].to_numpy().reshape(len(ys), len(xs))
                pred_matrix = np.where(mask_matrix, pred_matrix, np.nan)

            # Suavizado solo visual: mejora el aspecto del mapa sin tocar métricas ni datos internos.
            smooth_passes = 1 if resolution_m <= 25 else 0
            display_matrix = _smooth_prediction_matrix(pred_matrix, passes=smooth_passes)
            try:
                grid_df['prediction_display'] = display_matrix.reshape(-1)
            except Exception:
                pass

            if self.ai_canvas is not None:
                self.ai_canvas.get_tk_widget().destroy()
                self.ai_canvas = None
            self.ai_fig, ax = plt.subplots(figsize=(9.5, 6.6), dpi=100)
            cmap = 'turbo'
            # Dibujo con bordes reales de celda para que el Grid (m)
            # se aprecie de verdad: 20 m son bloques pequeños y 75 m bloques grandes.
            lon_edges = _centers_to_edges(lon_matrix[0, :])
            lat_edges = _centers_to_edges(lat_matrix[:, 0])
            mesh = ax.pcolormesh(lon_edges, lat_edges, display_matrix, shading='flat', cmap=cmap, alpha=0.72)
            cb = self.ai_fig.colorbar(mesh, ax=ax)
            cb.set_label(f"{bundle['target']} predicho")

            for building in bundle['buildings']:
                bmin_lat, bmin_lon, bmax_lat, bmax_lon = building['latlon_bbox']
                ax.add_patch(Rectangle((bmin_lon, bmin_lat), bmax_lon - bmin_lon, bmax_lat - bmin_lat, fill=False, edgecolor='dimgray', linewidth=0.4, alpha=0.8))

            sample_df = bundle['feature_df']
            ax.scatter(sample_df['LON'], sample_df['LAT'], s=8, c='black', alpha=0.35, label='Muestras reales')
            for tower in bundle['towers']:
                ax.scatter(tower['lon'], tower['lat'], marker='^', s=80, c='white', edgecolors='black', linewidth=1.0)
                ax.text(tower['lon'], tower['lat'], str(tower['id']), fontsize=7, ha='left', va='bottom', color='white', bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.45))

            sample_min_lon = float(sample_df['LON'].min())
            sample_max_lon = float(sample_df['LON'].max())
            sample_min_lat = float(sample_df['LAT'].min())
            sample_max_lat = float(sample_df['LAT'].max())
            pad_lon = max(0.00018, (sample_max_lon - sample_min_lon) * 0.08)
            pad_lat = max(0.00018, (sample_max_lat - sample_min_lat) * 0.08)
            xlim = (sample_min_lon - pad_lon, sample_max_lon + pad_lon)
            ylim = (sample_min_lat - pad_lat, sample_max_lat + pad_lat)

            if ctx is not None:
                try:
                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom='auto')
                    ax.set_xlim(*xlim)
                    ax.set_ylim(*ylim)
                except Exception:
                    ax.grid(True, linestyle='--', alpha=0.25)
            else:
                ax.grid(True, linestyle='--', alpha=0.25)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(f"Predicción IA de {bundle['target']} - {bundle['tech']} | Grid {resolution_m} m")
            ax.set_xlabel('Longitud')
            ax.set_ylabel('Latitud')
            ax.legend(loc='upper right')
            self.ai_fig.tight_layout()

            self.ai_canvas = FigureCanvasTkAgg(self.ai_fig, master=self.ai_plot_frame)
            self.ai_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
            self.ai_canvas.draw()

            # Actualizo el panel izquierdo para dejar claro qué grid se ha usado realmente.
            try:
                base_info = (self.ai_model_bundle or {}).get('info_text', '')
                valid_cells = int(np.isfinite(display_matrix).sum())
                total_cells = int(display_matrix.size)
                self._write_ai_info(base_info + f'\n\nMapa IA generado:\n- Grid usado: {resolution_m} m\n- Celdas X/Y: {len(xs)} x {len(ys)}\n- Celdas visibles: {valid_cells} / {total_cells}')
            except Exception:
                pass
            self._set_ai_busy(False, f'IA: mapa predictivo generado con grid de {resolution_m} m. Ya puedes exportarlo.', success=True, enable_predict=True, enable_export=True)
        except Exception as exc:
            self._set_ai_busy(False, f'IA: error al generar el mapa: {exc}', enable_predict=True, enable_export=self.ai_fig is not None)
            self._write_ai_info((self.ai_model_bundle or {}).get('info_text', '') + f'\n\nError al generar mapa:\n{exc}')
            raise

    def _export_ai_prediction(self):

        if self.ai_fig is None:
            messagebox.showwarning('Predicción IA', 'Primero genera el mapa IA.')
            return
        filepath = filedialog.asksaveasfilename(
            title='Guardar mapa predictivo',
            defaultextension='.png',
            filetypes=[('PNG', '*.png')],
            initialfile=f"AI_Coverage_{self.ai_tech_var.get()}_{self.ai_target_var.get()}.png"
        )
        if not filepath:
            return
        self.ai_fig.savefig(filepath, dpi=180, bbox_inches='tight')
        self._set_ai_status(f'IA: PNG exportado en {filepath}', 'success')

    def _create_control_panel(self):
        self.left_panel.grid_rowconfigure(0, weight=1)
        self.left_panel.grid_columnconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(self.left_panel, orient=tk.VERTICAL)
        scrollbar.grid(row=0, column=1, sticky='ns')
        canvas = tk.Canvas(self.left_panel, yscrollcommand=scrollbar.set, borderwidth=0, highlightthickness=0, bg='#1c2230')
        canvas.grid(row=0, column=0, sticky='nsew')
        self.scrollable_frame = ttk.Frame(canvas, style='Panel.TFrame')
        self.canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        scrollbar.config(command=canvas.yview)

        def configure_scroll_region(_event):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def configure_window_width(event):
            canvas.itemconfig(self.canvas_window, width=event.width)

        self.scrollable_frame.bind('<Configure>', configure_scroll_region)
        canvas.bind('<Configure>', configure_window_width)

        folder_box = ttk.LabelFrame(self.scrollable_frame, text='Carpetas', padding=10)
        folder_box.pack(fill='x', pady=(0, 12), padx=4)
        ttk.Button(folder_box, text='Seleccionar carpeta de entrada', style='Accent.TButton', command=self._browse_folder).pack(fill='x', pady=4)
        self.folder_path_label = ttk.Label(folder_box, text='Sin carpeta seleccionada', style='Panel.TLabel', wraplength=300)
        self.folder_path_label.pack(fill='x', pady=(4, 0))

        section2 = ttk.LabelFrame(self.scrollable_frame, text='Comparación', padding='10')
        section2.pack(fill='x', pady=(0, 12), padx=4)
        ttk.Label(section2, text='Número de sets a comparar:', style='Panel.TLabel').pack(anchor='w', pady=(2, 4))
        sets_frame = ttk.Frame(section2, style='Panel.TFrame')
        sets_frame.pack(fill='x', pady=4)
        self.num_sets_var = tk.StringVar(value='1')
        self.num_sets_spinbox = ttk.Spinbox(sets_frame, from_=1, to=12, width=5, textvariable=self.num_sets_var)
        self.num_sets_spinbox.pack(side='left', padx=(0, 8))
        self.num_sets_spinbox.bind('<Return>', lambda e: self._update_set_selection())
        self.num_sets_spinbox.bind('<FocusOut>', lambda e: self._update_set_selection())
        ttk.Button(sets_frame, text='Aplicar', command=self._update_set_selection).pack(side='left')

        self.sets_section = ttk.LabelFrame(self.scrollable_frame, text='Sets seleccionados', padding='10')
        self.sets_section.pack(fill='x', pady=(0, 12), padx=4)
        self.sets_container = ttk.Frame(self.sets_section, style='Panel.TFrame')
        self.sets_container.pack(fill='x')

        actions_box = ttk.LabelFrame(self.scrollable_frame, text='Acciones', padding=10)
        actions_box.pack(fill='x', pady=(0, 12), padx=4)
        self.start_button = ttk.Button(actions_box, text='Generar análisis', command=self._start_process, state=tk.DISABLED, style='Accent.TButton')
        self.start_button.pack(fill='x', pady=4)

        detected_box = ttk.LabelFrame(self.scrollable_frame, text='Campañas detectadas', padding=8)
        detected_box.pack(fill='both', expand=True, pady=(0, 12), padx=4)
        self.detected_listbox = tk.Listbox(detected_box, width=40, height=12, bg='#20293a', fg='#edf2f7', selectbackground='#2f81f7', activestyle='none', bd=0, highlightthickness=0)
        self.detected_listbox.pack(fill='both', expand=True)

        self.status_label = ttk.Label(self.scrollable_frame, text='Estado: selecciona una carpeta.', style='Muted.TLabel', wraplength=300)
        self.status_label.pack(fill='x', pady=(6, 10), padx=4)

        self._create_set_frames()

    def _create_enhanced_dashboard(self):
        container = ttk.Frame(self.analysis_tab, style='Panel.TFrame', padding=6)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        controls_frame = ttk.LabelFrame(container, text="Controles de animación", padding=8)
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.animate_button = ttk.Button(controls_frame, text="Iniciar simulación", command=self._on_animate_click, state=tk.DISABLED, style='Accent.TButton')
        self.animate_button.pack(side="left", padx=(0, 8))
        self.capture_button = ttk.Button(controls_frame, text="Iniciar captura", command=self._on_capture_click, state=tk.DISABLED)
        self.capture_button.pack(side="left", padx=8)
        self.pause_button = ttk.Button(controls_frame, text="Pausa", command=self._on_pause_click, state=tk.DISABLED)
        self.pause_button.pack(side="left", padx=8)
        self.stop_button = ttk.Button(controls_frame, text="Detener", command=self._on_stop_click, state=tk.DISABLED)
        self.stop_button.pack(side="left", padx=8)

        self.main_paned_window = ttk.PanedWindow(container, orient=tk.VERTICAL)
        self.main_paned_window.grid(row=1, column=0, sticky="nsew")

        self.top_frame = ttk.Frame(self.main_paned_window, style='Panel.TFrame')
        self.main_paned_window.add(self.top_frame, weight=3)
        self.top_frame.grid_rowconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(0, weight=1)
        self.top_frame.grid_columnconfigure(1, weight=1)

        self.top_left_frame = ttk.LabelFrame(self.top_frame, text="Mapa de handovers (todos los sets)")
        self.top_left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=2)
        self.top_left_frame.grid_rowconfigure(1, weight=1)
        self.top_left_frame.grid_columnconfigure(0, weight=1)

        self.top_right_frame = ttk.LabelFrame(self.top_frame, text="Heatmap del parámetro con estaciones base")
        self.top_right_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=2)
        self.top_right_frame.grid_rowconfigure(1, weight=1)
        self.top_right_frame.grid_columnconfigure(0, weight=1)

        self.bottom_frame = ttk.LabelFrame(self.main_paned_window, text="Parámetro frente al recorrido (todos los sets)")
        self.main_paned_window.add(self.bottom_frame, weight=2)
        self.bottom_frame.grid_rowconfigure(1, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=1)

        self._create_ho_map_controls()
        self._create_heatmap_controls()
        self._create_bottom_controls()

        self._add_placeholder(self.top_left_frame, "Genera el análisis para ver el mapa de handovers", row=1)
        self._add_placeholder(self.top_right_frame, "Selecciona set y parámetro y pulsa 'Iniciar simulación'", row=1)
        self._add_placeholder(self.bottom_frame, "Pulsa 'Iniciar simulación' para ver el parámetro a lo largo del recorrido", row=1)

    def _create_ho_map_controls(self):
        control_frame = ttk.Frame(self.top_left_frame, padding=5)
        control_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(control_frame, text="GIF del mapa de handovers:").pack(side="left", padx=(0, 8))
        self.export_ho_map_button = ttk.Button(control_frame, text="Exportar GIF",
                                               command=lambda: self._save_captured_gif('ho_map'),
                                               state=tk.DISABLED)
        self.export_ho_map_button.pack(side="left")

    def _create_heatmap_controls(self):
        control_frame = ttk.Frame(self.top_right_frame, padding=5)
        control_frame.grid(row=0, column=0, sticky="ew")
        left_frame = ttk.Frame(control_frame)
        left_frame.pack(side="left", fill="x", expand=True)

        set_frame = ttk.Frame(left_frame)
        set_frame.pack(fill="x", pady=2)
        ttk.Label(set_frame, text="Set:", width=10).pack(side="left")
        self.set_dropdown = ttk.Combobox(set_frame, state="readonly", width=36)
        self.set_dropdown.pack(side="left", fill="x", expand=True, padx=5)

        param_frame = ttk.Frame(left_frame)
        param_frame.pack(fill="x", pady=2)
        ttk.Label(param_frame, text="Parámetro:", width=10).pack(side="left")
        self.param_dropdown = ttk.Combobox(param_frame, state="readonly", width=24)
        self.param_dropdown.pack(side="left", fill="x", expand=True, padx=5)

        gif_frame = ttk.Frame(left_frame)
        gif_frame.pack(fill="x", pady=(6, 0))
        ttk.Label(gif_frame, text="GIF del heatmap:", width=10).pack(side="left")
        self.export_heatmap_button = ttk.Button(gif_frame, text="Exportar GIF",
                                                command=lambda: self._save_captured_gif('heatmap'),
                                                state=tk.DISABLED)
        self.export_heatmap_button.pack(side="left", padx=5)

    def _create_bottom_controls(self):
        control_frame = ttk.Frame(self.bottom_frame, padding=5)
        control_frame.grid(row=0, column=0, sticky="ew")
        ttk.Label(control_frame, text="GIF de la gráfica inferior:").pack(side="left", padx=(0, 8))
        self.export_line_plot_button = ttk.Button(control_frame, text="Exportar GIF",
                                                  command=lambda: self._save_captured_gif('line_plot'),
                                                  state=tk.DISABLED)
        self.export_line_plot_button.pack(side="left")

    def _add_placeholder(self, parent, text, row=0):
        ph = ttk.Label(parent, text=text, style='Muted.TLabel', font=('Segoe UI', 11))
        ph.grid(row=row, column=0, sticky="")

    def _clear_frame(self, frame):
        """Elimina todos los elementos que haya dentro de un frame."""
        for widget in frame.winfo_children():
            widget.destroy()

    def _clear_plot_areas_only(self):
        """Limpia solo las gráficas y textos provisionales, no los controles."""
        for widget in self.top_left_frame.winfo_children():
            if isinstance(widget, (tk.Canvas, FigureCanvasTkAgg)) or (isinstance(widget, ttk.Label) and "Generate analysis" in widget.cget("text")):
                widget.destroy()
        
        for widget in self.top_right_frame.winfo_children():
            if isinstance(widget, (tk.Canvas, FigureCanvasTkAgg)) or (isinstance(widget, ttk.Label) and "Select Set" in widget.cget("text")):
                widget.destroy()
        
        for widget in self.bottom_frame.winfo_children():
            if isinstance(widget, (tk.Canvas, FigureCanvasTkAgg)) or (isinstance(widget, ttk.Label) and "Click 'Animate'" in widget.cget("text")):
                widget.destroy()

    # --- Lógica de zoom del mapa de handovers ---
    
    def _ho_reset_zoom(self):
        """Devuelve el mapa de handovers al zoom original."""
        if self.ho_map_ax is None or self.ho_original_limits is None:
            return
        self.ho_map_ax.set_xlim(self.ho_original_limits['x'])
        self.ho_map_ax.set_ylim(self.ho_original_limits['y'])
        self._refresh_ho_map()
    
    # --- Lógica de zoom interactivo general ---
    
    def _toggle_global_zoom_mode(self):
        """Activa o desactiva el zoom interactivo en los dos mapas."""
        
        if self.heatmap_canvas is None or self.ho_map_canvas is None:
             if self.heatmap_zoom_mode_active:
                 self._deactivate_global_zoom_mode()
             return

        self.heatmap_zoom_mode_active = not self.heatmap_zoom_mode_active
        
        if self.heatmap_zoom_mode_active:
            self.heatmap_click_cid = self.heatmap_canvas.mpl_connect(
                'button_press_event', self._on_heatmap_zoom_click)
            self.heatmap_canvas.get_tk_widget().config(cursor="crosshair")
            
            self.ho_map_click_cid = self.ho_map_canvas.mpl_connect(
                'button_press_event', self._on_ho_map_zoom_click)
            self.ho_map_canvas.get_tk_widget().config(cursor="crosshair")

            self.heatmap_zoom_button.config(text="Pan")
            
        else:
            self._deactivate_global_zoom_mode()

    def _deactivate_global_zoom_mode(self):
        """Desactiva de forma limpia el modo zoom en ambos mapas."""
        
        if self.heatmap_click_cid and self.heatmap_canvas:
            self.heatmap_canvas.mpl_disconnect(self.heatmap_click_cid)
            self.heatmap_click_cid = None
        if self.heatmap_canvas:
            try:
                self.heatmap_canvas.get_tk_widget().config(cursor="")
            except tk.TclError:
                pass 

        if self.ho_map_click_cid and self.ho_map_canvas:
            self.ho_map_canvas.mpl_disconnect(self.ho_map_click_cid)
            self.ho_map_click_cid = None
        if self.ho_map_canvas:
            try:
                self.ho_map_canvas.get_tk_widget().config(cursor="")
            except tk.TclError:
                pass

        if self.heatmap_zoom_button:
            self.heatmap_zoom_button.config(text="Zoom")
            
        self.heatmap_zoom_mode_active = False

    def _on_ho_map_zoom_click(self, event):
        """Gestiona el clic sobre el mapa de handovers para acercar o alejar."""
        
        if event.inaxes != self.ho_map_ax:
            return
            
        x_center, y_center = event.xdata, event.ydata
        
        curr_xlim = self.ho_map_ax.get_xlim()
        curr_ylim = self.ho_map_ax.get_ylim()
        x_range = curr_xlim[1] - curr_xlim[0]
        y_range = curr_ylim[1] - curr_ylim[0]
        
        if event.button == 1: # Clic izquierdo para acercar
            new_x_range = x_range * 0.8
            new_y_range = y_range * 0.8
        elif event.button == 3: # Clic derecho para alejar
            new_x_range = x_range * 1.2
            new_y_range = y_range * 1.2
        else:
            return 

        self.ho_map_ax.set_xlim([x_center - new_x_range / 2, x_center + new_x_range / 2])
        self.ho_map_ax.set_ylim([y_center - new_y_range / 2, y_center + new_y_range / 2])
        
        self._refresh_ho_map()
        
    def _on_heatmap_zoom_click(self, event):
        """Gestiona el clic sobre el heatmap para acercar o alejar."""
        
        if event.inaxes != self.heatmap_ax:
            return
            
        x_center, y_center = event.xdata, event.ydata
        
        curr_xlim = self.heatmap_ax.get_xlim()
        curr_ylim = self.heatmap_ax.get_ylim()
        x_range = curr_xlim[1] - curr_xlim[0]
        y_range = curr_ylim[1] - curr_ylim[0]
        
        if event.button == 1: # Clic izquierdo para acercar
            new_x_range = x_range * 0.8
            new_y_range = y_range * 0.8
        elif event.button == 3: # Clic derecho para alejar
            new_x_range = x_range * 1.2
            new_y_range = y_range * 1.2
        else:
            return

        self.heatmap_ax.set_xlim([x_center - new_x_range / 2, x_center + new_x_range / 2])
        self.heatmap_ax.set_ylim([y_center - new_y_range / 2, y_center + new_y_range / 2])
        
        self._refresh_heatmap()
    
    def _heatmap_reset_zoom(self):
        """Devuelve el heatmap al zoom original."""
        
        if self.heatmap_zoom_mode_active:
            self._toggle_global_zoom_mode()
            
        if self.heatmap_ax is None or self.heatmap_original_limits is None:
            return
        self.heatmap_ax.set_xlim(self.heatmap_original_limits['x'])
        self.heatmap_ax.set_ylim(self.heatmap_original_limits['y'])
        self._refresh_heatmap()
    
    def _refresh_ho_map(self):
        """Refrescar handover map with proper zoom level."""
        if self.ho_map_ax is None:
            return
            
        self.ho_map_ax.images.clear()
        
        xlim = self.ho_map_ax.get_xlim()
        width_deg = xlim[1] - xlim[0]
        
        if width_deg < 0.003: zoom = 19
        elif width_deg < 0.006: zoom = 18
        elif width_deg < 0.012: zoom = 17
        elif width_deg < 0.025: zoom = 16
        elif width_deg < 0.05: zoom = 15
        elif width_deg < 0.1: zoom = 14
        elif width_deg < 0.2: zoom = 13
        else: zoom = 12

        add_sharp_basemap(self.ho_map_ax, zoom)
        
        self.ho_map_canvas.draw_idle()

    def _refresh_heatmap(self):
        """Refrescar heatmap with proper zoom level."""
        if self.heatmap_ax is None:
            return
            
        self.heatmap_ax.images.clear()
        
        xlim = self.heatmap_ax.get_xlim()
        width_deg = xlim[1] - xlim[0]
        
        if width_deg < 0.003: zoom = 19
        elif width_deg < 0.006: zoom = 18
        elif width_deg < 0.012: zoom = 17
        elif width_deg < 0.025: zoom = 16
        elif width_deg < 0.05: zoom = 15
        elif width_deg < 0.1: zoom = 14
        elif width_deg < 0.2: zoom = 13
        else: zoom = 12

        add_sharp_basemap(self.heatmap_ax, zoom)
        
        self.heatmap_canvas.draw_idle()

    # --- Lógica del panel de control ---
            
    def _update_set_selection(self):
        try:
            new_num_sets = int(self.num_sets_spinbox.get()) 
            if new_num_sets < 1 or new_num_sets > 12:
                messagebox.showwarning("Invalid Input", "Number of sets must be between 1 and 12")
                self.num_sets_spinbox.set(str(self.num_sets)) 
                return
            if new_num_sets != self.num_sets:
                self.num_sets = new_num_sets
                self._create_set_frames()
        except ValueError:
            messagebox.showwarning("Invalid Input", "Please enter a valid number")
            self.num_sets_spinbox.set(str(self.num_sets))
            return
        self._validate_inputs()

    def _create_set_frames(self):
        for frame in self.set_frames:
            frame.destroy()
        self.set_frames.clear()
        for i in range(self.num_sets):
            set_frame = ttk.Frame(self.sets_container)
            set_frame.pack(fill="x", pady=(0, 10))
            if i > 0:
                separator = ttk.Separator(set_frame, orient='horizontal')
                separator.pack(fill="x", pady=(0, 8))
            ttk.Label(set_frame, text=f"Test Set {i+1}", font=('Arial', 9, 'bold')).pack(anchor="w", pady=(0, 5))
            def add_combo_row(parent, label_text):
                row = ttk.Frame(parent)
                row.pack(fill='x', pady=2)
                ttk.Label(row, text=label_text, width=10).pack(side='left')
                combo = ttk.Combobox(row, state='readonly')
                combo.pack(side='left', fill='x', expand=True, padx=(5, 0))
                combo.bind('<<ComboboxSelected>>', self._on_combo_selected)
                return combo
            set_frame.tester_combo = add_combo_row(set_frame, 'Tester:')
            set_frame.operator_combo = add_combo_row(set_frame, 'Operator:')
            set_frame.date_combo = add_combo_row(set_frame, 'Date:')
            set_frame.tech_combo = add_combo_row(set_frame, 'Tech:')
            self.set_frames.append(set_frame)
        if self.folder_path:
            self._update_combobox_values()
    def _update_combobox_values(self):
        if not self.set_frames:
            return
        testers = sorted({t[0] for t in self.available_tests})
        for set_frame in self.set_frames:
            self._refresh_set_frame_options(set_frame, testers)

    def _refresh_set_frame_options(self, set_frame, testers=None):
        if testers is None:
            testers = sorted({t[0] for t in self.available_tests})

        current_tester = set_frame.tester_combo.get().strip()
        current_operator = set_frame.operator_combo.get().strip()
        current_date = set_frame.date_combo.get().strip()
        current_tech = set_frame.tech_combo.get().strip()

        set_frame.tester_combo['values'] = testers
        if current_tester not in testers:
            current_tester = ''
            set_frame.tester_combo.set('')

        candidate_tests = [t for t in self.available_tests if (not current_tester or t[0] == current_tester)]
        operators = sorted({t[1] for t in candidate_tests})
        set_frame.operator_combo['values'] = operators
        if current_operator not in operators:
            current_operator = ''
            set_frame.operator_combo.set('')

        candidate_tests = [t for t in candidate_tests if (not current_operator or t[1] == current_operator)]
        dates = sorted({t[2] for t in candidate_tests})
        set_frame.date_combo['values'] = dates
        if current_date not in dates:
            current_date = ''
            set_frame.date_combo.set('')

        candidate_tests = [t for t in candidate_tests if (not current_date or t[2] == current_date)]
        techs = sorted({t[3] for t in candidate_tests})
        set_frame.tech_combo['values'] = techs
        if current_tech not in techs:
            set_frame.tech_combo.set('')

        for combo, values in ((set_frame.tester_combo, testers), (set_frame.operator_combo, operators), (set_frame.date_combo, dates), (set_frame.tech_combo, techs)):
            if len(values) == 1 and not combo.get().strip():
                combo.set(values[0])

    def _on_combo_selected(self, event):
        widget = event.widget
        for set_frame in self.set_frames:
            if widget in (set_frame.tester_combo, set_frame.operator_combo, set_frame.date_combo, set_frame.tech_combo):
                self._refresh_set_frame_options(set_frame)
                break
        self.after(10, self._validate_inputs)

    def _get_combo_values(self):
        sets_params = []
        for set_frame in self.set_frames:
            tester = set_frame.tester_combo.get().strip()
            operator = set_frame.operator_combo.get().strip()
            date = set_frame.date_combo.get().strip()
            technology = set_frame.tech_combo.get().strip()
            if tester and operator and date and technology:
                sets_params.append({'tester': tester, 'operator': operator, 'date': date, 'technology': technology})
        return sets_params
    def _validate_inputs(self):
        sets_params = self._get_combo_values()
        if not self.folder_path:
            self.status_label.config(text="Estado: selecciona primero una carpeta de datos.", foreground="#ff7b72")
            self.start_button.config(state=tk.DISABLED)
            return
        if self.num_sets == 0:
            self.status_label.config(text="Estado: indica el número de sets y pulsa 'Aplicar'.", foreground="#ff7b72")
            self.start_button.config(state=tk.DISABLED)
            return
        if len(sets_params) != self.num_sets:
            self.status_label.config(text=f"Estado: completa los {self.num_sets} sets seleccionados.", foreground="#ff7b72")
            self.start_button.config(state=tk.DISABLED)
            return
        for i, params in enumerate(sets_params):
            set_combo = (params['tester'], params['operator'], params['date'], params['technology'])
            if set_combo not in self.available_tests:
                self.status_label.config(text=f"Estado: la combinación del set {i+1} no existe en la carpeta.", foreground="#ff7b72")
                self.start_button.config(state=tk.DISABLED)
                return
        set_combos = [(p['tester'], p['operator'], p['date'], p['technology']) for p in sets_params]
        if len(set_combos) != len(set(set_combos)):
            self.status_label.config(text="Estado: no se permiten sets duplicados.", foreground="#ff7b72")
            self.start_button.config(state=tk.DISABLED)
        else:
            self.status_label.config(text=f"Estado: listo para analizar {self.num_sets} set(s).", foreground="#7ee787")
            self.start_button.config(state=tk.NORMAL)
    def _update_available_tests(self):
        self.available_tests.clear()
        if self.detected_listbox is not None:
            self.detected_listbox.delete(0, tk.END)
        if not self.folder_path or not os.path.isdir(self.folder_path):
            self.status_label.config(text='Estado: carpeta no válida.', foreground='#ffb86c')
            return
        for filename in os.listdir(self.folder_path):
            meta = parse_campaign_filename(filename)
            if not meta:
                continue
            self.available_tests.add((meta['tester'], meta['operator'], meta['date'], meta['technology']))
        self._update_combobox_values()
        for set_frame in self.set_frames:
            if set_frame.tester_combo.get().strip() and set_frame.operator_combo.get().strip() and set_frame.date_combo.get().strip() and set_frame.tech_combo.get().strip():
                continue
            set_frame.tester_combo.set('')
            set_frame.operator_combo.set('')
            set_frame.date_combo.set('')
            set_frame.tech_combo.set('')
        for tester, operator, date, technology in sorted(self.available_tests):
            line = f"{tester} | {operator} | {date} | {technology}"
            if self.detected_listbox is not None:
                self.detected_listbox.insert(tk.END, line)
        if self.available_tests and len(self.available_tests) == 1:
            tester, operator, date, technology = next(iter(self.available_tests))
            if self.set_frames:
                self.set_frames[0].tester_combo.set(tester)
                self.set_frames[0].operator_combo.set(operator)
                self.set_frames[0].date_combo.set(date)
                self.set_frames[0].tech_combo.set(technology)
        self._validate_inputs()
        if self.available_tests:
            self.status_label.config(text=f"Estado: carpeta escaneada correctamente. {len(self.available_tests)} campaña(s) detectada(s).", foreground='#7ee787')
        else:
            self.status_label.config(text='Estado: no se han encontrado campañas válidas.', foreground='#ff7b72')

    def _browse_folder(self):
        new_folder_path = filedialog.askdirectory(title='Selecciona la carpeta con archivos de Drive Test')
        if new_folder_path:
            self.folder_path = new_folder_path
            self.folder_path_label.config(text=self.folder_path, foreground='#edf2f7')
            self.available_tests.clear()
            if self.detected_listbox is not None:
                self.detected_listbox.delete(0, tk.END)
            for set_frame in self.set_frames:
                set_frame.tester_combo.set('')
                set_frame.operator_combo.set('')
                set_frame.date_combo.set('')
                set_frame.tech_combo.set('')
            self._update_available_tests()

    def _start_process(self):
        self.status_label.config(text='Estado: procesando datos...', foreground='#ffb86c')
        self.start_button.config(state=tk.DISABLED)
        self.update_idletasks()
        if self.animation:
            self._on_stop_click(clear_plots=False)
        sets_params = self._get_combo_values()
        merge_tolerance = DEFAULT_MERGE_TOLERANCE_MINUTES
        try:
            self.analysis_processor = DriveTestAnalysis(self.folder_path, sets_params, merge_tolerance_minutes=merge_tolerance)
            if not self.analysis_processor.run_analysis():
                self.start_button.config(state=tk.NORMAL)
                self.status_label.config(text='Estado: fallo en el procesado. Revisa el formato.', foreground='#ff7b72')
                return
            self._plot_handover_analysis()
            if self.analysis_processor.data:
                first_label = list(self.analysis_processor.data.keys())[0]
                first_param = self.analysis_processor.get_available_parameters()[0] if self.analysis_processor.get_available_parameters() else 'RSRP'
                self._plot_heatmap(first_label, first_param)
                self._plot_line_graph(first_param)
            self._update_animation_controls()
            self._update_ai_controls()
            self._update_summary_cards(sets_params)
            self.generate_stats_button.config(state=tk.NORMAL)
            self.status_label.config(text='Estado: análisis completado. Listo para animar.', foreground='#7ee787')
            self.start_button.config(state=tk.NORMAL)
        except Exception as exc:
            self.start_button.config(state=tk.NORMAL)
            self.status_label.config(text=f'Estado: error: {exc}', foreground='#ff7b72')
            messagebox.showerror('Analysis Error', str(exc))

    def _update_summary_cards(self, sets_params):
        if not self.analysis_processor or not self.analysis_processor.is_ready:
            return
        labels = []
        for params in sets_params:
            labels.append(f"{params['tester']} | {params['operator']} | {params['date']} | {params['technology']}")
        self.current_campaign_var.set('  vs  '.join(labels))
        combined = self.analysis_processor.combined_data.copy()
        samples = len(combined)
        speedtests = 0
        if 'DOWNLOAD_SPEED' in combined.columns:
            speedtests = int(combined['DOWNLOAD_SPEED'].notna().sum())
        handovers = len(self.analysis_processor.get_handover_data()) if self.analysis_processor else 0
        distance_km = 0.0
        signal_avg = np.nan
        if {'LAT', 'LON'}.issubset(combined.columns):
            coords = combined[['LAT', 'LON']].dropna().reset_index(drop=True)
            if len(coords) > 1:
                lat1 = np.radians(coords['LAT'].iloc[:-1].to_numpy())
                lon1 = np.radians(coords['LON'].iloc[:-1].to_numpy())
                lat2 = np.radians(coords['LAT'].iloc[1:].to_numpy())
                lon2 = np.radians(coords['LON'].iloc[1:].to_numpy())
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                distance_km = float((6371000.0 * c).sum() / 1000.0)
        signal_col = None
        for candidate in ('RSRP', 'RSRP/RSCP'):
            if candidate in combined.columns:
                signal_col = candidate
                break
        if signal_col:
            signal_avg = pd.to_numeric(combined[signal_col], errors='coerce').mean()
        self.metric_vars['samples'].set(str(samples))
        self.metric_vars['speedtests'].set(str(speedtests))
        self.metric_vars['handovers'].set(str(handovers))
        self.metric_vars['distance'].set(f"{distance_km:.2f}")
        self.metric_vars['signal'].set('-' if pd.isna(signal_avg) else f"{signal_avg:.2f}")

    def _update_animation_controls(self):
        """Rellena los desplegables con los sets y parámetros cargados."""
        if not self.analysis_processor or not self.analysis_processor.is_ready:
            return
            
        set_labels = list(self.analysis_processor.data.keys())
        self.set_dropdown['values'] = set_labels
        if set_labels:
            self.set_dropdown.current(0)
            
        # Relleno el desplegable mostrando también las unidades
        self.raw_param_list = self.analysis_processor.get_available_parameters()
        display_params = [get_param_with_unit(p) for p in self.raw_param_list]
        self.param_dropdown['values'] = display_params
        
        if display_params:
            self.param_dropdown.current(0)
            
        if set_labels and display_params:
            self.set_dropdown.config(state="readonly")
            self.param_dropdown.config(state="readonly")
            self.animate_button.config(state=tk.NORMAL)
        else:
            self.animate_button.config(state=tk.DISABLED)
            if not display_params:
                messagebox.showwarning("Missing Data", "No animation parameters (RSRP, RSRQ, etc.) found in the data.")

    # Inserto las gráficas usando grid y posición fila/columna
    def _embed_plot(self, fig, frame, row=0, col=0):
        """Inserta una gráfica dentro del grid de un frame concreto."""
        # Busco y elimino el canvas anterior de esa celda
        for widget in frame.grid_slaves(row=row, column=col):
            if isinstance(widget, (tk.Canvas, FigureCanvasTkAgg)):
                widget.destroy()
                
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=row, column=col, sticky="nsew") # Uso grid para colocar la gráfica
        canvas.draw()
        return canvas


    def _compact_map_axes(self, ax, x_ticks=5, y_ticks=5):
        """Ajuste pequeño para que los números de los mapas no se monten entre sí."""
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_ticks))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_ticks))
        ax.tick_params(axis='both', which='major', labelsize=7, pad=1)
        ax.xaxis.get_offset_text().set_fontsize(7)
        ax.yaxis.get_offset_text().set_fontsize(7)
        ax.grid(False)

    def _centered_map_limits(self, lon_values, lat_values, pad_ratio=0.10, min_pad=0.00018):
        """
        Calcula límites centrados en la ruta.
        Lo hago aparte porque contextily a veces mete el tile completo y deja
        el recorrido desplazado dentro del mapa. Con estos límites, la ruta
        queda centrada y con un poco de aire por los cuatro lados.
        """
        lon = pd.to_numeric(pd.Series(lon_values), errors='coerce').dropna()
        lat = pd.to_numeric(pd.Series(lat_values), errors='coerce').dropna()

        if lon.empty or lat.empty:
            return None

        lon_min, lon_max = float(lon.min()), float(lon.max())
        lat_min, lat_max = float(lat.min()), float(lat.max())

        lon_center = (lon_min + lon_max) / 2.0
        lat_center = (lat_min + lat_max) / 2.0

        lon_half = max((lon_max - lon_min) / 2.0 * (1.0 + 2.0 * pad_ratio), min_pad)
        lat_half = max((lat_max - lat_min) / 2.0 * (1.0 + 2.0 * pad_ratio), min_pad)

        return {
            'x': [lon_center - lon_half, lon_center + lon_half],
            'y': [lat_center - lat_half, lat_center + lat_half]
        }

    def _apply_centered_limits_after_basemap(self, ax, limits):
        """
        Reaplico los límites después de añadir OSM.
        Si no se hace esto, el mapa base puede abrir demasiado el encuadre.
        """
        if limits is None:
            return
        ax.set_xlim(limits['x'])
        ax.set_ylim(limits['y'])
        ax.set_aspect('auto')

    def _compact_line_axes(self, ax):
        """Ajuste de la gráfica inferior para que entren título, ejes y leyenda."""
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', which='major', labelsize=8, pad=2)

    # --- Métodos de dibujo de gráficas ---

    def _plot_handover_analysis(self):
        """Dibuja el análisis de handovers en la zona superior izquierda."""
        for widget in self.top_left_frame.winfo_children():
            if isinstance(widget, (tk.Canvas, FigureCanvasTkAgg)) or (isinstance(widget, ttk.Label) and "Generate analysis" in widget.cget("text")):
                widget.destroy()

        if not self.analysis_processor or not self.analysis_processor.is_ready:
            self._add_placeholder(self.top_left_frame, "No data available for handover analysis.", row=1)
            return

        ho_data = self.analysis_processor.get_handover_data()
        
        if 'HANDOVER_FLAG' not in self.analysis_processor.combined_data.columns:
            self._add_placeholder(self.top_left_frame, "Cannot plot Handovers: 'XCI' column missing.", row=1)
            return

        self.ho_map_fig, self.ho_map_ax = plt.subplots(figsize=(6, 4), dpi=130)
        
        test_labels = list(self.analysis_processor.data.keys())
        color_map = self.analysis_processor.color_map
        total_hos = 0

        for label in test_labels:
            color = color_map.get(label)
            path_df = self.analysis_processor.data[label]
            self.ho_map_ax.plot(path_df['LON'], path_df['LAT'], color=color, 
                                alpha=0.7, linewidth=2, label=label)
            
            ho_subset = ho_data[ho_data['Test_Label'] == label]
            if not ho_subset.empty:
                self.ho_map_ax.scatter(ho_subset['LON'], ho_subset['LAT'], color=color, 
                                       label=f'HO ({label})', s=80, alpha=0.9, 
                                       edgecolors='black', linewidth=1.5, marker='D')
                total_hos += len(ho_subset)
        
        lon_data = self.analysis_processor.combined_data['LON']
        lat_data = self.analysis_processor.combined_data['LAT']

        # Encuadre centrado en la ruta, con un poco de margen por arriba, abajo,
        # derecha e izquierda. Así evito que el recorrido quede pegado a un borde.
        self.ho_original_limits = self._centered_map_limits(
            lon_data, lat_data, pad_ratio=0.10, min_pad=0.00018
        )
        
        self.ho_map_ax.set_xlim(self.ho_original_limits['x'])
        self.ho_map_ax.set_ylim(self.ho_original_limits['y'])

        width_deg = self.ho_original_limits['x'][1] - self.ho_original_limits['x'][0]
        if width_deg < 0.003: initial_zoom = 19
        elif width_deg < 0.006: initial_zoom = 18
        elif width_deg < 0.012: initial_zoom = 17
        elif width_deg < 0.025: initial_zoom = 16
        elif width_deg < 0.05: initial_zoom = 15
        elif width_deg < 0.1: initial_zoom = 14
        else: initial_zoom = 13

        add_sharp_basemap(self.ho_map_ax, initial_zoom)
        self._apply_centered_limits_after_basemap(self.ho_map_ax, self.ho_original_limits)

        self.ho_map_ax.set_title(f"Handover Events (Total: {total_hos})", fontsize=10, pad=4)
        self.ho_map_ax.set_xlabel("Longitude", fontsize=8, labelpad=2)
        self.ho_map_ax.set_ylabel("Latitude", fontsize=8, labelpad=2)
        self._compact_map_axes(self.ho_map_ax, x_ticks=5, y_ticks=4)
        self.ho_map_ax.legend(fontsize=6, loc='upper right', framealpha=0.85)
        self.ho_map_fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.18)
        
        # Inserto la gráfica en la fila 1
        self.ho_map_canvas = self._embed_plot(self.ho_map_fig, self.top_left_frame, row=1)
        
        if self.heatmap_zoom_mode_active:
            self.ho_map_click_cid = self.ho_map_canvas.mpl_connect(
                'button_press_event', self._on_ho_map_zoom_click)
            self.ho_map_canvas.get_tk_widget().config(cursor="crosshair")

    def _plot_cell_towers(self, selected_set_label):
        """Dibuja solo los iconos de las estaciones base en el heatmap."""
        if not self.analysis_processor or selected_set_label not in self.analysis_processor.cell_towers:
            return
            
        cell_towers = self.analysis_processor.cell_towers[selected_set_label]
        
        for symbol in self.cell_symbols.values():
            symbol.remove()
        for text in self.cell_labels.values():
            text.remove()
        
        self.cell_symbols.clear()
        self.cell_labels.clear()
        
        for cell_id, tower_info in cell_towers.items():
            # Dejo únicamente el símbolo de estación base. Quito el texto negro
            # porque se solapaba con la ruta y no aportaba información clara.
            symbol = self.heatmap_ax.text(
                tower_info['lon'], tower_info['lat'],
                '⅄', fontsize=23, ha='center', va='center',
                color=tower_info['color'], alpha=0.95
            )
            self.cell_symbols[cell_id] = symbol

    def _plot_heatmap(self, selected_set_label, param):
        """Dibuja el heatmap del parámetro en la zona superior derecha."""
        for widget in self.top_right_frame.winfo_children():
            if isinstance(widget, (tk.Canvas, FigureCanvasTkAgg)) or (isinstance(widget, ttk.Label) and "Select Set" in widget.cget("text")):
                widget.destroy()
        
        df = self.analysis_processor.data[selected_set_label]
        
        if param not in df.columns or df[param].isnull().all():
            self._add_placeholder(self.top_right_frame, f"No data for '{param}' in this set.", row=1)
            return

        self.heatmap_fig, self.heatmap_ax = plt.subplots(figsize=(6, 4), dpi=130)
        
        # Obtengo el nombre visible con unidades
        display_param = get_param_with_unit(param)
        
        if param == 'TECHNOLOGY':
            # Tratamiento especial cuando el parámetro es TECHNOLOGY
            for tech, color in TECHNOLOGY_COLORS.items():
                tech_data = df[df['TECHNOLOGY'] == tech]
                if not tech_data.empty:
                    self.heatmap_ax.scatter(tech_data['LON'], tech_data['LAT'], s=30, 
                                          color=color, label=tech, alpha=0.8)
            
            # Añado la leyenda de colores por tecnología
            self.heatmap_ax.legend(fontsize=8, loc='upper right')
            
        else:
            # Tratamiento normal para parámetros numéricos
            sc = self.heatmap_ax.scatter(df['LON'], df['LAT'], s=30, 
                                     c=df[param], cmap='viridis', 
                                     vmin=df[param].min(), vmax=df[param].max(),
                                     alpha=0.8)
            
            # Uso el nombre visible en la barra de color
            cbar = self.heatmap_fig.colorbar(sc, ax=self.heatmap_ax, label=display_param)
            cbar.ax.tick_params(labelsize=8)
        
        self._plot_cell_towers(selected_set_label)
        
        # Encuadre centrado del heatmap. Mantengo un margen pequeño para que
        # la ruta no quede pegada al borde, pero sin enseñar tanto alrededor.
        self.heatmap_original_limits = self._centered_map_limits(
            df['LON'], df['LAT'], pad_ratio=0.10, min_pad=0.00015
        )
        
        self.heatmap_ax.set_xlim(self.heatmap_original_limits['x'])
        self.heatmap_ax.set_ylim(self.heatmap_original_limits['y'])

        width_deg = self.heatmap_original_limits['x'][1] - self.heatmap_original_limits['x'][0]
        if width_deg < 0.003: initial_zoom = 19
        elif width_deg < 0.006: initial_zoom = 18
        elif width_deg < 0.012: initial_zoom = 17
        elif width_deg < 0.025: initial_zoom = 16
        elif width_deg < 0.05: initial_zoom = 15
        elif width_deg < 0.1: initial_zoom = 14
        else: initial_zoom = 13

        add_sharp_basemap(self.heatmap_ax, initial_zoom)
        self._apply_centered_limits_after_basemap(self.heatmap_ax, self.heatmap_original_limits)

        operator = self.analysis_processor.operator_names.get(selected_set_label, "Unknown")
        # Título corto para que no se corte dentro del panel del heatmap.
        # El nombre completo del set ya aparece en el desplegable, así que aquí dejo solo
        # lo importante para leerlo bien: parámetro, operador y tecnología.
        label_parts = [part.strip() for part in selected_set_label.split('|')]
        tech_label = label_parts[-1] if label_parts else ""
        self.heatmap_ax.set_title(f"{display_param} · {operator} · {tech_label}", fontsize=9, pad=2)
        self.heatmap_ax.set_xlabel("Longitude", fontsize=8, labelpad=1)
        self.heatmap_ax.set_ylabel("Latitude", fontsize=8, labelpad=1)
        self.heatmap_ax.tick_params(axis='both', labelsize=8, pad=1)
        self.heatmap_fig.tight_layout(pad=0.8)
        self.heatmap_fig.subplots_adjust(top=0.90, bottom=0.16, left=0.12, right=0.88)
        
        # Inserto la gráfica en la fila 1
        self.heatmap_canvas = self._embed_plot(self.heatmap_fig, self.top_right_frame, row=1)
        
        if self.heatmap_zoom_mode_active:
            self.heatmap_click_cid = self.heatmap_canvas.mpl_connect(
                'button_press_event', self._on_heatmap_zoom_click)
            self.heatmap_canvas.get_tk_widget().config(cursor="crosshair")

    def _plot_line_graph(self, param):
        """Dibuja la gráfica inferior del parámetro con líneas de estadística."""
        # Limpio solo la gráfica, no los controles
        for widget in self.bottom_frame.winfo_children():
            if isinstance(widget, (tk.Canvas, FigureCanvasTkAgg)) or (isinstance(widget, ttk.Label) and "Click 'Animate'" in widget.cget("text")):
                widget.destroy()
        
        self.line_plot_fig, self.line_plot_ax = plt.subplots(figsize=(12, 4), dpi=100)
        
        color_map = self.analysis_processor.color_map
        all_data = self.analysis_processor.data
        self.animation_max_frames = 0
        self.stat_lines.clear()

        # Obtengo el nombre visible con unidades
        display_param = get_param_with_unit(param)

        if param == 'TECHNOLOGY':
            # Tratamiento especial cuando el parámetro es TECHNOLOGY
            for label, df in all_data.items():
                if 'TECHNOLOGY' in df.columns and not df['TECHNOLOGY'].isnull().all():
                    # Convierto la tecnología a valores numéricos para dibujarla
                    tech_numeric = df['TECHNOLOGY'].map(TECHNOLOGY_ORDER).fillna(0)
                    
                    # Creo una gráfica tipo escalón para los cambios de tecnología
                    x_values = df.index
                    line_color = color_map[label]
                    
                    # La dibujo como escalón para que se vean los saltos inmediatos
                    self.line_plot_ax.step(x_values, tech_numeric, 
                                         where='post', color=line_color, 
                                         label=label, linewidth=2, alpha=0.8)
                    
                    if len(df) > self.animation_max_frames:
                        self.animation_max_frames = len(df)
            
            # Personalizo el eje Y cuando se muestra tecnología
            self.line_plot_ax.set_yticks(list(TECHNOLOGY_ORDER.values()))
            self.line_plot_ax.set_yticklabels(list(TECHNOLOGY_ORDER.keys()))
            self.line_plot_ax.set_ylim(0.5, 4.5)  # Añado algo de margen visual
            
        else:
            # Obtengo la unidad para las etiquetas
            unit_str = PARAMETER_UNITS.get(param, "")
            label_unit_str = f" {unit_str}" if unit_str else ""
            
            # Tratamiento normal para parámetros numéricos
            for label, df in all_data.items():
                if param in df.columns and not df[param].isnull().all():
                    x_values = df.index
                    line_color = color_map[label]
                    self.line_plot_ax.plot(x_values, df[param], color=line_color, 
                                          label=label, linewidth=2, alpha=0.8)
                    if len(df) > self.animation_max_frames:
                        self.animation_max_frames = len(df)
                    
                    min_val, max_val, avg_val = self.analysis_processor.get_statistics(label, param)
                    
                    if min_val is not None and max_val is not None and avg_val is not None:
                        base_rgb = to_rgba(line_color)
                        dark_color_hls = colorsys.rgb_to_hls(base_rgb[0], base_rgb[1], base_rgb[2])
                        dark_color = colorsys.hls_to_rgb(dark_color_hls[0], max(0.3, dark_color_hls[1] * 0.7), dark_color_hls[2])
                        light_color_hls = colorsys.rgb_to_hls(base_rgb[0], base_rgb[1], base_rgb[2])
                        light_color = colorsys.hls_to_rgb(light_color_hls[0], min(0.9, light_color_hls[1] * 1.3), light_color_hls[2])
                        
                        # Añado unidades en la leyenda
                        min_line = self.line_plot_ax.axhline(y=min_val, color=dark_color, linestyle='-', 
                                                            linewidth=1.5, alpha=0.8, 
                                                            label=f'{label} Min: {min_val:.2f}{label_unit_str}')
                        max_line = self.line_plot_ax.axhline(y=max_val, color=dark_color, linestyle='-', 
                                                            linewidth=1.5, alpha=0.8, 
                                                            label=f'{label} Max: {max_val:.2f}{label_unit_str}')
                        avg_line = self.line_plot_ax.axhline(y=avg_val, color=light_color, linestyle='--', 
                                                            linewidth=1.5, alpha=0.8, 
                                                            label=f'{label} Avg: {avg_val:.2f}{label_unit_str}')
                        
                        self.stat_lines[label] = {
                            'min': min_line,
                            'max': max_line, 
                            'avg': avg_line
                        }
            
        # Uso el nombre visible en el título y en el eje Y
        self.line_plot_ax.set_title(f"{display_param} frente al recorrido", fontsize=11, pad=4)
        self.line_plot_ax.set_xlabel("Progresión del recorrido (muestras)", fontsize=9, labelpad=3)
        self.line_plot_ax.set_ylabel(display_param, fontsize=9, labelpad=3)
        self._compact_line_axes(self.line_plot_ax)
        
        handles, labels = self.line_plot_ax.get_legend_handles_labels()
        self.line_plot_ax.legend(handles, labels, fontsize=6, loc='upper right', framealpha=0.85)
        
        self.line_plot_ax.grid(True, alpha=0.25)
        self.line_plot_ax.set_xlim(0, self.animation_max_frames)
        
        if param != 'TECHNOLOGY':
            all_param_values = pd.concat([df[param] for df in all_data.values() if param in df.columns])
            if not all_param_values.empty:
                y_min, y_max = all_param_values.min(), all_param_values.max()
                y_range = y_max - y_min
                if y_range > 0:
                    self.line_plot_ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        self.line_plot_fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.24)
        
        # Inserto la gráfica en la fila 1
        self.line_plot_canvas = self._embed_plot(self.line_plot_fig, self.bottom_frame, row=1)

    # --- Métodos de animación mejorados ---
    
    def _on_animate_click(self):
        """Inicia una animación nueva o reanuda una que estaba en pausa."""
        if self.animation_paused:
            self.animation.resume()
            self.animation_paused = False
            self.animate_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            return

        self._on_stop_click(clear_plots=False)
        
        selected_set = self.set_dropdown.get()
        # Obtengo el parámetro visible y lo relaciono con el nombre interno
        selected_display_param = self.param_dropdown.get()
        
        if not selected_set or not selected_display_param:
            messagebox.showerror("Error de entrada", "Selecciona un set y un parámetro.")
            return
            
        # A partir del nombre visible saco el nombre interno, por ejemplo RSRP
        selected_param = ""
        try:
            param_index = self.param_dropdown['values'].index(selected_display_param)
            selected_param = self.raw_param_list[param_index]
        except (ValueError, AttributeError, IndexError):
             # Alternativa por si la lista interna no está preparada o el índice falla
             selected_param = selected_display_param.split(' ')[0]
             if selected_param == 'Download': selected_param = 'DOWNLOAD_SPEED'
             if selected_param == 'Upload': selected_param = 'UPLOAD_SPEED'

        if not selected_param:
             messagebox.showerror("Error de entrada", "Could not find matching parameter.")
             return

        # Para la lógica interna uso siempre el parámetro real
        self._plot_heatmap(selected_set, selected_param)
        self._plot_line_graph(selected_param)
        
        # Limpio las capturas anteriores y desactivo exportaciones
        self.capture_buffers = {'ho_map': [], 'heatmap': [], 'line_plot': []}
        self.captured_gifs = {'ho_map': [], 'heatmap': [], 'line_plot': []}
        if self.export_ho_map_button: self.export_ho_map_button.config(state=tk.DISABLED)
        if self.export_heatmap_button: self.export_heatmap_button.config(state=tk.DISABLED)
        if self.export_line_plot_button: self.export_line_plot_button.config(state=tk.DISABLED)

        self.animation_data_cache = {
            'all_dfs': self.analysis_processor.data,
            'selected_df': self.analysis_processor.data[selected_set],
            'selected_param': selected_param, # Guardo el parámetro interno
            'selected_set': selected_set
        }
        
        self.ho_map_markers = {}
        for label in self.analysis_processor.data.keys():
            marker, = self.ho_map_ax.plot([], [], 'o', color='yellow', markersize=8, 
                                         markeredgecolor='black', markeredgewidth=1.5, alpha=0.9)
            self.ho_map_markers[label] = marker
            
        self.heatmap_marker, = self.heatmap_ax.plot([], [], 'o', color='yellow', markersize=10,
                                                   markeredgecolor='black', markeredgewidth=2, alpha=1.0)
        
        self.cell_connection_line, = self.heatmap_ax.plot([], [], 'r--', linewidth=2, alpha=0.7)
        
        self.red_line = self.line_plot_ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        # Si el parámetro es TECHNOLOGY, preparo la anotación de texto
        if selected_param == 'TECHNOLOGY' and self.heatmap_ax:
             self.tech_annotation = self.heatmap_ax.text(0, 0, '', 
                                                        fontsize=8, fontweight='bold', 
                                                        ha='center', va='center', 
                                                        color='white',
                                                        transform=self.heatmap_ax.transData)
             # Al inicio queda oculto
             self.tech_annotation.set_visible(False)
        
        self.animation = animation.FuncAnimation(
            self.line_plot_fig, 
            self._update_animation_frame,
            frames=self.animation_max_frames,
            interval=150,
            blit=False,
            repeat=True
        )
        
        self.animate_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL)
        self.animation_paused = False
        
        if self.heatmap_zoom_button:
            self.heatmap_zoom_button.config(state=tk.NORMAL)
        
        # Activo el botón de captura
        if self.capture_button:
            self.capture_button.config(text="Iniciar captura", state=tk.NORMAL)


    def _on_pause_click(self):
        """Pausa la animación."""
        if self.animation and not self.animation_paused:
            self.animation.pause()
            self.animation_paused = True
            self.animate_button.config(text="Reanudar", state=tk.NORMAL)
            self.pause_button.config(state=tk.DISABLED)
            # Desactivo captura mientras está en pausa
            if self.capture_button:
                self.capture_button.config(state=tk.DISABLED)

    def _on_stop_click(self, clear_plots=False):
        """Detiene y reinicia la animación."""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            
        self._deactivate_global_zoom_mode()
        if self.heatmap_zoom_button:
            self.heatmap_zoom_button.config(state=tk.DISABLED)
            
        if self.red_line:
            self.red_line.set_xdata([0, 0])
        for marker in self.ho_map_markers.values():
            marker.set_data([], [])
        if self.heatmap_marker:
            self.heatmap_marker.set_data([], [])
        if self.cell_connection_line:
            self.cell_connection_line.set_data([], [])
            
        if hasattr(self, 'tech_annotation'):
            self.tech_annotation.set_text('')
            self.tech_annotation.set_visible(False)
            
        if self.line_plot_canvas: 
            self.line_plot_canvas.draw_idle()
        if self.ho_map_canvas: 
            self.ho_map_canvas.draw_idle()
        if self.heatmap_canvas: 
            self.heatmap_canvas.draw_idle()

        self.animation_paused = False
        self.animate_button.config(text="Iniciar simulación", state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)

        # Desactivo y reinicio los botones de captura/exportación
        self.is_capturing = False
        if self.capture_button:
            self.capture_button.config(text="Iniciar captura", state=tk.DISABLED)
        if self.export_ho_map_button:
            self.export_ho_map_button.config(state=tk.DISABLED)
        if self.export_heatmap_button:
            self.export_heatmap_button.config(state=tk.DISABLED)
        if self.export_line_plot_button:
            self.export_line_plot_button.config(state=tk.DISABLED)
        
        # Limpio los buffers
        self.capture_buffers = {'ho_map': [], 'heatmap': [], 'line_plot': []}
        self.captured_gifs = {'ho_map': [], 'heatmap': [], 'line_plot': []}
        
        if clear_plots:
            self.animation_data_cache = {} # Limpio la caché
            current_set = self.set_dropdown.get() if self.set_dropdown else ""
            current_param_display = self.param_dropdown.get() if self.param_dropdown else ""
            
            self._clear_plot_areas_only()
            
            self._add_placeholder(self.top_left_frame, "Genera el análisis para ver el mapa de handovers", row=1)
            self._add_placeholder(self.top_right_frame, "Selecciona set y parámetro y pulsa 'Iniciar animación'", row=1)
            self._add_placeholder(self.bottom_frame, "Pulsa 'Iniciar animación' para ver el parámetro a lo largo del recorrido", row=1)
            
            self.ho_map_fig = None
            self.ho_map_ax = None
            self.ho_map_canvas = None
            self.heatmap_fig = None
            self.heatmap_ax = None
            self.heatmap_canvas = None
            self.line_plot_fig = None
            self.line_plot_ax = None
            self.line_plot_canvas = None
            
            self.cell_symbols.clear()
            self.cell_labels.clear()
            self.stat_lines.clear()
            
            if self.analysis_processor and self.analysis_processor.is_ready:
                set_labels = list(self.analysis_processor.data.keys())
                self.raw_param_list = self.analysis_processor.get_available_parameters()
                display_params = [get_param_with_unit(p) for p in self.raw_param_list]

                self.set_dropdown['values'] = set_labels
                self.param_dropdown['values'] = display_params
                
                if current_set and current_set in set_labels:
                    self.set_dropdown.set(current_set)
                elif set_labels:
                    self.set_dropdown.set(set_labels[0])
                    
                if current_param_display and current_param_display in display_params:
                    self.param_dropdown.set(current_param_display)
                elif display_params:
                    self.param_dropdown.set(display_params[0])
                
                self.set_dropdown.config(state="readonly")
                self.param_dropdown.config(state="readonly")
                self.animate_button.config(state=tk.NORMAL)


    def _update_animation_frame(self, frame_index):
        """Esta función se ejecuta en cada frame de la animación."""
        artists = []
        
        if self.red_line:
            self.red_line.set_xdata([frame_index, frame_index])
            artists.append(self.red_line)
        
        for label, marker in self.ho_map_markers.items():
            df = self.animation_data_cache['all_dfs'][label]
            if frame_index < len(df):
                point = df.iloc[frame_index]
                marker.set_data([point['LON']], [point['LAT']])
                artists.append(marker)
            else:
                marker.set_data([], [])
                
        df_selected = self.animation_data_cache['selected_df']
        selected_set = self.animation_data_cache['selected_set']
        selected_param = self.animation_data_cache['selected_param']
        
        if frame_index < len(df_selected) and self.heatmap_marker:
            point = df_selected.iloc[frame_index]
            
            if selected_param == 'TECHNOLOGY':
                # Para TECHNOLOGY uso un marcador más grande y muestro la tecnología actual
                tech = point.get('TECHNOLOGY', 'UNKNOWN')
                self.heatmap_marker.set_data([point['LON']], [point['LAT']])
                self.heatmap_marker.set_markersize(15)  # Marcador más grande
                self.heatmap_marker.set_color(TECHNOLOGY_COLORS.get(tech, '#FFA500'))
                
                # Añado la anotación con la tecnología
                if hasattr(self, 'tech_annotation'):
                    self.tech_annotation.set_position((point['LON'], point['LAT']))
                    self.tech_annotation.set_text(tech)
                    self.tech_annotation.set_color('white')
                    self.tech_annotation.set_visible(True)
                    artists.append(self.tech_annotation)
                
                artists.append(self.heatmap_marker)
            else:
                # Comportamiento normal para el resto de parámetros
                self.heatmap_marker.set_data([point['LON']], [point['LAT']])
                self.heatmap_marker.set_markersize(10)  # Vuelvo al tamaño normal
                self.heatmap_marker.set_color('yellow') # Restauro el color normal
                if hasattr(self, 'tech_annotation'):
                    self.tech_annotation.set_visible(False)
                artists.append(self.heatmap_marker)
            
            current_cell = self.analysis_processor.get_current_cell(selected_set, frame_index)
            if current_cell and self.cell_connection_line:
                self.cell_connection_line.set_data(
                    [point['LON'], current_cell['lon']],
                    [point['LAT'], current_cell['lat']]
                )
                artists.append(self.cell_connection_line)
            elif self.cell_connection_line:
                self.cell_connection_line.set_data([], [])
        else:
            if self.heatmap_marker:
                self.heatmap_marker.set_data([], [])
            if self.cell_connection_line:
                self.cell_connection_line.set_data([], [])
            if hasattr(self, 'tech_annotation'):
                self.tech_annotation.set_text('')
                self.tech_annotation.set_visible(False)
            
        # Si estoy grabando, capturo el frame actual
        if self.is_capturing:
            self._capture_current_frame()

        if self.line_plot_canvas:
            self.line_plot_canvas.draw_idle()
        if self.ho_map_canvas:
            self.ho_map_canvas.draw_idle()
        if self.heatmap_canvas:
            self.heatmap_canvas.draw_idle()
            
        return artists

    # --- Elimino los métodos antiguos de exportación ---
    # --- Nuevos métodos de captura y guardado ---

    def _on_capture_click(self):
        """Gestiona el botón de iniciar/detener captura."""
        if self.is_capturing:
            # --- Parada de la captura ---
            self.is_capturing = False
            self.capture_button.config(text="Iniciar captura")
            
            # Proceso los frames capturados desde los buffers
            self.captured_gifs['ho_map'] = self.capture_buffers['ho_map'].copy()
            self.captured_gifs['heatmap'] = self.capture_buffers['heatmap'].copy()
            self.captured_gifs['line_plot'] = self.capture_buffers['line_plot'].copy()
            
            # Limpio los buffers
            self.capture_buffers = {'ho_map': [], 'heatmap': [], 'line_plot': []}

            # Activo exportación si hay frames capturados
            if self.captured_gifs['ho_map'] and PILLOW_AVAILABLE:
                self.export_ho_map_button.config(state=tk.NORMAL)
            if self.captured_gifs['heatmap'] and PILLOW_AVAILABLE:
                self.export_heatmap_button.config(state=tk.NORMAL)
            if self.captured_gifs['line_plot'] and PILLOW_AVAILABLE:
                self.export_line_plot_button.config(state=tk.NORMAL)
                
            total_frames = len(self.captured_gifs.get('ho_map', [])) # Uso la longitud de uno de los buffers como referencia
            if total_frames == 0:
                messagebox.showinfo("Capture", "Capture stopped. No frames were recorded.")
            else:
                messagebox.showinfo("Capture Complete", f"Capture finished. {total_frames} frames recorded.\n"
                                    "You can now export the individual GIFs.")

        else:
            # --- Inicio de la captura ---
            if not PILLOW_AVAILABLE:
                messagebox.showerror("Capture Error", 
                                     "Pillow library not found. Please install it:\n\npip install Pillow")
                return
            
            self.is_capturing = True
            self.capture_button.config(text="Detener captura")
            
            # Limpio buffers y GIFs anteriores
            self.capture_buffers = {'ho_map': [], 'heatmap': [], 'line_plot': []}
            self.captured_gifs = {'ho_map': [], 'heatmap': [], 'line_plot': []}
            
            # Desactivo exportar mientras se está capturando
            if self.export_ho_map_button: self.export_ho_map_button.config(state=tk.DISABLED)
            if self.export_heatmap_button: self.export_heatmap_button.config(state=tk.DISABLED)
            if self.export_line_plot_button: self.export_line_plot_button.config(state=tk.DISABLED)

    def _capture_current_frame(self):
        """Captura el frame actual de las tres gráficas y lo guarda en buffers."""
        try:
            if self.ho_map_canvas:
                # print_to_buffer devuelve una tupla, así que la desempaqueto bien
                buf, (w, h) = self.ho_map_canvas.print_to_buffer()
                img = Image.frombytes("RGBA", (w, h), buf)
                self.capture_buffers['ho_map'].append(img)

            if self.heatmap_canvas:
                # print_to_buffer devuelve una tupla, así que la desempaqueto bien
                buf, (w, h) = self.heatmap_canvas.print_to_buffer()
                img = Image.frombytes("RGBA", (w, h), buf)
                self.capture_buffers['heatmap'].append(img)

            if self.line_plot_canvas:
                # print_to_buffer devuelve una tupla, así que la desempaqueto bien
                buf, (w, h) = self.line_plot_canvas.print_to_buffer()
                img = Image.frombytes("RGBA", (w, h), buf)
                self.capture_buffers['line_plot'].append(img)
                
        except Exception as e:
            print(f"Error capturing frame: {e}")
            # Detengo la captura para evitar una cascada de errores
            if self.is_capturing:
                self.is_capturing = False
                self.capture_button.config(text="Iniciar captura")
                messagebox.showerror("Capture Error", f"An error occurred during frame capture: {e}\nCapture has been stopped.")

    def _save_captured_gif(self, plot_type):
        """Guarda los frames capturados de una gráfica concreta en un GIF."""
        if not PILLOW_AVAILABLE:
            messagebox.showerror("Export Error", 
                                 "Pillow library not found. Please install it:\n\npip install Pillow")
            return
            
        plot_name = plot_type.replace('_', ' ')
        if not self.captured_gifs.get(plot_type):
            messagebox.showerror("Export Error", f"No captured frames found for the {plot_name}.\n\n"
                                 "Please run an animation, click 'Iniciar captura', "
                                 "wait, click 'Detener Capture', and then try exporting.")
            return

        save_path = filedialog.asksaveasfilename(
            title=f"Save Captured {plot_name} Animation",
            defaultextension=".gif",
            filetypes=[("GIF Animation", "*.gif")]
        )
        
        if not save_path:
            return # El usuario ha cancelado

        try:
            frames = self.captured_gifs[plot_type]
            # Duración en ms por frame; 100 ms equivale a 10 fps
            frames[0].save(
                save_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0
            )
            messagebox.showinfo("Export Complete", f"Captured {plot_name} animation saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to save GIF: {e}")


    # --- Funciones de frame para guardar la animación ---
    # --- Se han quitado los métodos antiguos de guardado/exportación ---

# --- Ejecución principal ---

if __name__ == '__main__':
    try:
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TLabel', font=('Arial', 9))
        style.configure('TButton', font=('Arial', 9, 'bold'), padding=6)
        style.configure('Accent.TButton', background='#4f46e5', foreground='white')
        style.map('Accent.TButton', 
                  background=[('active', '#3730a3'), ('!disabled', '#4f46e5')], 
                  foreground=[('!disabled', 'white')])
    except tk.TclError:
        pass

    app = DriveTestGUI()
    app.mainloop()