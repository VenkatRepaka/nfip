# %% [markdown]
# # DSC 411/511 Project
# # Notebook 00: Data Acquisition and Merging
#
# **Student:** Venkat Repaka
#
# ---
#
# This notebook collects every dataset the project uses and joins them into one table.
#
# **Why do we need more than NFIP?** My first research questions use only the NFIP claims file.
# The claims file describes the *building* and the *policy* very well. It says almost nothing
# about the *flood itself*: how much rain fell, how close a hurricane came, or how rich the
# neighbourhood is. Early models on NFIP alone gave weak answers. So we add four outside sources:
#
# | Source | What it adds | Join key |
# |---|---|---|
# | OpenFEMA NFIP Redacted Claims v3 | building, policy, damage, payments | (base table) |
# | PRISM daily precipitation, 4 km | rain at the claim location, on and before the loss date | latitude, longitude, dateOfLoss |
# | NOAA HURDAT2 Atlantic best track | nearest hurricane track, distance, wind, category | latitude, longitude, dateOfLoss |
# | Natural Earth coastline | distance from the claim to the ocean coast (a storm-surge proxy) | latitude, longitude |
# | Census ACS 5-year 2019-2023 | income, home value, owners, poverty, race, building age | block group, tract, county |
# | FEMA National Risk Index v1.20 | county coastal / inland flood risk, social vulnerability | county FIPS |
#
# Every download is **cached** in `data_raw/`. If a file is already there, it is not downloaded again.
# This makes the whole project easy to rerun.

# %% [markdown]
# ---
# # Section 0: Setup

# %%
import os
import io
import json
import time
import zipfile
import datetime
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import pyarrow.dataset as pa_dataset
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
pd.set_option('display.width', 160)

PROJECT_DIR = Path('/Users/home/workspace/School/DataMining/nfip_claims')
RAW_DIR = PROJECT_DIR / 'data_raw'
PROCESSED_DIR = PROJECT_DIR / 'data_processed'
TABLE_DIR = PROJECT_DIR / 'tables'
for folder in [RAW_DIR, PROCESSED_DIR, TABLE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

STUDY_START = datetime.date(2020, 1, 1)
STUDY_END = datetime.date(2025, 12, 31)
print('Study window:', STUDY_START, 'to', STUDY_END)


# %%
def download_if_missing(url, destination, min_size_bytes=1000, timeout=600):
    """Download a file only if we do not already have it. Returns the local path."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size >= min_size_bytes:
        return destination
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        temporary_path = destination.with_suffix(destination.suffix + '.part')
        with open(temporary_path, 'wb') as file_handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                file_handle.write(chunk)
        temporary_path.rename(destination)
    return destination


# %% [markdown]
# ## 0.1 The data snapshot (frozen)
#
# OpenFEMA updates the claims file almost every week. A new download gives slightly different numbers (between two
# downloads during this project the 2024-2025 test set grew from 105,137 to 105,482 claims). To make every result
# reproducible, we **freeze** the input files: the cell below records the download date, size and SHA-256 fingerprint
# of each raw file in `tables/00_data_snapshot.json`, plus the version of HURDAT2, NRI and ACS.
#
# - If the snapshot file does not exist yet, it is written now (from the files already on disk; nothing is downloaded).
# - If it exists, the cell checks that the files on disk are still the same snapshot and prints a warning if one changed.
#
# **All results in this project use the snapshot printed below.**

# %%
import hashlib
import pyarrow.parquet as pa_parquet

SNAPSHOT_PATH = TABLE_DIR / '00_data_snapshot.json'


def sha256_of_file(path, chunk_size=1 << 24):
    """SHA-256 fingerprint of a file, read in 16 MB pieces (works for the 2.5 GB policies file)."""
    digest = hashlib.sha256()
    with open(path, 'rb') as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path, version_note):
    """Name, version, size, download time (file modification time) and SHA-256 of one raw file."""
    path = Path(path)
    info = {'file': str(path.relative_to(PROJECT_DIR)), 'version': version_note, 'present': path.exists()}
    if path.exists():
        stat = path.stat()
        info.update({'size_bytes': stat.st_size, 'size_mb': round(stat.st_size / 1e6, 1),
                     'downloaded_local_time': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds'),
                     'sha256': sha256_of_file(path)})
    return info


def build_data_snapshot():
    claims_file = RAW_DIR / 'NfipClaimsV3.parquet'
    policies_file = RAW_DIR / 'policies' / 'NfipPoliciesV3.parquet'
    snapshot = {'snapshot_written_local_time': datetime.datetime.now().isoformat(timespec='seconds'), 'files': []}
    claims_info = describe_file(claims_file, 'OpenFEMA FIMA NFIP Redacted Claims v3, bulk parquet')
    if claims_file.exists():
        claims_info['rows'] = pa_parquet.ParquetFile(claims_file).metadata.num_rows
        claims_info['data_as_of'] = str(pa_dataset.dataset(claims_file).to_table(columns=['asOfDate'])
                                        .column('asOfDate').to_pandas().max())
    snapshot['files'].append(claims_info)
    policies_info = describe_file(policies_file, 'OpenFEMA FIMA NFIP Redacted Policies v3, bulk parquet (notebook 09)')
    if policies_file.exists():
        policies_info['rows'] = pa_parquet.ParquetFile(policies_file).metadata.num_rows
    snapshot['files'].append(policies_info)
    snapshot['files'].append(describe_file(RAW_DIR / 'nfip_fields.json', 'OpenFEMA data dictionary for NfipClaims v3'))
    snapshot['files'].append(describe_file(RAW_DIR / 'hurdat' / 'hurdat2-1851-2025.txt',
                                           'NOAA NHC HURDAT2 Atlantic 1851-2025, release file hurdat2-1851-2025-092326.txt'))
    snapshot['files'].append(describe_file(RAW_DIR / 'nri' / 'NRI_Table_Counties.zip',
                                           'FEMA National Risk Index v1.20 (December 2025), county table'))
    for acs_file in sorted((RAW_DIR / 'census').glob('acsdt5y2023-*.dat')):
        table_name = acs_file.stem.split('-')[1].upper()
        snapshot['files'].append(describe_file(acs_file, f'Census ACS 2019-2023 5-year Summary File, table {table_name}'))
    for other_file, note in [(RAW_DIR / 'census' / '2023_Gaz_tracts_national.zip', 'Census 2023 Gazetteer, tracts'),
                             (RAW_DIR / 'census' / 'tab20_tract20_tract10_natl.txt', 'Census 2020-2010 tract relationship file'),
                             (RAW_DIR / 'boundaries' / 'ne_10m_coastline.zip', 'Natural Earth 1:10m coastline'),
                             (RAW_DIR / 'boundaries' / 'cb_2023_us_state_20m.zip', 'Census 2023 cartographic boundaries, states'),
                             (RAW_DIR / 'boundaries' / 'cb_2023_us_county_20m.zip', 'Census 2023 cartographic boundaries, counties')]:
        snapshot['files'].append(describe_file(other_file, note))
    # PRISM has about 2,200 small daily files: we record their number, date range and one fingerprint of all names+sizes.
    prism_files = sorted((RAW_DIR / 'prism').glob('*.zip'))
    if prism_files:
        listing = '\n'.join(f'{p.name}:{p.stat().st_size}' for p in prism_files).encode()
        snapshot['prism_daily_grids'] = {'version': 'PRISM AN daily precipitation 4 km (provisional/stable as downloaded)',
                                         'files': len(prism_files), 'first_file': prism_files[0].name,
                                         'last_file': prism_files[-1].name,
                                         'total_size_mb': round(sum(p.stat().st_size for p in prism_files) / 1e6, 1),
                                         'sha256_of_names_and_sizes': hashlib.sha256(listing).hexdigest()}
    return snapshot


def print_data_snapshot(snapshot):
    print(f"Data snapshot written {snapshot['snapshot_written_local_time']}")
    file_table = pd.DataFrame(snapshot['files'])
    columns = [c for c in ['file', 'version', 'size_mb', 'downloaded_local_time', 'rows', 'data_as_of', 'sha256'] if c in file_table]
    file_table = file_table[columns].copy()
    file_table['sha256'] = file_table['sha256'].str[:16] + '...'
    if 'rows' in file_table:
        file_table['rows'] = file_table['rows'].map(lambda v: f'{int(v):,}' if pd.notna(v) else '')
    with pd.option_context('display.max_colwidth', 70, 'display.width', 250):
        print(file_table.fillna('').to_string(index=False))
    if 'prism_daily_grids' in snapshot:
        print('PRISM:', snapshot['prism_daily_grids'])


if SNAPSHOT_PATH.exists():
    data_snapshot = json.load(open(SNAPSHOT_PATH))
    print_data_snapshot(data_snapshot)
    print('\nChecking that the claims and policies files on disk are still this snapshot ...')
    for frozen in data_snapshot['files'][:2]:
        path_on_disk = PROJECT_DIR / frozen['file']
        if not frozen.get('present') or not path_on_disk.exists():
            continue
        same = sha256_of_file(path_on_disk) == frozen['sha256']
        print(f"  {frozen['file']}: {'same file (SHA-256 matches)' if same else 'WARNING: the file changed since the snapshot!'}")
elif (RAW_DIR / 'NfipClaimsV3.parquet').exists():
    data_snapshot = build_data_snapshot()
    json.dump(data_snapshot, open(SNAPSHOT_PATH, 'w'), indent=2)
    print_data_snapshot(data_snapshot)
    print(f'\nSnapshot saved to {SNAPSHOT_PATH}')
else:
    print('No raw files yet: the snapshot is written at the end of this notebook, after the downloads.')

# %% [markdown]
# ---
# # Section 1: NFIP Redacted Claims (OpenFEMA)
#
# - The brief mentioned `FimaNfipClaims` **v2**. FEMA has replaced it with **v3** (`NfipClaims`).
#   The v2 endpoint is deprecated, so we use v3. The columns are almost the same.
# - We use the **bulk Parquet file** (about 160 MB, 2.7 million rows) instead of the API.
#   The API returns at most 1,000 rows per call, so it would need about 2,700 calls.
# - The file has one row per claim. Latitude and longitude are **rounded to 0.1 degree**
#   (about 11 km) to protect privacy.

# %%
NFIP_BULK_URL = 'https://www.fema.gov/about/reports-and-data/openfema/v3/NfipClaimsV3.parquet'
NFIP_FIELDS_URL = ("https://www.fema.gov/api/open/v1/OpenFemaDataSetFields?$filter="
                   "openFemaDataSet%20eq%20'NfipClaims'%20and%20datasetVersion%20eq%203&$top=1000")

nfip_full_path = download_if_missing(NFIP_BULK_URL, RAW_DIR / 'NfipClaimsV3.parquet', min_size_bytes=10_000_000)
nfip_fields_path = download_if_missing(NFIP_FIELDS_URL, RAW_DIR / 'nfip_fields.json')
print('NFIP file:', nfip_full_path, f'{nfip_full_path.stat().st_size / 1e6:.0f} MB')

# %% [markdown]
# We read only the rows with `dateOfLoss` between 2020-01-01 and 2025-12-31.
# PyArrow filters while reading, so the 2.7 million rows never sit in memory together.

# %%
nfip_dataset = pa_dataset.dataset(nfip_full_path)
total_rows_in_file = nfip_dataset.count_rows()

date_filter = ((pa_dataset.field('dateOfLoss') >= STUDY_START) &
               (pa_dataset.field('dateOfLoss') <= STUDY_END))
claims_2020_2025 = nfip_dataset.to_table(filter=date_filter).to_pandas()

print(f'Rows in the full national file : {total_rows_in_file:,}')
print(f'Rows with loss in 2020-2025    : {len(claims_2020_2025):,}')
print(f'Columns                        : {claims_2020_2025.shape[1]}')
print('Data as of                     :', claims_2020_2025['asOfDate'].max())

# %% [markdown]
# OpenFEMA stores money and elevation columns as exact decimals. pandas works better with
# ordinary floats, so we convert them now.

# %%
decimal_columns = [column for column in claims_2020_2025.columns
                   if str(nfip_dataset.schema.field(column).type).startswith('decimal')]
for column in decimal_columns:
    claims_2020_2025[column] = pd.to_numeric(claims_2020_2025[column].astype('float64'), errors='coerce')
claims_2020_2025['dateOfLoss'] = pd.to_datetime(claims_2020_2025['dateOfLoss'])
print('Decimal columns converted to float:', len(decimal_columns))
print(decimal_columns)

# %% [markdown]
# ## 1.1 Which states do we keep?
#
# PRISM rainfall covers only the **48 contiguous states and DC (CONUS)**. Alaska, Hawaii, Puerto Rico,
# the Virgin Islands and Guam have no PRISM 4 km daily grid. We keep CONUS so that every claim
# can receive the same external features. This removes very few claims (see below).
#
# We keep **all** CONUS states and not only hurricane states. The HURDAT2 features then become
# "no storm nearby" for inland claims, which is a real and useful signal.

# %%
NON_CONUS_STATES = ['AK', 'HI', 'PR', 'VI', 'GU', 'AS', 'MP']
rows_by_state_group = claims_2020_2025['state'].isin(NON_CONUS_STATES).value_counts()
print('Claims outside CONUS:', int(rows_by_state_group.get(True, 0)))
print(claims_2020_2025.loc[claims_2020_2025['state'].isin(NON_CONUS_STATES), 'state'].value_counts().to_string())

claims_conus = claims_2020_2025[~claims_2020_2025['state'].isin(NON_CONUS_STATES)].copy()
print(f'\nClaims kept (CONUS, 2020-2025): {len(claims_conus):,}')
print('States kept:', claims_conus['state'].nunique())

# %%
claims_by_year = claims_conus.groupby(claims_conus['dateOfLoss'].dt.year).size().rename('claims')
print(claims_by_year.to_string())
print()
print('Top 10 states by number of claims:')
print(claims_conus['state'].value_counts().head(10).to_string())

# %% [markdown]
# **First observation.** Florida alone has more than half of all claims. 2024 (Helene and Milton)
# has the most claims. 2025 is low, partly because claims are still being reported and settled.

# %% [markdown]
# ---
# # Section 2: The claim locations
#
# All external data is joined by **place and date**. Many claims share the same rounded
# point, so we build a small table of unique points. This keeps the joins fast.

# %%
claims_conus['latitude'] = claims_conus['latitude'].round(1)
claims_conus['longitude'] = claims_conus['longitude'].round(1)
has_location = claims_conus['latitude'].notna() & claims_conus['longitude'].notna()
print(f'Claims with a location   : {has_location.sum():,} ({has_location.mean() * 100:.2f}%)')
print(f'Claims without a location: {(~has_location).sum():,}')

unique_points = (claims_conus.loc[has_location, ['latitude', 'longitude']]
                 .drop_duplicates().reset_index(drop=True))
unique_points['point_id'] = np.arange(len(unique_points))
print(f'Unique 0.1-degree points : {len(unique_points):,}')

claims_conus = claims_conus.merge(unique_points, on=['latitude', 'longitude'], how='left')

# %% [markdown]
# ---
# # Section 3: PRISM Daily Precipitation
#
# PRISM gives one grid of daily rain (in mm) for the whole of CONUS at about 4 km resolution.
#
# **A detail that matters:** a PRISM "day" is the 24 hours that **end at 12:00 UTC** on that date.
# Rain that falls on the evening of the loss date is therefore stored in the **next** day's file.
# So our rain windows include one day after the loss date:
#
# | Feature | PRISM days used | Meaning |
# |---|---|---|
# | `rain_loss_day_mm` | day t | rain on the reported loss date |
# | `rain_3day_mm` | t-1, t, t+1 | the storm's rain around the loss date |
# | `rain_7day_mm` | t-5 ... t+1 | a wet week: soil already full of water |
# | `rain_max_daily_7day_mm` | max over t-5 ... t+1 | the single worst day |
#
# **Point uncertainty.** NFIP coordinates are rounded to 0.1 degree (about 11 km). One PRISM
# cell is about 4 km. So we take the **mean of the 3 x 3 block of cells** around the point
# (about 12 km x 12 km). This matches the location uncertainty better than a single cell.
#
# We need every day from 2019-12-26 to 2026-01-01, about 2,200 files of about 1.2 MB each.

# %%
import rasterio

PRISM_DIR = RAW_DIR / 'prism'
PRISM_DIR.mkdir(exist_ok=True)
prism_first_day = STUDY_START - datetime.timedelta(days=6)
prism_last_day = STUDY_END + datetime.timedelta(days=1)
prism_days = pd.date_range(prism_first_day, prism_last_day, freq='D')
print('PRISM days needed:', len(prism_days))


def prism_url_for_day(day):
    day_text = day.strftime('%Y%m%d')
    return (f'https://data.prism.oregonstate.edu/time_series/us/an/4km/ppt/daily/'
            f'{day.year}/prism_ppt_us_25m_{day_text}.zip')


def download_one_prism_day(day):
    destination = PRISM_DIR / f'prism_ppt_us_25m_{day.strftime("%Y%m%d")}.zip'
    for attempt in range(4):
        try:
            download_if_missing(prism_url_for_day(day), destination, min_size_bytes=100_000, timeout=120)
            return destination
        except Exception as error:
            time.sleep(3 * (attempt + 1))
    return None


download_start = time.time()
with ThreadPoolExecutor(max_workers=4) as pool:      # 4 parallel downloads: polite to the server
    prism_files = list(pool.map(download_one_prism_day, prism_days))
missing_prism_days = [day for day, path in zip(prism_days, prism_files) if path is None]
print(f'PRISM files ready: {sum(path is not None for path in prism_files):,} '
      f'(missing {len(missing_prism_days)}) in {time.time() - download_start:.0f} s')

# %% [markdown]
# Now we read each daily grid once and pick the 3 x 3 mean at every unique claim point.
# The grid is the same every day, so we compute the row and column of each point only once.

# %%
prism_cache_path = PROCESSED_DIR / 'prism_daily_rain_at_points.npy'

first_zip = prism_files[0]
first_tif_name = [name for name in zipfile.ZipFile(first_zip).namelist() if name.endswith('.tif')][0]
with rasterio.open(f'zip://{first_zip}!/{first_tif_name}') as grid:
    grid_rows, grid_cols = rasterio.transform.rowcol(grid.transform, unique_points['longitude'].values,
                                                    unique_points['latitude'].values)
    grid_rows = np.clip(np.array(grid_rows), 1, grid.height - 2)
    grid_cols = np.clip(np.array(grid_cols), 1, grid.width - 2)
    grid_nodata = grid.nodata
    print('Grid size:', grid.height, 'x', grid.width, ' cell size (deg):', round(grid.res[0], 4))


def read_prism_day_at_points(zip_path):
    tif_name = [name for name in zipfile.ZipFile(zip_path).namelist() if name.endswith('.tif')][0]
    with rasterio.open(f'zip://{zip_path}!/{tif_name}') as grid:
        rain = grid.read(1).astype('float32')
    rain[rain == grid_nodata] = np.nan
    rain[rain < 0] = np.nan
    block_values = []
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            block_values.append(rain[grid_rows + row_offset, grid_cols + col_offset])
    return np.nanmean(np.vstack(block_values), axis=0)


if prism_cache_path.exists():
    rain_matrix = np.load(prism_cache_path)
else:
    rain_matrix = np.full((len(prism_days), len(unique_points)), np.nan, dtype='float32')
    for day_index, zip_path in enumerate(prism_files):
        if zip_path is not None:
            rain_matrix[day_index] = read_prism_day_at_points(zip_path)
    np.save(prism_cache_path, rain_matrix)

print('Rain matrix shape (days x points):', rain_matrix.shape)
print(f'Share of point-days with no value (water or outside grid): {np.isnan(rain_matrix).mean() * 100:.2f}%')

# %%
day_position = {day.date(): index for index, day in enumerate(prism_days)}
loss_day_index = claims_conus['dateOfLoss'].dt.date.map(day_position)
point_index = claims_conus['point_id']
usable = loss_day_index.notna() & point_index.notna()
loss_day_index_valid = loss_day_index[usable].astype(int).values
point_index_valid = point_index[usable].astype(int).values


def window_values(first_offset, last_offset):
    """Stack rain for days t+first_offset ... t+last_offset for every claim."""
    stacked = [rain_matrix[loss_day_index_valid + offset, point_index_valid]
               for offset in range(first_offset, last_offset + 1)]
    return np.vstack(stacked)


for column in ['rain_loss_day_mm', 'rain_3day_mm', 'rain_7day_mm', 'rain_max_daily_7day_mm']:
    claims_conus[column] = np.nan
claims_conus.loc[usable, 'rain_loss_day_mm'] = window_values(0, 0)[0]
claims_conus.loc[usable, 'rain_3day_mm'] = window_values(-1, 1).sum(axis=0)
claims_conus.loc[usable, 'rain_7day_mm'] = window_values(-5, 1).sum(axis=0)
claims_conus.loc[usable, 'rain_max_daily_7day_mm'] = window_values(-5, 1).max(axis=0)

print(claims_conus[['rain_loss_day_mm', 'rain_3day_mm', 'rain_7day_mm', 'rain_max_daily_7day_mm']]
      .describe(percentiles=[0.1, 0.5, 0.9, 0.99]).round(1).to_string())

# %% [markdown]
# **Sanity check.** Western North Carolina had record rain from Helene (Sept 2024), with
# 300-500 mm reported widely. If our join works, the Helene claims in NC should show that.

# %%
helene_nc = claims_conus[(claims_conus['floodEvent'] == 'Hurricane Helene') & (claims_conus['state'] == 'NC')]
print(f'Helene claims in NC: {len(helene_nc):,}')
print('Median 3-day rain (mm):', round(helene_nc['rain_3day_mm'].median(), 1))
print('Median 7-day rain (mm):', round(helene_nc['rain_7day_mm'].median(), 1))

# %% [markdown]
# ---
# # Section 4: NOAA HURDAT2 Hurricane Tracks
#
# HURDAT2 is the official "best track" of every Atlantic tropical storm: one position every
# 6 hours, with the maximum sustained wind (knots) and the central pressure.
#
# **Steps**
# 1. Parse the text file and keep storms from 2019 to 2025.
# 2. Interpolate each track to **hourly** positions, so that the "nearest point" is not
#    biased by the 6-hour gaps (a storm moves 100-200 km in 6 hours).
# 3. For every claim, look at all hourly positions within **+/- 3 days** of the loss date.
# 4. Keep the closest one. Record its distance, its wind at that time, the storm's lifetime
#    maximum wind and the Saffir-Simpson category.

# %%
HURDAT_URL = 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt'
hurdat_path = download_if_missing(HURDAT_URL, RAW_DIR / 'hurdat' / 'hurdat2-1851-2025.txt')


def parse_hurdat2(path, first_year=2019):
    track_rows = []
    current_storm_id, current_storm_name = None, None
    with open(path) as file_handle:
        for line in file_handle:
            parts = [part.strip() for part in line.split(',')]
            if parts[0].startswith('AL') and len(parts) >= 3 and len(parts[0]) == 8:
                current_storm_id, current_storm_name = parts[0], parts[1]
                continue
            year = int(parts[0][:4])
            if year < first_year:
                continue
            latitude = float(parts[4][:-1]) * (1 if parts[4].endswith('N') else -1)
            longitude = float(parts[5][:-1]) * (-1 if parts[5].endswith('W') else 1)
            track_rows.append({
                'storm_id': current_storm_id,
                'storm_name': current_storm_name,
                'time_utc': pd.to_datetime(parts[0] + parts[1], format='%Y%m%d%H%M'),
                'record_identifier': parts[2],
                'status': parts[3],
                'latitude': latitude,
                'longitude': longitude,
                'max_wind_kt': float(parts[6]),
                'min_pressure_mb': float(parts[7]) if parts[7] not in ('-999', '') else np.nan,
            })
    return pd.DataFrame(track_rows)


hurricane_tracks = parse_hurdat2(hurdat_path)
print(f'Track points 2019-2025: {len(hurricane_tracks):,} from {hurricane_tracks["storm_id"].nunique()} storms')
print(hurricane_tracks.groupby(hurricane_tracks['time_utc'].dt.year)['storm_id'].nunique().rename('storms').to_string())

# %%
hourly_track_parts = []
for storm_id, storm_track in hurricane_tracks.groupby('storm_id'):
    storm_track = storm_track.drop_duplicates('time_utc').set_index('time_utc').sort_index()
    hourly_index = pd.date_range(storm_track.index.min(), storm_track.index.max(), freq='h')
    hourly = (storm_track[['latitude', 'longitude', 'max_wind_kt']]
              .reindex(storm_track.index.union(hourly_index))
              .interpolate(method='time')
              .loc[hourly_index])
    hourly['storm_id'] = storm_id
    hourly['storm_name'] = storm_track['storm_name'].iloc[0]
    hourly['storm_lifetime_max_wind_kt'] = storm_track['max_wind_kt'].max()
    hourly_track_parts.append(hourly)
hourly_tracks = pd.concat(hourly_track_parts).rename_axis('time_utc').reset_index()
print(f'Hourly track points: {len(hourly_tracks):,}')


# %%
def saffir_simpson_category(wind_kt):
    """-1 = tropical depression, 0 = tropical storm, 1..5 = hurricane category."""
    bins = [-1, 33, 63, 82, 95, 112, 136, 1000]
    labels = [-1, 0, 1, 2, 3, 4, 5]
    return pd.cut(wind_kt, bins=bins, labels=labels).astype(float)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


STORM_WINDOW_DAYS = 3
point_dates = (claims_conus.loc[claims_conus['point_id'].notna(), ['point_id', 'latitude', 'longitude', 'dateOfLoss']]
               .drop_duplicates(['point_id', 'dateOfLoss']).reset_index(drop=True))
print(f'Unique (point, loss date) pairs: {len(point_dates):,}')

storm_results = []
track_times = hourly_tracks['time_utc'].values
for loss_date, group in point_dates.groupby('dateOfLoss'):
    window_start = loss_date - pd.Timedelta(days=STORM_WINDOW_DAYS)
    window_end = loss_date + pd.Timedelta(days=STORM_WINDOW_DAYS + 1)
    in_window = hourly_tracks[(hourly_tracks['time_utc'] >= window_start) & (hourly_tracks['time_utc'] < window_end)]
    if in_window.empty:
        continue
    distances = haversine_km(group['latitude'].values[:, None], group['longitude'].values[:, None],
                             in_window['latitude'].values[None, :], in_window['longitude'].values[None, :])
    closest = distances.argmin(axis=1)
    closest_rows = in_window.iloc[closest]
    storm_results.append(pd.DataFrame({
        'point_id': group['point_id'].values,
        'dateOfLoss': loss_date,
        'storm_id': closest_rows['storm_id'].values,
        'storm_name': closest_rows['storm_name'].values,
        'storm_distance_km': distances[np.arange(len(group)), closest],
        'storm_wind_at_closest_kt': closest_rows['max_wind_kt'].values,
        'storm_lifetime_max_wind_kt': closest_rows['storm_lifetime_max_wind_kt'].values,
        'storm_hours_from_loss_date': ((closest_rows['time_utc'].values - np.datetime64(loss_date))
                                       / np.timedelta64(1, 'h')) - 12,
    }))
storm_features = pd.concat(storm_results, ignore_index=True)
storm_features['storm_category_at_closest'] = saffir_simpson_category(storm_features['storm_wind_at_closest_kt'])
print(f'Point-dates with any storm within +/-{STORM_WINDOW_DAYS} days: {len(storm_features):,}')

claims_conus = claims_conus.merge(storm_features, on=['point_id', 'dateOfLoss'], how='left')
claims_conus['storm_within_200km'] = (claims_conus['storm_distance_km'] <= 200).astype(int)
print(f'Claims with a storm track within 200 km: {claims_conus["storm_within_200km"].mean() * 100:.1f}%')

# %% [markdown]
# **Sanity check:** does the nearest storm agree with FEMA's own `floodEvent` label?

# %%
named_hurricane_claims = claims_conus[claims_conus['floodEvent'].fillna('').str.startswith('Hurricane')].copy()
named_hurricane_claims['fema_name'] = named_hurricane_claims['floodEvent'].str.replace('Hurricane ', '').str.upper()
named_hurricane_claims['names_agree'] = named_hurricane_claims['fema_name'] == named_hurricane_claims['storm_name']
agreement_table = (named_hurricane_claims.groupby('floodEvent')
                   .agg(claims=('id', 'size'), name_agreement=('names_agree', 'mean'),
                        median_distance_km=('storm_distance_km', 'median'))
                   .sort_values('claims', ascending=False).head(12))
print(agreement_table.round(2).to_string())
print(f"\nOverall agreement on named-hurricane claims: {named_hurricane_claims['names_agree'].mean() * 100:.1f}%")

# %% [markdown]
# ---
# # Section 5: Distance to the Ocean Coast
#
# Storm surge (code 1, "tidal water overflow") can only reach places close to the sea.
# Distance to the coast is therefore a simple, physical proxy for surge exposure.
# We use the Natural Earth 1:10m coastline (ocean only, no lakes) and measure distance
# in an equal-area projection (EPSG:5070, metres).

# %%
import geopandas as gpd

coastline_zip = download_if_missing('https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip',
                                    RAW_DIR / 'boundaries' / 'ne_10m_coastline.zip')
coastline = gpd.read_file(f'zip://{coastline_zip}')
coastline = coastline.cx[-130:-60, 20:52].to_crs(5070)    # North America box, then metres

points_gdf = gpd.GeoDataFrame(unique_points.copy(),
                              geometry=gpd.points_from_xy(unique_points['longitude'], unique_points['latitude']),
                              crs=4326).to_crs(5070)
nearest = gpd.sjoin_nearest(points_gdf, coastline[['geometry']], how='left', distance_col='coast_distance_m')
nearest = nearest.drop_duplicates('point_id')
unique_points['coast_distance_km'] = (nearest.set_index('point_id')['coast_distance_m'] / 1000).reindex(unique_points['point_id']).values
claims_conus = claims_conus.merge(unique_points[['point_id', 'coast_distance_km']], on='point_id', how='left')
print(claims_conus['coast_distance_km'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).to_string())

# %% [markdown]
# ---
# # Section 6: Census ACS 5-Year (2019-2023)
#
# **A change from the plan.** The Census Data API now refuses requests without a key
# (it redirects to a "missing key" page). We did not want to put a personal key in the
# notebooks. Instead we use the **ACS Summary File** bulk tables. They are public, need no key,
# and contain every geography down to the **block group**.
#
# | Table | Variable we build |
# |---|---|
# | B01003 | total population |
# | B19013 | median household income |
# | B25077 | median home value (owner-occupied) |
# | B25003 | % owner-occupied housing units |
# | B25035 | median year the housing was built |
# | C17002 | % of people below the poverty line (income-to-poverty ratio < 1) |
# | B03002 | % non-Hispanic White |
#
# NFIP's `censusGeoid` is a 12-digit **block group** code. We try block group first, then the
# tract (first 11 digits), then the county (first 5 digits).

# %%
ACS_BASE_URL = 'https://www2.census.gov/programs-surveys/acs/summary_file/2023/table-based-SF/data/5YRData/'
ACS_TABLES = ['b01003', 'b19013', 'b25077', 'b25003', 'b25035', 'c17002', 'b03002']
CENSUS_DIR = RAW_DIR / 'census'


def read_acs_table(table_name):
    path = download_if_missing(f'{ACS_BASE_URL}acsdt5y2023-{table_name}.dat',
                               CENSUS_DIR / f'acsdt5y2023-{table_name}.dat', min_size_bytes=100_000)
    table = pd.read_csv(path, sep='|', dtype={'GEO_ID': str}, low_memory=False)
    # Keep counties (050), tracts (140) and block groups (150)
    table = table[table['GEO_ID'].str[:3].isin(['050', '140', '150'])]
    table['geo_level'] = table['GEO_ID'].str[:3].map({'050': 'county', '140': 'tract', '150': 'block_group'})
    table['geoid'] = table['GEO_ID'].str.split('US').str[1]
    return table


acs_raw = None
for table_name in ACS_TABLES:
    table = read_acs_table(table_name)
    keep = ['geoid', 'geo_level'] + [column for column in table.columns if '_E' in column]
    table = table[keep]
    acs_raw = table if acs_raw is None else acs_raw.merge(table, on=['geoid', 'geo_level'], how='outer')
print('ACS geographies:', acs_raw['geo_level'].value_counts().to_dict())

# %%
acs_numeric = acs_raw.drop(columns=['geoid', 'geo_level']).apply(pd.to_numeric, errors='coerce')
acs_numeric = acs_numeric.where(acs_numeric > -100_000)     # ACS uses -666666666 etc. for "not available"

acs_features = pd.DataFrame({
    'geoid': acs_raw['geoid'],
    'geo_level': acs_raw['geo_level'],
    'acs_population': acs_numeric['B01003_E001'],
    'acs_median_household_income': acs_numeric['B19013_E001'],
    'acs_median_home_value': acs_numeric['B25077_E001'],
    'acs_pct_owner_occupied': 100 * acs_numeric['B25003_E002'] / acs_numeric['B25003_E001'],
    'acs_median_year_built': acs_numeric['B25035_E001'],
    'acs_pct_below_poverty': 100 * (acs_numeric['C17002_E002'] + acs_numeric['C17002_E003']) / acs_numeric['C17002_E001'],
    'acs_pct_non_hispanic_white': 100 * acs_numeric['B03002_E003'] / acs_numeric['B03002_E001'],
})
acs_features['acs_median_year_built'] = acs_features['acs_median_year_built'].where(acs_features['acs_median_year_built'] > 1900)
print(acs_features.groupby('geo_level').median(numeric_only=True).round(1).T.to_string())

# %%
ACS_COLUMNS = [column for column in acs_features.columns if column.startswith('acs_')]
claims_conus['block_group_geoid'] = claims_conus['censusGeoid'].str[:12]
claims_conus['tract_geoid'] = claims_conus['censusGeoid'].str[:11]
claims_conus['county_geoid'] = claims_conus['censusGeoid'].str[:5]
claims_conus['county_geoid'] = claims_conus['county_geoid'].fillna(claims_conus['countyCode'])

acs_joined = pd.DataFrame(index=claims_conus.index, columns=ACS_COLUMNS, dtype='float64')
claims_conus['acs_join_level'] = 'none'
for level, key in [('block_group', 'block_group_geoid'), ('tract', 'tract_geoid'), ('county', 'county_geoid')]:
    level_table = acs_features[acs_features['geo_level'] == level].set_index('geoid')[ACS_COLUMNS]
    level_table = level_table[level_table['acs_median_household_income'].notna()]
    still_missing = claims_conus['acs_join_level'] == 'none'
    matched_values = level_table.reindex(claims_conus.loc[still_missing, key].values)
    found = matched_values['acs_median_household_income'].notna().values
    target_index = claims_conus.index[still_missing][found]
    acs_joined.loc[target_index, ACS_COLUMNS] = matched_values.values[found]
    claims_conus.loc[target_index, 'acs_join_level'] = level
claims_conus[ACS_COLUMNS] = acs_joined[ACS_COLUMNS].astype(float)
print('How each claim was matched to ACS:')
print((claims_conus['acs_join_level'].value_counts(normalize=True) * 100).round(2).to_string())

# %% [markdown]
# Most claims match at block group level. Claims that do not match usually have a 2010
# block group code that changed in 2020; the tract or county fallback catches them.

# %% [markdown]
# ---
# # Section 7: FEMA National Risk Index (NRI v1.20, December 2025)
#
# The NRI gives a modelled, long-term risk score for every county. These scores are fixed
# properties of the place, so they are known **before** any flood. We keep the flood-related
# ones and the social vulnerability and resilience scores.

# %%
nri_zip = download_if_missing('https://www.fema.gov/about/reports-and-data/openfema/nri/v120/NRI_Table_Counties.zip',
                              RAW_DIR / 'nri' / 'NRI_Table_Counties.zip', min_size_bytes=1_000_000)
NRI_COLUMNS = {
    'STCOFIPS': 'county_geoid',
    'RISK_SCORE': 'nri_overall_risk_score',
    'SOVI_SCORE': 'nri_social_vulnerability_score',
    'RESL_SCORE': 'nri_community_resilience_score',
    'CFLD_RISKS': 'nri_coastal_flood_risk_score',
    'IFLD_RISKS': 'nri_inland_flood_risk_score',
    'HRCN_RISKS': 'nri_hurricane_risk_score',
    'CFLD_AFREQ': 'nri_coastal_flood_annual_frequency',
    'IFLD_AFREQ': 'nri_inland_flood_annual_frequency',
}
with zipfile.ZipFile(nri_zip) as archive:
    nri_counties = pd.read_csv(archive.open('NRI_Table_Counties.csv'), usecols=list(NRI_COLUMNS), dtype={'STCOFIPS': str})
nri_counties = nri_counties.rename(columns=NRI_COLUMNS)
nri_counties['county_geoid'] = nri_counties['county_geoid'].str.zfill(5)
claims_conus = claims_conus.merge(nri_counties, on='county_geoid', how='left')
print(f'Claims matched to an NRI county: {claims_conus["nri_overall_risk_score"].notna().mean() * 100:.2f}%')

# %% [markdown]
# ---
# # Section 8: Boundaries for Maps
#
# Census cartographic boundary files (1:20 million) for states and counties. Used only for maps.

# %%
for layer in ['county', 'state']:
    download_if_missing(f'https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_{layer}_20m.zip',
                        RAW_DIR / 'boundaries' / f'cb_2023_us_{layer}_20m.zip')
hurricane_tracks.to_parquet(PROCESSED_DIR / 'hurdat2_tracks_2019_2025.parquet', index=False)
print('Boundaries and storm tracks saved.')

# %% [markdown]
# ---
# # Section 9: Save the Merged Table
#
# One row per claim. NFIP columns keep their original names. New columns have clear prefixes:
# `rain_`, `storm_`, `coast_`, `acs_`, `nri_`.

# %%
merged_path = PROCESSED_DIR / 'claims_merged_2020_2025.parquet'
claims_conus.to_parquet(merged_path, index=False)

added_columns = [column for column in claims_conus.columns
                 if column.startswith(('rain_', 'storm_', 'coast_', 'acs_', 'nri_'))]
join_summary = pd.DataFrame({
    'feature': added_columns,
    'non_missing_pct': [round(claims_conus[column].notna().mean() * 100, 2) for column in added_columns],
})
join_summary.to_csv(TABLE_DIR / '00_external_join_coverage.csv', index=False)
print(f'Saved {merged_path.name}: {len(claims_conus):,} rows x {claims_conus.shape[1]} columns')
print(join_summary.to_string(index=False))

# %%
# If the snapshot could not be written at the top (first run, files were downloaded above), write it now.
if not SNAPSHOT_PATH.exists():
    data_snapshot = build_data_snapshot()
    json.dump(data_snapshot, open(SNAPSHOT_PATH, 'w'), indent=2)
    print_data_snapshot(data_snapshot)
else:
    print(f'Data snapshot: {SNAPSHOT_PATH.name} (claims file as of {data_snapshot["files"][0].get("data_as_of")})')

# %% [markdown]
# ---
# # Summary
#
# | Step | Rows |
# |---|---|
# | National NFIP v3 file (all years, as of 2026-11-16) | 2,731,064 |
# | Loss date in 2020-2025 | 277,088 |
# | Kept CONUS (PRISM coverage) | 276,119 |
# | ...of which have a location (lat/lon) | 274,312 |
#
# These counts belong to the frozen data version in `tables/00_data_snapshot.json` (Section 0.1). If OpenFEMA
# republishes the file, the counts will change; compare the SHA-256 in the snapshot to know if you have the same data.
#
# **What we built.** One claim table with 7 kinds of information: building, policy, rain,
# storm, coast distance, neighbourhood (ACS) and county risk (NRI).
#
# **What comes next.** Notebook 01 cleans this table. It decides, column by column, what to
# drop, what to fix and what to impute, and it lists the features that would leak the answer.
