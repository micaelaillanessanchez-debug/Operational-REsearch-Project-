# Operational-REsearch-Project-

## Description
This repository contains the information about the project where our main goal is to analice the corelation to the temperature and irraciance map of france.

## Tool used to get the database
The analysis is based on reanalysis data obtained from Climate Data Store (CDS), a European repository of climate and atmospheric data. 
Specifically, the project employs the following datasets

-considering the solar radiation: 
(https://ads.atmosphere.copernicus.eu/datasets/cams-gridded-solar-radiation?tab=download)

-considering temperature:
https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4-monthly?tab=download

where the area chosen is defined by the next coordinates


the year 2011

### Step 1 – Import required libraries
This step installs and imports the Python libraries required to download and
process climate data from the Copernicus Climate Data Store.

- `cdsapi` is used to access and download climate datasets
- `netcdf4` enables reading NetCDF climate data files
- `xarray` is used to manipulate multi-dimensional climate data
- `pandas` is used for data analysis and tabular processing
AGREGAR LAS OTRAS LIBRARIES
```python
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cdsapi
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

```
### Step 2 – Download solar radiation data from Copernicus

This step connects to the Copernicus Atmosphere Data Store (CAMS) using the CDS API
and downloads global horizontal solar irradiation data for a selected area and time period.

```python
client = cdsapi.Client(url, key, verify=False)

dataset = "cams-gridded-solar-radiation"
request = {
    "variable": ["global_horizontal_irradiation"],
    "sky_type": ["clear"],
    "version": ["4.6"],
    "year": ["2011"],
    "month": [
        "01"
    ],
    "area": [51.0, -5, 42.0, 8.0]

}
client.retrieve(dataset, request).download()
```
### Step 3 – Spatial aggregation and GHI energy calculation

This step processes the Global Horizontal Irradiance (GHI) data by aggregating
the irradiance values over time for each latitude–longitude grid point and
converting them into accumulated solar energy.

```python
ds_GHI = xr.open_dataset(
    'v4.6_GHI_clear_2011_01.area-subset.51.0.8.0.42.0.-5.nc',
    engine='h5netcdf'
)

df_converted_GHI = ds_GHI.to_dataframe()

df_sum = (
    df_converted_GHI
    .groupby(["latitude", "longitude"])["global_horizontal_clear_sky_irradiation"]
    .sum()
    .reset_index()
)

df_sum["GHI_energy_Wh_m2"] = (
    df_sum["global_horizontal_clear_sky_irradiation"] * 0.25
)
```
### Step 4 – Creation of the Latitud Longitud Grid

This step reshapes the aggregated GHI energy data into a 
latitude–longitude grid. The spatial axes are reordered.
```python

ghi_grid = df_sum.pivot(
    index='latitude',
    columns='longitude',
    values='GHI_energy_Wh_m2'
hi_grid = ghi_grid[sorted(ghi_grid.columns)]     # longitudes croissantes
ghi_grid = ghi_grid.sort_index(ascending=True)   # latitudes croissantes (Sud → Nord)

```
### heatmap CODE POUR IRRADIATION HEAT MAP
plt.figure(figsize=(12,8))
sns.heatmap(
    ghi_grid[::-1],  # inverse les lignes pour que le Sud soit en bas
    cmap="YlOrRd",
    cbar_kws={'label': 'GHI cumulative (Wh/m²)'}
)
plt.title("Global Horizontal Irradiation (Cumulative)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
