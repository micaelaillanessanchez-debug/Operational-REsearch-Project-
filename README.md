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

```python
import math
from datetime import datetime, timedelta

import cdsapi
import xarray as xr
import pandas as pd

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
### Step 3- 
This step reads the downloaded NetCDF files using xarray, converts each solar
irradiance component into a pandas DataFrame, and combines them into a single
dataset. The final dataset is exported as a CSV file for further analysis. 
```python
ds_GHI = xr.open_dataset('v4.6_GHI_clear_2011_01.area-subset.51.0.8.0.42.0.-5.nc', engine='h5netcdf')
df_converted_GHI = ds_GHI.to_dataframe()
df_converted_GHI
```
