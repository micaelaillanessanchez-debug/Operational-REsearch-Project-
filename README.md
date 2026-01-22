# Operational-Research-Project- Lucas Caredda - Micaela Illanes
CODE LINK: https://colab.research.google.com/drive/1gMk1vfGNk7KZkYFkVCpmbJ9QJ91zM3zF?usp=sharing#scrollTo=AaVIO3rascFM
## Description
This repository contains the information about the project where our main goal is to analice the corelation to the temperature and irradiance of France. After that, our objective was to analice the relationship betweem temperature and Pv performace


## Tools used to get the database
The analysis is based on reanalysis data obtained from Climate Data Store (CDS), a European repository of climate and atmospheric data. 
Specifically, the project employs the following datasets

-considering the solar radiation: 
(https://ads.atmosphere.copernicus.eu/datasets/cams-gridded-solar-radiation?tab=download)

-considering temperature:
https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4-monthly?tab=download

-coordinates website: geojson.io

Where the area chosen is defined by the next coordinates
lat_min, lat_max = 42.0, 51.0
lon_min, lon_max = -5.0, 8.0

The Proyect was done considering data from the year 2022

### Step 1 – Import required libraries
This step installs and imports the Python libraries required to download and
process climate data from the Copernicus Climate Data Store.

-cdsapi is used to access and download climate datasets from the Copernicus Climate Data Store.

-netcdf4 enables reading and handling NetCDF climate data files.

-xarray is used to manipulate multi-dimensional climate datasets, particularly time-series and gridded data.

-pandas is used for data analysis, aggregation, and tabular data processing.

-numpy provides efficient numerical operations and array manipulation.

-cartopy is used for geospatial projections and map-based visualizations.

-geopandas enables the handling and analysis of geospatial vector data.

-matplotlib is used for plotting and visualizing data.

-seaborn provides high-level statistical data visualization built on top of Matplotlib

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
    "year": ["2022"],
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
### Step 5 – Creation of the irradiation heatmap 
This step generates a heat map of cumulative Global Horizontal Irradiation (GHI) using a latitude–longitude grid. 
To ensure correct geographical orientation, the latitude axis is inverted so that the South is displayed at the bottom and the North at the top, consistent with conventional map representations. 
A sequential color map is applied, where warmer colors indicate higher solar irradiation levels. This visualization allows the identification of zones with higher and lower solar resource availability

```python
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
```

<img width="983" height="723" alt="image" src="https://github.com/user-attachments/assets/bd365b70-a28c-41a4-821f-6818655e60fe" />

### Step 6 Generation of georeferenced Irradiation heat map 
This step generates a georeferenced heat map of cumulative Global Horizontal Irradiation (GHI).
The spatial domain corresponds to the CAMS study area, defined by minimum and maximum latitude and longitude values.

The GHI values are plotted using pcolormesh, which associates each grid cell with its exact geographical coordinates. The latitude and longitude vectors are explicitly created from the grid structure to ensure correct spatial alignment. Unlike matrix-based plots, no axis inversion is required because the data are already ordered in increasing latitude.

A sequential color scale (yellow to red) is applied, where warmer colors indicate higher cumulative solar irradiation. 
The resulting map allows a clear visualization of the spatial variability of solar resources across the territory, providing essential insight for regional-scale photovoltaic potential assessment.
```python
# CODE POUR IRRADIATION HEATMAP + CARTE DES PAYS
# Zone CAMS
lat_min, lat_max = 42.0, 51.0
lon_min, lon_max = -5.0, 8.0

plt.figure(figsize=(18,10))

# Axe géographique Cartopy
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Ajouter carte
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, edgecolor='black')
ax.add_feature(cfeature.COASTLINE)

# -----------------------
# Heatmap CAMS
# -----------------------
# Créer les coordonnées exactes de la grille
lon = np.array(sorted(ghi_grid.columns))
lat = np.array(sorted(ghi_grid.index))  # latitudes croissantes

# Meshgrid pour imshow
Lon, Lat = np.meshgrid(lon, lat)

# Afficher la heatmap
pcm = ax.pcolormesh(
    Lon, Lat, ghi_grid.values,  # pas besoin d'inverser
    cmap='YlOrRd',
    alpha=1,
    shading='auto'              # très important pour que les pixels correspondent
)

# Ajouter colorbar
plt.colorbar(pcm, ax=ax, label='GHI cumulative (Wh/m²)')

# Axes et titre
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Global Horizontal Irradiation (Cumulative) over France")
plt.show()
```
<img width="1313" height="820" alt="image" src="https://github.com/user-attachments/assets/7e175008-1436-4f45-b0b7-7eae914c97a7" />


### Step 7- Generation of georeferenced Irradiation heat map for the hole year 2022
<img width="1291" height="820" alt="image" src="https://github.com/user-attachments/assets/aaa81bd2-e7e5-43e6-a6b6-b8b0702f6a87" />


### Step 8- Generation of the mean temperature map in 2022
```python
c = cdsapi.Client(url='https://ads.atmosphere.copernicus.eu/api', key='56ff124d-08b8-4fe3-a524-e45272516d72', verify=False)

dataset = "cams-global-reanalysis-eac4-monthly"
request = {
    "variable": ["2m_temperature"],
    "year": ["2022"],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "product_type": ["monthly_mean"],
    "data_format": "grib",
    "area": [51.0, -5, 42.0, 8.0]
}

c.retrieve(dataset, request).download()
```
<img width="1280" height="808" alt="image" src="https://github.com/user-attachments/assets/0b9b88d1-8c5f-4e88-b2e6-788e3fceb3ef" />

### Step 9- Generation of a combined Temperature degradation x GHI for 2022

We tried to combine both temperature and irradiation data to create a new, more relevant map for solar panels—a map integrating the effect of temperature on PV performance. To do this, we extrapolated the temperature column from our temperature DataFrame onto the irradiation grid. Be careful here: the two DataFrames do not have the same spatial resolution:

0.75° for temperature

0.1° for irradiation

```python
# Interpoler la température sur la grille GHI
t2m_on_ghi_grid = t2m_annual_mean_c.interp(
    latitude=ghi_energy.latitude,
    longitude=ghi_energy.longitude,
    method="nearest"
)

t2m_on_ghi_grid
```
After doing so we correct the GHI with a chosen performance degradation factor : -0.35% per °C above 25°C 

```python
temp_factor_on_ghi = -0.35 * (
    (t2m_on_ghi_grid + 273.15) - (273 + 25)
)

GHI_temp_corrected = ghi_energy * temp_factor_on_ghi

df_final = (
    GHI_temp_corrected
    .to_dataframe(name="GHI_temp_corrected")
    .reset_index()
)

df_final
```
We obtain our new map :

<img width="1280" height="813" alt="image" src="https://github.com/user-attachments/assets/c34634c7-d1e7-4b83-819f-2dcb006afe76" />

### Recognising the panels
Installing dependencies
This step installs the required Python libraries used throughout the project. Roboflow is used to manage and download the annotated dataset, while Ultralytics provides the YOLO framework used for training the segmentation model. 
```python
!pip install roboflow
!pip install ultralytics
!pip install roboflow
```
Connecting to Roboflow and downloading the dataset: This code initializes the Roboflow API . It selects a particular dataset version and downloads it in YOLOv11 format, which includes the images, annotations

```python
from roboflow import Roboflow
rf = Roboflow(api_key="buMGF1FhJBQTVvngwajD")
project = rf.workspace("myprojects-20ycu").project("usmb-3ou0v")
version = project.version(1)
dataset = version.download("yolov11")
```
Model initialization and training: This code imports the YOLO framework from the Ultralytics library and initializes a pretrained YOLOv11 segmentation model. A lightweight model variant (yolo11s-seg) is selected to balance computational efficiency and segmentation performance. The model is then trained using the dataset configuration defined in the data.yaml file. Training parameters such as the number of epochs, input image resolution, batch size, and number of data-loading workers are defined 
```python
import os
from ultralytics import YOLO # Ensure YOLO is imported

# carregar modelo
#model = YOLO("yolo11l-seg.pt") # Initialize the model
model =YOLO("yolo11s-seg.pt")
# treinar
model.train(
    data=os.path.join(dataset.location, "data.yaml"),
    epochs=2,         # mais épocas para convergência melhor
    imgsz=416,         # imagens maiores para detalhes
    batch=4,           # aproveitando os 16GB da GPU
    workers=8,          # pode aumentar se sua máquina tiver muitos núcleos
    project="usmb-3ou0v",
    name="exp_highres"
```
<img width="519" height="615" alt="image" src="https://github.com/user-attachments/assets/086846e7-4e2c-4c9f-a7e5-d43e3d91b0e6" />
