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
## Code 
###  Código Python
```md
### Python example

```python
import math
from datetime import datetime, timedelta

import cdsapi
import xarray as xr
import pandas as pd

