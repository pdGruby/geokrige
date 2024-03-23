import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

data_df = """
station,lon,lat,temp
BIAŁYSTOK,23.15,53.13,7.23
BIELSKO-BIAŁA,19.04,49.81,8.45
CHOJNICE,17.57,53.7,7.59
ELBLĄG,19.55,54.22,8.02
GORZÓW WIELKOPOLSKI,15.24,52.73,8.92
HEL,18.79,54.64,8.36
JELENIA GÓRA,15.73,50.9,7.66
KALISZ,18.08,51.75,8.74
KATOWICE,19.03,50.24,8.53
KIELCE-SUKÓW,20.69,50.80,7.77
KOSZALIN,16.18,54.19,8.35
KOŁO,18.64,52.2,8.43
KOŁOBRZEG,15.38,54.16,8.48
KRAKÓW-BALICE,19.81,50.07,8.41
KĘTRZYN,21.37,54.07,7.52
KŁODZKO,16.64,50.43,7.78
LEGNICA,16.17,51.2,9.16
LESKO,22.33,49.47,7.74
LESZNO,16.58,51.84,8.79
LUBLIN-RADAWIEC,22.4,51.22,7.80
LĘBORK,17.74,54.53,7.86
MŁAWA,20.36,53.13,7.68
NOWY SĄCZ,20.71,49.61,8.50
OLSZTYN,20.48,53.78,7.59
OPOLE,17.92,50.67,9.15
POZNAŃ,16.73,52.41,8.88
PŁOCK,19.7,52.55,8.28
RACIBÓRZ,18.22,50.09,8.71
RZESZÓW-JASIONKA,22.04,50.11,8.36
SANDOMIERZ,21.75,50.68,8.11
SIEDLCE,22.28,52.16,7.87
SULEJÓW,19.89,51.35,8.02
SUWAŁKI,22.93,54.1,6.68
SZCZECIN,14.55,53.43,9.06
SŁUBICE,14.57,52.36,9.11
TARNÓW,20.96,50.03,8.86
TERESPOL,23.62,52.07,7.90
TORUŃ,18.6,53.01,8.44
USTKA,16.86,54.56,8.34
WARSZAWA,20.99,52.17,8.48
WIELUŃ,18.57,51.22,8.59
WROCŁAW,16.88,51.11,9.11
WŁODAWA,23.53,51.54,7.86
ZAKOPANE,19.95,49.3,5.76
ZIELONA GÓRA,15.51,51.94,8.90
ŁEBA,17.53,54.76,8.03
ŁÓDŹ,19.36,51.72,8.32
ŚWINOUJŚCIE,14.25,53.91,8.70
"""

data_df = [line.split(',') for line in data_df.splitlines()[1:]]
data_df = pd.DataFrame(data_df[1:], columns=data_df[0])
data_df = data_df.set_index('station')

data_df['lon'] = data_df['lon'].astype('float64')
data_df['lat'] = data_df['lat'].astype('float64')
data_df['temp'] = data_df['temp'].astype('float64')

geometry = [Point(lon, lat) for lon, lat in zip(data_df['lon'], data_df['lat'])]
data_gdf = gpd.GeoDataFrame(data_df.reset_index(), crs='EPSG:4326', geometry=geometry)

data_gdf = data_gdf[['station', 'temp', 'geometry']]
