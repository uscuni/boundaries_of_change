import geopandas as gpd
import pandas as pd
from scipy.spatial import distance

def get_buildings(path_to_buildings):
    buildings = gpd.read_parquet(path_to_buildings)
    
    final_clusters = pd.read_parquet(
        "/data/uscuni-ulce/processed_data/clusters/cluster_mapping_v8.pq")
    
    level_columns = []
    
    for level in final_clusters.columns:
        buildings[f"level_{level}"] = buildings.final_without_noise.map(
            final_clusters[level]
        )
        level_columns.append(f"level_{level}")
    
    level_7 = buildings.pop("final_without_noise")
    buildings["level_7"] = level_7
    level_columns.append("level_7")
    return buildings, level_columns

def get_building_counts_per_region(buildings, boundary, level="level_4"):
    joined = gpd.sjoin(buildings, boundary, how="left", predicate="within")
    grouped = joined.groupby("NAME")[level].value_counts()
    cnt_table = pd.DataFrame(grouped.unstack().fillna(0).T)
    
    return cnt_table

def normalize_building_counts(count_table):
    region_building_sums = count_table.sum(axis=0)
    cnt_table_norm = count_table.div(region_building_sums, axis=1)
    return cnt_table_norm.T