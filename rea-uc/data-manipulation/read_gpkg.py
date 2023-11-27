import json
import fiona
import geopandas as gp
from sqlalchemy import create_engine


engine = create_engine("postgresql://rea:rea@147.102.6.64:5555")

count  = 0
count2 = 0
count1 = 0
total = 0 
# No need to pass "layer='etc'" if there's only one layer
features = []
with fiona.open('stateland.gpkg') as layer:
    
    for feature in layer:
        count2 += 1
        if feature['properties']['mainusetype'] is not None:
            if '1122' not in feature['properties']['mainusetype']:
                
                feature['properties']['mainusetype'] = 0
                count +=1
        else:
            count1 += 1
        features.append(feature)

    total = count2 - (count + count1)                
               
    print("total entries are:", count2)        
    print("not multi apartment:" , count)
    print("null values are:", count1)
    print("postive values:",total)
    gdf = gp.GeoDataFrame.from_features([feature for feature in features], crs=25833)
    gdf = gdf[['geometry', 'address', 'cadastrename', 'mainusetype', 'totalarea', 'usefularea', 'surfacenumberoffloors']]
    gdf = gdf[gdf['address'].notna()]
    gdf.drop(gdf[gdf['mainusetype'] == 0].index, inplace=True)
    #gdf.set_crs(epsg=0, inplace=True)
    #gdf.to_crs(epsg=0)
    del gdf['mainusetype']
    gdf.rename(columns={'cadastrename': 'cadastre_number', 'totalarea' : 'total_area', 'usefularea' : 'useful_area', 'surfacenumberoffloors' : 'floors'}, inplace=True)
    print(gdf.head)
    gdf.to_postgis("buildings", engine, index = False, if_exists = 'append') 
    #print(gdf.iloc[0])

   
