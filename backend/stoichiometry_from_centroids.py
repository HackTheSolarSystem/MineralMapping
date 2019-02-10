"""
Return stoichiometric compositions from cluster values
Meant for use with dbscan_clustering.py and mineral_mapping_script.py 
to determine the likely composition of clusters based on a stoichiometric
conversion of weight percent to moles
"""

import pandas as pd
import numpy as np
import periodictable
from collections import Counter

object_name = 'obj2' #User input

#Read in CSV. CSV columns should be list of elements + cluster, index should be pixels
# Data should be weight percents of each element in each pixel. Comes from dbscan_clustering.py
df_cl = pd.read_csv('challenge_data/df_{}_cluster.csv'.format(object_name))
df_cl.set_index('cluster', inplace = True) 
centroids = pd.DataFrame(columns = df_cl.columns[0:-1])
clusters = []
#Calculate mean weight percent for each cluster
for val in df_cl.index.unique():
    clusters.append(val)
    mean = df_cl[df_cl.index == val].mean(axis = 0)
    centroids = centroids.append(mean, ignore_index = True)

# Get molar masses for each element we're considering    
molar_masses = pd.DataFrame({'Si': [periodictable.Si.mass], 'P': [periodictable.P.mass],
    'Cr': [periodictable.Cr.mass], 'Al': [periodictable.Al.mass], 'S': [periodictable.S.mass], 
    'Ti': [periodictable.Ti.mass], 'Ca': [periodictable.Ca.mass], 'Mg': [periodictable.Mg.mass], 
    'Ni': [periodictable.Ni.mass], 'Fe': [periodictable.Fe.mass]})

#Get moles from weight percent
centroid_moles = centroids.div(molar_masses.values, axis = 1)
centroid_moles[centroid_moles == 0] = np.nan
#Scale by setting min element moles to 1. Future steps should involve rounding a la stoichiometry
centroid_moles = centroid_moles.div(centroid_moles.min(axis = 1), axis = 0).dropna(how = 'all', axis = 'columns')
#Add clusters ID and count of pixels in each cluster for reference
centroid_moles['cluster'] = clusters
centroid_moles['pixels per cluster'] = centroid_moles['cluster'].map(l)
#Save as CSV
centroid_moles.to_csv('stoichiometric_output_{}.csv'.format(object_input))


x = centroid_moles.fillna(0).round(2)



x.head(10)

