import PIL
from PIL import Image
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
im = Image.open('challenge_data/dataset_1_opaques/obj1_8bt_Ca.tif')
data = np.asarray(im)

data = Image.open('challenge_data/dataset_1_opaques/standards_8bt_Ti.tif')
data1 = Image.open('challenge_data/dataset_1_opaques/standards_8bt_Ti.tif')

Image.open('challenge_data/dataset_1_opaques/standards_8bt_Ca.tif')
Image.open('challenge_data/dataset_1_opaques/standards_8bt_Fe.tif')


## imagine we have a dataframe of standards 
## Mineral | Percent weight | Mean intensity | Standard deviation

#for element in element maps:



weights = pd.read_csv('challenge_data/weights_to_minerals.csv')
mineral_standards = pd.read_csv('challenge_data/mineral_standards.csv')
mineral_dict = dict(zip(np.unique(mineral_standards['mineral']), 
    ["Ca_Ti_O_3", "Fe_", "Fe_3O_4", "Fe_S_", "Ni_S_", "Ni_", "Ca_Fe_Mg_Mn_Ni_Si_", "Ti_O_2"]))
weights['mineral'] = weights['mineral'].map(mineral_dict)
mineral_standards['mineral'] = mineral_standards['mineral'].map(mineral_dict)
elements = [val for val in mineral_standards.columns if val != 'mineral']
coefs = pd.DataFrame(index = ['coeff'], columns = elements)
lr = LinearRegression(fit_intercept = False)
for element in elements:
    element_df = mineral_standards[mineral_standards['mineral'].str.contains(element + "_")]
    if element_df.empty:
        continue
    minerals = element_df['mineral'].unique()
    xis = np.empty(0)
    yis = np.empty(0)
    for mine in minerals:
        weight = weights[weights['mineral'] == mine][element]
        intensities = element_df[element_df['mineral'] == mine][element]
        fig = plt.figure()
        intensities.hist()
        plt.ylim(0,1300)
        plt.title(element + " " + mine)
        xis = np.append(xis, np.array(intensities))
        yis = np.append(yis, np.repeat(weight, len(intensities)))

    xis, yis = xis.reshape(-1,1), yis.reshape(-1,1)
    reg = lr.fit(xis,yis)
    pred = reg.predict(xi_pred)
    reg.coef_
    coefs[element] = float(reg.coef_)

coefs


pred = reg.predict(xi_pred)
xi_pred =  np.arange(0,400).reshape(-1,1)
fig = plt.figure()
plt.plot(xis,yis, 'o', alpha = .01)
plt.plot(xi_pred, pred, '*')
plt.ylim(0, 100)
plt.xlim(0, 400)




