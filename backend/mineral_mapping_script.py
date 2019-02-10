# import libraries
import PIL
from PIL import Image
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from pathlib import Path
from skimage.io import imread, imshow

# open example image
im = Image.open('challenge_data/dataset_1_opaques/obj1_8bt_Ca.tif')
data = np.asarray(im)

data = Image.open('challenge_data/dataset_1_opaques/standards_8bt_Ti.tif')
data1 = Image.open('challenge_data/dataset_1_opaques/standards_8bt_Ti.tif')

Image.open('challenge_data/dataset_1_opaques/standards_8bt_Ca.tif')
Image.open('challenge_data/dataset_1_opaques/standards_8bt_Fe.tif')


## imagine we have a dataframe of standards
## Mineral | Percent weight | Mean intensity | Standard deviation

#for element in element maps:


# read in percent weights by element of the minerals in the standard
weights = pd.read_csv('challenge_data/weights_to_minerals.csv')
weights.head()
# read in the pixel intensities by element in the standard
mineral_standards = pd.read_csv('challenge_data/mineral_standards.csv')
mineral_standards.head()
# create dictionary to standardize file names to chemical formulas
mineral_dict = dict(zip(np.unique(mineral_standards['mineral']),
    ["Ca_Ti_O_3", "Fe_", "Fe_3O_4", "Fe_S_", "Ni_S_", "Ni_", "Ca_Fe_Mg_Mn_Ni_Si_", "Ti_O_2"]))
# use dictionary to change mineral column
weights['mineral'] = weights['mineral'].map(mineral_dict)
mineral_standards['mineral'] = mineral_standards['mineral'].map(mineral_dict)
elements = [val for val in mineral_standards.columns if val != 'mineral']
coefs = pd.DataFrame(index = ['coeff'], columns = elements)
lr = LinearRegression(fit_intercept = False)
# loop through elements to create linear regression of percent weight vs pixel intensity
# in the minerals in the standard
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
    #pred = reg.predict(xi_pred)
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

mineral_standards.columns[:-1]
mineral_standards.shape
percent_weight_pred = pd.DataFrame(columns = mineral_standards.columns)
percent_weight_pred

mineral_standards.columns

# apply coefficients from linear regression to pixel intensities from standard
for col in mineral_standards.columns[:-1]:
    percent_weight_pred[col] = mineral_standards[col].apply(lambda x: x*coefs[col])

percent_weight_pred

percent_weight_pred.to_csv("predicted_percentweight_standard.csv")


# Define where your images are
root = Path("/Users/hellenfellows/OneDrive - AMNH/BridgeUp/HackathonRepo/MineralMapping/challenge_data/")
image_path = root / "dataset_1_opaques"
list(image_path.glob('*'))

obj2_minerals = [i for i in list(image_path.glob('obj2_32bt*.tif'))]
meteorite_element = [{'name': s.name.split('_')[2].split('.')[0], 'image':imread(s)} for s in image_path.glob('obj2_32bt*.tif')]
meteorite_element

obj2_intensities = pd.DataFrame(columns = [val['name'] for val in meteorite_element])
for m in meteorite_element:
    element = m['name']
    obj2_intensities[element] = list(np.ravel(m['image']))



obj2_intensities.head()

obj2_percent_weight_pred = obj2_intensities.copy()
obj2_percent_weight_pred.head()

coefs = coefs[obj2_intensities.columns]
coefs

obj2_intensities.shape
coefs.shape
obj2_intensities.mul(coefs.values, axis = 1)
new = obj2_intensities.mul(coefs.values, axis = 1)
obj2_percent_weight_pred

obj2_intensities.columns
col = 'Si'
coefs[col]
obj2_intensities[col].apply(lambda x: float(x*coefs[col].values))

# apply coefficients from linear regression to pixel intensities from object 2
obj2_percent_weight_pred = obj2_intensities.apply(lambda x: x*coefs.values[0], axis = 1)

obj2_intensities.head()
obj2_percent_weight_pred.head()

obj2_percent_weight_pred.to_csv("./challenge_data/predicted_percentweight_obj2.csv")
