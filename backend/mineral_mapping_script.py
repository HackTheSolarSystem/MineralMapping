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

# read in percent weights by element of the minerals in the standard
weights = pd.read_csv('challenge_data/weights_to_minerals.csv')
weights.head()
# read in the pixel intensities by element in the standard
mineral_standards = pd.read_csv('challenge_data/mineral_standards.csv')
mineral_standards.head()
# create dictionary to standardize file names to chemical formulas
# needed to separate each element in the formula with an _ to make looping easier
mineral_dict = dict(zip(np.unique(mineral_standards['mineral']),
    ["Ca_Ti_O_3", "Fe_", "Fe_3O_4", "Fe_S_", "Ni_S_", "Ni_", "Ca_Fe_Mg_Mn_Ni_Si_", "Ti_O_2"]))
# use dictionary to change mineral columns to underscore format
weights['mineral'] = weights['mineral'].map(mineral_dict)
mineral_standards['mineral'] = mineral_standards['mineral'].map(mineral_dict)
# list of elements
# need to ignore the "mineral" column of the data
elements = [val for val in mineral_standards.columns if val != 'mineral']
coefs = pd.DataFrame(index = ['coeff'], columns = elements)
# make a linear regression forcing the intercept to be zero
# since zero intensity should correspond to zero percent weight
lr = LinearRegression(fit_intercept = False)
# loop through elements to create linear regression of percent weight vs pixel intensity
# in the minerals in the standard
for element in elements:
    element_df = mineral_standards[mineral_standards['mineral'].str.contains(element + "_")]
    # if the element has no percent weights, skip it
    if element_df.empty:
        continue
    minerals = element_df['mineral'].unique()
    xis = np.empty(0)
    yis = np.empty(0)
    for mine in minerals:
        # get percent weights of the element in that mineral
        weight = weights[weights['mineral'] == mine][element]
        intensities = element_df[element_df['mineral'] == mine][element]
        # create histogram of element intensities in each mineral
        fig = plt.figure()
        intensities.hist()
        plt.ylim(0,1300)

        #plt.title(element + " " + mine + "std = " + str(intensities.std()))
        name = element + "-in-" + mine + "_std_"+ str(round(intensities.std())) + ".png"
        plt.savefig("images/" + name)
        xis = np.append(xis, np.array(intensities))
        yis = np.append(yis, np.repeat(weight, len(intensities)))

    xis, yis = xis.reshape(-1,1), yis.reshape(-1,1)
    # fit linear regression on percent weight vs intensity
    reg = lr.fit(xis,yis)
    xi_pred =  np.arange(0,900).reshape(-1,1)
    # create predictions for range of intensity values
    pred = reg.predict(xi_pred)
    # plot regression lines over range of intensities for elements
    fig = plt.figure()
    plt.plot(xis,yis, 'o', alpha = .01)
    plt.plot(xi_pred, pred, '*')
    plt.title(element)
    plt.ylim(0, max(yis) + 20)
    plt.xlim(0, max(xis))
    plt.savefig("./images/" + element + "weight_intensity_regression")
    reg.coef_
    # get the linear regression coefficient for each element
    coefs[element] = float(reg.coef_)

coefs

# create empty data frame to fill with predicted percent weights
percent_weight_pred = pd.DataFrame(columns = mineral_standards.columns)
# reorder coefficients to match order of columns in mineral_standards
coefs = coefs[mineral_standards.columns[:-1]]

# apply coefficients from linear regression to pixel intensities from standard
percent_weight_pred = mineral_standards.drop(['mineral'], axis = 1).apply(lambda x: x*coefs.values[0], axis = 1)

# if the predicted percent weight is over 100, set it to 100
percent_weight_pred[percent_weight_pred > 100] = 100
# add a mineral column
percent_weight_pred['mineral'] = mineral_standards['mineral']
percent_weight_pred.to_csv("predicted_percentweight_standard.csv")


# Define where your images are
root = Path("/Users/hellenfellows/OneDrive - AMNH/BridgeUp/HackathonRepo/MineralMapping/challenge_data/")
image_path = root / "dataset_1_opaques"
list(image_path.glob('*'))

# calculate percent weights for object 1
# list of the 32bt files in object 1
obj1_minerals = [i for i in list(image_path.glob('obj1_32bt*.tif'))]
obj1_minerals
# create a dictionary with each element and the pixel intensities for that element in object 1
meteorite_element = [{'name': s.name.split('_')[2].split('.')[0], 'image':imread(s)} for s in image_path.glob('obj1_32bt*.tif')]
meteorite_element

# read in the mask for object 1
mask = imread(root / "dataset_1_opaques/obj1_mask.tif")
# the mask for object 1 is 3 pixels larger than the object 1 image, so we trim it
mask = mask[:-3,:]
# initialize pixels
pixels = []
# add intensities for each element for pixels not masked out
for element in meteorite_element:
    pixels.append(element['image'][mask > 0])

# combine pixels into a dataframe
obj1_intensities = pd.DataFrame(np.dstack(pixels)[0], columns=[i['name'] for i in meteorite_element])
# set column names into correct format
# S and P used to be called Sul and Pho
obj1_intensities.columns = ['Ca', 'Ti', 'Al', 'Cr', 'S', 'Si', 'P', 'Fe', 'Ni', 'Mg']
obj1_intensities.head()

# create a predicted percent weight dataframe of the same size as the intensity dataframe
obj1_percent_weight_pred = obj1_intensities.copy()
# reorder coefficients in the order of the column names
coefs = coefs[obj1_intensities.columns]

# apply coefficients from linear regression to pixel intensities from object 1
obj1_percent_weight_pred = obj1_intensities.apply(lambda x: x*coefs.values[0], axis = 1)
# replace all cells with greater than 100 predicted weight with 100
obj1_percent_weight_pred[obj1_percent_weight_pred > 100] = 100
obj1_percent_weight_pred.to_csv("./challenge_data/predicted_percentweight_obj1.csv")

# calculate percent weight for object 2

obj2_minerals = [i for i in list(image_path.glob('obj2_32bt*.tif'))]
meteorite_element = [{'name': s.name.split('_')[2].split('.')[0], 'image':imread(s)} for s in image_path.glob('obj2_32bt*.tif')]

# read in mask for object 2
mask = imread(root / "dataset_1_opaques/obj2_mask.tif")
pixels = []
for element in meteorite_element:
    pixels.append(element['image'][mask > 0])
obj2_intensities = pd.DataFrame(np.dstack(pixels)[0], columns=[i['name'] for i in meteorite_element])

# create data frame for predicted percent weights
obj2_percent_weight_pred = obj2_intensities.copy()
# reorder coefficients
coefs = coefs[obj2_intensities.columns]

# apply coefficients from linear regression to pixel intensities from object 2
obj2_percent_weight_pred = obj2_intensities.apply(lambda x: x*coefs.values[0], axis = 1)

# replace all cells with greater than 100 predicted weight with 100
obj2_percent_weight_pred[obj2_percent_weight_pred > 100] = 100


obj2_percent_weight_pred.to_csv("/challenge_data/predicted_percentweight_obj2.csv")
