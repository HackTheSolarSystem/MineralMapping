## Mineral Mapping via Clustering

### Addressing [Meteorite Mineral Mapping](https://github.com/amnh/HackTheSolarSystem/wiki/Meteorite-Mineral-Mapping)

### Created by Ahnighito
* Peter Kang
* Cecina Babich Morrow
* Katy Abbott
* Jeremy Neiman
* Meret Götschel
* Jackson Lee
* John Underwood

### Solution Description

In this project we explored a few different options for classifying minerals based on the intensities of constituent elements in
EMPA results. All of our approaches involved treating the intensity images as a set of n-dimesional points (1 dimension per
consituent element) and classifying within that space. The approaches we explored were:
1. Linear classification via SVM
2. Random forest classification
3. Nearest neighbor classification
4. Cluster inference via DBSCAN clustering

Broadly, all of the classifiers were trained based on images of standards gathered from pure samples of minerals that shared
constituent elements with the minerals of interest. Each of 7 elements across 8 standards was combined into a 7-dimensional point,
and the classifiers were tasked with finding models that separated the resultant cloud of points into clusters that correlated
with the standards they came from.

#### Weight Distribution Inference of Minerals
Many of our approaches required some initial hypotheses about the distributions of minerals that weren't given in the standards.
This presented a challenge since we had to build a process for translating theoretical distributions of elements to intensities we
could use to train our models. In order to do this, we used the data gathered from our standards and correlated it with knowledge
about the composition of the minerals they represented, e.g. the chemical formula. From there, we used linear regression to find
a function that translated intensities to percent weights and vice versa.

#### Linear Classification via SVM
Given the generally discrete nature of chemical formulae the hypothesis behind linear classification was that the clusters would form
around discrete intensity values of each element, which a linear model would be able to split very effectively. We found that this
performed very well when we ran validation on the standards, but the model was sadly ineffective when applied to the actual meteorites.
Much of this can probably be attributed to a lack of labeled data for many of the minerals of interest. In order to get around the
lack of labelled data, we resorted to generating "reasonable" datasets based on the variance of the minerals in the standards centered
around the theoretical intensities given by weight distribution inference.

#### Random Forest Classification
Random forest classification builds many decision trees that produce a non-linear model for classification. In practice we found that
this handled the real-object mineral case more robustly than the linear classification case.

#### Nearest Neighbor Classification
Nearest neighbor was implemented by comparing intensities in the image to the theoretical intensities given by our inference above. Each
point was compared to each of the elements in question, and was classified as the mineral whose point it was closest to. This approach
was very performant, since training only involved plotting one neighbor point per element, and evaluation likewise only involved
comparing against one neighbor point per element.

#### Cluster Inference via DBSCAN Clustering
Our other approaches listed above all have the restriction of requiring labelled data for training. This solution approached the problem
from another direction, taking it for granted that the data is separated into clusters, and just attempting to identify those clusters.
DBSCAN is a cluster-finding algorithm that starts with a set of points, representing clusters, and attepts to grow the clusters by
adding new points to a cluster that are within a radius epsilon of points already within the cluster. When this approach worked it
worked very well, but in our experience it was a fragile approach that required a lot of fine-tuning of the epsilon parameter. Another
weakness of this approach is that correlating each cluster with a mineral is non-trivial, but theoretically could be done by attempting
to map the centers of the found clusters to weight distributions of known minerals. However, the benefit of not having to know what
minerals are of interest beforehand could be a huge bonus in cases where composition isn't well known, or there's a possibility of
unexpected elements existing in the sample.

### Installation Instructions

1. Install Python following the instructions at https://www.python.org/downloads/
2. Using Python's package manager `pip`, install the following libraries:
  - jupyter
  - matplotlib
  - numpy
  - pandas
  - periodictable
  - scikit-learn
  - scikit-image
3. Much of the code is in the form of Jupyter notebook files, which can be accessed by running the `jupyter notebook` command and
opening the .ipynb file in the browser window that opens.

### Frontend Overview

Additionally, this project looked to create a web app that would allow users to interact with the results of their analysis. The app renders each pixel of the output and colors it according to its mineral identification. Hovering over each pixel tells the user which mineral is there.

Running this Angular app requires npm and [node.js](https://nodejs.org/en) v8.x+. From the root directory in your terminal, follow the below instructions:

`cd frontend`

`npm install`

`npm start` or if you have the [angular cli installed](https://angular.io/guide/quickstart) `ng serve`

Currently, the back and front ends of this repository are not connected by an API; the frontend renders the results from files preloaded into the `/frontend/src/app/constants` directory in JSON format.
