# Introduction
This document primarily lists resources for performing deep learning (DL) on satellite imagery. To a lesser extent Machine learning (ML, e.g. random forests, stochastic gradient descent) are also discussed, as are classical image processing techniques.

# Top links
* https://github.com/chrieke/awesome-satellite-imagery-datasets
* https://github.com/deepVector/geospatial-machine-learning
* [A modern geospatial workflow](https://gist.github.com/jacquestardie/0d1c0cb413b3b9b06edf)

# Datasets
* [Various datasets listed here](https://www.maptiler.com/gallery/satellite/)

## Sentinel-hub
* One of the best known open data sets. Sentinel-hub provides access to a range of Sentinel data and may be the best overall source of imagery + data.
* Sentinel 2 -> [raw data - requester pays](https://registry.opendata.aws/sentinel-2/) Also paid access via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py). [GBDX also has Sentinel imagery](https://notebooks.geobigdata.io/hub/pricing).

## Kaggle
Kaggle hosts several large satellite image datasets ([> 1 GB](https://www.kaggle.com/datasets?sortBy=relevance&group=public&search=image&page=1&pageSize=20&size=large&filetype=all&license=all)). A list if general image datasets is [here](https://gisgeography.com/free-satellite-imagery-data-list/). A list of land-use datasets is [here](https://gisgeography.com/free-global-land-cover-land-use-data/).

### Kaggle - Deepsat - classification challenge
Each sample image is 28x28 pixels and consists of 4 bands - red, green, blue and near infrared. The training and test labels are one-hot encoded 1x6 vectors. Each image patch is size normalized to 28x28 pixels. Data in `.mat` Matlab format. JPEG?
* [Sat4](https://www.kaggle.com/crawford/deepsat-sat4) 500,000 image patches covering four broad land cover classes - **barren land, trees, grassland and a class that consists of all land cover classes other than the above three** [Example notebook](https://www.kaggle.com/robmarkcole/satellite-image-classification)
* [Sat6](https://www.kaggle.com/crawford/deepsat-sat6) 405,000 image patches each of size 28x28 and covering 6 landcover classes - **barren land, trees, grassland, roads, buildings and water bodies.**

### Kaggle - Amazon from space - classification challenge
* https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
* 3-5 meter resolution GeoTIFF images
* 12 classes including - **cloudy, primary + waterway** etc

### Kaggle - DSTL - segmentation challenge
* https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
* 45 satellite images covering 1km x 1km in both 3-band and 16-band formats.
* 10 Labelled classes include - **Buildings, Road, Trees, Crops, Waterway, Vehicles**

### Kaggle - Airbus Ship Detection Challenge
* https://www.kaggle.com/c/airbus-ship-detection/overview
* I believe there was a problem with this dataset, which led to many complaints that the competition was ruined.

### Kaggle - Draper - place images in order of time
* https://www.kaggle.com/c/draper-satellite-image-chronology/data
* Images are grouped into sets of five, each of which have the same setId. Each image in a set was taken on a different day (but not necessarily at the same time each day). The images for each set cover approximately the same area but are not exactly aligned.

## Alternative datasets
There are a variety of datasets suitable for land classification problems.

### UC Merced
* http://weegee.vision.ucmerced.edu/datasets/landuse.html
* This is a 21 class land use image dataset meant for research purposes.
* There are 100 RGB TIFF images for each class
* Each image measures 256x256 pixels with a pixel resolution of 1 foot

### Google Datasets
* A wide variety of datasets, including satellite imagery in optical, radar bands etc.
* https://cloud.google.com/public-datasets/

### AWS datasets
* Landsat -> free viewer at [remotepixel](https://viewer.remotepixel.ca/#3/40/-70.5) and [libra](https://libra.developmentseed.org/)
* Optical, radar, segmented etc. https://aws.amazon.com/earth/
* [SpaceNet](https://spacenetchallenge.github.io/datasets/datasetHomePage.html)

### Quilt
* Several people have uploaded datasets to [Quilt](https://quiltdata.com/search/?q=satellite)

# Online computing resources
Generally a GPU is required for DL. [Googles colab](https://colab.research.google.com/) is free but limited compute time (12 hours) and somewhat non persistent,

### Kaggle
* Free to use
* GPU Kernels (may run for 1 hour which limits usefulness?)
* Tensorflow, pytorch & fast.ai available
* Advantage that many datasets are already available
* [Read](https://medium.com/@hortonhearsafoo/announcing-fast-ai-part-1-now-available-as-kaggle-kernels-8ef4ca3b9ce6)

### Clouderizer
* https://clouderizer.com/
* Clouderizer is a cloud computing management service, it takes care of installing the required packages to a cloud computing instance (like Amazon AWS or Google Colab). Clouderizer is free for 200 hours per month (Robbie plan) and does not require a credit card to sign up.
* Run projects locally, on cloud or both.
* SSH terminal, Jupyter Notebooks and Tensorboard are securely accessible from Clouderizer Web Console.

### AWS
* GPU available
* https://aws.amazon.com/ec2/?ft=n

### Microsoft Azure
* GPU available (link?)
* Focus on CNTK?
* https://azure.microsoft.com/en-us/free/?b=16.45
* https://docs.microsoft.com/en-us/azure/machine-learning/preview/scenario-aerial-image-classification

### Google
* [ML engine](https://cloud.google.com/ml-engine/) - sklearn, tensorflow, keras
* Collaboratory ([notebooks](https://colab.research.google.com) with GPU as a backend for free for 12 hours at a time),
* Tensorflow available
* pytorch can be installed, [useful articles](https://towardsdatascience.com/fast-ai-lesson-1-on-google-colab-free-gpu-d2af89f53604)

### Floydhub
* https://www.floydhub.com/
* Cloud GPUs
* Jupyter Notebooks
* Tensorboard
* Version Control for DL
* Deploy Models as REST APIs
* Public Datasets

### Paperspace
* https://www.paperspace.com/
* 1-Click Jupyter Notebooks
* GPU on demand
* [Python API](https://github.com/Paperspace/paperspace-python)

## Crestle
* https://www.crestle.com/
* Cloud GPU & persistent file store
* Fast.ai lessons pre-installed

## Salamander
* https://salamander.ai/

# Interesting DL projects
### RoboSat
* https://github.com/mapbox/robosat
* Generic ecosystem for feature extraction from aerial and satellite imagery.

### RoboSat.Pink
* A fork of robotsat
* https://github.com/datapink/robosat.pink

### DeepOSM
* https://github.com/trailbehind/DeepOSM
* Train a deep learning net with OpenStreetMap features and satellite imagery.

### DeepNetsForEO - segmentation
* https://github.com/nshaud/DeepNetsForEO
* Uses SegNET for working on remote sensing images using deep learning.

### Skynet-data
* https://github.com/developmentseed/skynet-data
* Data pipeline for machine learning with OpenStreetMap

# Production
### Custom REST API
* Basic https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html with code [here](https://github.com/jrosebr1/simple-keras-rest-api)
* Advanced https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/
* https://github.com/galiboo/olympus

### Tensorflow Serving
* https://www.tensorflow.org/serving/
* Official version is python 2 but python 3 build [here](https://github.com/illagrenan/tensorflow-serving-api-python3)
* Another approach is [to use Docker](https://www.tensorflow.org/serving/docker)

TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. Multiple models, or indeed multiple versions of the same model, can be served simultaneously.  TensorFlow Serving comes with a scheduler that groups individual inference requests into batches for joint execution on a GPU

### Floydhub
* Allows exposing model via rest API

### modeldepot
* https://modeldepot.io
* ML models hosted

# Image formats & catalogues
* We certainly want to consider cloud optimised GeoTiffs https://www.cogeo.org/
* https://terria.io/ for pretty catalogues
* [Remote pixel](https://remotepixel.ca/projects/index.html#satsearch)
* [Sentinel-hub eo-browser](https://apps.sentinel-hub.com/eo-browser/)

# STAC - SpatioTemporal Asset Catalog
* Specification describing the layout of a catalogue comprising of static files. The aim is that the catalogue is crawlable so it can be indexed by a search engine and make imagery discoverable, without requiring yet another API interface.
* An initiative of https://www.radiant.earth/ in particular https://github.com/cholmes
* Spec at https://github.com/radiantearth/stac-spec
* Browser at https://github.com/radiantearth/stac-browser
* Talk at https://docs.google.com/presentation/d/1O6W0lMeXyUtPLl-k30WPJIyH1ecqrcWk29Np3bi6rl0/edit#slide=id.p
* Example catalogue at https://landsat-stac.s3.amazonaws.com/catalog.json
* Chat https://gitter.im/SpatioTemporal-Asset-Catalog/Lobby
* Several useful repos on https://github.com/sat-utils

# State of the art
What are companies doing?
* Overall trend to using AWS S3 backend for image storage. Just speculating, but a [serverless pipeline](https://github.com/aws-samples/amazon-rekognition-video-analyzer) appears to be where companies are headed for routine compute tasks, whilst providing a Jupyter notebook approach for custom analysis. [Cloud optimised geotiffs to become the standard?](http://blog.digitalglobe.com/developers/cloud-optimized-geotiffs-and-the-path-to-accessible-satellite-imagery-analytics/)
* DigitalGlobe have a cloud hosted Jupyter notebook platform called [GBDX](https://platform.digitalglobe.com/gbdx/). Cloud hosting means they can guarantee the infrastructure supports their algorithms, and they appear to be close/closer to deploying DL. [Tutorial notebooks here](https://notebooks.geobigdata.io/hub/tutorials/list).
* Planet have a [Jupyter notebook platform](https://developers.planet.com/) which can be deployed locally and requires an [API key](https://developers.planet.com/docs/quickstart/getting-started/) (14 days free). They have a python wrapper (2.7?!) to their rest API. They are mostly focussed on classical & fast algorithms?

# Interesting projects
* Pangeo - resources for parallel processing using Dask and Xarray http://pangeo.io/index.html
* Open Data Cube - serve up cubes of data https://www.opendatacube.org/
* [Process Satellite data using AWS Lambda functions](https://github.com/RemotePixel/remotepixel-api)
* [OpenDroneMap](https://github.com/OpenDroneMap/ODM) - generate maps, point clouds, 3D models and DEMs from drone, balloon or kite images.

# Techniques
This section explores the different techniques (DL, ML & classical) people are applying to common problems in satellite imagery analysis. Classification problems are the most simply addressed via DL, object detection is harder, and cloud detection harder still (niche interest).

## Land classification
* Very common problem, assign land classification to a pixel based on pixel value, can be addressed via [simple sklearn cluster algorithm](https://github.com/acgeospatial/Satellite_Imagery_Python/blob/master/Clustering_KMeans-Sentinel2.ipynb) or [deep learning](https://towardsdatascience.com/land-use-land-cover-classification-with-deep-learning-9a5041095ddb).
* Land use is related to classification, but we are trying to detect a scene, e.g. housing, forestry. I have tried CNN -> [See my notebooks](https://github.com/robmarkcole/satellite-image-deep-learning/tree/master/land_classification)
* [Sea-Land segmentation using DL](https://arxiv.org/pdf/1709.00201.pdf)

## Change detection
* Monitor water levels, coast lines, size of urban areas, wildfire damage. For greyscale images detection can be as simple as `diff_image = abs(image1 - image2)` but most commonly further processing is applied.
* Using PCA -> https://appliedmachinelearning.blog/2017/11/25/unsupervised-changed-detection-in-multi-temporal-satellite-images-using-pca-k-means-python-code/
* Using CNN -> https://github.com/vbhavank/Unstructured-change-detection-using-CNN
* [Siamese neural network to detect changes in aerial images](https://github.com/vbhavank/Siamese-neural-network-for-change-detection)
* https://www.spaceknow.com/

## Image registration
* [Wikipedia article on registration](https://en.wikipedia.org/wiki/Image_registration) -> register for change detection or [image stitching](https://mono.software/2018/03/14/Image-stitching/)
* Traditional approach -> define control points, employ RANSAC algorithm
* [Phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) used to estimate the translation between two images with sub-pixel accuracy, useful for [allows accurate registration of low resolution imagery onto high resolution imagery](https://onlinelibrary.wiley.com/doi/10.1002/9781118724194.ch11), or register a [sub-image on a full image](https://www.mathworks.com/help/images/registering-an-image-using-normalized-cross-correlation.html) -> Unlike many spatial-domain algorithms, the phase correlation method is resilient to noise, occlusions, and other defects. [Applied to Landsat images here](https://github.com/JamieTurrin/Phase-Correlation).

## Object detection
* A typical task is detecting boats on the ocean, which should be simpler than land based challenges owing to blank background in images, but is still challenging and no convincing robust solutions available.
* Intro articles [here](https://medium.com/earthcube-stories/how-hard-it-is-for-an-ai-to-detect-ships-on-satellite-images-7265e34aadf0) and [here](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7).
* [DigitalGlobe article](http://gbdxstories.digitalglobe.com/boats/) - they use a combination classical techniques (masks, erodes) to reduce the search space (identifying water via [NDWI](https://en.wikipedia.org/wiki/Normalized_difference_water_index) which requires SWIR) then apply a binary DL classifier on candidate regions of interest. They deploy the final algo [as a task](https://github.com/platformstories/boat-detector) on their GBDX platform. They propose that in the future an R-CNN may be suitable for the whole process.
* [Planet use non DL felzenszwalb algorithm to detect ships](https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ship-detector/01_ship_detector.ipynb)
* [Segmentation of buildings on kaggle](https://www.kaggle.com/kmader/synthetic-word-ocr/kernels)
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”.
* [Deep learning for satellite imagery via image segmentation](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)


## Cloud detection
* A subset of the object detection problem, but surprisingly challenging
* From [this article on sentinelhub](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13) there are three popular classical algorithms that detects thresholds in multiple bands in order to identify clouds. In the same article they propose using semantic segmentation combined with a CNN for a cloud classifier (excellent review paper [here](https://arxiv.org/pdf/1704.06857.pdf)), but state that this requires too much compute resources.
* [This article](https://www.mdpi.com/2072-4292/8/8/666) compares a number of ML algorithms, random forests, stochastic gradient descent, support vector machines, Bayesian method.
* DL..

## Super resolution
* https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-1-ec5c5cd3cd2
* https://modeldepot.io/joe/vdsr-for-super-resolution

## Pansharpening
* Does not require DL, classical algos suffice, [see this notebook](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/panchromatic_sharpening.ipynb)
* https://github.com/mapbox/rio-pansharpen

## Stereo imaging for terrain mapping
* Map terrain from stereo images to produce a digital elevation model (DEM) -> high resolution & paired images required, typically 0.3 m, e.g. [Worldview](https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/37/DG-WV2ELEVACCRCY-WP.pdf) or [GeoEye](https://www.pobonline.com/articles/100233-when-is-satellite-stereo-imagery-the-best-option-for-3d-modeling).
* Process of creating a DEM [here](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLI-B1/327/2016/isprs-archives-XLI-B1-327-2016.pdf) and [here](https://www.geoimage.com.au/media/brochure_pdfs/Geoimage_DEM_brochure_Oct10_LR.pdf).
* https://github.com/MISS3D/s2p -> produces elevation models from images taken by high resolution optical satellites -> demo code on https://gfacciol.github.io/IS18/
* [Intro to depth from stereo](https://github.com/IntelRealSense/librealsense/blob/master/doc/depth-from-stereo.md)
* [Automatic 3D Reconstruction from Multi-Date Satellite Images](http://dev.ipol.im/~facciolo/pub/CVPRW2017.pdf)
* [Semi-global matching with neural networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Seki_SGM-Nets_Semi-Global_Matching_CVPR_2017_paper.pdf)
* [Predict the fate of glaciers](https://github.com/geohackweek/glacierhack_2018)
* [monodepth - Unsupervised single image depth prediction with CNNs](https://github.com/mrharicot/monodepth)
* [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://github.com/jzbontar/mc-cnn)

## NVDI - vegetation index
* Simple band math `ndvi = np.true_divide((ir - r), (ir + r))`
* [Example notebook](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/ndvi_calculation.ipynb)


# For fun
* [Style transfer - see the world in a new way](https://gist.github.com/jacquestardie/6227891818625e4c19c1b1d5bebe4fe4)

# Useful References
* https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0
* https://github.com/taspinar/sidl/blob/master/notebooks/2_Detecting_road_and_roadtypes_in_sattelite_images.ipynb
* [Geonotebooks](https://github.com/OpenGeoscience/geonotebook) with [Docker container](https://github.com/OpenGeoscience/geonotebook/tree/master/devops/docker)
* [Sentinel NetCDF data](https://github.com/acgeospatial/Sentinel-5P/blob/master/Sentinel_5P.ipynb)
