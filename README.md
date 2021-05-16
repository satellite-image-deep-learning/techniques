# Introduction
This document lists resources for performing deep learning (DL) on satellite imagery. To a lesser extent classical Machine learning (ML, e.g. random forests) are also discussed, as are classical image processing techniques.

# Table of contents
* [Top links](https://github.com/robmarkcole/satellite-image-deep-learning#top-links)
* [Datasets](https://github.com/robmarkcole/satellite-image-deep-learning#datasets)
* [Interesting deep learning projects](https://github.com/robmarkcole/satellite-image-deep-learning#interesting-deep-learning-projects)
* [Techniques](https://github.com/robmarkcole/satellite-image-deep-learning#techniques)
* [Image formats, data management and catalogues](https://github.com/robmarkcole/satellite-image-deep-learning#image-formats-data-management-and-catalogues)
* [State of the art](https://github.com/robmarkcole/satellite-image-deep-learning#state-of-the-art)
* [Online platforms for Geo analysis](https://github.com/robmarkcole/satellite-image-deep-learning#online-platforms-for-geo-analysis)
* [Free online computing resources](https://github.com/robmarkcole/satellite-image-deep-learning#free-online-computing-resources)
* [Production](https://github.com/robmarkcole/satellite-image-deep-learning#production)
* [Useful open source software](https://github.com/robmarkcole/satellite-image-deep-learning#useful-open-source-software)
* [Movers and shakers on Github](https://github.com/robmarkcole/satellite-image-deep-learning#movers-and-shakers-on-github)
* [Companies on Github](https://github.com/robmarkcole/satellite-image-deep-learning#companies-on-github)
* [Courses](https://github.com/robmarkcole/satellite-image-deep-learning#courses)
* [Online communities](https://github.com/robmarkcole/satellite-image-deep-learning#online-communities)
* [Jobs](https://github.com/robmarkcole/satellite-image-deep-learning#jobs)
* [Neural nets in space](https://github.com/robmarkcole/satellite-image-deep-learning#neural-nets-in-space)
* [About the author](https://github.com/robmarkcole/satellite-image-deep-learning#about-the-author)

# Top links
* [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)
* [awesome-earthobservation-code](https://github.com/acgeospatial/awesome-earthobservation-code)
* [awesome-sentinel](https://github.com/Fernerkundung/awesome-sentinel)
* [A modern geospatial workflow](https://gist.github.com/jacquestardie/0d1c0cb413b3b9b06edf)
* [geospatial-machine-learning](https://github.com/deepVector/geospatial-machine-learning)
* [Long list of satellite missions with example imagery](https://www.satimagingcorp.com/satellite-sensors/)
* [AWS datasets](https://registry.opendata.aws/)
* [paperswithcode](https://paperswithcode.com/area/computer-vision) aggregates SoTA Computer Vision techniques

# Datasets
**Warning** satellite image files can be LARGE, even a small data set may comprise 50 GB of imagery

## WorldView
* A commercial satellite owned by [DigitalGlobe](https://www.digitalglobe.com/)
* https://en.wikipedia.org/wiki/WorldView-3
* 0.3m PAN, 1.24 MS, 3.7m SWIR. Off-Nadir (stereo) available.
* Owned by [DigitalGlobe](https://www.digitalglobe.com/)
* [Getting Started with SpaceNet](https://medium.com/@sumit.arora/getting-started-with-aws-spacenet-and-spacenet-dataset-visualization-basics-7ddd2e5809a2)
* [Dataset on AWS](https://spacenet.ai/datasets/) -> see [this getting started notebook](https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53) and this notebook on the [off-Nadir dataset](https://medium.com/the-downlinq/introducing-the-spacenet-off-nadir-imagery-and-buildings-dataset-e4a3c1cb4ce3)
* [cloud_optimized_geotif here](http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/) used in the 3D modelling notebook [here](https://gfacciol.github.io/IS18/).
* [Package of utilities](https://github.com/SpaceNetChallenge/utilities) to assist working with the SpaceNet dataset.
* [WorldView cloud optimized geotiffs](http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/) used in the 3D modelling notebook [here](https://gfacciol.github.io/IS18/).
* For more Worldview imagery see Kaggle DSTL competition.

## Sentinel
* As part of the [EU Copernicus program](https://en.wikipedia.org/wiki/Copernicus_Programme), multiple Sentinel satellites are capturing imagery -> see [wikipedia](https://en.wikipedia.org/wiki/Copernicus_Programme#Sentinel_missions).
* 13 bands, Spatial resolution of 10 m, 20 m and 60 m, 290 km swath, the temporal resolution is 5 days
* [awesome-sentinel](https://github.com/Fernerkundung/awesome-sentinel) - a curated list of awesome tools, tutorials and APIs related to data from the Copernicus Sentinel Satellites.
* [Sentinel-2 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs/) and [Sentinel-2 L2A 120m Mosaic](https://registry.opendata.aws/sentinel-s2-l2a-mosaic-120/)
* [Open access data on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2?prefix=tiles%2F31%2FT%2FCJ%2F)
* Paid access via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py).
* [Example loading sentinel data in a notebook](https://github.com/binder-examples/getting-data/blob/master/Sentinel2.ipynb)
* [so2sat on Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/so2sat) - So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and the corresponding local climate zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions of the world.
* [eurosat](https://www.tensorflow.org/datasets/catalog/eurosat) - EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples. Dataset and usage in [EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/EuroSAT), where a CNN achieves a classification accuracy 98.57%.
* [bigearthnet](https://www.tensorflow.org/datasets/catalog/bigearthnet) - The BigEarthNet is a new large-scale Sentinel-2 benchmark archive, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels.
* [Jupyter Notebooks for working with Sentinel-5P Level 2 data stored on S3](https://github.com/Sentinel-5P/data-on-s3). The data can be browsed [here](https://meeo-s5p.s3.amazonaws.com/index.html#/?t=catalogs)
* [Sentinel NetCDF data](https://github.com/acgeospatial/Sentinel-5P/blob/master/Sentinel_5P.ipynb)
* [Analyzing Sentinel-2 satellite data in Python with Keras](https://github.com/jensleitloff/CNN-Sentinel)
* [Xarray backend to Copernicus Sentinel-1 satellite data products](https://github.com/bopen/xarray-sentinel)

## Landsat
* Long running US program -> see [Wikipedia](https://en.wikipedia.org/wiki/Landsat_program) and read [the official webpage](https://www.usgs.gov/core-science-systems/nli/landsat)
* 8 bands, 15 to 60 meters, 185km swath, the temporal resolution is 16 days
* DECEMBER 2020: USGS publishes Landsat Collection 2 Dataset with 'significant geometric and radiometric improvements'. COG and STAC data format. [Announcement](https://www.usgs.gov/news/usgs-releases-most-advanced-landsat-archive-date) and [website](https://www.usgs.gov/core-science-systems/nli/landsat/landsat-collection-2-level-1-data?qt-science_support_page_related_con=1#qt-science_support_page_related_con). Beware data on Google and AWS (below) may be in different formats.
* [Landsat 4, 5, 7, and 8 imagery on Google](https://cloud.google.com/storage/docs/public-datasets/landsat), see [the GCP bucket here](https://console.cloud.google.com/storage/browser/gcp-public-data-landsat/), with Landsat 8 imagery in COG format analysed in [this notebook](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb)
* [Landsat 8 imagery on AWS](https://registry.opendata.aws/landsat-8/), with many tutorials and tools listed
* https://github.com/kylebarron/landsat-mosaic-latest -> Auto-updating cloudless Landsat 8 mosaic from AWS SNS notifications
* [Visualise landsat imagery using Datashader](https://examples.pyviz.org/landsat/landsat.html#landsat-gallery-landsat)
* [Landsat-mosaic-tiler](https://github.com/kylebarron/landsat-mosaic-tiler) -> The repo host all the code for [landsatlive.live](https://landsatlive.live) website and APIs.

## Spacenet
* Spacenet is an online hub for data, challenges, algorithms, and tools.
* [spacenet.ai website](https://spacenet.ai/) covering the series of SpaceNet challenges, lots of useful resources (blog, video and papers)
* [The SpaceNet 7 Multi-Temporal Urban Development Challenge: Dataset Release](https://medium.com/the-downlinq/the-spacenet-7-multi-temporal-urban-development-challenge-dataset-release-9e6e5f65c8d5)
* SpaceNet - WorldView-3 [article here](https://spark-in.me/post/spacenet-three-challenge), and [semantic segmentation using Raster Vision](https://docs.rastervision.io/en/0.8/quickstart.html)

## Planet
* [Planet’s high-resolution, analysis-ready mosaics of the world’s tropics](https://www.planet.com/nicfi/), supported through Norway’s International Climate & Forests Initiative. [BBC coverage](https://www.bbc.co.uk/news/science-environment-54651453)

## Merced
* Land use classification dataset with 21 classes and 100 RGB TIFF images for each class
* Each image measures 256x256 pixels with a pixel resolution of 1 foot
* http://weegee.vision.ucmerced.edu/datasets/landuse.html
* Available as a Tensorflow dataset -> https://www.tensorflow.org/datasets/catalog/uc_merced
* Also [available as a multi-label dataset](https://towardsdatascience.com/multi-label-land-cover-classification-with-deep-learning-d39ce2944a3d)
* Read [Vision Transformers for Remote Sensing Image Classification](https://www.mdpi.com/2072-4292/13/3/516/htm) where a Vision Transformer classifier achieves 98.49% classification accuracy on Merced

## PatternNet
* Land use classification dataset with 38 classes and 800 RGB JPG images for each class
* https://sites.google.com/view/zhouwx/dataset?authuser=0
* Publication: [PatternNet: A Benchmark Dataset for Performance Evaluation of Remote Sensing Image Retrieval](https://arxiv.org/abs/1706.03424)

## Kaggle
Kaggle hosts over > 100 satellite image datasets, [search results here](https://www.kaggle.com/search?q=satellite+image+in%3Adatasets).
The [kaggle blog](http://blog.kaggle.com) is an interesting read.

### Kaggle - Amazon from space - classification challenge
* https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
* 3-5 meter resolution GeoTIFF images from planet Dove satellite constellation
* 12 classes including - **cloudy, primary + waterway** etc
* [1st place winner interview - used 11 custom CNN](http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/)
* [FastAI Multi-label image classification](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95)
* [Multi-Label Classification of Satellite Photos of the Amazon Rainforest](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)

### Kaggle - DSTL - segmentation challenge
* https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
* Rating - medium, many good examples (see the Discussion as well as kernels), but as this competition was run a couple of years ago many examples use python 2
* WorldView 3 - 45 satellite images covering 1km x 1km in both 3 (i.e. RGB) and 16-band (400nm - SWIR) images
* 10 Labelled classes include - **Buildings, Road, Trees, Crops, Waterway, Vehicles**
* [Interview with 1st place winner who used segmentation networks](http://blog.kaggle.com/2017/04/26/dstl-satellite-imagery-competition-1st-place-winners-interview-kyle-lee/) - 40+ models, each tweaked for particular target (e.g. roads, trees)
* [Deepsense 4th place solution](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)

### Kaggle - Airbus Ship Detection Challenge
* https://www.kaggle.com/c/airbus-ship-detection/overview
* Rating - medium, most solutions using deep-learning, many kernels, [good example kernel](https://www.kaggle.com/kmader/baseline-u-net-model-part-1)
* I believe there was a problem with this dataset, which led to many complaints that the competition was ruined

### Kaggle - Draper - place images in order of time
* https://www.kaggle.com/c/draper-satellite-image-chronology/data
* Rating - hard. Not many useful kernels.
* Images are grouped into sets of five, each of which have the same setId. Each image in a set was taken on a different day (but not necessarily at the same time each day). The images for each set cover approximately the same area but are not exactly aligned.
* Kaggle interviews for entrants who [used XGBOOST](http://blog.kaggle.com/2016/09/15/draper-satellite-image-chronology-machine-learning-solution-vicens-gaitan/) and a [hybrid human/ML approach](http://blog.kaggle.com/2016/09/08/draper-satellite-image-chronology-damien-soukhavong/)

### Kaggle - Deepsat - classification challenge
Not satellite but airborne imagery. Each sample image is 28x28 pixels and consists of 4 bands - red, green, blue and near infrared. The training and test labels are one-hot encoded 1x6 vectors. Each image patch is size normalized to 28x28 pixels. Data in `.mat` Matlab format. JPEG?
* [Imagery source](https://csc.lsu.edu/~saikat/deepsat/)
* [Sat4](https://www.kaggle.com/crawford/deepsat-sat4) 500,000 image patches covering four broad land cover classes - **barren land, trees, grassland and a class that consists of all land cover classes other than the above three**
* [Sat6](https://www.kaggle.com/crawford/deepsat-sat6) 405,000 image patches each of size 28x28 and covering 6 landcover classes - **barren land, trees, grassland, roads, buildings and water bodies.**
* [Deep Gradient Boosted Learning article](https://alan.do/deep-gradient-boosted-learning-4e33adaf2969)

### Kaggle - Understanding Clouds from Satellite Images
In this challenge, you will build a model to classify cloud organization patterns from satellite images.
* https://www.kaggle.com/c/understanding_cloud_organization/
* [3rd place solution on Github by naivelamb](https://github.com/naivelamb/kaggle-cloud-organization)

### Kaggle - miscellaneous
* https://www.kaggle.com/reubencpereira/spatial-data-repo -> Satellite + loan data
* https://www.kaggle.com/towardsentropy/oil-storage-tanks -> Image data of industrial tanks with bounding box annotations, estimate tank fill % from shadows
* https://www.kaggle.com/rhammell/ships-in-satellite-imagery -> Classify ships in San Franciso Bay using Planet satellite imagery
* https://www.kaggle.com/rhammell/planesnet -> Detect aircraft in Planet satellite image chips
* https://www.kaggle.com/datamunge/overheadmnist -> A Benchmark Satellite Dataset as Drop-In Replacement for MNIST

## Tensorflow datasets
* [resisc45](https://www.tensorflow.org/datasets/catalog/resisc45) - RESISC45 dataset is a publicly available benchmark for Remote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class.
* [eurosat](https://www.tensorflow.org/datasets/catalog/eurosat) - EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples.
* [bigearthnet](https://www.tensorflow.org/datasets/catalog/bigearthnet) - The BigEarthNet is a new large-scale Sentinel-2 benchmark archive, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels.

## AWS datasets
* [Earth on AWS](https://aws.amazon.com/earth/) is the AWS equivalent of Google Earth Engine
* Currently 36 satellite datasets on the [Registry of Open Data on AWS](https://registry.opendata.aws)

## Microsoft
* [USBuildingFootprints](https://github.com/Microsoft/USBuildingFootprints) -> computer generated building footprints in all 50 US states, GeoJSON format, generated using semantic segmentation
* Checkout Microsofts [Planetary Computer](https://innovation.microsoft.com/en-us/planetary-computer) project

## Google Earth Engine (GEE)
* https://developers.google.com/earth-engine/
* Various imagery and climate datasets, including Landsat & Sentinel imagery
* Since there is a whole community around GEE I will not reproduce it here but point to [awesome-google-earth-engine](https://github.com/gee-community/awesome-google-earth-engine) and list very select references relevant to deep learning
* [How to Use Google Earth Engine and Python API to Export Images to Roboflow](https://blog.roboflow.com/how-to-use-google-earth-engine-with-roboflow/) -> to acquire training data
* [Reduce Satellite Image Resolution with Google Earth Engine](https://towardsdatascience.com/reduce-satellite-image-resolution-with-google-earth-engine-95a129ef488) -> a crucial step before applying machine learning to satellite imagery
* [ee-fastapi](https://github.com/csaybar/ee-fastapi) is a simple FastAPI web application for performing flood detection using Google Earth Engine in the backend.
* [How to Download High-Resolution Satellite Data for Anywhere on Earth](https://towardsdatascience.com/how-to-download-high-resolution-satellite-data-for-anywhere-on-earth-5e6dddee2803)

## Radiant Earth
* https://www.radiant.earth/
* Datasets and also models on https://mlhub.earth/

## FAIR1M ‘world’s largest satellite image database’
* [Database of 15,000 high-definition images with 1 million labelled ‘scenes’ will be open to the international community in June 2021](https://amp-scmp-com.cdn.ampproject.org/c/s/amp.scmp.com/news/china/science/article/3131819/china-makes-worlds-largest-satellite-image-database-train-ai)
* [FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery](https://arxiv.org/abs/2103.05569)

### DEM (digital elevation maps)
* Shuttle Radar Topography Mission: [data - open access](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1-arc?qt-science_center_objects=0#qt-science_center_objects)
* Copernicus Digital Elevation Model (DEM) on S3, represents the surface of the Earth including buildings, infrastructure and vegetation. Data is provided as Cloud Optimized GeoTIFFs. [link](https://registry.opendata.aws/copernicus-dem/)

## Weather Datasets
* UK metoffice -> https://www.metoffice.gov.uk/datapoint
* NASA (make request and emailed when ready) -> https://search.earthdata.nasa.gov
* NOAA (requires BigQuery) -> https://www.kaggle.com/noaa/goes16/home
* Time series weather data for several US cities -> https://www.kaggle.com/selfishgene/historical-hourly-weather-data

## Time series datasets
* [BreizhCrops](https://github.com/dl4sits/BreizhCrops) -> A Time Series Dataset for Crop Type Mapping

## UAV & Drone datasets
* Many on https://www.visualdata.io
* [AU-AIR dataset](https://bozcani.github.io/auairdataset) -> a multi-modal UAV dataset for object detection.
* [ERA](https://lcmou.github.io/ERA_Dataset/) ->  A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos.
* [Aerial Maritime Drone Dataset](https://public.roboflow.ai/object-detection/aerial-maritime)
* [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/)
* [RetinaNet for pedestrian detection](https://towardsdatascience.com/pedestrian-detection-in-aerial-images-using-retinanet-9053e8a72c6)
* [Aerial Maritime Drone Dataset](https://public.roboflow.com/object-detection/aerial-maritime/1)
* [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) -> identify fire and other emergencies from a drone
* [OpenDroneMap](https://github.com/OpenDroneMap/ODM) -> generate maps, point clouds, 3D models and DEMs from drone, balloon or kite images.
* [Dataset of thermal and visible aerial images for multi-modal and multi-spectral image registration and fusion](https://www.sciencedirect.com/science/article/pii/S2352340920302201) -> The dataset consists of 30 visible images and their metadata, 80 thermal images and their metadata, and a visible georeferenced orthoimage.
* [BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos](https://ieeexplore.ieee.org/document/9093284) -> TIR videos of humans and animals with several challenging scenarios like scale variations, background clutter due to thermal reflections, large camera rotations, and motion blur
* [ERA: A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos](https://lcmou.github.io/ERA_Dataset/)

## Synthetic data
* [The Synthinel-1 dataset: a collection of high resolution synthetic overhead imagery for building segmentation](https://arxiv.org/ftp/arxiv/papers/2001/2001.05130.pdf)
* [RarePlanes](https://www.cosmiqworks.org/RarePlanes/) ->  incorporates both real and synthetically generated satellite imagery including aircraft.
* Checkout Microsoft [AirSim](https://microsoft.github.io/AirSim/), which is a simulator for drones, cars and more, built on Unreal Engine

# Interesting deep learning projects
### TorchSat
* [TorchSat](https://github.com/sshuair/torchsat) is an open-source deep learning framework for satellite imagery analysis based on PyTorch

### Raster Vision by Azavea
* https://www.azavea.com/projects/raster-vision/
* An open source Python framework for building computer vision models on aerial, satellite, and other large imagery sets.
* Accessible through the [Raster Foundry](https://rasterfoundry.azavea.com/)
* [Example use cases on open data](https://github.com/azavea/raster-vision-examples)

### spaceml.org
* http://spaceml.org/
* A Machine Learning toolbox and developer community building the next generation AI applications for space science and exploration.

### DeepNetsForEO (no activity since 2019)
* https://github.com/nshaud/DeepNetsForEO
* Uses SegNET for working on remote sensing images using deep learning.

### Skynet-data (no activity since 2018)
* https://github.com/developmentseed/skynet-data
* Data pipeline for machine learning with OpenStreetMap

### RoboSat (no longer maintained)
* https://github.com/mapbox/robosat
* Semantic segmentation on aerial and satellite imagery. Extracts features such as: buildings, parking lots, roads, water, clouds
* [robosat-jupyter-notebook](https://github.com/Element84/robosat-jupyter-notebook) -> walks through all of the steps in an excellent blog post on the Robosat feature extraction and machine learning pipeline.
* Note there is/was fork of Robosat, originally named RoboSat.pink, and subsequently neat-EO.pink although this appears to be dead/archived

### DeepOSM (no activity since 2017)
* https://github.com/trailbehind/DeepOSM
* Train a deep learning net with OpenStreetMap features and satellite imagery.

# Techniques
This section explores the different deep and machine learning techniques people are applying to common problems in satellite imagery analysis. Note that almost all imagery data on the internet is in RGB format, and common techniques designed for working with this 3 band imagery may fail or need significant adaptation to work with multiband data (e.g. 13-band Sentinel 2). Also the vast majority of the literature uses supervised learning with the requirement for large volumes of annotated data, which is a bottleneck to development and deployment. We are just starting to see unsupervised approached applied to remote sensing data.

## Land classification
Assign a label to an image, e.g. this is an image of a forest
* Land classification using a [simple sklearn cluster algorithm](https://github.com/acgeospatial/Satellite_Imagery_Python/blob/master/Clustering_KMeans-Sentinel2.ipynb) or [deep learning](https://towardsdatascience.com/land-use-land-cover-classification-with-deep-learning-9a5041095ddb).
* [Land Use Classification using Convolutional Neural Network in Keras](https://github.com/tavgreen/landuse_classification)
* [Sea-Land segmentation using DL](https://arxiv.org/pdf/1709.00201.pdf)
* [A U-net based on Tensorflow for objection detection (or segmentation) of satellite images - DSTL dataset but python 2.7](https://github.com/rogerxujiang/dstl_unet)
* [What’s growing there? Using eo-learn and fastai to identify crops from multi-spectral remote sensing data (Sentinel 2)](https://towardsdatascience.com/whats-growing-there-a5618a2e6933)
* [FastAI Multi-label image classification](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95)
* [Land use classification using Keras](https://examples.pyviz.org/landuse_classification/Image_Classification.html#landuse-classification-gallery-image-classification) and UC Merced dataset
* [Detecting Informal Settlements from Satellite Imagery using fine-tuning of ResNet-50 classifier](https://blog.goodaudience.com/detecting-informal-settlements-using-satellite-imagery-and-convolutional-neural-networks-d571a819bf44) with [repo](https://github.com/dymaxionlabs/ap-latam)
* Land use classification of UC Merced dataset using [Keras](https://examples.pyviz.org/landuse_classification/Image_Classification.html#landuse-classification-gallery-image-classification) or alternatively [fastai](https://medium.com/spatial-data-science/deep-learning-for-geospatial-data-applications-multi-label-classification-2b0a1838fcf3)
* [Water Detection in High Resolution Satellite Images using the waterdetect python package](https://towardsdatascience.com/water-detection-in-high-resolution-satellite-images-using-the-waterdetect-python-package-7c5a031e3d16) -> The main idea is to combine water indexes (NDWI, MNDWI, etc.) with reflectance bands (NIR, SWIR, etc.) into an automated clustering process
* [Contrastive Sensor Fusion](https://github.com/descarteslabs/contrastive_sensor_fusion) -> Code implementing Contrastive Sensor Fusion, an approach for unsupervised learning of multi-sensor representations targeted at remote sensing imagery.
* [Codebase for land cover classification with U-Net](https://github.com/jaeeolma/lulc_ml)
* [Multi-Label Classification of Satellite Photos of the Amazon Rainforest](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/) -> uses the Planet dataset & TF 2 & Keras
* [UrbanLandUse](https://github.com/wri/UrbanLandUse) -> This repository contains a comprehensive set of instructions for creating and applying ML models that characterize land use / land cover (LULC) in urban areas.
* [Land cover classification of Sundarbans satellite imagery using K-Nearest Neighbor(K-NNC), Support Vector Machine (SVM), and Gradient Boosting classification algorithms with Python](https://towardsdatascience.com/land-cover-classification-in-satellite-imagery-using-python-ae39dbf2929)
* [Ground Truth Labeling of Satellite Imagery using K-Means Clustering with Python](https://towardsdatascience.com/ground-truth-labeling-of-satellite-imagery-using-k-means-clustering-with-python-b4ab02e1314f)
* [Deep Learning for Land Cover Classification of Satellite Imagery Using Python](https://towardsdatascience.com/deep-learning-for-land-cover-classification-of-satellite-imagery-using-python-e7ca9f7bfa0a)
* [Multi-label Land Cover Classification with Deep Learning](https://towardsdatascience.com/multi-label-land-cover-classification-with-deep-learning-d39ce2944a3d) using the redesigned Multi-label UC Merced dataset with 17 land cover classes
* [Vision Transformers Use Case: Satellite Image Classification without CNNs](https://joaootavionf007.medium.com/vision-transformers-use-case-satellite-image-classification-without-cnns-2c4dbeb06f87)
* [hyperspectral_deeplearning_review](https://github.com/mhaut/hyperspectral_deeplearning_review) -> Code of December 2019 paper ["Deep Learning Classifiers for Hyperspectral Imaging: A Review"](https://www.sciencedirect.com/science/article/pii/S0924271619302187)
* [Deep Learning-Based Classification of Hyperspectral Data](https://github.com/hantek/deeplearn_hsi)
* [AutoEncoders for Land Cover Classification of Hyperspectral Images](https://towardsdatascience.com/autoencoders-for-land-cover-classification-of-hyperspectral-images-part-1-c3c847ebc69b) -> An autoencoder nerual net is used to reduce 103 band data to 60 features (dimensionality reduction), keras
* [Tree species classification from from airborne LiDAR and hyperspectral data using 3D convolutional neural networks](https://github.com/jaeeolma/tree-detection-evo#individual-tree-detection-and-matching-field-data-to-detected-tree-crowns)
* [hyperspectral-autoencoders](https://github.com/lloydwindrim/hyperspectral-autoencoders) -> Tools for training and using unsupervised autoencoders and supervised deep learning classifiers for hyperspectral data, built on tensorflow. Autoencoders are unsupervised neural networks that are useful for a range of applications such as unsupervised feature learning and dimensionality reduction.
* [Applying Deep Learning on Satellite Imagery Classification](https://medium.com/analytics-vidhya/applying-deep-learning-on-satellite-imagery-951faa0cbb31) -> using EuroSAT dataset of RGB and multi spectral covering 13 spectral bands, [with repo](https://github.com/raoofnaushad/Land-Cover-Classification-using-Sentinel-2-Dataset)
* [Land Cover Classification of Satellite Imagery using Convolutional Neural Networks](https://towardsdatascience.com/land-cover-classification-of-satellite-imagery-using-convolutional-neural-networks-91b5bb7fe808) using Keras and a dataset captured over Salinas Valley, California

## Semantic segmentation
Whilst classification will assign a label to a whole image, semantic segmentation will assign a label to each **pixel**
* [Instance segmentation with keras - links to satellite examples](https://github.com/matterport/Mask_RCNN)
* [Semantic Segmentation on Aerial Images using fastai](https://medium.com/swlh/semantic-segmentation-on-aerial-images-using-fastai-a2696e4db127)
* https://github.com/Paulymorphous/Road-Segmentation
* [UNSOAT used fast.ai to train a Unet to perform semantic segmentation on satellite imageries to detect water](https://forums.fast.ai/t/unosat-used-fastai-ai-for-their-floodai-model-discussion-on-how-to-move-forward/78468) - [paper](https://www.mdpi.com/2072-4292/12/16/2532) + [notebook](https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/Fastai%20training.ipynb), accuracy 0.97, precision 0.91, recall 0.92.
* [Identification of roads and highways using Sentinel-2 imagery (10m) super-resolved using the SENX4 model up to x4 the initial spatial resolution (2.5m)](https://tracasa.es/innovative-stories/sen2roadlasviastambiensevendesdesentinel-2/)
* [find-unauthorized-constructions-using-aerial-photography](https://medium.com/towards-artificial-intelligence/find-unauthorized-constructions-using-aerial-photography-and-deep-learning-with-code-part-2-b56ca80c8c99) -> U-Net & Keras
* [WildFireDetection](https://github.com/yueureka/WildFireDetection) -> Using U-Net Model to Detect Wildfire from Satellite Imagery, with streamlit UI
* [Pixel level segmentation on Azure](https://github.com/Azure/pixel_level_land_classification)
* [DigitalGlobe article](http://gbdxstories.digitalglobe.com/boats/) - they use a combination classical techniques (masks, erodes) to reduce the search space (identifying water via [NDWI](https://en.wikipedia.org/wiki/Normalized_difference_water_index) which requires SWIR) then apply a binary DL classifier on candidate regions of interest. They deploy the final algo [as a task](https://github.com/platformstories/boat-detector) on their GBDX platform. They propose that in the future an R-CNN may be suitable for the whole process.
* [Сrор field boundary detection: approaches overview and main challenges](https://soilmate.medium.com/%D1%81r%D0%BE%D1%80-field-boundary-detection-approaches-overview-and-main-challenges-53736725cb06)
* [A Practical Method for High-Resolution Burned Area Monitoring Using Sentinel-2 and VIIRS](https://www.mdpi.com/2072-4292/13/9/1608) with [code](https://github.com/mnpinto/FireHR)
* [instance-segmentation-maskrcnn](https://github.com/saraivaufc/instance-segmentation-maskrcnn) -> Instance segmentation of center pivot irrigation system in Brazil using Landsat images and Convolutional Neural Network

## Object detection
A good introduction to the challenge of performing object detection on aerial imagery is given in [this paper](https://arxiv.org/abs/1902.06042v2). In summary, images are large and objects may comprise only a few pixels, easily confused with random features in background. An example task is detecting boats on the ocean, which should be simpler than land based detection owing to the relatively blank background in images, but is still challenging.
* Intro articles [here](https://medium.com/earthcube-stories/how-hard-it-is-for-an-ai-to-detect-ships-on-satellite-images-7265e34aadf0) and [here](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7).
* [Super-Resolution and Object Detection](https://medium.com/the-downlinq/super-resolution-and-object-detection-a-love-story-part-4-8ad971eef81e) -> Super-resolution is a relatively inexpensive enhancement that can improve object detection performance
* [Anomaly Detection on Mars using a GAN](https://omdena.com/projects/anomaly-detection-mars/)
* [Tackling the Small Object Problem in Object Detection](https://blog.roboflow.com/tackling-the-small-object-problem-in-object-detection)
* [Satellite Imagery Multiscale Rapid Detection with Windowed Networks (SIMRDWN)](https://github.com/avanetten/simrdwn) -> combines some of the leading object detection algorithms into a unified framework designed to detect objects both large and small in overhead imagery
* Several useful articles on [awesome-tiny-object-detection](https://github.com/kuanhungchen/awesome-tiny-object-detection)
* [Challenges with SpaceNet 4 off-nadir satellite imagery: Look angle and target azimuth angle](https://medium.com/the-downlinq/challenges-with-spacenet-4-off-nadir-satellite-imagery-look-angle-and-target-azimuth-angle-2402bc4c3cf6) -> building prediction in images taken at nearly identical look angles — for example, 29 and 30 degrees — produced radically different performance scores.
* [YOLTv4](https://github.com/avanetten/yoltv4) -> YOLTv4 is designed to detect objects in aerial or satellite imagery in arbitrarily large images that far exceed the ~600×600 pixel size typically ingested by deep learning object detection frameworks. Read [Announcing YOLTv4: Improved Satellite Imagery Object Detection](https://towardsdatascience.com/announcing-yoltv4-improved-satellite-imagery-object-detection-f5091e913fad)
* [Benchmarks for Object Detection in Aerial Images](https://github.com/dingjiansw101/AerialDetection) -> codebase created to build benchmarks for object detection in aerial images

#### Object detection - buildings
* [Machine Learning For Rooftop Detection and Solar Panel Installment](https://omdena.com/blog/machine-learning-rooftops/) discusses tiling large images and generating annotations from OSM data. Features of the roofs were calculated using a combination of contour detection and classification
* [Building footprint detection with fastai on the challenging SpaceNet7 dataset](https://deeplearning.berlin/satellite%20imagery/computer%20vision/fastai/2021/02/17/Building-Detection-SpaceNet7.html)
* [DeepSolar is a deep learning framework that analyzes satellite imagery to identify the GPS locations and sizes of solar panels](http://web.stanford.edu/group/deepsolar/ds)
* [Building Extraction with YOLT2 and SpaceNet Data](https://medium.com/the-downlinq/building-extraction-with-yolt2-and-spacenet-data-a926f9ffac4f)
* [Segmentation of buildings on kaggle](https://www.kaggle.com/kmader/synthetic-word-ocr/kernels)
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”.
* [AIcrowd dataset of building outlines](https://www.aicrowd.com/challenges/mapping-challenge-old) -> 300x300 pixel RGB images with annotations in MS-COCO format

#### Object detection - boats, planes & vehicles
* [Detecting Ships in Satellite Imagery](https://medium.com/dataseries/detecting-ships-in-satellite-imagery-7f0ca04e7964) using the Planet dataset and Keras
* [Truck Detection with Sentinel-2 during COVID-19 crisis](https://github.com/hfisser/Truck_Detection_Sentinel2_COVID19) -> moving objects in Sentinel-2 data causes a specific reflectance relationship in the RGB, which looks like a rainbow, and serves as a marker for trucks. Improve accuracy by only analysing roads.
* [Planet use non DL felzenszwalb algorithm to detect ships](https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ship-detector/01_ship_detector.ipynb)
* [EESRGAN](https://github.com/Jakaria08/EESRGAN) -> Small-Object Detection in Remote Sensing Images with End-to-End Edge-Enhanced GAN and Object Detector Network
* [Ship detection using k-means clustering & CNN classifier on patches](https://towardsdatascience.com/data-science-and-satellite-imagery-985229e1cd2f)

#### Object detection - trees & green areas
* [DeepForest](https://deepforest.readthedocs.io/en/latest/index.html) is a python package for training and predicting individual tree crowns from airborne RGB imagery
* [Official repository for the "Identifying trees on satellite images" challenge from Omdena](https://github.com/cienciaydatos/ai-challenge-trees)
* [Counting-Trees-using-Satellite-Images](https://github.com/A2Amir/Counting-Trees-using-Satellite-Images) -> create an inventory of incoming and outgoing trees for an annual tree inspections, uses keras
* [2020 Nature paper - An unexpectedly large count of trees in the West African Sahara and Sahel](https://www.nature.com/articles/s41586-020-2824-5) -> tree detection framework based on U-Net & tensorflow 2 with code [here](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel/tree/v1.0.0)
* [Detecting solar panels from satellite imagery](https://towardsdatascience.com/weekend-project-detecting-solar-panels-from-satellite-imagery-f6f5d5e0da40)
* [Find sports fields using Mask R-CNN and overlay on open-street-map](https://github.com/jremillard/images-to-osm)
* [DeepForest](https://github.com/weecology/DeepForest) -> Python Package for Tree Crown Detection in Airborne RGB imagery

## Cloud detection
* From [this article on sentinelhub](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13) there are three popular classical algorithms that detects thresholds in multiple bands in order to identify clouds. In the same article they propose using semantic segmentation combined with a CNN for a cloud classifier (excellent review paper [here](https://arxiv.org/pdf/1704.06857.pdf)), but state that this requires too much compute resources.
* [This article](https://www.mdpi.com/2072-4292/8/8/666) compares a number of ML algorithms, random forests, stochastic gradient descent, support vector machines, Bayesian method.
* [Segmentation of Clouds in Satellite Images Using Deep Learning](https://medium.com/swlh/segmentation-of-clouds-in-satellite-images-using-deep-learning-a9f56e0aa83d) -> a U-Net is employed to interpret and extract the information embedded in the satellite images in a multi-channel fashion, and finally output a pixel-wise mask indicating the existence of cloud.
* [Cloud Detection in Satellite Imagery](https://www.azavea.com/blog/2021/02/08/cloud-detection-in-satellite-imagery/) compares FPN and CheapLab architectures on Sentinel-2 L1C and L2A imagery

## Change detection
Monitor water levels, coast lines, size of urban areas, wildfire damage. Note, clouds change often too..!
* [awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection)
* Using PCA (python 2, requires updating) -> https://appliedmachinelearning.blog/2017/11/25/unsupervised-changed-detection-in-multi-temporal-satellite-images-using-pca-k-means-python-code/
* Using CNN -> https://github.com/vbhavank/Unstructured-change-detection-using-CNN
* [Siamese neural network to detect changes in aerial images](https://github.com/vbhavank/Siamese-neural-network-for-change-detection)
* https://www.spaceknow.com/
* [LANDSAT Time Series Analysis for Multi-temporal Land Cover Classification using Random Forest](https://github.com/agr-ayush/Landsat-Time-Series-Analysis-for-Multi-Temporal-Land-Cover-Classification)
* [Change Detection in 3D: Generating Digital Elevation Models from Dove Imagery](https://www.planet.com/pulse/publications/change-detection-in-3d-generating-digital-elevation-models-from-dove-imagery/)
* [Change Detection in Hyperspectral Images Using Recurrent 3D Fully Convolutional Networks](https://www.mdpi.com/2072-4292/10/11/1827)
* [QGIS 2 plugin for applying change detection algorithms on high resolution satellite imagery](https://github.com/dymaxionlabs/massive-change-detection)
* [Change-Detection-Review](https://github.com/MinZHANG-WHU/Change-Detection-Review) -> A review of change detection methods, including codes and open data sets for deep learning.
* [Flood Detection and Monitoring using Satellite Imagery with Python](https://towardsdatascience.com/flood-detection-and-monitoring-using-satellite-imagery-with-python-5cd32157206c)
* [LamboiseNet](https://github.com/hbaudhuin/LamboiseNet) -> Master thesis about change detection in satellite imagery using Deep Learning

## Wealth and economic activity measurement
The goal is to predict economic activity from satellite imagery rather than conducting labour intensive ground surveys
* [Using publicly available satellite imagery and deep learning to understand economic well-being in Africa, Nature Comms 22 May 2020](https://www.nature.com/articles/s41467-020-16185-w) -> Used CNN on Ladsat imagery (night & day) to predict asset wealth of African villages
* [Combining Satellite Imagery and machine learning to predict poverty](https://towardsdatascience.com/combining-satellite-imagery-and-machine-learning-to-predict-poverty-884e0e200969) -> review article
* [Measuring Human and Economic Activity from Satellite Imagery to Support City-Scale Decision-Making during COVID-19 Pandemic](https://arxiv.org/abs/2004.07438)
* [Predicting Food Security Outcomes Using CNNs for Satellite Tasking](https://arxiv.org/pdf/1902.05433.pdf)
* [Crop yield Prediction with Deep Learning](https://github.com/JiaxuanYou/crop_yield_prediction) -> The necessary code for the paper Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data, AAAI 2017 (Best Student Paper Award in Computational Sustainability Track).
* https://github.com/taspinar/sidl/blob/master/notebooks/2_Detecting_road_and_roadtypes_in_sattelite_images.ipynb
* [Measuring the Impacts of Poverty Alleviation Programs with Satellite Imagery and Deep Learning](https://github.com/luna983/beyond-nightlight)
* [Traffic density estimation as a regression problem](https://omdena.com/blog/ai-road-safety/)
* [Crop Yield Prediction Using Deep Neural Networks and LSTM](https://omdena.com/blog/deep-learning-yield-prediction/) and [Building a Crop Yield Prediction App in Senegal Using Satellite Imagery and Jupyter](https://omdena.com/blog/yield-prediction/)
* [Advanced Deep Learning Techniques for Predicting Maize Crop Yield using Sentinel-2 Satellite Imagery](https://zionayomide.medium.com/advanced-deep-learning-techniques-for-predicting-maize-crop-yield-using-sentinel-2-satellite-1b63ac8b0789)

## Super-resolution
Super-resolution imaging is a class of techniques that enhance the resolution of an imaging system, and can be applied as a pre-processing step to improve the detection of small objects. For an introduction to this topic [read this excellent article](https://bleedai.com/super-resolution-going-from-3x-to-8x-resolution-in-opencv/)
* https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-1-ec5c5cd3cd2 -> Nov 2016 blog post by CosmiQ Works with a nice introduction to the topic. Proposes and demonstrates a new architecture with perturbation layers with practical guidance on the methodology and [code](https://github.com/CosmiQ/super-resolution). [Three part series](https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-3-2e2f61eee1d3)
* [Super Resolution for Satellite Imagery - srcnn repo](https://github.com/WarrenGreen/srcnn)
* [TensorFlow implementation of "Accurate Image Super-Resolution Using Very Deep Convolutional Networks" adapted for working with geospatial data](https://github.com/CosmiQ/VDSR4Geo) 
* [Random Forest Super-Resolution (RFSR repo)](https://github.com/jshermeyer/RFSR) including [sample data](https://github.com/jshermeyer/RFSR/tree/master/SampleImagery)
* [Super-Resolution (python) Utilities for managing large satellite images](https://github.com/jshermeyer/SR_Utils)
* [Enhancing Sentinel 2 images by combining Deep Image Prior and Decrappify](https://medium.com/omdena/pushing-the-limits-of-open-source-data-enhancing-satellite-imagery-through-deep-learning-9d8a3bbc0e0a). Repo for [deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior) and article on [decrappify](https://www.fast.ai/2019/05/03/decrappify/)
* The keras docs have a great tutorial - [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/)
* [HighRes-net](https://github.com/ElementAI/HighRes-net) -> Pytorch implementation of HighRes-net, a neural network for multi-frame super-resolution, trained and tested on the European Space Agency’s Kelvin competition
* [super-resolution-using-gan](https://github.com/saraivaufc/super-resolution-using-gan) -> Super-Resolution of Sentinel-2 Using Generative Adversarial Networks
* [Super-resolution of Multispectral Satellite Images Using Convolutional Neural Networks](https://up42.com/blog/tech/super-resolution-of-multispectral-satellite-images-using-convolutional) with [paper](https://arxiv.org/abs/2002.00580)
* [Small-Object Detection in Remote Sensing Images with End-to-End Edge-Enhanced GAN and Object Detector Network](https://paperswithcode.com/paper/small-object-detection-in-remote-sensing) -> enhanced super-resolution GAN (ESRGAN)
* [pytorch-enhance](https://github.com/isaaccorley/pytorch-enhance) -> Library of Image Super-Resolution Models, Datasets, and Metrics for Benchmarking or Pretrained Use

## Image-to-image translation using GANS
Generative Adversarial Networks, or GANS, can be used to translate images, e.g. from SAR to RGB.
* [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/) -> how to develop a Pix2Pix model for translating satellite photographs to Google map images. A good intro to GANS
* [SAR to RGB Translation using CycleGAN](https://www.esri.com/arcgis-blog/products/api-python/imagery/sar-to-rgb-translation-using-cyclegan/) -> uses a CycleGAN model in the ArcGIS API for Python
* [A growing problem of ‘deepfake geography’: How AI falsifies satellite images](https://www.washington.edu/news/2021/04/21/a-growing-problem-of-deepfake-geography-how-ai-falsifies-satellite-images/)
* [Kaggle Pix2Pix Maps](https://www.kaggle.com/alincijov/pix2pix-maps) -> dataset for pix2pix to take a google map satellite photo and build a street map

## SAR & Denoising
I group these together since I most often see denoising in the context of SAR imagery.
* Convolutional autoencoder network can be employed to image denoising, [read about this on the Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html)
* [Removing speckle noise from Sentinel-1 SAR using a CNN](https://medium.com/upstream/denoising-sentinel-1-radar-images-5f764faffb3e)
* A dataset which is specifically made for deep learning on SAR and optical imagery is the SEN1-2 dataset, which contains corresponding patch pairs of Sentinel 1 (VV) and 2 (RGB) data. It is the largest manually curated dataset of S1 and S2 products, with corresponding labels for land use/land cover mapping, SAR-optical fusion, segmentation and classification tasks. Data: https://mediatum.ub.tum.de/1474000
* [so2sat on Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/so2sat) -> So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and the corresponding local climate zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions of the world.
* [You do not need clean images for SAR despeckling with deep learning](https://towardsdatascience.com/you-do-not-need-clean-images-for-sar-despeckling-with-deep-learning-fe9c44350b69) -> How Speckle2Void learned to stop worrying and love the noise
* [PySAR - InSAR (Interferometric Synthetic Aperture Radar) timeseries analysis in python](https://github.com/hfattahi/PySAR)
* [Synthetic Aperture Radar (SAR) Analysis With Clarifai](https://www.clarifai.com/blog/synthetic-aperture-radar-sar-analysis-with-clarifai)
* [Labeled SAR imagery dataset of ten geophysical phenomena from Sentinel-1 wave mode](https://www.seanoe.org/data/00456/56796/) consists of more than 37,000 SAR vignettes divided into ten defined geophysical categories

## Unsupervised learning
Also referred to as self-supervised learning, [Yann LeCun in 2016](https://braindump.jethro.dev/posts/lecun_cake_analogy/) described un/self-supervised as the 'bulk of the cake': *If we think of our brain as a cake, then the cake base is unsupervised learning. The machine predicts any part of its input for any observed part (e.g. future frames in videos), all without the use of labelled data. Supervised learning forms the icing on the cake, and reinforcement learning is the cherry on top.*
* [Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data](https://github.com/ElementAI/seasonal-contrast) -> Seasonal Contrast (SeCo) is an effective pipeline to leverage unlabeled data for in-domain pre-training of remote sensing representations. Models trained with SeCo achieve better performance than their ImageNet pre-trained counterparts and state-of-the-art self-supervised learning methods on multiple downstream tasks. [paper](https://arxiv.org/abs/2103.16607)

## ML best practice & general techniques
* [4-ways-to-improve-class-imbalance](https://towardsdatascience.com/4-ways-to-improve-class-imbalance-for-image-data-9adec8f390f1) discusses the pros and cons of several rebalancing techniques, applied to an aerial dataset. Reason to read: models can reach an accuracy ceiling where majority classes are easily predicted but minority classes poorly predicted. Overall model accuracy may not improve until steps are taken to account for class imbalance.
* [Seven steps towards a satellite imagery dataset](https://omdena.com/blog/satellite-imagery-dataset/)
* [Implementing Transfer Learning from RGB to Multi-channel Imagery](https://towardsdatascience.com/implementing-transfer-learning-from-rgb-to-multi-channel-imagery-f87924679166) -> takes a resnet50 model pre-trained on an input of 224x224 pixels with 3 channels (RGB) and updates it for a new input of 480x400 pixels and 15 channels (12 new + RGB) using keras
* [How to implement augmentations for Multispectral Satellite Images Segmentation using Fastai-v2 and Albumentations](https://towardsdatascience.com/how-to-implement-augmentations-for-multispectral-satellite-images-segmentation-using-fastai-v2-and-ea3965736d1)
* [Principal Component Analysis: In-depth understanding through image visualization](https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f) applied to Landsat TM images, [with repo](https://github.com/Skumarr53/Principal-Component-Analysis-testing-on-Image-data)
* [Leveraging Geolocation Data for Machine Learning: Essential Techniques](https://towardsdatascience.com/leveraging-geolocation-data-for-machine-learning-essential-techniques-192ce3a969bc) -> A Gentle Guide to Feature Engineering and Visualization with Geospatial data, in Plain English
* [3 Tips to Optimize Your Machine Learning Project for Data Labeling](https://www.azavea.com/blog/2020/07/21/3-tips-to-optimize-your-machine-learning-project-for-data-labeling/)
* [Image Classification Labeling: Single Class versus Multiple Class Projects](https://www.azavea.com/blog/2020/06/08/image-classification-labeling-single-class-versus-multiple-class-projects/)
* [Labeling Satellite Imagery for Machine Learning](https://www.azavea.com/blog/2020/03/24/labeling-satellite-imagery-for-machine-learning/)
* [Image Augmentations for Aerial Datasets](https://blog.roboflow.com/image-augmentations-for-aerial-datasets/)

# Miscellaneous (generally) non ML Techniques
## Pansharpening
Image fusion of low res multispectral with high res pan band.
* Several algorithms described [in the ArcGIS docs](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/fundamentals-of-panchromatic-sharpening.htm), with the simplest being taking the mean of the pan and RGB pixel value.
* Does not require DL, classical algos suffice, [see this notebook](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/panchromatic_sharpening.ipynb) and [this kaggle kernel](https://www.kaggle.com/resolut/panchromatic-sharpening)
* https://github.com/mapbox/rio-pansharpen

## NVDI - vegetation index
* Simple band math `ndvi = np.true_divide((ir - r), (ir + r))` but challenging due to the size of the imagery.
* [Example notebook local](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/ndvi_calculation.ipynb)
* [Landsat data in cloud optimised (COG) format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article here](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).
* [Visualise water loss with Holoviews](https://examples.pyviz.org/walker_lake/Walker_Lake.html#walker-lake-gallery-walker-lake)

## Image registration
Image registration is the process of transforming different sets of data into one coordinate system. Typical use is overlapping images taken at different times or with different cameras.
* [Wikipedia article on registration](https://en.wikipedia.org/wiki/Image_registration) -> register for change detection or [image stitching](https://mono.software/2018/03/14/Image-stitching/)
* Traditional approach -> define control points, employ RANSAC algorithm
* [Phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) is used to estimate the translation between two images with sub-pixel accuracy. Can be used for accurate registration of low resolution imagery onto high resolution imagery, or to register a [sub-image on a full image](https://www.mathworks.com/help/images/registering-an-image-using-normalized-cross-correlation.html) -> Unlike many spatial-domain algorithms, the phase correlation method is resilient to noise, occlusions, and other defects.
* [cnn-registration](https://github.com/yzhq97/cnn-registration) -> A image registration method using convolutional neural network features written in Python2, Tensorflow 1.5

## Terrain mapping, Lidar & DEMs
Measure surface contours.
* [Wikipedia DEM article](https://en.wikipedia.org/wiki/Digital_elevation_model) and [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) article
* [Intro to depth from stereo](https://github.com/IntelRealSense/librealsense/blob/master/doc/depth-from-stereo.md)
* Map terrain from stereo images to produce a digital elevation model (DEM) -> high resolution & paired images required, typically 0.3 m, e.g. [Worldview](https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/37/DG-WV2ELEVACCRCY-WP.pdf) or [GeoEye](https://www.pobonline.com/articles/100233-when-is-satellite-stereo-imagery-the-best-option-for-3d-modeling).
* Process of creating a DEM [here](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLI-B1/327/2016/isprs-archives-XLI-B1-327-2016.pdf) and [here](https://www.geoimage.com.au/media/brochure_pdfs/Geoimage_DEM_brochure_Oct10_LR.pdf).
* [ArcGIS can generate DEMs from stereo images](http://pro.arcgis.com/en/pro-app/help/data/imagery/generate-elevation-data-using-the-dems-wizard.htm)
* https://github.com/MISS3D/s2p -> produces elevation models from images taken by high resolution optical satellites -> demo code on https://gfacciol.github.io/IS18/
* [Automatic 3D Reconstruction from Multi-Date Satellite Images](http://dev.ipol.im/~facciolo/pub/CVPRW2017.pdf)
* [Semi-global matching with neural networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Seki_SGM-Nets_Semi-Global_Matching_CVPR_2017_paper.pdf)
* [Predict the fate of glaciers](https://github.com/geohackweek/glacierhack_2018)
* [monodepth - Unsupervised single image depth prediction with CNNs](https://github.com/mrharicot/monodepth)
* [Stereo Matching by Training a Convolutional Neural Network to Compare Image Patches](https://github.com/jzbontar/mc-cnn)
* [Terrain and hydrological analysis based on LiDAR-derived digital elevation models (DEM) - Python package](https://github.com/giswqs/lidar)
* [Phase correlation in scikit-image](https://scikit-image.org/docs/0.13.x/auto_examples/transform/plot_register_translation.html)
* [s2p](https://github.com/cmla/s2p) -> a Python library and command line tool that implements a stereo pipeline which produces elevation models from images taken by high resolution optical satellites such as Pléiades, WorldView, QuickBird, Spot or Ikonos
* The [Mapbox API](https://docs.mapbox.com/help/troubleshooting/access-elevation-data/) provides images and elevation maps, [article here](https://towardsdatascience.com/creating-high-resolution-satellite-images-with-mapbox-and-python-750b3ac83dd7)
* [Reconstructing 3D buildings from aerial LiDAR with Mask R-CNN](https://medium.com/geoai/reconstructing-3d-buildings-from-aerial-lidar-with-ai-details-6a81cb3079c0)

# Image formats, data management and catalogues
* [GeoServer](http://geoserver.org/) -> an open source server for sharing geospatial data
* Open Data Cube - serve up cubes of data https://www.opendatacube.org/
* https://terria.io/ for pretty catalogues
* [Remote pixel](https://remotepixel.ca/projects/index.html#satsearch)
* [Sentinel-hub eo-browser](https://apps.sentinel-hub.com/eo-browser/)
* Large datasets may come in HDF5 format, can view with -> https://www.hdfgroup.org/downloads/hdfview/
* Climate data is often in netcdf format, which can be opened using xarray
* The xarray docs list a number of ways that data [can be stored and loaded](http://xarray.pydata.org/en/latest/io.html#).
* [TileDB](https://tiledb.com/) -> a 'Universal Data Engine' to store, analyze and share any data (beyond tables), with any API or tool (beyond SQL) at planet-scale (beyond clusters), open source and managed options. [Recently hiring](https://discourse.pangeo.io/t/job-openings-at-tiledb-inc/787) to work with xarray, dask, netCDF and cloud native storage
* [BigVector database](https://deepai.org/bigvector) -> A fully-managed, highly-scalable, and cost-effective database for vectors. Vectorize structured data or orbital imagery and discover new insights
* Read about [Serverless PostGIS on AWS Aurora](https://blog.addresscloud.com/serverless-postgis/)
* [Hub](https://github.com/activeloopai/Hub) -> The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Works locally or on any cloud. Read [Faster Machine Learning Using Hub by Activeloop: A code walkthrough of using the hub package for satellite imagery](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005)
* [A Comparison of Spatial Functions: PostGIS, Athena, PrestoDB, BigQuery vs RedShift](https://ual.sg/post/2020/07/03/a-comparison-of-spatial-functions-postgis-athena-prestodb-bigquery-vs-redshift/)

## Cloud Optimised GeoTiff (COG)
A Cloud Optimized GeoTIFF (COG) is a regular GeoTIFF that supports HTTP range requests, enabling downloading of specific tiles rather than the full file. COG generally work normally in GIS software such as QGIS, but are larger than regular GeoTIFFs
* https://www.cogeo.org/
* [cog-best-practices](https://github.com/pangeo-data/cog-best-practices)
* [rio-cogeo](https://cogeotiff.github.io/rio-cogeo/) -> Cloud Optimized GeoTIFF (COG) creation and validation plugin for Rasterio.
* [aiocogeo](https://github.com/geospatial-jeff/aiocogeo) -> Asynchronous cogeotiff reader (python asyncio)
* [Landsat data in cloud optimised (COG) format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article Cloud Native Geoprocessing of Earth Observation Satellite Data with Pangeo](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).
* [Working with COGS and STAC in python using geemap](https://geemap.org/notebooks/44_cog_stac/)
* [Load, Experiment, and Download Cloud Optimized Geotiffs (COG) using Python with Google Colab](https://towardsdatascience.com/access-satellite-imagery-with-aws-and-google-colab-4660178444f5) -> short read which covers finding COGS, opening with Rasterio and doing some basic manipulations, all in a Colab Notebook.
* [Exploring USGS Terrain Data in COG format using hvPlot](https://discourse.holoviz.org/t/exploring-usgs-terrain-data-in-cog-format-using-hvplot/1727) -> local COG from public AWS bucket, open with rioxarray, visualise with [hvplot](https://hvplot.holoviz.org/). See [the Jupyter notebook](https://nbviewer.jupyter.org/gist/rsignell-usgs/9657896371bb4f38437505146555264c)
* [aws-lambda-docker-rasterio](https://github.com/addresscloud/aws-lambda-docker-rasterio) -> AWS Lambda Container Image with Python Rasterio for querying Cloud Optimised GeoTiffs. See [this presentation](https://blog.addresscloud.com/rasters-revealed-2021/)
* [cogbeam](https://github.com/GoogleCloudPlatform/cogbeam) -> a python based Apache Beam pipeline, optimized for Google Cloud Dataflow, which aims to expedite the conversion of traditional GeoTIFFs into COGs
* [cogserver](https://github.com/rouault/cogserver) -> Expose a GDAL file as a HTTP accessible on-the-fly COG
* [Displaying a gridded dataset on a web-based map - Step by step guide for displaying large GeoTIFFs, using Holoviews, Bokeh, and Datashader](https://towardsdatascience.com/displaying-a-gridded-dataset-on-a-web-based-map-ad6bbe90247f)

## STAC - SpatioTemporal Asset Catalog specification
The STAC specification provides a common metadata specification, API, and catalog format to describe geospatial assets, so they can more easily indexed and discovered.
* A good place to start is to view the [Planet Disaster Data catalogue](https://planet.stac.cloud/) which has the [catalogue source on Github](https://github.com/cholmes/pdd-stac) and uses the [stac-browser](https://github.com/radiantearth/stac-browser)
* Spec at https://github.com/radiantearth/stac-spec
* [Getting Started with STAC APIs](https://www.azavea.com/blog/2021/04/05/getting-started-with-stac-apis/) intro article
* [SpatioTemporal Asset Catalog API specification](https://github.com/radiantearth/stac-api-spec) -> an API to make geospatial assets openly searchable and crawlable
* [stacindex](https://stacindex.org/) -> STAC Catalogs, Collections, APIs, Software and Tools
* Several useful repos on https://github.com/sat-utils
* [Intake-STAC](https://github.com/intake/intake-stac) -> Intake-STAC provides an opinionated way for users to load Assets from STAC catalogs into the scientific Python ecosystem. It uses the intake-xarray plugin and supports several file formats including GeoTIFF, netCDF, GRIB, and OpenDAP.
* [sat-utils/sat-search](https://github.com/sat-utils/sat-search) -> Sat-search is a Python 3 library and a command line tool for discovering and downloading publicly available satellite imagery using STAC compliant API
* [franklin](https://github.com/azavea/franklin) -> A STAC/OGC API Features Web Service focused on ease-of-use for end-users.
* [stacframes](https://github.com/azavea/stacframes) -> A Python library for working with STAC Catalogs via Pandas DataFrames
* [sat-api-pg](https://github.com/developmentseed/sat-api-pg) -> A Postgres backed STAC API
* [stactools](https://github.com/stac-utils/stactools) -> Command line utility and Python library for STAC
* [pystac](https://github.com/stac-utils/pystac) -> Python library for working with any STAC Catalog
* [STAC Examples for Nightlights data](https://github.com/developmentseed/nightlights_stac_examples) -> minimal example STAC implementation for the [Light Every Night](https://registry.opendata.aws/wb-light-every-night/) dataset of all VIIRS DNB and DMSP-OLS nighttime satellite data
* [stackstac](https://github.com/gjoseph92/stackstac) -> Turn a STAC catalog into a dask-based xarray

# State of the art
What are companies doing?
* A [serverless pipeline](https://github.com/aws-samples/amazon-rekognition-video-analyzer) appears to be where companies are headed for routine compute tasks. Checkout [process Satellite data using AWS Lambda functions](https://github.com/RemotePixel/remotepixel-api). Just beware of runtime limits and [cold starts](https://mikhail.io/serverless/coldstarts/aws/)
* Traditional data formats aren't designed for processing, so new standards are developing such as COGS
* Google provide training on how to use Apache Spark on Google Cloud Dataproc to distribute a computationally intensive (satellite) image processing task onto a cluster of machines -> https://google.qwiklabs.com/focuses/5834?parent=catalog
* Read about [Planet on Google](https://cloud.google.com/customers/planet) and also how [Airbus use Google](https://cloud.google.com/customers/airbus) as the backend for their [OneAtlas](https://oneatlas.airbus.com/) data portal

# Online platforms for Geo analysis
* [This article discusses some of the available platforms](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2)
* [Pangeo](http://pangeo.io/index.html) -> There is no single software package called “pangeo”; rather, the Pangeo project serves as a coordination point between scientists, software, and computing infrastructure. Includes open source resources for parallel processing using Dask and Xarray. Pangeo recently announced their 2.0 goals: pivoting away from directly operating cloud-based JupyterHubs, and towards eductaion and research
* [Airbus Sandbox](https://sandbox.intelligence-airbusds.com/web/) -> will provide access to imagery
* [Descartes Labs](https://www.descarteslabs.com/) -> access to EO imagery from a variety of providers via python API
* DigitalGlobe have a cloud hosted Jupyter notebook platform called [GBDX](https://gbdxdocs.digitalglobe.com/docs/about-the-gbdx-platform). Cloud hosting means they can guarantee the infrastructure supports their algorithms, and they appear to be close/closer to deploying DL.
* Planet have a [Jupyter notebook platform](https://developers.planet.com/) which can be deployed locally.
* [jupyteo.com](https://www.jupyteo.com) -> hosted Jupyter environment with many features for working with EO data
* [eurodatacube.com](https://eurodatacube.com) -> data & platform for EO analytics in Jupyter env, paid
* [Unfolded Studio](https://studio.unfolded.ai/) -> next generation geospatial analytics and visualization platform building on open source geospatial technologies including kepler.gl, deck.gl and H3. Processing is down browser side enabling excellent performance. [Rasters support added April 2021](https://www.unfolded.ai/blog/2021-04-28-raster-layer/)
* [up42](https://up42.com/) is a developer platform and marketplace, offering all the building blocks for powerful, scalable geospatial products
* [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) -> direct Google Earth Engine competitor in the making?

# Free online computing resources
Generally a GPU is required for DL, and this section lists a couple of free Jupyter environments with GPU available. There is a good overview of online Jupyter development environments [on the fast.ai site](https://course19.fast.ai). I personally use Colab Pro with data hosted on Google Drive

### Google Colab
* Collaboratory [notebooks](https://colab.research.google.com) with GPU as a backend for free for 12 hours at a time. Note that the GPU may be shared with other users, so if you aren't getting good performance try reloading.
* Also a pro tier for $10 a month -> https://colab.research.google.com/signup
* Tensorflow, pytorch & fast.ai available but you may need to update them

### Kaggle - also Google!
* Free to use
* GPU Kernels - may run for 1 hour
* Tensorflow, pytorch & fast.ai available but you may need to update them
* Advantage that many datasets are already available

# Production
For an overview on serving deep learning models checkout [Practical-Deep-Learning-on-the-Cloud](https://github.com/PacktPublishing/-Practical-Deep-Learning-on-the-Cloud).

### Rest API on dedicated server
A conceptually simple approach to serving up deep learning model inference code is to wrap it in a rest API. That can be implemented in python (flask or FastAPI), and hosted on a dedicated server e.g. EC2 instance
* Basic API: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html with code [here](https://github.com/jrosebr1/simple-keras-rest-api)
* Advanced API with request queuing: https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/

### Framework specific serving options
* [Tensorflow](https://www.tensorflow.org/tfx/guide/serving) and [pytorch](https://github.com/pytorch/serve) specific serving

### AWS
* Host your data on [S3](https://aws.amazon.com/s3/) and metadata in a relational db such as [postgres](https://aws.amazon.com/rds/postgresql/)
* For batch processing use [Batch](https://aws.amazon.com/batch/) to run python scripts. Break out units of processing into [Lambda](https://aws.amazon.com/lambda/) functions. Note that lambda may not be a particularly quick solution for deep learning applications, since you do not have the option to batch inference on a GPU. There is also a hard runtime limit of 15 minutes, and creating a container with all the required dependencies can be a challenge. To get started read [Using container images to run PyTorch models in AWS Lambda](https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/)
* Use [Step functions](https://aws.amazon.com/step-functions/) to orchestrate data pipelines on batch and lambda. If this is too limited or you want to write pipelines in python (rather than json used by step functions) checkout [Prefect](https://www.prefect.io/)
* [Sagemaker](https://aws.amazon.com/sagemaker/) is a hosted Jupyter environment for training and deployment of ML models.
* [Deep learning AMIs](https://aws.amazon.com/machine-learning/amis/) are EC2 instances with deep learning frameworks preinstalled. They do require more setup from the user than Sagemaker but in return allow access to the underlying hardware, which makes debugging issues more straightforward. There is a [good guide to setting up your AMI instance on the Keras blog](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html)
* [Rekognition](https://aws.amazon.com/rekognition/custom-labels-features/) custom labels is a 'no code' annotation, training and inferencing service. Read [Training models using Satellite (Sentinel-2) imagery on Amazon Rekognition Custom Labels](https://ryfeus.medium.com/training-models-using-satellite-imagery-on-amazon-rekognition-custom-labels-dd44ac6a3812). For a comparison with Azure and Google alternatives [read this article](https://blog.roboflow.com/automl-vs-rekognition-vs-custom-vision/)
* When developing you will definitely want to use [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) and probably [aws-data-wrangler](https://github.com/awslabs/aws-data-wrangler)
* For managing infrastructure use [Terraform](https://www.terraform.io/). Alternatively if you wish to use TypeScript, JavaScript, Python, Java, or C# checkout [AWS CDK](https://aws.amazon.com/cdk/), although I found relatively few examples to get going using python
* [AWS Ground Station now supports data delivery to Amazon S3](https://aws.amazon.com/about-aws/whats-new/2021/04/aws-ground-station-now-supports-data-delivery-to-amazon-s3/)

### chip-n-scale-queue-arranger by developmentseed
* https://github.com/developmentseed/chip-n-scale-queue-arranger
* an orchestration pipeline for running machine learning inference at scale
* [Supports fast.ai models](https://github.com/developmentseed/fastai-serving)

# Useful paid software
* [ArcGIS](https://www.arcgis.com/index.html) -> mapping and analytics software, with both local and cloud hosted options. Checkout [Geospatial deep learning with arcgis.learn](https://developers.arcgis.com/python/guide/geospatial-deep-learning/). It [appears](https://www.esri.com/arcgis-blog/products/api-python/imagery/sar-to-rgb-translation-using-cyclegan/) ArcGIS are using fastai for their deep learning backend. [ArcGIS Jupyter Notebooks](https://www.esri.com/arcgis-blog/products/arcgis-enterprise/analytics/introducing-arcgis-notebooks/) in ArcGIS Enterprise are built to run big data analysis, deep learning models, and dynamic visualization tools. 

# Useful open source software
[A note on licensing](https://www.gislounge.com/businesses-using-open-source-gis/): The two general types of licenses for open source are copyleft and permissive. Copyleft requires that subsequent derived software products also carry the license forward, e.g. the GNU Public License (GNU GPLv3). For permissive, options to modify and use the code as one please are more open, e.g. MIT & Apache 2. Checkout [choosealicense.com/](https://choosealicense.com/)
* [QGIS](https://qgis.org/en/site/)- Create, edit, visualise, analyse and publish geospatial information. [Python scripting](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/intro.html#scripting-in-the-python-console) and [plugins](https://plugins.qgis.org/plugins/). Open source alternative to ArcGIS.
* [Orfeo toolbox](https://www.orfeo-toolbox.org/) - remote sensing toolbox with python API (just a wrapper to the C code). Do activites such as [pansharpening](https://www.orfeo-toolbox.org/CookBook/Applications/app_Pansharpening.html), ortho-rectification, image registration, image segmentation & classification. Not much documentation.
* [QUICK TERRAIN READER - view DEMS, Windows](http://appliedimagery.com/download/)
* [dl-satellite-docker](https://github.com/sshuair/dl-satellite-docker) -> docker files for geospatial analysis, including tensorflow, pytorch, gdal, xgboost...
* [AIDE V2 - Tools for detecting wildlife in aerial images using active learning](https://github.com/microsoft/aerial_wildlife_detection)
* [Land Cover Mapping web app from Microsoft](https://github.com/microsoft/landcover)
* [Solaris](https://github.com/CosmiQ/solaris) -> An open source ML pipeline for overhead imagery by [CosmiQ Works](https://www.cosmiqworks.org/), similar to Rastervision but with some unique very vool features
* [openSAR](https://github.com/EarthBigData/openSAR) -> Synthetic Aperture Radar (SAR) Tools and Documents from Earth Big Data LLC (http://earthbigdata.com/)
* [qhub](https://qhub.dev) -> QHub enables teams to build and maintain a cost effective and scalable compute/data science platform in the cloud.
* [imagej](https://imagej.net) -> a very versatile image viewer and processing program
* [Geo Data Viewer](https://github.com/RandomFractals/geo-data-viewer) extension for VSCode which enables opening and viewing various geo data formats with nice visualisations
* [Datasette](https://datasette.io/) is a tool for exploring and publishing data as an interactive website and accompanying API, with SQLite backend. Various plugins extend its functionality, for example to allow displaying geospatial info, render images (useful for thumbnails), and add user authentication.
* [Photoprism](https://github.com/photoprism/photoprism) is a privately hosted app for browsing, organizing, and sharing your photo collection, with support for tiffs
* [dbeaver](https://github.com/dbeaver/dbeaver) is a free universal database tool and SQL client with [geospatial features](https://github.com/dbeaver/dbeaver/wiki/Working-with-Spatial-GIS-data)
* [Grafana](https://grafana.com/) can be used to make interactive dashboards, checkout [this example showing Point data](https://blog.timescale.com/blog/grafana-variables-101/). Note there is an [AWS managed service for Grafana](https://aws.amazon.com/grafana/)
* [litestream](https://litestream.io/) -> Continuously stream SQLite changes to S3-compatible storage
  
## GDAL & Rasterio
* So improtant this pair gets their own section. GDAL is THE command line tool for reading and writing raster and vector geospatial data formats. If you are using python you will probably want to use Rasterio which provides a pythonic wrapper for GDAL
* [GDAL](https://gdal.org) and [on twitter](https://twitter.com/gdaltips)
* GDAL is a dependency of Rasterio and can be difficult to build and install. I recommend using conda, brew (on OSX) or docker in these situations
* GDAL docker quickstart: `docker pull osgeo/gdal` then `docker run --rm -v $(pwd):/data/ osgeo/gdal gdalinfo /data/cog.tiff`
* [Even Rouault](https://github.com/rouault) maintains GDAL, please consider [sponsoring him](https://github.com/sponsors/rouault)
* [Rasterio](https://rasterio.readthedocs.io/en/latest/) -> reads and writes GeoTIFF and other raster formats and provides a Python API based on Numpy N-dimensional arrays and GeoJSON. There are a variety of plugins that extend Rasterio functionality.
* [rio-cogeo](https://cogeotiff.github.io/rio-cogeo/) -> Cloud Optimized GeoTIFF (COG) creation and validation plugin for Rasterio.
* [rioxarray](https://github.com/corteva/rioxarray) -> geospatial xarray extension powered by rasterio
* [aws-lambda-docker-rasterio](https://github.com/addresscloud/aws-lambda-docker-rasterio) -> AWS Lambda Container Image with Python Rasterio for querying Cloud Optimised GeoTiffs. See [this presentation](https://blog.addresscloud.com/rasters-revealed-2021/)
* [godal](https://github.com/airbusgeo/godal) -> golang wrapper for GDAL
* [Write rasterio to xarray](https://github.com/robintw/XArrayAndRasterio/blob/master/rasterio_to_xarray.py)
* [Loam: A Client-Side GDAL Wrapper for Javascript](https://github.com/azavea/loam)

## Python general utilities
* [PyShp](https://github.com/GeospatialPython/pyshp) -> The Python Shapefile Library (PyShp) reads and writes ESRI Shapefiles in pure Python
* [s2p](https://github.com/cmla/s2p) -> a Python library and command line tool that implements a stereo pipeline which produces elevation models from images taken by high resolution optical satellites such as Pléiades, WorldView, QuickBird, Spot or Ikonos
* [EarthPy](https://github.com/earthlab/earthpy) -> A set of helper functions to make working with spatial data in open source tools easier. read[Exploratory Data Analysis (EDA) on Satellite Imagery Using EarthPy](https://towardsdatascience.com/exploratory-data-analysis-eda-on-satellite-imagery-using-earthpy-c0e186fe4293)
* [pygeometa](https://geopython.github.io/pygeometa/) -> provides a lightweight and Pythonic approach for users to easily create geospatial metadata in standards-based formats using simple configuration files
* [pesto](https://airbusdefenceandspace.github.io/pesto/) -> PESTO is designed to ease the process of packaging a Python algorithm as a processing web service into a docker image. It contains shell tools to generate all the boiler plate to build an OpenAPI processing web service compliant with the Geoprocessing-API. By [Airbus Defence And Space](https://github.com/AirbusDefenceAndSpace)
* [GEOS](https://geos.readthedocs.io/en/latest/index.html) -> Google Earth Overlay Server (GEOS) is a python-based server for creating Google Earth overlays of tiled maps. Your can also display maps in the web browser, measure distances and print maps as high-quality PDF’s.
* [GeoDjango](https://docs.djangoproject.com/en/3.1/ref/contrib/gis/) intends to be a world-class geographic Web framework. Its goal is to make it as easy as possible to build GIS Web applications and harness the power of spatially enabled data. [Some features of GDAL are supported.](https://docs.djangoproject.com/en/3.1/ref/contrib/gis/gdal/)
* [rasterstats](https://pythonhosted.org/rasterstats/) -> summarize geospatial raster datasets based on vector geometries
* [turfpy](https://turfpy.readthedocs.io/en/latest/index.html) -> a Python library for performing geospatial data analysis which reimplements turf.js
* [image-similarity-measures](https://github.com/up42/image-similarity-measures) -> Implementation of eight evaluation metrics to access the similarity between two images. [Blog post here](https://up42.com/blog/tech/image-similarity-measures)

## Python low level numerical & data formats
* [xarray](http://xarray.pydata.org/en/stable/) -> N-D labeled arrays and datasets. Read [Handling multi-temporal satellite images with Xarray](https://medium.com/@bonnefond.virginie/handling-multi-temporal-satellite-images-with-xarray-30d142d3391). Checkout [xarray_leaflet](https://github.com/davidbrochart/xarray_leaflet) for tiled map plotting
* [xarray-spatial](https://github.com/makepath/xarray-spatial) -> Fast, Accurate Python library for Raster Operations. Implements algorithms using Numba and Dask, free of GDAL
* [xarray-beam](https://github.com/google/xarray-beam) -> Distributed Xarray with Apache Beam by Google
* [Geowombat](https://geowombat.readthedocs.io/) -> geo-utilities applied to air- and space-borne imagery, uses Rasterio, Xarray and Dask for I/O and distributed computing with named coordinates
* [NumpyTiles](https://github.com/planetlabs/numpytiles-spec) -> a specification for providing multiband full-bit depth raster data in the browser
* [Zarr](https://zarr.readthedocs.io/en/stable/) -> Zarr is a format for the storage of chunked, compressed, N-dimensional arrays. Zarr depends on NumPy

## Python image handling and manipulation
* [Pillow is the Python Imaging Library](https://pillow.readthedocs.io/en/stable/) -> this will be your go-to package for image manipulation in python
* [opencv-python](https://github.com/opencv/opencv-python) is pre-built CPU-only OpenCV packages for Python
* [kornia](https://github.com/kornia/kornia) is a differentiable computer vision library for PyTorch, like openCV but on the GPU. Perform image transformations, epipolar geometry, depth estimation, and low-level image processing such as filtering and edge detection that operate directly on tensors.
* [tifffile](https://github.com/cgohlke/tifffile) -> Read and write TIFF files
* [xtiff](https://github.com/BodenmillerGroup/xtiff) -> A small Python 3 library for writing multi-channel TIFF stacks
* [geotiff](https://github.com/Open-Source-Agriculture/geotiff) -> A noGDAL tool for reading and writing geotiff files
* [image_slicer](https://github.com/samdobson/image_slicer) -> Split images into tiles. Join the tiles back together.
* [tiler](https://github.com/nuno-faria/tiler) -> split images into tiles and merge tiles into a large image
* [felicette](https://github.com/plant99/felicette) -> Satellite imagery for dummies. Generate JPEG earth imagery from coordinates/location name with publicly available satellite data.
* [imagehash](https://github.com/JohannesBuchner/imagehash) -> Image hashes tell whether two images look nearly identical.
* [xbatcher](https://github.com/pangeo-data/xbatcher) -> Xbatcher is a small library for iterating xarray DataArrays in batches. The goal is to make it easy to feed xarray datasets to machine learning libraries such as Keras.
* [fake-geo-images](https://github.com/up42/fake-geo-images) -> A module to programmatically create geotiff images which can be used for unit tests
* [sahi](https://github.com/obss/sahi) -> A vision library for performing sliced inference on large images/small objects
* [imagededup](https://github.com/idealo/imagededup) -> Finding duplicate images made easy! Uses perceptual hashing

## Python deep learning toolsets
* [torchvision-enhance](https://github.com/sshuair/torchvision-enhance) -> Enhance PyTorch vision for semantic segmentation, multi-channel images and TIF file
* [DeepHyperX](https://github.com/eecn/Hyperspectral-Classification) -> A Python/pytorch tool to perform deep learning experiments on various hyperspectral datasets.
* [image-super-resolution](https://github.com/idealo/image-super-resolution) -> Super-scale your images and run experiments with Residual Dense and Adversarial Networks.

## Python data discovery and ingestion
* [landsat_ingestor](https://github.com/landsat-pds/landsat_ingestor) -> Scripts and other artifacts for landsat data ingestion into Amazon public hosting
* [satpy](https://github.com/pytroll/satpy) -> a python library for reading and manipulating meteorological remote sensing data and writing it to various image and data file formats
* [GIBS-Downloader](https://github.com/spaceml-org/GIBS-Downloader) -> a command-line tool which facilitates the downloading of NASA satellite imagery and offers different functionalities in order to prepare the images for training in a machine learning pipeline

## Python graphing and visualisation
* [hvplot](https://hvplot.holoviz.org/) -> A high-level plotting API for the PyData ecosystem built on HoloViews. Allows overlaying data on map tiles, see [Exploring USGS Terrain Data in COG format using hvPlot](https://discourse.holoviz.org/t/exploring-usgs-terrain-data-in-cog-format-using-hvplot/1727)
* [Pyviz](https://examples.pyviz.org/) examples include several interesting geospatial visualisations
* [napari](https://napari.org) -> napari is a fast, interactive, multi-dimensional image viewer for Python. It’s designed for browsing, annotating, and analyzing large multi-dimensional images. By integrating closely with the Python ecosystem, napari can be easily coupled to leading machine learning and image analysis tools. Note that to view a 3GB COG I had to install the [napari-tifffile-reader](https://github.com/GenevieveBuckley/napari-tifffile-reader) plugin.
* [pixel-adjust](https://github.com/cisaacstern/pixel-adjust) -> Interactively select and adjust specific pixels or regions within a single-band raster. Built with rasterio, matplotlib, and panel.
* [Plotly Dash](https://plotly.com/dash/) can be used for making interactive dashboards
* [folium](https://python-visualization.github.io/folium/) -> a python wrapper to the excellent [leaflet.js](https://leafletjs.com/) which makes it easy to visualize data that’s been manipulated in Python on an interactive leaflet map. Also checkout the [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) component for adding folium maps to your streamlit apps
* [ipyearth](https://github.com/davidbrochart/ipyearth) -> An IPython Widget for Earth Maps
* [geopandas-view](https://github.com/martinfleis/geopandas-view) -> Interactive exploration of GeoPandas GeoDataFrames
* [geogif](https://github.com/gjoseph92/geogif) -> Turn xarray timestacks into GIFs

## Python cluster computing with Dask
* Get started by reading [Democratizing Satellite Imagery Analysis with Dask](https://coiled.io/blog/democratizing-satellite-imagery-analysis-with-dask/)
* [Dask](https://docs.dask.org/en/latest/) works with your favorite PyData libraries to provide performance at scale for the tools you love -> checkout [Read and manipulate tiled GeoTIFF datasets](https://examples.dask.org/applications/satellite-imagery-geotiff.html#) and [accelerating-science-dask](https://coiled.io/blog/accelerating-science-dask-gentemann/)
* [Coiled](https://coiled.io) is a managed Dask service.
* [Dask with PyTorch for large scale image analysis](https://blog.dask.org/2021/03/29/apply-pretrained-pytorch-model)
* [stackstac](https://github.com/gjoseph92/stackstac) -> Turn a STAC catalog into a dask-based xarray
* [dask-geopandas](https://github.com/jsignell/dask-geopandas) -> Parallel GeoPandas with Dask
* [dask-image](https://github.com/dask/dask-image) -> many SciPy ndimage functions implemented

## Algorithms in python
* [WaterDetect](https://github.com/cordmaur/WaterDetect) -> an end-to-end algorithm to generate open water cover mask, specially conceived for L2A Sentinel 2 imagery. It can also be used for Landsat 8 images and for other multispectral clustering/segmentation tasks.
* [GatorSense Hyperspectral Image Analysis Toolkit](https://github.com/GatorSense/hsi_toolkit_py) -> This repo contains algorithms for Anomaly Detectors, Classifiers, Dimensionality Reduction, Endmember Extraction, Signature Detectors, Spectral Indices
* [arosics](https://danschef.git-pages.gfz-potsdam.de/arosics/doc/about.html) -> Perform automatic subpixel co-registration of two satellite image datasets based on an image matching approach
* [detectree](https://github.com/martibosch/detectree) -> Tree detection from aerial imagery
* [pylandstats](https://github.com/martibosch/pylandstats) -> compute landscape metrics

## Tools for image annotation
If you are performing object detection you will need to annotate images with bounding boxes. Check that your annotation tool of choice supports large image (likely geotiff) files, as not all will. Note that GeoJSON is widely used by remote sensing researchers but this annotation format is not commonly supported in general computer vision frameworks, and in practice you may have to convert the annotation format to use the data with your chosen framework. There are both closed and open source tools for creating and converting annotation formats.
* A long list of tools is [here](https://simonwenkel.com/2019/07/19/list-of-annotation-tools-for-machine-learning-research.html)
* [GroundWork](https://groundwork.azavea.com/) is designed for annotating and labeling geospatial data like satellite imagery, from Azavea
* [Labelme Image Annotation for Geotiffs](https://medium.com/@wvsharber/labelme-image-annotation-for-geotiffs-b460ba83804f) -> uses [Labelme](https://github.com/wkentaro/labelme)
* [Label Maker](https://github.com/developmentseed/label-maker) -> downloads OpenStreetMap QA Tile information and satellite imagery tiles and saves them as an `.npz` file for use in machine learning training.
* [CVAT](https://github.com/openvinotoolkit/cvat) is worth investigating, and have an [open issue](https://github.com/openvinotoolkit/cvat/issues/531) to support large TIFF files. [This article on Roboflow](https://blog.roboflow.com/cvat/) gives a good intro to CVAT.
* [Deep Block](https://app.deepblock.net) is a general purpose AI platform that includes a tool for COCOJSON export for aerial imagery. Checkout [this video](https://www.youtube.com/watch?v=gg5qSV-yw4U&feature=youtu.be)
* AWS supports image annotation via the [Rekognition Custom Labels console](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/gs-console.html)
* [Roboflow](https://roboflow.com) can be used to convert between annotation formats
* Other annotation tools include [supervise.ly](https://supervise.ly) (web UI), [rectlabel](https://rectlabel.com) (OSX desktop app) and [VoTT](https://github.com/Microsoft/VoTT)
* [Label Studio](https://github.com/heartexlabs/label-studio) is a multi-type data labeling and annotation tool with standardized output format, webpage at [labelstud.io](https://labelstud.io/)
* [Deeplabel](https://github.com/jveitchmichaelis/deeplabel) is a cross-platform tool for annotating images with labelled bounding boxes. Deeplabel also supports running inference using state-of-the-art object detection models like Faster-RCNN and YOLOv4. With support out-of-the-box for CUDA, you can quickly label an entire dataset using an existing model.
* [Alturos.ImageAnnotation](https://github.com/AlturosDestinations/Alturos.ImageAnnotation) is a collaborative tool for labeling image data on S3 for yolo
* [rectlabel](https://rectlabel.com/) is a desktop app for MacOS to label images for bounding box object detection and segmentation
* [pigeonXT](https://github.com/dennisbakhuis/pigeonXT) can be used to create custom image classification annotators within Jupyter notebooks
* [ipyannotations](https://github.com/janfreyberg/ipyannotations) -> Image annotations in python using jupyter notebooks
* [diffgram](https://github.com/diffgram/diffgram) supports cloud backends
* [Label-Detect](https://github.com/Jakaria08/Label-Detect) -> is a graphical image annotation tool and using this tool a user can also train and test large satellite images, fork of the popular labelImg tool
* [Swipe-Labeler](https://github.com/spaceml-org/Swipe-Labeler) -> Swipe Labeler is a Graphical User Interface based tool that allows rapid labeling of image data

# Movers and shakers on Github
* [Adam Van Etten](https://github.com/avanetten) is doing interesting things in object detection and segmentation
* [Andrew Cutts](https://github.com/acgeospatial) cohosts the [Scene From Above podcast](https://scenefromabove.podbean.com) and has many interesting repos
* [Ankit Kariryaa](https://github.com/ankitkariryaa) published a recent nature paper on tree detection
* [Chris Holmes](https://github.com/cholmes) is doing great things at Planet
* [Christoph Rieke](https://github.com/chrieke) maintains a very popular imagery repo and has published his thesis on segmentation
* [Even Rouault](https://github.com/rouault) maintains several of the most critical tools in this domain such as GDAL, please consider [sponsoring him](https://github.com/sponsors/rouault)
* [Jake Shermeyer](https://github.com/jshermeyer) many interesting repos
* [Mort Canty](https://github.com/mortcanty) is an expert in change detection
* [Nicholas Murray](https://www.murrayensis.org/) is an Australia-based scientist with a focus on delivering the science necessary to inform large scale environmental management and conservation
* [Qiusheng Wu](https://github.com/giswqs) is an Assistant Professor in the Department of Geography at the University of Tennessee
* [Robin Wilson](https://github.com/robintw) is a former academic who is very active in the satellite imagery space

# Companies on Github
For a full list of companies, on and off Github, checkout [awesome-geospatial-companies](https://github.com/chrieke/awesome-geospatial-companies). The following lists companies with interesting Github profiles.
* [Airbus Defence And Space](https://github.com/AirbusDefenceAndSpace)
* [Azavea](https://github.com/azavea) -> lots of interesting repos around STAC
* [Development Seed](https://github.com/developmentseed)
* [Descartes Labs](https://github.com/descarteslabs)
* [Digital Globe](https://github.com/DigitalGlobe)
* [Mapbox](https://github.com/mapbox) -> thanks for Rasterio!
* [Planet Labs](https://github.com/planetlabs) -> thanks for COGS!
* [up42](https://github.com/up42) -> Airbus spinout providing 'The easiest way to build geospatial solutions'

# Courses
* [Introduction to Geospatial Raster and Vector Data with Python](https://carpentries-incubator.github.io/geospatial-python/aio/index.html) -> an intro course on a single page
* [Manning: Monitoring Changes in Surface Water Using Satellite Image Data](https://liveproject.manning.com/course/106/monitoring-changes-in-surface-water-using-satellite-image-data?)
* [Automating GIS processes](https://automating-gis-processes.github.io/2016/index.html) includes a lesson on automating raster data processing
* For deep learning checkout the [fastai course](https://course.fast.ai/) which uses the fast.ai library & pytorch
* [pyimagesearch.com](https://www.pyimagesearch.com/) hosts courses and plenty of material using opencv and keras
* Official [opencv](https://opencv.org/courses/) courses
* [TensorFlow Developer Professional Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice)

# Books
* [Image Analysis, Classification and Change Detection in Remote Sensing With Algorithms for Python, Fourth Edition, By Morton John Canty](https://www.routledge.com/Image-Analysis-Classification-and-Change-Detection-in-Remote-Sensing-With/Canty/p/book/9781138613225) -> code [here](https://github.com/mortcanty/CRC4Docker)
* I highly recommend [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) by François Chollet

# Online communities
* [fast AI geospatial study group](https://forums.fast.ai/t/geospatial-deep-learning-resources-study-group/31044)
* [Kaggle Intro to Satellite imagery Analysis group](https://www.kaggle.com/getting-started/131455)
* [Omdena](https://omdena.com/) brings together small teams of engineers to work on AI projects

# Jobs
* [Pangeo discourse](https://discourse.pangeo.io/c/news/jobs) lists multiple jobs, global
* [Current Job Openings at Planet](https://boards.greenhouse.io/planetlabs)

# Neural nets in space
Processing on satellite allows less data to be downlinked. E.g. super-resolution image might take 4-8 images to generate, then a single image is downlinked.
* [Lockheed Martin and USC to Launch Jetson-Based Nanosatellite for Scientific Research Into Orbit - Aug 2020](https://news.developer.nvidia.com/lockheed-martin-usc-jetson-nanosatellite/) - One app that will run on the GPU-accelerated satellite is SuperRes, an AI-based application developed by Lockheed Martin, that can automatically enhance the quality of an image.
* [Intel to place movidius in orbit to filter images of clouds at source - Oct 2020](https://techcrunch.com/2020/10/20/intel-is-providing-the-smarts-for-the-first-satellite-with-local-ai-processing-on-board/) - Getting rid of these images before they’re even transmitted means that the satellite can actually realize a bandwidth savings of up to 30%
* Whilst not involving neural nets the [PyCubed](https://www.notion.so/PyCubed-4cbfac7e9b684852a2ab2193bd485c4d) project gets a mention here as it is putting python on space hardware such as the [V-R3x](https://www.nasa.gov/ames/v-r3x)

# About the author
My background is in optical physics, and I hold a PhD from Cambridge on the topic of [localised surface Plasmons](https://pubs.acs.org/doi/abs/10.1021/nl0710506). Since academia I have held a variety of roles, including doing research at [Sharp Labs Europe](https://www.sle.sharp.co.uk/), developing optical systems at [Surrey Satellites](https://www.sstl.co.uk/) (SSTL), and working at an IOT startup. It was whilst at SSTL that I started this repository as a personal resource. Over time I have steadily gravitated towards data analytics and software engineering with python, and I now work as a senior data scientist at [Satellite Vu](https://www.satellitevu.com/). Please feel free to connect with me on Twitter & LinkedIn, and please do let me know if this repository is useful to your work.

<!-- markdown-link-check-disable -->
[![Linkedin: robmarkcole](https://img.shields.io/badge/-Robin%20Cole-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/robmarkcole/)](https://www.linkedin.com/in/robmarkcole/)
[![Twitter Follow](https://img.shields.io/twitter/follow/robmarkcole?label=Follow)](https://twitter.com/robmarkcole)
<!-- markdown-link-check-enable -->
