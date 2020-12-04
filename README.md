# Introduction
This document lists resources for performing deep learning (DL) on satellite imagery. To a lesser extent classical Machine learning (ML, e.g. random forests) are also discussed, as are classical image processing techniques.

# Table of contents
* [Top links](https://github.com/robmarkcole/satellite-image-deep-learning#top-links)
* [Datasets](https://github.com/robmarkcole/satellite-image-deep-learning#datasets)
* [Interesting deep learning projects](https://github.com/robmarkcole/satellite-image-deep-learning#interesting-deep-learning-projects)
* [Techniques](https://github.com/robmarkcole/satellite-image-deep-learning#techniques)
* [Image formats and catalogues](https://github.com/robmarkcole/satellite-image-deep-learning#image-formats-data-management-and-catalogues)
* [State of the art](https://github.com/robmarkcole/satellite-image-deep-learning#state-of-the-art)
* [Online platforms for Geo analysis](https://github.com/robmarkcole/satellite-image-deep-learning#online-platforms-for-geo-analysis)
* [Free online computing resources](https://github.com/robmarkcole/satellite-image-deep-learning#free-online-computing-resources)
* [Production](https://github.com/robmarkcole/satellite-image-deep-learning#production)
* [Useful open source software](https://github.com/robmarkcole/satellite-image-deep-learning#useful-open-source-software)
* [Movers and shakers on Github](https://github.com/robmarkcole/satellite-image-deep-learning#movers-and-shakers-on-github)
* [Courses](https://github.com/robmarkcole/satellite-image-deep-learning#courses)
* [Online communities](https://github.com/robmarkcole/satellite-image-deep-learning#online-communities)
* [Companies](https://github.com/robmarkcole/satellite-image-deep-learning#companies)
* [Jobs](https://github.com/robmarkcole/satellite-image-deep-learning#jobs)
* [Neural nets in space](https://github.com/robmarkcole/satellite-image-deep-learning#neural-nets-in-space)
* [About the author](https://github.com/robmarkcole/satellite-image-deep-learning#about-the-author)

# Top links
* https://github.com/chrieke/awesome-satellite-imagery-datasets
* [A modern geospatial workflow](https://gist.github.com/jacquestardie/0d1c0cb413b3b9b06edf)
* [geospatial-machine-learning](https://github.com/deepVector/geospatial-machine-learning)
* [Long list of satellite missions with example imagery](https://www.satimagingcorp.com/satellite-sensors/)
* [AWS datasets](https://registry.opendata.aws/)
* [awesome-earthobservation-code](https://github.com/acgeospatial/awesome-earthobservation-code)
* [awesome-sentinel](https://github.com/Fernerkundung/awesome-sentinel)

# Datasets
* **Warning** satellite image files can be LARGE, even a small data set may comprise 50 GB of imagery
* [Various datasets listed here](https://www.maptiler.com/gallery/satellite/) and at [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)

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
* [Sentinel-2 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs/)
* [Open access data on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2?prefix=tiles%2F31%2FT%2FCJ%2F)
* Paid access via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py).
* [Example loading sentinel data in a notebook](https://github.com/binder-examples/getting-data/blob/master/Sentinel2.ipynb)
* [so2sat on Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/so2sat) - So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and the corresponding local climate zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions of the world.
* [eurosat](https://www.tensorflow.org/datasets/catalog/eurosat) - EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples. Dataset and usage in [EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/EuroSAT), where a CNN achieves a classification accuracy 98.57%.
* [bigearthnet](https://www.tensorflow.org/datasets/catalog/bigearthnet) - The BigEarthNet is a new large-scale Sentinel-2 benchmark archive, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels.
* [Jupyter Notebooks for working with Sentinel-5P Level 2 data stored on S3](https://github.com/Sentinel-5P/data-on-s3). The data can be browsed [here](https://meeo-s5p.s3.amazonaws.com/index.html#/?t=catalogs)
* [Sentinel NetCDF data](https://github.com/acgeospatial/Sentinel-5P/blob/master/Sentinel_5P.ipynb)

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
* [SpaceNet - WorldView-3](https://spacenetchallenge.github.io/datasets/datasetHomePage.html) and [article here](https://spark-in.me/post/spacenet-three-challenge). Also example [semantic segmentation using Raster Vision](https://docs.rastervision.io/en/0.8/quickstart.html)

## Planet
* [Planet’s high-resolution, analysis-ready mosaics of the world’s tropics](https://www.planet.com/nicfi/), supported through Norway’s International Climate & Forests Initiative. [BBC coverage](https://www.bbc.co.uk/news/science-environment-54651453)

### Shuttle Radar Topography Mission (digital elevation maps)
* [Data - open access](http://srtm.csi.cgiar.org/srtmdata/)

## Kaggle
Kaggle hosts over 60 satellite image datasets, [search results here](https://www.kaggle.com/search?q=satellite+image+in%3Adatasets).
The [kaggle blog](http://blog.kaggle.com) is an interesting read.

### Kaggle - Amazon from space - classification challenge
* https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
* 3-5 meter resolution GeoTIFF images from planet Dove satellite constellation
* 12 classes including - **cloudy, primary + waterway** etc
* [1st place winner interview - used 11 custom CNN](http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/)
* [FastAI Multi-label image classification](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95)

### Kaggle - DSTL - segmentation challenge
* https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
* Rating - medium, many good examples (see the Discussion as well as kernels), but as this competition was run a couple of years ago many examples use python 2
* WorldView 3 - 45 satellite images covering 1km x 1km in both 3 (i.e. RGB) and 16-band (400nm - SWIR) images
* 10 Labelled classes include - **Buildings, Road, Trees, Crops, Waterway, Vehicles**
* [Interview with 1st place winner who used segmentation networks](http://blog.kaggle.com/2017/04/26/dstl-satellite-imagery-competition-1st-place-winners-interview-kyle-lee/) - 40+ models, each tweaked for particular target (e.g. roads, trees)
* [Deepsense 4th place solution](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
* My analysis [here](https://github.com/robmarkcole/Useful-python/tree/master/Kaggle/dstl-satellite-imagery-feature-detection)

### Kaggle - Airbus Ship Detection Challenge
* https://www.kaggle.com/c/airbus-ship-detection/overview
* Rating - medium, most solutions using deep-learning, many kernels, [good example kernel](https://www.kaggle.com/kmader/baseline-u-net-model-part-1).
* I believe there was a problem with this dataset, which led to many complaints that the competition was ruined.

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

### Kaggle - other
* Satellite + loan data -> https://www.kaggle.com/reubencpereira/spatial-data-repo

## Alternative datasets
There are a variety of datasets suitable for land classification problems.

### Tensorflow datasets
* There are a number of remote sensing datasets
* [resisc45](https://www.tensorflow.org/datasets/catalog/resisc45) - RESISC45 dataset is a publicly available benchmark for Remote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class.
* [eurosat](https://www.tensorflow.org/datasets/catalog/eurosat) - EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples.
* [bigearthnet](https://www.tensorflow.org/datasets/catalog/bigearthnet) - The BigEarthNet is a new large-scale Sentinel-2 benchmark archive, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels.

### UCMerced
* http://weegee.vision.ucmerced.edu/datasets/landuse.html
* Available as a Tensorflow dataset -> https://www.tensorflow.org/datasets/catalog/uc_merced
* This is a 21 class land use image dataset meant for research purposes.
* There are 100 RGB TIFF images for each class
* Each image measures 256x256 pixels with a pixel resolution of 1 foot
* Image classification of UCMerced using [Keras](https://examples.pyviz.org/landuse_classification/Image_Classification.html#landuse-classification-gallery-image-classification) or alternatively [fastai](https://medium.com/spatial-data-science/deep-learning-for-geospatial-data-applications-multi-label-classification-2b0a1838fcf3)

### AWS datasets
* Landsat -> free viewer at [remotepixel](https://viewer.remotepixel.ca/#3/40/-70.5) and [libra](https://libra.developmentseed.org/)
* Optical, radar, segmented etc. https://aws.amazon.com/earth/
* Spacenet data is hosted on S3

### Quilt
* Several people have uploaded datasets to [Quilt](https://open.quiltdata.com/search?q=satellite)

### Google Earth Engine
* https://developers.google.com/earth-engine/
* Various imagery and climate datasets, including Landsat & Sentinel imagery
* [Python API](https://developers.google.com/earth-engine/python_install) but  all compute happens on Googles servers
* [Google Earth Engine Community on Github](https://github.com/gee-community)
* [awesome-google-earth-engine](https://github.com/gee-community/awesome-google-earth-engine) - Curated list of Google Earth Engine resources
* [ee-tensorflow-notebooks](https://github.com/gee-community/ee-tensorflow-notebooks) - Repository to place example notebooks for Deep Learning applications with TensorFlow and Earth Engine.

### Weather Datasets
* UK met-odffice -> https://www.metoffice.gov.uk/datapoint
* NASA (make request and emailed when ready) -> https://search.earthdata.nasa.gov
* NOAA (requires BigQuery) -> https://www.kaggle.com/noaa/goes16/home
* Time series weather data for several US cities -> https://www.kaggle.com/selfishgene/historical-hourly-weather-data

### UAV & Drone datasets
* Many on https://www.visualdata.io
* [AU-AIR dataset](https://bozcani.github.io/auairdataset) -> a multi-modal UAV dataset for object detection.
* [ERA](https://lcmou.github.io/ERA_Dataset/) ->  A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos.
* [Aerial Maritime Drone Dataset](https://public.roboflow.ai/object-detection/aerial-maritime)
* [Stanford Drone Dataset](http://cvgl.stanford.edu/projects/uav_data/)
* [RetinaNet for pedestrian detection](https://towardsdatascience.com/pedestrian-detection-in-aerial-images-using-retinanet-9053e8a72c6)
* [Aerial Maritime Drone Dataset](https://public.roboflow.com/object-detection/aerial-maritime/1)
* [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) - identify fire and other emergencies from a drone
* [OpenDroneMap](https://github.com/OpenDroneMap/ODM) - generate maps, point clouds, 3D models and DEMs from drone, balloon or kite images.

### Synthetic data
* [The Synthinel-1 dataset: a collection of high resolution synthetic overhead imagery for building segmentation](https://arxiv.org/ftp/arxiv/papers/2001/2001.05130.pdf)
* [RarePlanes](https://www.cosmiqworks.org/RarePlanes/) ->  incorporates both real and synthetically generated satellite imagery including aircraft.

# Interesting deep learning projects
### Raster Vision by Azavea
* https://www.azavea.com/projects/raster-vision/
* An open source Python framework for building computer vision models on aerial, satellite, and other large imagery sets.
* Accessible through the [Raster Foundry](https://www.rasterfoundry.com/)
* [Example use cases on open data](https://github.com/azavea/raster-vision-examples)

### RoboSat
* https://github.com/mapbox/robosat
* Semantic segmentation on aerial and satellite imagery. Extracts features such as: buildings, parking lots, roads, water, clouds
* [robosat-jupyter-notebook](https://github.com/Element84/robosat-jupyter-notebook) -> walks through all of the steps in an excellent blog post on the Robosat feature extraction and machine learning pipeline.
* Note there is/was fork of Robosat, originally named RoboSat.pink, and subsequently https://neat-EO.pink although this appears to be down/archived

### DeepOSM
* https://github.com/trailbehind/DeepOSM
* Train a deep learning net with OpenStreetMap features and satellite imagery.

### DeepNetsForEO - segmentation
* https://github.com/nshaud/DeepNetsForEO
* Uses SegNET for working on remote sensing images using deep learning.

### Skynet-data
* https://github.com/developmentseed/skynet-data
* Data pipeline for machine learning with OpenStreetMap

# Techniques
This section explores the different techniques (DL, ML & classical) people are applying to common problems in satellite imagery analysis. Classification problems are the most simply addressed via DL, object detection is harder, and cloud detection harder still (niche interest).

## Land classification
Assign a label to an image, e.g. this is an image of a forest.
* Land classification using a [simple sklearn cluster algorithm](https://github.com/acgeospatial/Satellite_Imagery_Python/blob/master/Clustering_KMeans-Sentinel2.ipynb) or [deep learning](https://towardsdatascience.com/land-use-land-cover-classification-with-deep-learning-9a5041095ddb).
* Land use is related to classification, but we are trying to detect a scene, e.g. housing, forestry. I have tried CNN -> [See my notebooks](https://github.com/robmarkcole/satellite-imagery-projects/tree/main/land_classification)
* [Land Use Classification using Convolutional Neural Network in Keras](https://github.com/tavgreen/landuse_classification)
* [Sea-Land segmentation using DL](https://arxiv.org/pdf/1709.00201.pdf)
* [Pixel level segmentation on Azure](https://github.com/Azure/pixel_level_land_classification)
* [Deep Learning-Based Classification of Hyperspectral Data](https://github.com/hantek/deeplearn_hsi)
* [A U-net based on Tensorflow for objection detection (or segmentation) of satellite images - DSTL dataset but python 2.7](https://github.com/rogerxujiang/dstl_unet)
* [What’s growing there? Using eo-learn and fastai to identify crops from multi-spectral remote sensing data (Sentinel 2)](https://towardsdatascience.com/whats-growing-there-a5618a2e6933)
* [FastAI Multi-label image classification](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95)
* [Land use classification using Keras](https://examples.pyviz.org/landuse_classification/Image_Classification.html#landuse-classification-gallery-image-classification)
* [Detecting Informal Settlements from Satellite Imagery using fine-tuning of ResNet-50 classifier](https://blog.goodaudience.com/detecting-informal-settlements-using-satellite-imagery-and-convolutional-neural-networks-d571a819bf44) with [repo](https://github.com/dymaxionlabs/ap-latam)
* Image classification of UCMerced using [Keras](https://examples.pyviz.org/landuse_classification/Image_Classification.html#landuse-classification-gallery-image-classification) or alternatively [fastai](https://medium.com/spatial-data-science/deep-learning-for-geospatial-data-applications-multi-label-classification-2b0a1838fcf3)
* [Water Detection in High Resolution Satellite Images using the waterdetect python package](https://towardsdatascience.com/water-detection-in-high-resolution-satellite-images-using-the-waterdetect-python-package-7c5a031e3d16) -> The main idea is to combine water indexes (NDWI, MNDWI, etc.) with reflectance bands (NIR, SWIR, etc.) into an automated clustering process

## Semantic segmentation
Whilst classification will assign a label to a whole image, semantic segmentation will assign a label to each pixel
* [Instance segmentation with keras - links to satellite examples](https://github.com/matterport/Mask_RCNN)
* [Semantic Segmentation on Aerial Images using fastai](https://medium.com/swlh/semantic-segmentation-on-aerial-images-using-fastai-a2696e4db127)
* https://github.com/Paulymorphous/Road-Segmentation
* [UNSOAT used fast.ai to train a Unet to perform semantic segmentation on satellite imageries to detect water](https://forums.fast.ai/t/unosat-used-fastai-ai-for-their-floodai-model-discussion-on-how-to-move-forward/78468) - [paper](https://www.mdpi.com/2072-4292/12/16/2532) + [notebook](https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/Fastai%20training.ipynb), accuracy 0.97, precision 0.91, recall 0.92.

## Change detection
Monitor water levels, coast lines, size of urban areas, wildfire damage. Note, clouds change often too..!
* Using PCA (python 2, requires updating) -> https://appliedmachinelearning.blog/2017/11/25/unsupervised-changed-detection-in-multi-temporal-satellite-images-using-pca-k-means-python-code/
* Using CNN -> https://github.com/vbhavank/Unstructured-change-detection-using-CNN
* [Siamese neural network to detect changes in aerial images](https://github.com/vbhavank/Siamese-neural-network-for-change-detection)
* https://www.spaceknow.com/
* [LANDSAT Time Series Analysis for Multi-temporal Land Cover Classification using Random Forest](https://github.com/agr-ayush/Landsat-Time-Series-Analysis-for-Multi-Temporal-Land-Cover-Classification)
* [Change Detection in 3D: Generating Digital Elevation Models from Dove Imagery](https://www.planet.com/pulse/publications/change-detection-in-3d-generating-digital-elevation-models-from-dove-imagery/)
* [Change Detection in Hyperspectral Images Using Recurrent 3D Fully Convolutional Networks](https://www.mdpi.com/2072-4292/10/11/1827)
* [PySAR - InSAR (Interferometric Synthetic Aperture Radar) timeseries analysis in python](https://github.com/hfattahi/PySAR)
* [QGIS 2 plugin for applying change detection algorithms on high resolution satellite imagery](https://github.com/dymaxionlabs/massive-change-detection)

## Image registration
Image registration is the process of transforming different sets of data into one coordinate system. Typical use is overlapping images taken at different times or with different cameras.
* [Wikipedia article on registration](https://en.wikipedia.org/wiki/Image_registration) -> register for change detection or [image stitching](https://mono.software/2018/03/14/Image-stitching/)
* Traditional approach -> define control points, employ RANSAC algorithm
* [Phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) used to estimate the translation between two images with sub-pixel accuracy, useful for [allows accurate registration of low resolution imagery onto high resolution imagery](https://onlinelibrary.wiley.com/doi/10.1002/9781118724194.ch11), or register a [sub-image on a full image](https://www.mathworks.com/help/images/registering-an-image-using-normalized-cross-correlation.html) -> Unlike many spatial-domain algorithms, the phase correlation method is resilient to noise, occlusions, and other defects. [Applied to Landsat images here](https://github.com/JamieTurrin/Phase-Correlation).

## Object detection
A good introduction to the challenge of performing object detection on aerial imagery is given in [this paper](https://arxiv.org/abs/1902.06042v2). In summary, images are large and objects may comprise only a few pixels, easily confused with random features in background. An example task is detecting boats on the ocean, which should be simpler than land based detection owing to the relatively blank background in images, but is still challenging.
* Intro articles [here](https://medium.com/earthcube-stories/how-hard-it-is-for-an-ai-to-detect-ships-on-satellite-images-7265e34aadf0) and [here](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7).
* [DigitalGlobe article](http://gbdxstories.digitalglobe.com/boats/) - they use a combination classical techniques (masks, erodes) to reduce the search space (identifying water via [NDWI](https://en.wikipedia.org/wiki/Normalized_difference_water_index) which requires SWIR) then apply a binary DL classifier on candidate regions of interest. They deploy the final algo [as a task](https://github.com/platformstories/boat-detector) on their GBDX platform. They propose that in the future an R-CNN may be suitable for the whole process.
* [Planet use non DL felzenszwalb algorithm to detect ships](https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ship-detector/01_ship_detector.ipynb)
* [Segmentation of buildings on kaggle](https://www.kaggle.com/kmader/synthetic-word-ocr/kernels)
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”.
* [Deep learning for satellite imagery via image segmentation](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
* [Building Extraction with YOLT2 and SpaceNet Data](https://medium.com/the-downlinq/building-extraction-with-yolt2-and-spacenet-data-a926f9ffac4f)
* [Find sports fields using Mask R-CNN and overlay on open-street-map](https://github.com/jremillard/images-to-osm)
* [Detecting solar panels from satellite imagery](https://towardsdatascience.com/weekend-project-detecting-solar-panels-from-satellite-imagery-f6f5d5e0da40)
* [Anomaly Detection on Mars using a GAN](https://omdena.com/projects/anomaly-detection-mars/)
* [Tackling the Small Object Problem in Object Detection](https://blog.roboflow.com/tackling-the-small-object-problem-in-object-detection)
* [Satellite Imagery Multiscale Rapid Detection with Windowed Networks (SIMRDWN)](https://github.com/avanetten/simrdwn) -> combines some of the leading object detection algorithms into a unified framework designed to detect objects both large and small in overhead imagery
* [2020 Nature paper - An unexpectedly large count of trees in the West African Sahara and Sahel](https://www.nature.com/articles/s41586-020-2824-5) -> tree detection framework based on U-Net & tensorflow 2 with code [here](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel/tree/v1.0.0)
* [Truck Detection with Sentinel-2 during COVID-19 crisis](https://github.com/hfisser/Truck_Detection_Sentinel2_COVID19) -> moving objects in Sentinel-2 data causes a specific reflectance relationship in the RGB, which looks like a rainbow, and serves as a marker for trucks. Improve accuracy by only analysing roads.
* [Counting-Trees-using-Satellite-Images](https://github.com/A2Amir/Counting-Trees-using-Satellite-Images) -> create an inventory of incoming and outgoing trees for an annual tree inspections, uses keras
* Several useful articles on [awesome-tiny-object-detection](https://github.com/kuanhungchen/awesome-tiny-object-detection)

## Cloud detection
A subset of the object detection problem, but surprisingly challenging
* From [this article on sentinelhub](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13) there are three popular classical algorithms that detects thresholds in multiple bands in order to identify clouds. In the same article they propose using semantic segmentation combined with a CNN for a cloud classifier (excellent review paper [here](https://arxiv.org/pdf/1704.06857.pdf)), but state that this requires too much compute resources.
* [This article](https://www.mdpi.com/2072-4292/8/8/666) compares a number of ML algorithms, random forests, stochastic gradient descent, support vector machines, Bayesian method.

## Wealth and economic activity measurement
The goal is to predict economic activity from satellite imagery rather than conducting labour intensive ground surveys
* [Using publicly available satellite imagery and deep learning to understand economic well-being in Africa, Nature Comms 22 May 2020](https://www.nature.com/articles/s41467-020-16185-w) -> Used CNN on Ladsat imagery (night & day) to predict asset wealth of African villages
* [Combining Satellite Imagery and machine learning to predict poverty](https://towardsdatascience.com/combining-satellite-imagery-and-machine-learning-to-predict-poverty-884e0e200969) -> review article
* [Measuring Human and Economic Activity from Satellite Imagery to Support City-Scale Decision-Making during COVID-19 Pandemic](https://arxiv.org/abs/2004.07438)
* [Predicting Food Security Outcomes Using CNNs for Satellite Tasking](https://arxiv.org/pdf/1902.05433.pdf)
* [Crop yield Prediction with Deep Learning](https://github.com/JiaxuanYou/crop_yield_prediction) -> The necessary code for the paper Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data, AAAI 2017 (Best Student Paper Award in Computational Sustainability Track).
* https://github.com/taspinar/sidl/blob/master/notebooks/2_Detecting_road_and_roadtypes_in_sattelite_images.ipynb

## Super resolution
Super-resolution imaging is a class of techniques that enhance the resolution of an imaging system. Very hot topic of research.
* https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-1-ec5c5cd3cd2 -> Nov 2016 blog post by CosmiQ Works with a nice introduction to the topic. Proposes and demonstrates a new architecture with perturbation layers with practical guidance on the methodology and [code](https://github.com/CosmiQ/super-resolution). [Three part series](https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-3-2e2f61eee1d3)
* [Super Resolution for Satellite Imagery - srcnn repo](https://github.com/WarrenGreen/srcnn)
* [TensorFlow implementation of "Accurate Image Super-Resolution Using Very Deep Convolutional Networks" adapted for working with geospatial data](https://github.com/CosmiQ/VDSR4Geo) 
* [Random Forest Super-Resolution (RFSR repo)](https://github.com/jshermeyer/RFSR) including [sample data](https://github.com/jshermeyer/RFSR/tree/master/SampleImagery)
* [Super-Resolution (python) Utilities for managing large satellite images](https://github.com/jshermeyer/SR_Utils)

## Pansharpening
Image fusion of low res multispectral with high res pan band.
* Several algorithms described [in the ArcGIS docs](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/fundamentals-of-panchromatic-sharpening.htm), with the simplest being taking the mean of the pan and RGB pixel value.
* Does not require DL, classical algos suffice, [see this notebook](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/panchromatic_sharpening.ipynb) and [this kaggle kernel](https://www.kaggle.com/resolut/panchromatic-sharpening)
* https://github.com/mapbox/rio-pansharpen

## Stereo imaging for terrain mapping & DEMs
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

## Lidar
* [Reconstructing 3D buildings from aerial LiDAR with Mask R-CNN](https://medium.com/geoai/reconstructing-3d-buildings-from-aerial-lidar-with-ai-details-6a81cb3079c0)

## NVDI - vegetation index
* Simple band math `ndvi = np.true_divide((ir - r), (ir + r))` but challenging due to the size of the imagery.
* [Example notebook local](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/ndvi_calculation.ipynb)
* [Landsat data in cloud optimised (COG) format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article here](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).
* [Visualise water loss with Holoviews](https://examples.pyviz.org/walker_lake/Walker_Lake.html#walker-lake-gallery-walker-lake)

## SAR
* [Removing speckle noise from Sentinel-1 SAR using a CNN](https://medium.com/upstream/denoising-sentinel-1-radar-images-5f764faffb3e)
* A dataset which is specifically made for deep learning on SAR and optical imagery is the SEN1-2 dataset, which contains corresponding patch pairs of Sentinel 1 (VV) and 2 (RGB) data. It is the largest manually curated dataset of S1 and S2 products, with corresponding labels for land use/land cover mapping, SAR-optical fusion, segmentation and classification tasks. Paper: https://elib.dlr.de/128117/1/SEN12MS_Preprint.pdf Data: https://mediatum.ub.tum.de/1474000
* [so2sat on Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/so2sat) - So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and the corresponding local climate zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions of the world.
* [Using Machine Learning to Automatically Detect Volcanic Unrest in a Time Series of Interferograms](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019JB017519)

# Image formats, data management and catalogues
* [GeoServer](http://geoserver.org/) -> an open source server for sharing geospatial data.
* https://terria.io/ for pretty catalogues
* [Remote pixel](https://remotepixel.ca/projects/index.html#satsearch)
* [Sentinel-hub eo-browser](https://apps.sentinel-hub.com/eo-browser/)
* Large datasets may come in HDF5 format, can view with -> https://www.hdfgroup.org/downloads/hdfview/
* Climate data is often in netcdf format, which can be [opened using xarray](https://moderndata.plot.ly/weather-maps-in-python-with-mapbox-gl-xarray-and-netcdf4/)
* The xarray docs list a number of ways that data [can be stored and loaded](http://xarray.pydata.org/en/latest/io.html#).
* [TileDB](https://tiledb.com/) -> a 'Universal Data Engine' to store, analyze and share any data (beyond tables), with any API or tool (beyond SQL) at planet-scale (beyond clusters), open source and managed options. [Recently hiring](https://discourse.pangeo.io/t/job-openings-at-tiledb-inc/787) to work with xarray, dask, netCDF and cloud native storage
* Open Data Cube - serve up cubes of data https://www.opendatacube.org/

## Cloud Optimised GeoTiff (COG)
* https://www.cogeo.org/
* TLDR: A Cloud Optimized GeoTIFF (COG) is a regular GeoTIFF file, aimed at being hosted on a HTTP file server (or Cloud object storage like S3), with an internal organization that enables more efficient workflows on the cloud. In particular they support HTTP range requests, enabling downloading of specific tiles rather than the full file. COGs work normally in GIS software such as QGIS.
* [Intro presentation from Saheel Ahmed](https://github.com/saheelBreezo/Cloud-Optimised-Geotiff/blob/master/Talk/Cloud_Optimized_GeoTIFF_Blue_Sky_Analytics.pdf)
* [cog-best-practices](https://github.com/pangeo-data/cog-best-practices)
* [rio-cogeo](https://cogeotiff.github.io/rio-cogeo/) -> Cloud Optimized GeoTIFF (COG) creation and validation plugin for Rasterio.
* [aiocogeo](https://github.com/geospatial-jeff/aiocogeo) -> Asynchronous cogeotiff reader (python asyncio)
* [Landsat data in cloud optimised (COG) format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article Cloud Native Geoprocessing of Earth Observation Satellite Data with Pangeo](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).

## STAC - SpatioTemporal Asset Catalog specification
The STAC specification provides a common metadata specification, API, and catalog format to describe geospatial assets, so they can more easily indexed and discovered. A 'spatiotemporal asset' is any file that represents information about the earth captured in a certain space and time. (from intake-stac docs)
* The aim is that the catalogue is crawlable so it can be indexed by a search engine and make imagery discoverable, without requiring yet another API interface.
* An initiative of https://www.radiant.earth/ in particular https://github.com/cholmes
* Spec at https://github.com/radiantearth/stac-spec
* Browser at https://github.com/radiantearth/stac-browser
* [stacindex](https://stacindex.org/) -> STAC Catalogs, Collections, APIs, Software and Tools
* Talk at https://docs.google.com/presentation/d/1O6W0lMeXyUtPLl-k30WPJIyH1ecqrcWk29Np3bi6rl0/edit#slide=id.p
* Example catalogue at https://landsat-stac.s3.amazonaws.com/catalog.json
* Chat https://gitter.im/SpatioTemporal-Asset-Catalog/Lobby
* Several useful repos on https://github.com/sat-utils
* [Intake-STAC](https://github.com/intake/intake-stac) -> Intake-STAC provides an opinionated way for users to load Assets from STAC catalogs into the scientific Python ecosystem. It uses the intake-xarray plugin and supports several file formats including GeoTIFF, netCDF, GRIB, and OpenDAP.
* [sat-utils/sat-search](https://github.com/sat-utils/sat-search) -> Sat-search is a Python 3 library and a command line tool for discovering and downloading publicly available satellite imagery using STAC compliant API

# State of the art
What are companies doing?
* Overall trend to using cloud (i.e. AWS, Google or Azure) storage buckets for hosting imagery
* [Airbus are using a Google backend](https://cloud.google.com/customers/airbus)
* [Planet are also on Google](https://cloud.google.com/customers/planet), not too surprising as Google own significant stock in Planet
* A [serverless pipeline](https://github.com/aws-samples/amazon-rekognition-video-analyzer) appears to be where companies are headed for routine compute tasks, whilst providing a Jupyter notebook approach for custom analysis. Checkout [process Satellite data using AWS Lambda functions](https://github.com/RemotePixel/remotepixel-api)
* Traditional data formats aren't designed for processing, so new standards are developing such as [cloud optimised geotiffs](http://blog.digitalglobe.com/developers/cloud-optimized-geotiffs-and-the-path-to-accessible-satellite-imagery-analytics/) and [zarr](https://github.com/zarr-developers/zarr)
* Google provide training on how to use Apache Spark on Google Cloud Dataproc to distribute a computationally intensive (satellite) image processing task onto a cluster of machines -> https://google.qwiklabs.com/focuses/5834?parent=catalog

# Online platforms for Geo analysis
* [This article discusses some of the available platforms](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2) -> TLDR Pangeo rocks, but must BYO imagery
* Pangeo - open source resources for parallel processing using Dask and Xarray http://pangeo.io/index.html
* [Airbus Sandbox](https://sandbox.intelligence-airbusds.com/web/) -> will provide access to imagery
* [Descartes Labs](https://www.descarteslabs.com/) -> access to EO imagery from a variety of providers via python API -> not clear which imagery is available (Airbus + others?) or pricing
* DigitalGlobe have a cloud hosted Jupyter notebook platform called [GBDX](https://platform.digitalglobe.com/gbdx/). Cloud hosting means they can guarantee the infrastructure supports their algorithms, and they appear to be close/closer to deploying DL. [Tutorial notebooks here](https://notebooks.geobigdata.io/hub/tutorials/list). Only Sentinel-2 and Landsat data on free tier.
* Planet have a [Jupyter notebook platform](https://developers.planet.com/) which can be deployed locally.
* Earth-i [Spectrum](https://earthi.space/spectrum/) appears to allow processing of imagery, with the capability to perform segmentation, change detection, object recognition. [This promo video](https://vimeo.com/420726376) contains some screenshots of the application. 

# Free online computing resources
Generally a GPU is required for DL, and this section lists a couple of free Jupyter environments with GPU available. There is a good overview of online Jupyter development environments [on the fast.ai site](https://course-v3.fast.ai/index.html). I personally use Colab with data hosted on Google Drive

### Google Colab
* Collaboratory [notebooks](https://colab.research.google.com) with GPU as a backend for free for 12 hours at a time. Note that the GPU may be shared with other users, so if you aren't getting good performance try reloading.
* Also a pro tier for $10 a month -> https://colab.research.google.com/signup
* Tensorflow  pytorch can be installed

### Kaggle - also Google!
* Free to use
* GPU Kernels - may run for 1 hour
* Tensorflow, pytorch & fast.ai available
* Advantage that many datasets are already available

### Paperspace
* Free tier available
* https://docs.paperspace.com/gradient/instances/free-instances

# Production
Once you have a trained model how do you expose it to the internet and other services? Usually through a rest API. This section lists a number of hosting options.

### Custom REST API
* Basic https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html with code [here](https://github.com/jrosebr1/simple-keras-rest-api)
* Advanced https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/

### Tensorflow Serving
* https://www.tensorflow.org/serving/
* TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. Multiple models, or indeed multiple versions of the same model, can be served simultaneously.  TensorFlow Serving comes with a scheduler that groups individual inference requests into batches for joint execution on a GPU

### Pytorch serve
* https://github.com/pytorch/serve

### AWS sagemaker
* https://aws.amazon.com/blogs/machine-learning/bring-your-own-deep-learning-framework-to-amazon-sagemaker-with-model-server-for-apache-mxnet/

### Paperspace gradient
* https://docs.paperspace.com/machine-learning/wiki/model-deployment

### chip-n-scale-queue-arranger by developmentseed
* https://github.com/developmentseed/chip-n-scale-queue-arranger
* an orchestration pipeline for running machine learning inference at scale
* [Supports fast.ai models](https://github.com/developmentseed/fastai-serving)

# Useful open source software
* [GDAL](https://gdal.org) -> THE tool for reading and writing raster and vector geospatial data formats
* [QGIS](https://qgis.org/en/site/)- Create, edit, visualise, analyse and publish geospatial information. [Python scripting and plugins](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/intro.html#scripting-in-the-python-console).
* [Orfeo toolbox](https://www.orfeo-toolbox.org/) - remote sensing toolbox with python API (just a wrapper to the C code). Do activites such as [pansharpening](https://www.orfeo-toolbox.org/CookBook/Applications/app_Pansharpening.html), ortho-rectification, image registration, image segmentation & classification. Not much documentation.
* [QUICK TERRAIN READER - view DEMS, Windows](http://appliedimagery.com/download/)
* [dl-satellite-docker](https://github.com/sshuair/dl-satellite-docker) -> docker files for geospatial analysis, including tensorflow, pytorch, gdal, xgboost...
* [AIDE V2 - Tools for detecting wildlife in aerial images using active learning](https://github.com/microsoft/aerial_wildlife_detection/tree/multiProject)
* [Land Cover Mapping web app from Microsoft](https://github.com/microsoft/landcover)
* [Solaris](https://github.com/CosmiQ/solaris) -> An open source ML pipeline for overhead imagery by [CosmiQ Works](https://www.cosmiqworks.org/), similar to Rastervision but with some unique very vool features
* [openSAR](https://github.com/EarthBigData/openSAR) -> Synthetic Aperture Radar (SAR) Tools and Documents from Earth Big Data LLC (http://earthbigdata.com/)


## Python low level numerical & data manipulation
* [Dask](https://docs.dask.org/en/latest/) -> [Read and manipulate tiled GeoTIFF datasets](https://examples.dask.org/applications/satellite-imagery-geotiff.html#)
* [Rasterio](https://rasterio.readthedocs.io/en/latest/) -> reads and writes GeoTIFF and other raster formats and provides a Python API based on Numpy N-dimensional arrays and GeoJSON.
* [xarray](http://xarray.pydata.org/en/stable/) -> N-D labeled arrays and datasets. Read [Handling multi-temporal satellite images with Xarray](https://medium.com/@bonnefond.virginie/handling-multi-temporal-satellite-images-with-xarray-30d142d3391)
* [xarray-spatial](https://github.com/makepath/xarray-spatial) -> Fast, Accurate Python library for Raster Operations. Implements algorithms using Numba and Dask, free of GDAL
* [Geowombat](https://geowombat.readthedocs.io/) -> geo-utilities applied to air- and space-borne imagery, uses Rasterio, Xarray and Dask for I/O and distributed computing with named coordinates
* [NumpyTiles](https://github.com/planetlabs/numpytiles-spec) -> a specification for providing multiband full-bit depth raster data in the browser
* [Zarr](https://zarr.readthedocs.io/en/stable/) -> Zarr is a format for the storage of chunked, compressed, N-dimensional arrays. Zarr depends on NumPy

## Python general utilities
* [gcsts for google cloud storage sile-system](https://github.com/dask/gcsfs) -> Pythonic file-system interface for Google Cloud Storage
* [satpy](https://github.com/pytroll/satpy) - a python library for reading and manipulating meteorological remote sensing data and writing it to various image and data file formats
* [Pyviz](https://examples.pyviz.org/) examples include several interesting geospatial visualisations
* [geemap](https://github.com/giswqs/geemap): A Python package for interactive mapping with Google Earth Engine, ipyleaflet, and ipywidgets. See the [Landsat timelapse example](https://github.com/giswqs/geemap/blob/master/examples/notebooks/27_timelapse_app.ipynb)
* [rio-color](https://github.com/mapbox/rio-color) -> Color correction plugin for Rasterio
* [WaterDetect](https://github.com/cordmaur/WaterDetect) -> an end-to-end algorithm to generate open water cover mask, specially conceived for L2A Sentinel 2 imagery. It can also be used for Landsat 8 images and for other multispectral clustering/segmentation tasks.
* [DeepHyperX](https://github.com/eecn/Hyperspectral-Classification) -> A Python/pytorch tool to perform deep learning experiments on various hyperspectral datasets.
* [landsat_ingestor](https://github.com/landsat-pds/landsat_ingestor) -> Scripts and other artifacts for landsat data ingestion into Amazon public hosting
* [PyShp](https://github.com/GeospatialPython/pyshp) -> The Python Shapefile Library (PyShp) reads and writes ESRI Shapefiles in pure Python
* [s2p](https://github.com/cmla/s2p) -> a Python library and command line tool that implements a stereo pipeline which produces elevation models from images taken by high resolution optical satellites such as Pléiades, WorldView, QuickBird, Spot or Ikonos
* [TorchSat](https://github.com/sshuair/torchsat) is an open-source deep learning framework for satellite imagery analysis based on PyTorch.
* [torchvision-enhance](https://github.com/sshuair/torchvision-enhance) -> Enhance PyTorch vision for semantic segmentation, multi-channel images and TIF file,...
* [felicette](https://github.com/plant99/felicette) -> Satellite imagery for dummies. Generate JPEG earth imagery from coordinates/location name with publicly available satellite data.
* [napari](https://napari.org) -> napari is a fast, interactive, multi-dimensional image viewer for Python. It’s designed for browsing, annotating, and analyzing large multi-dimensional images. By integrating closely with the Python ecosystem, napari can be easily coupled to leading machine learning and image analysis tools. [Example viewing Landsat-8 imagery](https://napari.org/tutorials/gallery#geospatial-data)

## Tools for image annotation
If you are performing object detection you will need to annotate images. Check that your annotation tool of choice supports large image (likely geotiff) files, as not all will. Note also that GEOJSON is widely used by remote sensing researchers but this annotation format is not commonly supported in general computer vision frameworks.
* [Labelme Image Annotation for Geotiffs](https://medium.com/@wvsharber/labelme-image-annotation-for-geotiffs-b460ba83804f) -> uses [Labelme](https://github.com/wkentaro/labelme)
* [CVAT](https://github.com/openvinotoolkit/cvat) is worth investigating, and have an [open issue](https://github.com/openvinotoolkit/cvat/issues/531) to support large TIFF files. [This article on Roboflow](https://blog.roboflow.com/cvat/) gives a good intro to CVAT.
* [Deep Block](https://app.deepblock.net) is a general purpose AI platform that includes a tool for COCOJSON export for aerial imagery. Checkout [this video](https://www.youtube.com/watch?v=gg5qSV-yw4U&feature=youtu.be)

# Movers and shakers
* [Adam Van Etten](https://github.com/avanetten) is doing interesting things in object detection and segmentation
* [Ankit Kariryaa](https://github.com/ankitkariryaa) published a recent nature paper on tree detection
* [Chris Holmes](https://github.com/cholmes) is doing great things at Planet
* [Christoph Rieke](https://github.com/chrieke) maintains a very popular imagery repo and has published his thesis on segmentation
* [Jake Shermeyer](https://github.com/jshermeyer) many interesting repos
* [Nicholas Murray](https://www.murrayensis.org/) is an Australia-based scientist with a focus on delivering the science necessary to inform large scale environmental management and conservation
* [Qiusheng Wu](https://github.com/giswqs) is an Assistant Professor in the Department of Geography at the University of Tennessee
* [Robin Wilson](https://github.com/robintw) is a former academic who is very active in the satellite imagery space

# Courses
* [Manning: Monitoring Changes in Surface Water Using Satellite Image Data](https://liveproject.manning.com/course/106/monitoring-changes-in-surface-water-using-satellite-image-data?)
* [Working with Geospatial Data in Python on Datacamp](https://www.datacamp.com/courses/working-with-geospatial-data-in-python?tap_a=5644-dce66f&tap_s=411670-1f1ebc)

# Competitions
* [Spacenet 7: Multi-Temporal Urban Development Challenge](https://www.topcoder.com/challenges/21ba3d19-3f7a-4abc-8d28-7c932887f0f6) - registration deadline Oct 28 2020. Track individual building construction over time from Planet imagery, challenge because of the small pixel area of each object, the high object density within images, and the dramatic image-to-image difference compared to frame-to-frame variation in video object tracking. 

# Online communities
* [fast AI geospatial study group](https://forums.fast.ai/t/geospatial-deep-learning-resources-study-group/31044)

# Companies
* https://github.com/chrieke/geospatial-companies -> List of 500+ geospatial companies by Christoph Rieke
* [Dymaxion Analytics](https://dymaxionlabs.com/) -> a machine learning API for developing bespoke object detection models for satellite and drone imagery.
* [Element84](https://www.element84.com/) -> consultancy
* [CosmiQ Works](https://www.cosmiqworks.org/) -> an IQT Lab focused on developing, prototyping, and evaluating emerging open source artificial intelligence capabilities for geospatial use cases.

# Jobs
* [Pangeo discourse](https://discourse.pangeo.io/c/news/jobs) lists multiple jobs, global

# Neural nets in space
Processing on satellite allows less data to be downlinked. E.g. super-resolution image might take 4-8 images to generate, then a single image is downlinked.
* [Lockheed Martin and USC to Launch Jetson-Based Nanosatellite for Scientific Research Into Orbit - Aug 2020](https://news.developer.nvidia.com/lockheed-martin-usc-jetson-nanosatellite/) - One app that will run on the GPU-accelerated satellite is SuperRes, an AI-based application developed by Lockheed Martin, that can automatically enhance the quality of an image.
* [Intel to place movidius in orbit to filter images of clouds at source - Oct 2020](https://techcrunch.com/2020/10/20/intel-is-providing-the-smarts-for-the-first-satellite-with-local-ai-processing-on-board/) - Getting rid of these images before they’re even transmitted means that the satellite can actually realize a bandwidth savings of up to 30%,

# About the author
My background is optical physics, and I have a PhD from Cambridge on the topic of Plasmon enhanced Raman spectroscopy. After doing a post doc I left academia and took a variety of roles, from industrial research at [Sharp Labs Europe](https://www.sle.sharp.co.uk/), to medical physics, to building optical telescopes at [Surrey Satellites](https://www.sstl.co.uk/) (SSTL). It was whilst at SSTL that I started this repo as a personal resource. I left SSTL, actually was made redundant along with 30% of the company, and after a brief stint at an IOT start up, I now work as a data engineer. Deep learning is currently a hobby, but I have ambitions to move into this domain when the right opportunity presents itself. My own satellite imagery projects are [here](https://github.com/robmarkcole/satellite-imagery-projects), and feel free to connect with me [on LinkedIn](https://www.linkedin.com/in/robmarkcole/).