# Introduction
This document primarily lists resources for performing deep learning (DL) on satellite imagery. To a lesser extent Machine learning (ML, e.g. random forests, stochastic gradient descent) are also discussed, as are classical image processing techniques.

# Top links
* https://github.com/chrieke/awesome-satellite-imagery-datasets
* [A modern geospatial workflow](https://gist.github.com/jacquestardie/0d1c0cb413b3b9b06edf)
* [geospatial-machine-learning](https://github.com/deepVector/geospatial-machine-learning)
* [Long list of satellite missions with example imagery](https://www.satimagingcorp.com/satellite-sensors/)
* [AWS datasets](https://registry.opendata.aws/)

# Table of contents
* [Datasets](https://github.com/robmarkcole/satellite-image-deep-learning#datasets)
* [Online computing resources](https://github.com/robmarkcole/satellite-image-deep-learning#online-computing-resources)
* [Interesting dl projects](https://github.com/robmarkcole/satellite-image-deep-learning#interesting-dl-projects)
* [Production](https://github.com/robmarkcole/satellite-image-deep-learning#production)
* [Image formats and catalogues](https://github.com/robmarkcole/satellite-image-deep-learning#image-formats--catalogues)
* [State of the art](https://github.com/robmarkcole/satellite-image-deep-learning#state-of-the-art)
* [Online platforms for Geo analysis](https://github.com/robmarkcole/satellite-image-deep-learning#online-platforms-for-geo-analysis)
* [Techniques](https://github.com/robmarkcole/satellite-image-deep-learning#techniques)
* [Useful references](https://github.com/robmarkcole/satellite-image-deep-learning#useful-references)

# Datasets
* **Warning** satellite image files can be LARGE, even a small data set may comprise 50 GB of imagery
* [Various datasets listed here](https://www.maptiler.com/gallery/satellite/) and at [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)

## WorldView - SpaceNet
* https://en.wikipedia.org/wiki/WorldView-3
* 0.3m PAN, 1.24 MS, 3.7m SWIR. Off-Nadir (stereo) available.
* Owned by [DigitalGlobe](https://www.digitalglobe.com/)
* [Intro to SpaceNet](https://medium.com/@sumit.arora/getting-started-with-aws-spacenet-and-spacenet-dataset-visualization-basics-7ddd2e5809a2)
* [SpaceNet dataset on AWS](https://spacenetchallenge.github.io/datasets/datasetHomePage.html) -> see [this getting started notebook](https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53) and this notebook on the [off-Nadir dataset](https://medium.com/the-downlinq/introducing-the-spacenet-off-nadir-imagery-and-buildings-dataset-e4a3c1cb4ce3). Note -> REQUESTER PAYS
* [cloud_optimized_geotif here](http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/) used in the 3D modelling notebook [here](https://gfacciol.github.io/IS18/).
* [Package of utilities](https://github.com/SpaceNetChallenge/utilities) to assist working with the SpaceNet dataset.
* For more Worldview imagery see Kaggle DSTL competition.

## Sentinel
* As part of the [EU Copernicus program](https://en.wikipedia.org/wiki/Copernicus_Programme), multiple Sentinel satellites are capturing imagery -> see [wikipedia](https://en.wikipedia.org/wiki/Copernicus_Programme#Sentinel_missions).
* 13 bands, Spatial resolution of 10 m, 20 m and 60 m, 290 km swath, the temporal resolution is 5 days
* [Open access data on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2?prefix=tiles%2F31%2FT%2FCJ%2F)
* Paid access via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py).
* [Example loading sentinel data in a notebook](https://github.com/binder-examples/getting-data/blob/master/Sentinel2.ipynb)

## Landsat
* Long running US program -> see [Wikipedia](https://en.wikipedia.org/wiki/Landsat_program)
* 8 bands, 15 to 60 meters, 185km swath, the temporal resolution is 16 days
* [Imagery on GCP](https://cloud.google.com/storage/docs/public-datasets/landsat), see [the GCP bucket here](https://console.cloud.google.com/storage/browser/gcp-public-data-landsat/), with imagery analysed in [this notebook](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) on Pangeo


### Shuttle Radar Topography Mission (digital elevation maps)
* [Data - open access](http://srtm.csi.cgiar.org/srtmdata/)

## Kaggle
Kaggle hosts several large satellite image datasets ([> 1 GB](https://www.kaggle.com/datasets?sortBy=relevance&group=public&search=image&page=1&pageSize=20&size=large&filetype=all&license=all)). A list if general image datasets is [here](https://gisgeography.com/free-satellite-imagery-data-list/). A list of land-use datasets is [here](https://gisgeography.com/free-global-land-cover-land-use-data/). The [kaggle blog](http://blog.kaggle.com) is an interesting read.

### Kaggle - Amazon from space - classification challenge
* https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
* 3-5 meter resolution GeoTIFF images from planet Dove satellite constellation
* 12 classes including - **cloudy, primary + waterway** etc
* [1st place winner interview - used 11 custom CNN](http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/)

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
* [Sat4](https://www.kaggle.com/crawford/deepsat-sat4) 500,000 image patches covering four broad land cover classes - **barren land, trees, grassland and a class that consists of all land cover classes other than the above three** [Example notebook](https://www.kaggle.com/robmarkcole/satellite-image-classification)
* [Sat6](https://www.kaggle.com/crawford/deepsat-sat6) 405,000 image patches each of size 28x28 and covering 6 landcover classes - **barren land, trees, grassland, roads, buildings and water bodies.**
* [Deep Gradient Boosted Learning article](https://alan.do/deep-gradient-boosted-learning-4e33adaf2969)

### Kaggle - other
* Satellite + loan data -> https://www.kaggle.com/reubencpereira/spatial-data-repo

## Alternative datasets
There are a variety of datasets suitable for land classification problems.

### UC Merced
* http://weegee.vision.ucmerced.edu/datasets/landuse.html
* This is a 21 class land use image dataset meant for research purposes.
* There are 100 RGB TIFF images for each class
* Each image measures 256x256 pixels with a pixel resolution of 1 foot

### AWS datasets
* Landsat -> free viewer at [remotepixel](https://viewer.remotepixel.ca/#3/40/-70.5) and [libra](https://libra.developmentseed.org/)
* Optical, radar, segmented etc. https://aws.amazon.com/earth/
* [SpaceNet - WorldView-3](https://spacenetchallenge.github.io/datasets/datasetHomePage.html) and [article here](https://spark-in.me/post/spacenet-three-challenge). Also example [semantic segmentation using Raster Vision](https://docs.rastervision.io/en/0.8/quickstart.html)

### Quilt
* Several people have uploaded datasets to [Quilt](https://quiltdata.com/search/?q=satellite)

### Google Earth Engine
* https://developers.google.com/earth-engine/
* Various imagery and climate datasets, including Landsat & Sentinel imagery
* [Python API](https://developers.google.com/earth-engine/python_install) but  all compute happens on Googles servers

### Weather Datasets
* UK met-odffice -> https://www.metoffice.gov.uk/datapoint
* NASA (make request and emailed when ready) -> https://search.earthdata.nasa.gov
* NOAA (requires BigQuery) -> https://www.kaggle.com/noaa/goes16/home
* Time series weather data for several US cities -> https://www.kaggle.com/selfishgene/historical-hourly-weather-data

# Online computing resources
Generally a GPU is required for DL, and this section lists Jupyter environments with GPU available. TLDR: Colab for free computing with GPU, all others have strings attached.

### Google Colab
* Collaboratory [notebooks](https://colab.research.google.com) with GPU as a backend for free for 12 hours at a time. Note that the GPU may be shared with other users, so if you aren't getting good performance try reloading.
* Tensorflow available & pytorch can be installed, [useful articles](https://towardsdatascience.com/fast-ai-lesson-1-on-google-colab-free-gpu-d2af89f53604)

### Kaggle - also Google!
* Free to use
* GPU Kernels - may run for 1 hour
* Tensorflow, pytorch & fast.ai available
* Advantage that many datasets are already available
* [Read](https://medium.com/@hortonhearsafoo/announcing-fast-ai-part-1-now-available-as-kaggle-kernels-8ef4ca3b9ce6)

### Floydhub
* https://www.floydhub.com/
* Pricing -> https://www.floydhub.com/pricing
* Free plan allows 1 job and 10GB storage, but pay for GPU.
* Cloud GPUs (AWS backend)
* Tensorboard
* Version Control for DL
* Deploy Models as REST APIs
* Public Datasets

### Clouderizer
* https://clouderizer.com/
* Clouderizer $5 month for 200 hours (Robbie plan)
* Run projects locally, on cloud or both.
* SSH terminal, Jupyter Notebooks and Tensorboard are securely accessible from Clouderizer Web Console.

### Paperspace
* https://www.paperspace.com/
* 1-Click Jupyter Notebooks, GPU on demand
* Pay as you go -> https://www.paperspace.com/pricing
* [Python API](https://github.com/Paperspace/paperspace-python)

## Crestle
* https://www.crestle.com/
* Pricing -> https://www.crestle.com/pricing
* Min plan is $5 per month with 200 hours per month. Pay $0.59/hour for GPU and storage $0.014/GB/day
* Cloud GPU & persistent file store
* Fast.ai lessons pre-installed

# Interesting DL projects
### Raster Vision by Azavea
* https://www.azavea.com/projects/raster-vision/
* An open source Python framework for building computer vision models on aerial, satellite, and other large imagery sets.
* Accessible through the [Raster Foundry](https://www.rasterfoundry.com/)
* [Example use cases on open data](https://github.com/azavea/raster-vision-examples)

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
* Large datasets may come in HDF5 format, can view with -> https://www.hdfgroup.org/downloads/hdfview/
* Climate data is often in netcdf format, which can be [opened using xarray](https://moderndata.plot.ly/weather-maps-in-python-with-mapbox-gl-xarray-and-netcdf4/)
* The xarray docs list a number of ways that data [can be stored and loaded](http://xarray.pydata.org/en/latest/io.html#).

## STAC - SpatioTemporal Asset Catalog
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
* Overall trend to using AWS S3 backend for image storage. There are a variety of tools for exploring and having teams collaborate on data on S3, e.g. [T4](https://github.com/quiltdata/t4).
* Bucking the trend, [Descartes & Airbus are using a google backend](https://spacenews.com/descartes-labs-platform-adds-airbus-imagery/) -> checkout [gcsts for google cloud storage sile-system](https://github.com/dask/gcsfs)
* Just speculating, but a [serverless pipeline](https://github.com/aws-samples/amazon-rekognition-video-analyzer) appears to be where companies are headed for routine compute tasks, whilst providing a Jupyter notebook approach for custom analysis.
* Traditional data formats aren't designed for processing, so new standards are developing such as [cloud optimised geotiffs](http://blog.digitalglobe.com/developers/cloud-optimized-geotiffs-and-the-path-to-accessible-satellite-imagery-analytics/) and [zarr](https://github.com/zarr-developers/zarr)

# Online platforms for Geo analysis
* [This article discusses some of the available platforms](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2) -> TLDR Pangeo rocks, but must BYO imagery
* Pangeo - open source resources for parallel processing using Dask and Xarray http://pangeo.io/index.html
* [Airbus Sandbox](https://sandbox.intelligence-airbusds.com/web/) -> will provide access to imagery
* [Descartes Labs](https://www.descarteslabs.com/) -> access to EO imagery from a variety of providers via python API -> not clear which imagery is available (Airbus + others?) or pricing
* DigitalGlobe have a cloud hosted Jupyter notebook platform called [GBDX](https://platform.digitalglobe.com/gbdx/). Cloud hosting means they can guarantee the infrastructure supports their algorithms, and they appear to be close/closer to deploying DL. [Tutorial notebooks here](https://notebooks.geobigdata.io/hub/tutorials/list). Only Sentinel-2 and Landsat data on free tier.
* Planet have a [Jupyter notebook platform](https://developers.planet.com/) which can be deployed locally and requires an [API key](https://developers.planet.com/docs/quickstart/getting-started/) (14 days free). They have a python wrapper (2.7..) to their rest API. No price after 14 day trial.


# Techniques
This section explores the different techniques (DL, ML & classical) people are applying to common problems in satellite imagery analysis. Classification problems are the most simply addressed via DL, object detection is harder, and cloud detection harder still (niche interest).

## Land classification
* Very common problem, assign land classification to a pixel based on pixel value, can be addressed via [simple sklearn cluster algorithm](https://github.com/acgeospatial/Satellite_Imagery_Python/blob/master/Clustering_KMeans-Sentinel2.ipynb) or [deep learning](https://towardsdatascience.com/land-use-land-cover-classification-with-deep-learning-9a5041095ddb).
* Land use is related to classification, but we are trying to detect a scene, e.g. housing, forestry. I have tried CNN -> [See my notebooks](https://github.com/robmarkcole/satellite-image-deep-learning/tree/master/land_classification)
* [Land Use Classification using Convolutional Neural Network in Keras](https://github.com/tavgreen/landuse_classification)
* [Sea-Land segmentation using DL](https://arxiv.org/pdf/1709.00201.pdf)
* [Pixel level segmentation on Azure](https://github.com/Azure/pixel_level_land_classification)
* [Deep Learning-Based Classification of Hyperspectral Data](https://github.com/hantek/deeplearn_hsi)
* [A U-net based on Tensorflow for objection detection (or segmentation) of satellite images - DSTL dataset but python 2.7](https://github.com/rogerxujiang/dstl_unet)

## Semantic segmentation
* Pixel-wise classification
* [Instance segmentation with keras - links to satellite examples](https://github.com/matterport/Mask_RCNN)


## Change detection
* Monitor water levels, coast lines, size of urban areas, wildfire damage. Note, clouds change often too..!
* Using PCA (python 2, requires updating) -> https://appliedmachinelearning.blog/2017/11/25/unsupervised-changed-detection-in-multi-temporal-satellite-images-using-pca-k-means-python-code/
* Using CNN -> https://github.com/vbhavank/Unstructured-change-detection-using-CNN
* [Siamese neural network to detect changes in aerial images](https://github.com/vbhavank/Siamese-neural-network-for-change-detection)
* https://www.spaceknow.com/
* [LANDSAT Time Series Analysis for Multi-temporal Land Cover Classification using Random Forest](https://github.com/agr-ayush/Landsat-Time-Series-Analysis-for-Multi-Temporal-Land-Cover-Classification)
* [Change Detection in 3D: Generating Digital Elevation Models from Dove Imagery](https://www.planet.com/pulse/publications/change-detection-in-3d-generating-digital-elevation-models-from-dove-imagery/)
* [Change Detection in Hyperspectral Images Using Recurrent 3D Fully Convolutional Networks](https://www.mdpi.com/2072-4292/10/11/1827)
* [PySAR - InSAR (Interferometric Synthetic Aperture Radar) timeseries analysis in python](https://github.com/hfattahi/PySAR)

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
* [Building Extraction with YOLT2 and SpaceNet Data](https://medium.com/the-downlinq/building-extraction-with-yolt2-and-spacenet-data-a926f9ffac4f)
* [Find sports fields using Mask R-CNN and overlay on open-street-map](https://github.com/jremillard/images-to-osm)


## Cloud detection
* A subset of the object detection problem, but surprisingly challenging
* From [this article on sentinelhub](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13) there are three popular classical algorithms that detects thresholds in multiple bands in order to identify clouds. In the same article they propose using semantic segmentation combined with a CNN for a cloud classifier (excellent review paper [here](https://arxiv.org/pdf/1704.06857.pdf)), but state that this requires too much compute resources.
* [This article](https://www.mdpi.com/2072-4292/8/8/666) compares a number of ML algorithms, random forests, stochastic gradient descent, support vector machines, Bayesian method.
* DL..

## Super resolution
* https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-1-ec5c5cd3cd2
* https://modeldepot.io/joe/vdsr-for-super-resolution

## Pansharpening
* Image fusion of low res multispectral with high res pan band. Several algorithms described [in the ArcGIS docs](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/fundamentals-of-panchromatic-sharpening.htm), with the simplest being taking the mean of the pan and RGB pixel value.
* Does not require DL, classical algos suffice, [see this notebook](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/panchromatic_sharpening.ipynb) and [this kaggle kernel](https://www.kaggle.com/resolut/panchromatic-sharpening)
* https://github.com/mapbox/rio-pansharpen

## Stereo imaging for terrain mapping & DEMs
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
* [Phase correlation in scikit-image](http://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html?highlight=cross%20correlation)

## Lidar
* [Reconstructing 3D buildings from aerial LiDAR with Mask R-CNN)](https://medium.com/geoai/reconstructing-3d-buildings-from-aerial-lidar-with-ai-details-6a81cb3079c0)

## NVDI - vegetation index
* Simple band math `ndvi = np.true_divide((ir - r), (ir + r))` but challenging due to the size of the imagery.
* [Example notebook local](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/ndvi_calculation.ipynb)
* [Landsat data in cloud optimised format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article here](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).

## SAR
* [Removing speckle noise from Sentinel-1 SAR using a CNN](https://medium.com/upstream/denoising-sentinel-1-radar-images-5f764faffb3e)

# For fun
* [Style transfer - see the world in a new way](https://gist.github.com/jacquestardie/6227891818625e4c19c1b1d5bebe4fe4)

# Useful open source software
* [QGIS](https://qgis.org/en/site/)- Create, edit, visualise, analyse and publish geospatial information. [Python scripting and plugins](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/intro.html#scripting-in-the-python-console).
* [Orfeo toolbox](https://www.orfeo-toolbox.org/) - remote sensing toolbox with python API (just a wrapper to the C code). Do activites such as [pansharpening](https://www.orfeo-toolbox.org/CookBook/Applications/app_Pansharpening.html), ortho-rectification, image registration, image segmentation & classification. Not much documentation.
* [QUICK TERRAIN READER - view DEMS, Windows](http://appliedimagery.com/download/)

# Useful References
* https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0
* https://github.com/taspinar/sidl/blob/master/notebooks/2_Detecting_road_and_roadtypes_in_sattelite_images.ipynb
* [Geonotebooks](https://github.com/OpenGeoscience/geonotebook) with [Docker container](https://github.com/OpenGeoscience/geonotebook/tree/master/devops/docker)
* [Sentinel NetCDF data](https://github.com/acgeospatial/Sentinel-5P/blob/master/Sentinel_5P.ipynb)
* Open Data Cube - serve up cubes of data https://www.opendatacube.org/
* [Process Satellite data using AWS Lambda functions](https://github.com/RemotePixel/remotepixel-api)
* [OpenDroneMap](https://github.com/OpenDroneMap/ODM) - generate maps, point clouds, 3D models and DEMs from drone, balloon or kite images.
