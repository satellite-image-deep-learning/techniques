# Datasets
* https://github.com/chrieke/awesome-satellite-imagery-datasets
* [Various datasets listed here](https://www.maptiler.com/gallery/satellite/)

Kaggle hosts several large satellite image datasets ([> 1 GB](https://www.kaggle.com/datasets?sortBy=relevance&group=public&search=image&page=1&pageSize=20&size=large&filetype=all&license=all)). A list if general image datasets is [here](https://gisgeography.com/free-satellite-imagery-data-list/). A list of land-use datasets is [here](https://gisgeography.com/free-global-land-cover-land-use-data/).

### Kaggle - Deepsat - classification challenge
Each sample image is 28x28 pixels and consists of 4 bands - red, green, blue and near infrared. The training and test labels are one-hot encoded 1x6 vectors. Each image patch is size normalized to 28x28 pixels. Data in `.mat` Matlab format. JPEG?
* [Sat4](https://www.kaggle.com/crawford/deepsat-sat4) 500,000 image patches covering four broad land cover classes - **barren land, trees, grassland and a class that consists of all land cover classes other than the above three** [Example notebook](https://www.kaggle.com/robmarkcole/satellite-image-classification)
* [Sat6](https://www.kaggle.com/crawford/deepsat-sat6) 405,000 image patches each of size 28x28 and covering 6 landcover classes - **barren land, trees, grassland, roads, buildings and water bodies.**

### Kaggle - Draper - place images in order of time
* https://www.kaggle.com/c/draper-satellite-image-chronology/data
* Images are grouped into sets of five, each of which have the same setId. Each image in a set was taken on a different day (but not necessarily at the same time each day). The images for each set cover approximately the same area but are not exactly aligned.

### Kaggle - DSTL - segmentation challenge
* https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
* 45 satellite images covering 1km x 1km in both 3-band and 16-band formats.
* 10 Labelled classes include - **Buildings, Road, Trees, Crops, Waterway, Vehicles**

### Kaggle - Amazon from space - classification challenge
* https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
* 3-5 meter resolution GeoTIFF images
* 12 classes including - **cloudy, primary + waterway** etc

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
* Sentinel 2 -> [raw data - requester pays](https://registry.opendata.aws/sentinel-2/) Also paid access via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py)
* Optical, radar, segmented etc. https://aws.amazon.com/earth/
* [SpaceNet](https://spacenetchallenge.github.io/datasets/datasetHomePage.html)

# Online computing resources
List of available cloud resources

### Kaggle
* GOU Kernels (may run for 1 hour which limits usefulness?)
* Tensorflow, pytorch & fast.ai available
* Advantage that many datasets are already available
* Free to use
* [Read](https://medium.com/@hortonhearsafoo/announcing-fast-ai-part-1-now-available-as-kaggle-kernels-8ef4ca3b9ce6)

### Clouderizer
* https://clouderizer.com/
* Clouderizer is a cloud computing management service, it takes care of installing the required packages to a cloud computing instance (like Amazon AWS or Google Colab). Clouderizer is free for 200 hours per month (Robbie plan) and does not require a credit card to sign up.
* Run projects locally, on cloud or both.
* SSH terminal, Jupyter Notebooks and Tensorboard are securely accessible from Clouderizer Web Console.

### AWS
* GPU available (link?)
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

# Interesting projects
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

### Floydhub??

### modeldepot
* https://modeldepot.io
* ML models hosted

# Image formats & catalogues
* We certainly want to consider cloud optimised GeoTiffs https://www.cogeo.org/
* https://terria.io/ for pretty catalogues
* [Remote pixel](https://remotepixel.ca/projects/index.html#satsearch)
* [Sentinel-hub eo-browser](https://apps.sentinel-hub.com/eo-browser/)

# State of the art
What are companies doing?
* Many using AWS S3 backend for image storage, see [DigitalGlobe](http://blog.digitalglobe.com/developers/cloud-optimized-geotiffs-and-the-path-to-accessible-satellite-imagery-analytics/). some have these integrated with (likely) Geoserver, e.g. [Planet](https://github.com/planetlabs). Just speculating, but a [serverless pipeline](https://github.com/aws-samples/amazon-rekognition-video-analyzer) appears to be the way to go. For simplifying deployment services like [Algorithmia](https://algorithmia.com/) may be useful
* [Voyager](https://www.voyagersearch.com/using-aws-batch-to-generate-image-thumbnails-for-voyager)
* Pangeo - resources for parallel processing of images http://pangeo.io/index.html
* Open Data Cube - serve up cubes of data https://www.opendatacube.org/
* [Process Satellite data using AWS Lambda functions](https://github.com/RemotePixel/remotepixel-api)

## Cloud detection
* Either (1) algorithm that detects thresholds in multiple bands or (2) machine learning single pixel classification -> https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13


# Useful References
* https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0
* https://github.com/taspinar/sidl/blob/master/notebooks/2_Detecting_road_and_roadtypes_in_sattelite_images.ipynb
* [Geonotebooks](https://github.com/OpenGeoscience/geonotebook) with [Docker container](https://github.com/OpenGeoscience/geonotebook/tree/master/devops/docker)
