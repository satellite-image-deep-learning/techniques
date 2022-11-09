# Datasets
This section contains a short list of datasets relevant to deep learning, particularly those which come up regularly in the literature. **Warning** satellite image files can be LARGE, and even a small datasets may comprise 50GB+ of imagery

## Lists of datasets
* [Earth Observation Database](https://eod-grss-ieee.com/)
* [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets)
* [Awesome_Satellite_Benchmark_Datasets](https://github.com/Seyed-Ali-Ahmadi/Awesome_Satellite_Benchmark_Datasets)
* [Callisto-Dataset-Collection](https://github.com/Agri-Hub/Callisto-Dataset-Collection) -> datasets that use Copernicus/sentinel data

## Sentinel
* As part of the [EU Copernicus program](https://en.wikipedia.org/wiki/Copernicus_Programme), multiple Sentinel satellites are capturing imagery -> see [wikipedia](https://en.wikipedia.org/wiki/Copernicus_Programme#Sentinel_missions).
* 13 bands, Spatial resolution of 10 m, 20 m and 60 m, 290 km swath, the temporal resolution is 5 days
* [awesome-sentinel](https://github.com/Fernerkundung/awesome-sentinel) -> a curated list of awesome tools, tutorials and APIs related to data from the Copernicus Sentinel Satellites.
* [Sentinel-2 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs/) and [Sentinel-2 L2A 120m Mosaic](https://registry.opendata.aws/sentinel-s2-l2a-mosaic-120/)
* [Open access data on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2?prefix=tiles%2F31%2FT%2FCJ%2F) 
* Paid access to Sentinel & Landsat data via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py)
* [Example loading sentinel data in a notebook](https://github.com/binder-examples/getting-data/blob/master/Sentinel2.ipynb)
* [so2sat on Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/so2sat) - So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and the corresponding local climate zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions of the world.
* [BigEarthNet](https://www.tensorflow.org/datasets/catalog/bigearthnet) - The BigEarthNet is a new large-scale Sentinel-2 benchmark archive, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels. Also available [in torchgeo](https://torchgeo.readthedocs.io/en/latest/api/datasets.html#bigearthnet)
* [Jupyter Notebooks for working with Sentinel-5P Level 2 data stored on S3](https://github.com/Sentinel-5P/data-on-s3). The data can be browsed [here](https://meeo-s5p.s3.amazonaws.com/index.html#/?t=catalogs)
* [Sentinel NetCDF data](https://github.com/acgeospatial/Sentinel-5P/blob/master/Sentinel_5P.ipynb)
* [Analyzing Sentinel-2 satellite data in Python with Keras](https://github.com/jensleitloff/CNN-Sentinel)
* [Xarray backend to Copernicus Sentinel-1 satellite data products](https://github.com/bopen/xarray-sentinel)
* [SEN2VENµS](https://zenodo.org/record/6514159#.YoRxM5PMK3I) -> a dataset for the training of Sentinel-2 super-resolution algorithms
* [SEN12MS](https://github.com/zhu-xlab/SEN12MS) -> A Curated Dataset of Georeferenced Multi-spectral Sentinel-1/2 Imagery for Deep Learning and Data Fusion. Checkout [SEN12MS toolbox](https://github.com/schmitt-muc/SEN12MS) and many referenced uses on [paperswithcode.com](https://paperswithcode.com/dataset/sen12ms)
* [Sen4AgriNet](https://github.com/Orion-AI-Lab/S4A) -> A Sentinel-2 multi-year, multi-country benchmark dataset for crop classification and segmentation with deep learning, with [website](https://www.sen4agrinet.space.noa.gr/) and [models](https://github.com/Orion-AI-Lab/S4A-Models)
* [earthspy](https://github.com/AdrienWehrle/earthspy) -> Monitor and study any place on Earth and in Near Real-Time (NRT) using the Sentinel Hub services developed by the EO research team at Sinergise
* [Space2Ground](https://github.com/Agri-Hub/Space2Ground) -> dataset with Space (Sentinel-1/2) and Ground (street-level images) components, annotated with crop-type labels for agriculture monitoring.
* [sentinel2tools](https://github.com/QuantuMobileSoftware/sentinel2tools) -> downloading & basic processing of Sentinel 2 imagesry. Read [Sentinel2tools: simple lib for downloading Sentinel-2 satellite images](https://medium.com/geekculture/sentinel2tools-simple-lib-for-downloading-sentinel-2-satellite-images-f8a6be3ee894)
* [open-sentinel-map](https://github.com/VisionSystemsInc/open-sentinel-map) -> The OpenSentinelMap dataset contains Sentinel-2 imagery and per-pixel semantic label masks derived from OpenStreetMap
* [MSCDUnet](https://github.com/Lihy256/MSCDUnet) -> change detection datasets containing VHR, multispectral (Sentinel-2) and SAR (Sentinel-1)
* [OMBRIA](https://github.com/geodrak/OMBRIA) -> Sentinel-1 & 2 dataset for adressing the flood mapping problem
* [Canadian-cropland-dataset](https://github.com/bioinfoUQAM/Canadian-cropland-dataset) -> a novel patch-based dataset compiled using optical satellite images of Canadian agricultural croplands retrieved from Sentinel-2
* [Sentinel-2 Cloud Cover Segmentation Dataset](https://mlhub.earth/data/ref_cloud_cover_detection_challenge_v1) on Radiant mlhub
* [The Azavea Cloud Dataset](https://www.azavea.com/blog/2021/08/02/the-azavea-cloud-dataset/) which is used to train this [cloud-model](https://github.com/azavea/cloud-model)

## Landsat
* Long running US program -> see [Wikipedia](https://en.wikipedia.org/wiki/Landsat_program)
* 8 bands, 15 to 60 meters, 185km swath, the temporal resolution is 16 days
* [Landsat 4, 5, 7, and 8 imagery on Google](https://cloud.google.com/storage/docs/public-datasets/landsat), see [the GCP bucket here](https://console.cloud.google.com/storage/browser/gcp-public-data-landsat/), with Landsat 8 imagery in COG format analysed in [this notebook](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb)
* [Landsat 8 imagery on AWS](https://registry.opendata.aws/landsat-8/), with many tutorials and tools listed
* https://github.com/kylebarron/landsat-mosaic-latest -> Auto-updating cloudless Landsat 8 mosaic from AWS SNS notifications
* [Visualise landsat imagery using Datashader](https://examples.pyviz.org/landsat/landsat.html#landsat-gallery-landsat)
* [Landsat-mosaic-tiler](https://github.com/kylebarron/landsat-mosaic-tiler) -> This repo hosts all the code for landsatlive.live website and APIs.

## Maxar
* Satellites owned by [Maxar](https://www.maxar.com/) (formerly DigitalGlobe) include [GeoEye-1](https://en.wikipedia.org/wiki/GeoEye-1), [WorldView-2](https://en.wikipedia.org/wiki/WorldView-2), [3](https://en.wikipedia.org/wiki/WorldView-3) & [4](https://en.wikipedia.org/wiki/WorldView-4)
* [Open Data images for humanitarian response](https://www.maxar.com/open-data)
* Maxar ARD (COG plus data masks, with STAC) [sample data in S3](https://ard.maxar.com/docs/sdk/examples/outputs/)
* [Dataset on AWS](https://spacenet.ai/datasets/) -> see [this getting started notebook](https://medium.com/the-downlinq/getting-started-with-spacenet-data-827fd2ec9f53) and this notebook on the [off-Nadir dataset](https://medium.com/the-downlinq/introducing-the-spacenet-off-nadir-imagery-and-buildings-dataset-e4a3c1cb4ce3)
* [cloud_optimized_geotif here](http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/) used in the 3D modelling notebook [here](https://gfacciol.github.io/IS18/).
* [WorldView cloud optimized geotiffs](http://menthe.ovh.hw.ipol.im/IARPA_data/cloud_optimized_geotif/) used in the 3D modelling notebook [here](https://gfacciol.github.io/IS18/).
* For more Worldview imagery see Kaggle DSTL competition.

## Planet
* [Planet’s high-resolution, analysis-ready mosaics of the world’s tropics](https://www.planet.com/nicfi/), supported through Norway’s International Climate & Forests Initiative. [BBC coverage](https://www.bbc.co.uk/news/science-environment-54651453)
* Planet have made imagery available via kaggle competitions

## UC Merced
* Land use classification dataset with 21 classes and 100 RGB TIFF images for each class
* Each image measures 256x256 pixels with a pixel resolution of 1 foot
* http://weegee.vision.ucmerced.edu/datasets/landuse.html
* Available as a Tensorflow dataset -> https://www.tensorflow.org/datasets/catalog/uc_merced
* Also [available as a multi-label dataset](https://towardsdatascience.com/multi-label-land-cover-classification-with-deep-learning-d39ce2944a3d)
* Read [Vision Transformers for Remote Sensing Image Classification](https://www.mdpi.com/2072-4292/13/3/516/htm) where a Vision Transformer classifier achieves 98.49% classification accuracy on Merced

## EuroSAT
* Land use classification dataset of Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples. Available in RGB and 13 band versions
* [EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/EuroSAT) -> publication where a CNN achieves a classification accuracy 98.57%
* Repos using fastai [here](https://github.com/shakasom/Deep-Learning-for-Satellite-Imagery) and [here](https://www.luigiselmi.eu/eo/lulc-classification-deeplearning.html)
* [evolved_channel_selection](http://matpalm.com/blog/evolved_channel_selection/) -> explores the trade off between mixed resolutions and whether to use a channel at all, with [repo](https://github.com/matpalm/evolved_channel_selection)
* RGB version available as [dataset in pytorch](https://pytorch.org/vision/stable/generated/torchvision.datasets.EuroSAT.html#torchvision.datasets.EuroSAT) with the 13 band version [in torchgeo](https://torchgeo.readthedocs.io/en/latest/api/datasets.html#eurosat). Checkout the tutorial on [data augmentation with this dataset](https://torchgeo.readthedocs.io/en/latest/tutorials/transforms.html)
* RGB and 13 band versions [in tensorflow](https://www.tensorflow.org/datasets/catalog/eurosat)

## PatternNet
* Land use classification dataset with 38 classes and 800 RGB JPG images for each class
* https://sites.google.com/view/zhouwx/dataset?authuser=0
* Publication: [PatternNet: A Benchmark Dataset for Performance Evaluation of Remote Sensing Image Retrieval](https://arxiv.org/abs/1706.03424)

## Million-AID
* https://captain-whu.github.io/DiRS/
* a new large-scale benchmark dataset containing million instances for RS scene classification
* 51 scene categories organized by the hierarchical category
* [Pretrained models](https://github.com/ViTAE-Transformer/ViTAE-Transformer-Remote-Sensing)
* Also see [AID](https://captain-whu.github.io/AID/), [AID-Multilabel-Dataset](https://github.com/Hua-YS/AID-Multilabel-Dataset) & [DFC15-multilabel-dataset](https://github.com/Hua-YS/DFC15-Multilabel-Dataset)

## DIOR object detection dataset
* https://gcheng-nwpu.github.io/
* https://arxiv.org/abs/1909.00133
* "DIOR" is a large-scale benchmark dataset for object detection in optical remote sensing images, which consists of 23,463 images and 192,518 object instances annotated with horizontal bounding boxes
* [ors-detection](https://github.com/Vlad15lav/ors-detection) -> Object Detection on the DIOR dataset using YOLOv3
* [dior_detect](https://github.com/hm-better/dior_detect) -> benchmarks for object detection on DIOR dataset
* [Tools](https://github.com/CrazyStoneonRoad/Tools) -> for dealing with the DIOR

## Multiscene
* https://multiscene.github.io/ & https://github.com/Hua-YS/Multi-Scene-Recognition
* MultiScene dataset aims at two tasks: Developing algorithms for multi-scene recognition & Network learning with noisy labels

## FAIR1M object detection dataset
* [FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery](https://arxiv.org/abs/2103.05569)
* Download at gaofen-challenge.com
* [2020Gaofen](https://github.com/AICyberTeam/2020Gaofen) -> 2020 Gaofen Challenge data, baselines, and metrics

## DOTA object detection dataset
* https://captain-whu.github.io/DOTA/index.html
* A Large-Scale Benchmark and Challenges for Object Detection in Aerial Images
* [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) for loading dataset
* [Arxiv paper](https://arxiv.org/abs/1711.10398)
* [Pretrained models in mmrotate](https://github.com/open-mmlab/mmrotate)
* [DOTA2VOCtools](https://github.com/Complicateddd/DOTA2VOCtools) -> dataset split and transform to voc format
* Segmentation annotations available in iSAID dataset

## iSAID instance segmentation dataset
* https://captain-whu.github.io/iSAID/dataset.html
* A Large-scale Dataset for Instance Segmentation in Aerial Images
* Uses images from the DOTA dataset

## HRSC RGB ship object detection dataset
* https://www.kaggle.com/datasets/guofeng/hrsc2016
* [Pretrained models in mmrotate](https://github.com/open-mmlab/mmrotate)
* [Rotation-RetinaNet-PyTorch](https://github.com/HsLOL/Rotation-RetinaNet-PyTorch)

## SAR Ship Detection Dataset (SSDD)
* https://github.com/TianwenZhang0825/Official-SSDD
* [Rotation-RetinaNet-PyTorch](https://github.com/HsLOL/Rotation-RetinaNet-PyTorch)

## LEVIR ship dataset
* [LEVIR-Ship](https://github.com/WindVChen/LEVIR-Ship)
* A dataset for tiny ship detection under medium-resolution remote sensing images
<!-- markdown-link-check-disable -->
* Hosted on [Nucleus](https://dashboard.scale.com/nucleus/ds_cbsghny30nf00b1x3w7g?utm_source=open_dataset&utm_medium=github&utm_campaign=levir_ships)
<!-- markdown-link-check-enable -->

## SAR Aircraft Detection Dataset 
* https://github.com/hust-rslab/SAR-aircraft-data
* 2966 nonoverlapped 224×224 slices are collected with 7835 aircraft targets

## xView Challenge Datasets for Humanitarian Assistance and Disaster Response
* [xView1](http://xviewdataset.org/) - Objects in context for overhead imagery. A fine-grained object detection dataset with 60 object classes along an ontology of 8 class types. Over 1,000,000 objects across over 1,400 km^2 of 0.3m resolution imagery. Paper available on [arXiv](https://arxiv.org/abs/1802.07856).
* [xView2/xBD](https://xview2.org/) - Finding and assessing damaged buildings on pre- and post-natural disaster imagery. With over 850,000 annotated buildings across over 45,000 km^2 of 0.3m resolution imagery, this dataset provides precise segmentation masks and damage labels on a four-level spectrum. Paper available on [arXiv](https://arxiv.org/abs/1911.09296).
* [xView3](https://iuu.xview.us/) - Detecting dark vessels engaged in illegal, unreported, and unregulated (IUU) fishing activities on synthetic aperture radar (SAR) imagery. With human and algorithm annotated instances of vessels and fixed infrastructure across 43,200,000 km^2 of Sentinel-1 imagery, this multi-modal dataset enables algorithms to detect and classify dark vessels. Paper available on [arXiv](https://arxiv.org/abs/2206.00897).
* All reference code, dataset processing utilities, and winning model codes + weights are available on the (xView GitHub organization page)[https://github.com/DIUx-xView).

## Vehicle Detection in Aerial Imagery (VEDAI)
* Link in the repo below possibly broken
* [pytorch-vedai](https://github.com/MichelHalmes/pytorch-vedai) -> object detection on the VEDAI dataset: Vehicle Detection in Aerial Imagery

## Cars Overhead With Context (COWC)
* http://gdo152.ucllnl.org/cowc/
* https://github.com/LLNL/cowc
* Large set of annotated cars from overhead
* Established baseline for detection and counting tasks

## AI-TOD - tiny object detection
* https://github.com/jwwangchn/AI-TOD
* The mean size of objects in AI-TOD is about 12.8 pixels, which is much smaller than other datasets
* [NWD](https://github.com/jwwangchn/NWD) -> code for 2021 [paper](https://arxiv.org/abs/2110.13389): A Normalized Gaussian Wasserstein Distance for Tiny Object Detection. Uses AI-TOD dataset
* [AI-TOD-v2](https://chasel-tsui.github.io/AI-TOD-v2/) -> meticulously relabelling of the v1 dataset

## Counting from Sky
* A Large-scale Dataset for Remote Sensing Object Counting and A Benchmark Method
* https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method

## AIRS (Aerial Imagery for Roof Segmentation)
* https://www.airs-dataset.com
* Public dataset for roof segmentation from very-high-resolution aerial imagery (7.5cm)
* AIRS dataset covers almost the full area of Christchurch, the largest city in the South Island of New Zealand.
* [Also on Kaggle](https://www.kaggle.com/atilol/aerialimageryforroofsegmentation)
* [Rooftop-Instance-Segmentation](https://github.com/MasterSkepticista/Rooftop-Instance-Segmentation) -> VGG-16, Instance Segmentation, uses the Airs dataset

## Inria building/not building segmentation dataset
* https://project.inria.fr/aerialimagelabeling/contest/
* RGB GeoTIFF at spatial resolution of 0.3 m
* Data covering Austin, Chicago, Kitsap County, Western & Easter Tyrol, Innsbruck, San Francisco & Vienna
* [SemSegBuildings](https://github.com/SharpestProjects/SemSegBuildings) -> Project using fast.ai framework for semantic segmentation on Inria building segmentation dataset
* [UNet_keras_for_RSimage](https://github.com/loveswine/UNet_keras_for_RSimage) -> keras code for binary semantic segmentation

## AICrowd Mapping Challenge \building segmentation dataset
* Dataset release as part of the [mapping-challenge](https://www.aicrowd.com/challenges/mapping-challenge)
* 300x300 pixel RGB images with annotations in COCO format
* Imagery appears to be global but with significant fraction from North America
* Winning solution published by neptune.ai [here](https://github.com/neptune-ai/open-solution-mapping-challenge), achieved precision 0.943 and recall 0.954 using Unet with Resnet.
* [mappingchallenge](https://github.com/krishanr/mappingchallenge) -> YOLOv5 applied to the AICrowd Mapping Challenge dataset

## BONAI - building footprint dataset
* https://github.com/jwwangchn/BONAI
* BONAI (Buildings in Off-Nadir Aerial Images) is a dataset for building footprint extraction (BFE) in off-nadir aerial images

## GID15 large scale semantic segmentation dataset
* https://captain-whu.github.io/GID15/

## LEVIR-CD building change detection dataset
* https://justchenhao.github.io/LEVIR/
* [FCCDN_pytorch](https://github.com/chenpan0615/FCCDN_pytorch) -> pytorch implemention of FCCDN for change detection task
* [RSICC](https://github.com/Chen-Yang-Liu/RSICC) -> the Remote Sensing Image Change Captioning dataset uses LEVIR-CD imagery

## ISPRS
* https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx
* Semantic segmentation dataset
* 38 patches of 6000x6000 pixels, each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic, and a DSM. Resolution 5 cm

## SpaceNet
* [spacenet.ai](https://spacenet.ai/) is an online hub for data, challenges, algorithms, and tools. Note that [CosmiQ Ended its Leadership of SpaceNet](https://medium.com/the-downlinq/closing-time-cosmiq-works-is-closing-down-and-ending-its-leadership-of-spacenet-a53ba239745b), handing over the reigns to Maxar
* SpaceNet ran a series consisting of seven challenges with datasets and utilities provided. Challenges covered (1&2) building segmentation, (3) road segmentation, (4) off-nadir buildings, (5) road network extraction, (6)multi-senor mapping, (7) multi-temporal urban change
* Building datasets covered a number of cities including: Rio, Paris, Vegas, Shanghai, Khartoum, Atlana, Moscow, Mumbai & Rotterdam
* [The SpaceNet 7 Multi-Temporal Urban Development Challenge: Dataset Release](https://medium.com/the-downlinq/the-spacenet-7-multi-temporal-urban-development-challenge-dataset-release-9e6e5f65c8d5)
* [spacenet-three-topcoder](https://github.com/snakers4/spacenet-three-topcoder) solution
* [official utilities](https://github.com/SpaceNetChallenge/utilities) -> Packages intended to assist in the preprocessing of SpaceNet satellite imagery data corpus to a format that is consumable by machine learning algorithms
* [andraugust spacenet-utils](https://github.com/andraugust/spacenet-utils) -> Display geotiff image with building-polygon overlay & label buildings using kNN on the pixel spectra
* [Spacenet-Building-Detection](https://github.com/IdanC1s2/Spacenet-Building-Detection) -> uses keras and [Spacenet 1 dataset](https://spacenet.ai/spacenet-buildings-dataset-v1/)

## WorldStrat Dataset
* https://github.com/worldstrat/worldstrat
* Nearly 10,000 km² of free high-resolution satellite imagery of unique locations which ensure stratified representation of all types of land-use across the world: from agriculture to ice caps, from forests to multiple urbanization densities.
* Each high-resolution image (1.5 m/pixel) comes with multiple temporally-matched low-resolution images from the freely accessible lower-resolution Sentinel-2 satellites (10 m/pixel)
* Several super-resolution benchmark models trained on it
* [Quick tour of the WorldStrat Dataset](https://robmarkcole.com/posts/2022-08-01-worldstrat.html) -> blog post by robmarkcole

## Tensorflow datasets
* [resisc45](https://www.tensorflow.org/datasets/catalog/resisc45) -> RESISC45 dataset is a publicly available benchmark for Remote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class.
* [eurosat](https://www.tensorflow.org/datasets/catalog/eurosat) -> EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples.
* [BigEarthNet](https://www.tensorflow.org/datasets/catalog/bigearthnet) -> a large-scale Sentinel-2 land use classification dataset, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels. Official website includes version of the dataset with Sentinel 1 & 2 chips
* [so2sat](https://www.tensorflow.org/datasets/catalog/so2sat) -> a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by Sentinel 1 & 2

## AWS datasets
* [Earth on AWS](https://aws.amazon.com/earth/) is the AWS equivalent of Google Earth Engine
* Currently 36 satellite datasets on the [Registry of Open Data on AWS](https://registry.opendata.aws)

## Microsoft datasets
* [US Building Footprints](https://github.com/Microsoft/USBuildingFootprints) -> building footprints in all 50 US states, GeoJSON format, generated using semantic segmentation. Also [Australia](https://github.com/microsoft/AustraliaBuildingFootprints), [Canadian](https://github.com/Microsoft/CanadianBuildingFootprints), [Uganda-Tanzania](https://github.com/microsoft/Uganda-Tanzania-Building-Footprints), [Kenya-Nigeria](https://github.com/microsoft/KenyaNigeriaBuildingFootprints) and [GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints) are available. Checkout [RasterizingBuildingFootprints](https://github.com/mehdiheris/RasterizingBuildingFootprints) to convert vector shapefiles to raster layers
* [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) is a Dask-Gateway enabled JupyterHub deployment focused on supporting scalable geospatial analysis, [source repo](https://github.com/microsoft/planetary-computer-hub)
* [landcover-orinoquia](https://github.com/microsoft/landcover-orinoquia) -> Land cover mapping of the Orinoquía region in Colombia, in collaboration with Wildlife Conservation Society Colombia. An #AIforEarth project
* [RoadDetections dataset by Microsoft](https://github.com/microsoft/RoadDetections)

## Google datasets
* [open-buildings](https://sites.research.google/open-buildings/) -> A dataset of building footprints to support social good applications covering 64% of the African continent. Read [Mapping Africa’s Buildings with Satellite Imagery](https://ai.googleblog.com/2021/07/mapping-africas-buildings-with.html)

## Google Earth Engine (GEE)
Since there is a whole community around GEE I will not reproduce it here but list very select references. Get started at https://developers.google.com/earth-engine/
* Various imagery and climate datasets, including Landsat & Sentinel imagery
* Supports large scale processing with classical algorithms, e.g. clustering for land use. For deep learning, you export datasets from GEE as tfrecords, train on your preferred GPU platform, then upload inference results back to GEE
* [awesome-google-earth-engine](https://github.com/gee-community/awesome-google-earth-engine)
* [Awesome-GEE](https://github.com/giswqs/Awesome-GEE)
* [awesome-earth-engine-apps](https://github.com/philippgaertner/awesome-earth-engine-apps)
* [How to Use Google Earth Engine and Python API to Export Images to Roboflow](https://blog.roboflow.com/how-to-use-google-earth-engine-with-roboflow/) -> to acquire training data
* [ee-fastapi](https://github.com/csaybar/ee-fastapi) is a simple FastAPI web application for performing flood detection using Google Earth Engine in the backend.
* [How to Download High-Resolution Satellite Data for Anywhere on Earth](https://towardsdatascience.com/how-to-download-high-resolution-satellite-data-for-anywhere-on-earth-5e6dddee2803)
* [wxee](https://github.com/aazuspan/wxee) -> Export data from GEE to xarray using wxee then train with pytorch or tensorflow models. Useful since GEE only suports tfrecord export natively

## Radiant Earth
* https://www.radiant.earth/
* Datasets and also models on https://mlhub.earth/

## Image captioning datasets
* [RSICD](https://github.com/201528014227051/RSICD_optimal) -> 10921 images with five sentences descriptions per image. Used in  [Fine tuning CLIP with Remote Sensing (Satellite) images and captions](https://huggingface.co/blog/fine-tune-clip-rsicd), models at [this repo](https://github.com/arampacha/CLIP-rsicd)
* [RSICC](https://github.com/Chen-Yang-Liu/RSICC) -> the Remote Sensing Image Change Captioning dataset contains 10077 pairs of bi-temporal remote sensing images and 50385 sentences describing the differences between images. Uses LEVIR-CD imagery

## Weather Datasets
* UK metoffice -> https://www.metoffice.gov.uk/datapoint
* NASA (make request and emailed when ready) -> https://search.earthdata.nasa.gov
* NOAA (requires BigQuery) -> https://www.kaggle.com/noaa/goes16/home
* Time series weather data for several US cities -> https://www.kaggle.com/selfishgene/historical-hourly-weather-data
* [DeepWeather](https://github.com/adamhazimeh/DeepWeather) -> improve weather forecasting accuracy by analyzing satellite images

## Forest datasets
* [awesome-forests](https://github.com/blutjens/awesome-forests) -> A curated list of ground-truth forest datasets for the machine learning and forestry community
* [ReforesTree](https://github.com/gyrrei/ReforesTree) -> A dataset for estimating tropical forest biomass based on drone and field data

## Geospatial datasets
* [Resource Watch](https://resourcewatch.org/data/explore) provides a wide range of geospatial datasets and a UI to visualise them

## Time series & change detection datasets
* [BreizhCrops](https://github.com/dl4sits/BreizhCrops) -> A Time Series Dataset for Crop Type Mapping
* The SeCo dataset contains image patches from Sentinel-2 tiles captured at different timestamps at each geographical location. [Download SeCo here](https://github.com/ElementAI/seasonal-contrast)
* [Onera Satellite Change Detection Dataset](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) comprises 24 pairs of multispectral images taken from the Sentinel-2 satellites between 2015 and 2018
* [SYSU-CD](https://github.com/liumency/SYSU-CD) -> The dataset contains 20000 pairs of 0.5-m aerial images of size 256×256 taken between the years 2007 and 2014 in Hong Kong

### DEM (digital elevation maps)
* Shuttle Radar Topography Mission, search online at usgs.gov
* Copernicus Digital Elevation Model (DEM) on S3, represents the surface of the Earth including buildings, infrastructure and vegetation. Data is provided as Cloud Optimized GeoTIFFs. [link](https://registry.opendata.aws/copernicus-dem/)
* [Awesome-DEM](https://github.com/DahnJ/Awesome-DEM)

## UAV & Drone datasets
* Many on https://www.visualdata.io
* [AU-AIR dataset](https://bozcani.github.io/auairdataset) -> a multi-modal UAV dataset for object detection.
* [ERA](https://lcmou.github.io/ERA_Dataset/) ->  A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos.
* [Aerial Maritime Drone Dataset](https://public.roboflow.ai/object-detection/aerial-maritime) 
* [RetinaNet for pedestrian detection](https://towardsdatascience.com/pedestrian-detection-in-aerial-images-using-retinanet-9053e8a72c6)
* [Aerial Maritime Drone Dataset](https://public.roboflow.com/object-detection/aerial-maritime/1)
* [EmergencyNet](https://github.com/ckyrkou/EmergencyNet) -> identify fire and other emergencies from a drone
* [OpenDroneMap](https://github.com/OpenDroneMap/ODM) -> generate maps, point clouds, 3D models and DEMs from drone, balloon or kite images.
* [Dataset of thermal and visible aerial images for multi-modal and multi-spectral image registration and fusion](https://www.sciencedirect.com/science/article/pii/S2352340920302201) -> The dataset consists of 30 visible images and their metadata, 80 thermal images and their metadata, and a visible georeferenced orthoimage.
* [BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos](https://ieeexplore.ieee.org/document/9093284) -> TIR videos of humans and animals with several challenging scenarios like scale variations, background clutter due to thermal reflections, large camera rotations, and motion blur
* [ERA: A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos](https://lcmou.github.io/ERA_Dataset/)
* [DroneVehicle](https://github.com/VisDrone/DroneVehicle) -> Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning
* [UAVOD10](https://github.com/weihancug/10-category-UAV-small-weak-object-detection-dataset-UAVOD10) -> 10 class of objects at 15 cm resolution. Classes are; building, ship, vehicle, prefabricated house, well, cable tower, pool, landslide, cultivation mesh cage, and quarry.
* [Busy-parking-lot-dataset---vehicle-detection-in-UAV-video](https://github.com/zhu-xlab/Busy-parking-lot-dataset---vehicle-detection-in-UAV-video)
* [OpenAerialMap](https://map.openaerialmap.org/) -> a set of tools for searching, sharing, and using openly licensed satellite and unmanned aerial vehicle (UAV) imagery
* [dd-ml-segmentation-benchmark](https://github.com/dronedeploy/dd-ml-segmentation-benchmark) -> DroneDeploy Machine Learning Segmentation Benchmark
* [SeaDronesSee](https://github.com/Ben93kie/SeaDronesSee) -> Vision Benchmark for Maritime Search and Rescue

## Other datasets
* [land-use-land-cover-datasets](https://github.com/r-wenger/land-use-land-cover-datasets)
* [EORSSD-dataset](https://github.com/rmcong/EORSSD-dataset) -> Extended Optical Remote Sensing Saliency Detection (EORSSD) Dataset
* [RSD46-WHU](https://github.com/RSIA-LIESMARS-WHU/RSD46-WHU) -> 46 scene classes for image classification, free for education, research and commercial use
* [RSOD-Dataset](https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-) -> dataset for object detection in PASCAL VOC format. Aircraft, playgrounds, overpasses & oiltanks
* [VHR-10_dataset_coco](https://github.com/chaozhong2010/VHR-10_dataset_coco) -> Object detection and instance segmentation dataset based on NWPU VHR-10 dataset. RGB & SAR
* [HRSID](https://github.com/chaozhong2010/HRSID) -> high resolution sar images dataset for ship detection, semantic segmentation, and instance segmentation tasks
* [MAR20](https://gcheng-nwpu.github.io/) -> Military Aircraft Recognition dataset
* [RSSCN7](https://github.com/palewithout/RSSCN7) -> Dataset of the article “Deep Learning Based Feature Selection for Remote Sensing Scene Classification”
* [Sewage-Treatment-Plant-Dataset](https://github.com/peijinwang/Sewage-Treatment-Plant-Dataset) -> object detection
* [TGRS-HRRSD-Dataset](https://github.com/CrazyStoneonRoad/TGRS-HRRSD-Dataset) -> High Resolution Remote Sensing Detection (HRRSD)
* [MUSIC4HA](https://github.com/gistairc/MUSIC4HA) -> MUltiband Satellite Imagery for object Classification (MUSIC) to detect Hot Area
* [MUSIC4GC](https://github.com/gistairc/MUSIC4GC) -> MUltiband Satellite Imagery for object Classification (MUSIC) to detect Golf Course
* [MUSIC4P3](https://github.com/gistairc/MUSIC4P3) -> MUltiband Satellite Imagery for object Classification (MUSIC) to detect Photovoltaic Power Plants (solar panels)
* [ABCDdataset](https://github.com/gistairc/ABCDdataset) -> damage detection dataset to identify whether buildings have been washed-away by tsunami
* [OGST](https://data.mendeley.com/datasets/bkxj8z84m9/3) -> Oil and Gas Tank Dataset
* [LS-SSDD-v1.0-OPEN](https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN) -> Large-Scale SAR Ship Detection Dataset
* [S2Looking](https://github.com/S2Looking/Dataset) -> A Satellite Side-Looking Dataset for Building Change Detection, [paper](https://arxiv.org/abs/2107.09244)
* [Zurich Summer Dataset](https://sites.google.com/site/michelevolpiresearch/data/zurich-dataset) -> Semantic segmentation of urban scenes
* [AISD](https://github.com/RSrscoder/AISD) -> Aerial Imagery dataset for Shadow Detection
* [Awesome-Remote-Sensing-Relative-Radiometric-Normalization-Datasets](https://github.com/ArminMoghimi/Awesome-Remote-Sensing-Relative-Radiometric-Normalization-Datasets)
* [SearchAndRescueNet](https://github.com/michaelthoreau/SearchAndRescueNet) -> Satellite Imagery for Search And Rescue Dataset, with example Faster R-CNN model
* [geonrw](https://ieee-dataport.org/open-access/geonrw) -> orthorectified aerial photographs, LiDAR derived digital elevation models and segmentation maps with 10 classes. With [repo](https://github.com/gbaier/geonrw)
* [Thermal power plans dataset](https://github.com/wenxinYin/AIR-TPPDD)
* [University1652-Baseline](https://github.com/layumi/University1652-Baseline) -> A Multi-view Multi-source Benchmark for Drone-based Geo-localization
* [benchmark_ISPRS2021](https://github.com/whuwuteng/benchmark_ISPRS2021) -> A new stereo dense matching benchmark dataset for deep learning
* [WHU-SEN-City](https://github.com/whu-csl/WHU-SEN-City) -> A paired SAR-to-optical image translation dataset which covers 34 big cities of China
* [SAR_vehicle_detection_dataset](https://github.com/whu-csl/SAR_vehicle_detection_dataset) -> 104 SAR images for vehicle detection, collected from Sandia MiniSAR/FARAD SAR images and MSTAR images
* [ERA-DATASET](https://github.com/zhu-xlab/ERA-DATASET) -> A Dataset and Deep Learning Benchmark for Event Recognition in Aerial Videos
* [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) -> a large-scale dataset for self-supervised learning in Earth observation
* [UBC-dataset](https://github.com/AICyberTeam/UBC-dataset) -> a dataset for building detection and classification from very high-resolution satellite imagery with the focus on object-level interpretation of individual buildings
* [AIR-CD](https://github.com/AICyberTeam/AIR-CD) -> a challenging cloud detection data set called AIR-CD, with higher spatial resolution and more representative landcover types
* [AIR-PolSAR-Seg](https://github.com/AICyberTeam/AIR-PolSAR-Seg) -> a challenging PolSAR terrain segmentation dataset
* [HRC_WHU](https://github.com/dr-lizhiwei/HRC_WHU) -> High-Resolution Cloud Detection Dataset comprising 150 RGB images and a resolution varying from 0.5 to 15 m in different global regions
* [AeroRIT](https://github.com/aneesh3108/AeroRIT) -> A New Scene for Hyperspectral Image Analysis
* [Building_Dataset](https://github.com/QiaoWenfan/Building_Dataset) -> High-speed Rail Line Building Dataset Display
* [Haiming-Z/MtS-WH-reference-map](https://github.com/Haiming-Z/MtS-WH-reference-map) -> a reference map for change detection based on MtS-WH
* [MtS-WH-Dataset](https://github.com/rulixiang/MtS-WH-Dataset) -> Multi-temporal Scene WuHan (MtS-WH) Dataset
* [Multi-modality-image-matching](https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods) -> image matching dataset including several remote sensing modalities
* [RID](https://github.com/TUMFTM/RID) -> Roof Information Dataset for CV-Based Photovoltaic Potential Assessment. With [paper](https://www.mdpi.com/2072-4292/14/10/2299)
* [APKLOT](https://github.com/langheran/APKLOT) -> A dataset for aerial parking block segmentation
* [QXS-SAROPT](https://github.com/yaoxu008/QXS-SAROPT) -> Optical and SAR pairing dataset from the [paper](https://arxiv.org/abs/2103.08259): The QXS-SAROPT Dataset for Deep Learning in SAR-Optical Data Fusion
* [SAR-ACD](https://github.com/AICyberTeam/SAR-ACD) -> SAR-ACD consists of 4322 aircraft clips with 6 civil aircraft categories and 14 other aircraft categories
* [SODA](https://shaunyuan22.github.io/SODA/) -> A large-scale Small Object Detection dataset. SODA-A comprises 2510 high-resolution images of aerial scenes, which has 800203 instances annotated with oriented rectangle box annotations over 9 classes.
* [Data-CSHSI](https://github.com/YuxiangZhang-BIT/Data-CSHSI) -> Open source datasets for Cross-Scene Hyperspectral Image Classification, includes Houston, Pavia & HyRank datasets
* [SynthWakeSAR](https://data.bris.ac.uk/data/dataset/30kvuvmatwzij2mz1573zqumfx) -> A Synthetic SAR Dataset for Deep Learning Classification of Ships at Sea, with [paper](https://www.mdpi.com/2072-4292/14/16/3999)
* [SAR2Opt-Heterogeneous-Dataset](https://github.com/MarsZhaoYT/SAR2Opt-Heterogeneous-Dataset) -> SAR-optical images to be used as a benchmark in change detection and image transaltion on remote sensing images
* [urban-tree-detection-data](https://github.com/jonathanventura/urban-tree-detection-data) -> Dataset for training and evaluating tree detectors in urban environments with aerial imagery
* [Landsat 8 Cloud Cover Assessment Validation Data](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data)
* [Attribute-Cooperated-Classification-Datasets](https://github.com/CrazyStoneonRoad/Attribute-Cooperated-Classification-Datasets) -> Three datasets based on AID, UCM, and Sydney. For each image, there is a label of scene classification and a label vector of attribute items.
* [RarePlanes](https://www.cosmiqworks.org/rareplanes-public-user-guide/) is a dataset of real (Maxar) and simulated images of planes. Utility functions at [VisionSystemsInc - RarePlanes](https://github.com/VisionSystemsInc/RarePlanes)
* [dynnet](https://github.com/aysim/dynnet) -> DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation
* [open_earth_map](https://github.com/bao18/open_earth_map) -> a benchmark dataset for global high-resolution land cover mapping

## Kaggle
Kaggle hosts over > 200 satellite image datasets, [search results here](https://www.kaggle.com/search?q=satellite+image+in%3Adatasets).
The [kaggle blog](http://blog.kaggle.com) is an interesting read.

### Kaggle - Amazon from space - classification challenge
* https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data
* 3-5 meter resolution GeoTIFF images from planet Dove satellite constellation
* 12 classes including - **cloudy, primary + waterway** etc
* [1st place winner interview - used 11 custom CNN](http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/)
* [FastAI Multi-label image classification](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95)
* [Multi-Label Classification of Satellite Photos of the Amazon Rainforest](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/)
* [Understanding the Amazon Rainforest with Multi-Label Classification + VGG-19, Inceptionv3, AlexNet & Transfer Learning](https://towardsdatascience.com/understanding-the-amazon-rainforest-with-multi-label-classification-vgg-19-inceptionv3-5084544fb655)
* [amazon-classifier](https://github.com/mikeskaug/amazon-classifier) -> compares random forest with CNN
* [multilabel-classification](https://github.com/muneeb706/multilabel-classification) -> compares various CNN architecutres
* [Planet-Amazon-Kaggle](https://github.com/Skumarr53/Planet-Amazon-Kaggle) -> uses fast.ai
* [deforestation_deep_learning](https://github.com/schumanzhang/deforestation_deep_learning)
* [Track-Human-Footprint-in-Amazon-using-Deep-Learning](https://github.com/sahanasub/Track-Human-Footprint-in-Amazon-using-Deep-Learning)
* [Amazon-Rainforest-CNN](https://github.com/cldowdy/Amazon-Rainforest-CNN) -> uses a 3-layer CNN in Tensorflow
* [rainforest-tagging](https://github.com/minggli/rainforest-tagging) -> Convolutional Neural Net and Recurrent Neural Net in Tensorflow for satellite images multi-label classification

### Kaggle - DSTL segmentation challenge
* https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
* Rating - medium, many good examples (see the Discussion as well as kernels), but as this competition was run a couple of years ago many examples use python 2
* WorldView 3 - 45 satellite images covering 1km x 1km in both 3 (i.e. RGB) and 16-band (400nm - SWIR) images
* 10 Labelled classes include - **Buildings, Road, Trees, Crops, Waterway, Vehicles**
* [Interview with 1st place winner who used segmentation networks](http://blog.kaggle.com/2017/04/26/dstl-satellite-imagery-competition-1st-place-winners-interview-kyle-lee/) - 40+ models, each tweaked for particular target (e.g. roads, trees)
* [ZF_UNET_224_Pretrained_Model 2nd place solution](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model) ->
* [3rd place soluton](https://github.com/osin-vladimir/kaggle-satellite-imagery-feature-detection) -> which explored pansharpening & calculating reflectance indices, with [arxiv paper](https://arxiv.org/abs/1706.06169) 
* [Deepsense 4th place solution](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
* [Entry by lopuhin](https://github.com/lopuhin/kaggle-dstl) using UNet with batch-normalization
* [Multi-class semantic segmentation of satellite images using U-Net](https://github.com/rogerxujiang/dstl_unet) using DSTL dataset, tensorflow 1 & python 2.7. Accompanying [article](https://towardsdatascience.com/dstl-satellite-imagery-contest-on-kaggle-2f3ef7b8ac40)
* [Deep-Satellite-Image-Segmentation](https://github.com/antoine-spahr/Deep-Satellite-Image-Segmentation)
* [Dstl-Satellite-Imagery-Feature-Detection-Improved](https://github.com/dsp6414/Dstl-Satellite-Imagery-Feature-Detection-Improved)
* [Satellite-imagery-feature-detection](https://github.com/ArangurenAndres/Satellite-imagery-feature-detection)
* [Satellite_Image_Classification](https://github.com/aditya-sawadh/Satellite_Image_Classification) -> using XGBoost and ensemble classification methods
* [Unet-for-Satellite](https://github.com/justinishikawa/Unet-for-Satellite)
* [building-segmentation](https://github.com/jimpala/building-segmentation) -> TensorFlow U-Net implementation trained to segment buildings in satellite imagery

### Kaggle - DeepSat land cover classification
* https://www.kaggle.com/datasets/crawford/deepsat-sat4 & https://www.kaggle.com/datasets/crawford/deepsat-sat6
* [DeepSat-Kaggle](https://github.com/athulsudheesh/DeepSat-Kaggle) -> uses Julia
* [deepsat-aws-emr-pyspark](https://github.com/hellosaumil/deepsat-aws-emr-pyspark) -> Using PySpark for Image Classification on Satellite Imagery of Agricultural Terrains

### Kaggle - Airbus ship detection challenge
* https://www.kaggle.com/c/airbus-ship-detection/overview
* Rating - medium, most solutions using deep-learning, many kernels, [good example kernel](https://www.kaggle.com/kmader/baseline-u-net-model-part-1)
* I believe there was a problem with this dataset, which led to many complaints that the competition was ruined
* [Deep Learning for Ship Detection and Segmentation](https://towardsdatascience.com/deep-learning-for-ship-detection-and-segmentation-71d223aca649) -> treated as instance segmentation problem, with [notebook](https://github.com/abhinavsagar/kaggle-notebooks/blob/master/ship_segmentation.ipynb)
* [Lessons Learned from Kaggle’s Airbus Challenge](https://towardsdatascience.com/lessons-learned-from-kaggles-airbus-challenge-252e25c5efac)
* [Airbus-Ship-Detection](https://github.com/kheyer/Airbus-Ship-Detection) -> This solution scored 139 out of 884 for the competition, combines ResNeXt50 based classifier and a U-net segmentation model
* [Ship-Detection-Project](https://github.com/ZTong1201/Ship-Detection-Project) -> uses Mask R-CNN and UNet model
* [Airbus_SDC](https://github.com/WillieMaddox/Airbus_SDC)
* [Airbus_SDC_dup](https://github.com/WillieMaddox/Airbus_SDC_dup) -> Project focused on detecting duplicate regions of overlapping satellite imagery. Applied to Airbus ship detection dataset
* [airbus-ship-detection](https://github.com/jancervenka/airbus-ship-detection) -> CNN with REST API
* [Ship-Detection-from-Satellite-Images-using-YOLOV4](https://github.com/debasis-dotcom/Ship-Detection-from-Satellite-Images-using-YOLOV4) -> uses Kaggle Airbus Ship Detection dataset
* [Image Segmentation: Kaggle experience](https://towardsdatascience.com/image-segmentation-kaggle-experience-9a41cb8924f0) -> Medium article by gold medal winner Vlad Shmyhlo

### Kaggle - Shipsnet classification dataset
* https://www.kaggle.com/rhammell/ships-in-satellite-imagery -> Classify ships in San Franciso Bay using Planet satellite imagery
* 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification, 3 meter pixel size
* [shipsnet-detector](https://github.com/rhammell/shipsnet-detector) -> Detect container ships in Planet imagery using machine learning

### Kaggle - Ships in Google Earth
* https://www.kaggle.com/tomluther/ships-in-google-earth
* 794 jpegs showing various sized ships in satellite imagery, annotations in Pascal VOC format for object detection models
* [kaggle-ships-in-Google-Earth-yolov5](https://github.com/robmarkcole/kaggle-ships-in-Google-Earth-yolov5)

### Kaggle - Ships in San Franciso Bay
* https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery
* 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification, provided by Planet
* [DeepLearningShipDetection](https://github.com/PenguinDan/DeepLearningShipDetection)
* [Ship-Detection-Using-Satellite-Imagery](https://github.com/Dhruvisha29/Ship-Detection-Using-Satellite-Imagery)

### Kaggle - Swimming pool and car detection using satellite imagery
* https://www.kaggle.com/kbhartiya83/swimming-pool-and-car-detection
* 3750 satellite images of residential areas with annotation data for swimming pools and cars
* [Object detection on Satellite Imagery using RetinaNet](https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-1-training-e589975afbd5)

### Kaggle - Planesnet classification dataset
* https://www.kaggle.com/rhammell/planesnet -> Detect aircraft in Planet satellite image chips
* 20x20 RGB images, the "plane" class includes 8000 images and the "no-plane" class includes 24000 images
* [Dataset repo](https://github.com/rhammell/planesnet) and [planesnet-detector](https://github.com/rhammell/planesnet-detector) demonstrates a small CNN classifier on this dataset
* [ergo-planes-detector](https://github.com/evilsocket/ergo-planes-detector) -> An ergo based project that relies on a convolutional neural network to detect airplanes from satellite imagery, uses the PlanesNet dataset
* [Using AWS SageMaker/PlanesNet to process Satellite Imagery](https://github.com/kskalvar/aws-sagemaker-planesnet-imagery)
* [Airplane-in-Planet-Image](https://github.com/MaxLenormand/Airplane-in-Planet-Image) -> pytorch model

### Kaggle - CGI Planes in Satellite Imagery w/ BBoxes
* https://www.kaggle.com/datasets/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes
* 500 computer generated satellite images of planes
* [Faster RCNN to detect airplanes](https://github.com/ShubhankarRawat/Airplane-Detection-for-Satellites)
* [aircraft-detection-from-satellite-images-yolov3](https://github.com/emrekrtorun/aircraft-detection-from-satellite-images-yolov3)

### Kaggle - Draper challenge to place images in order of time
* https://www.kaggle.com/c/draper-satellite-image-chronology/data
* Rating - hard. Not many useful kernels.
* Images are grouped into sets of five, each of which have the same setId. Each image in a set was taken on a different day (but not necessarily at the same time each day). The images for each set cover approximately the same area but are not exactly aligned.
* Kaggle interviews for entrants who [used XGBOOST](http://blog.kaggle.com/2016/09/15/draper-satellite-image-chronology-machine-learning-solution-vicens-gaitan/) and a [hybrid human/ML approach](http://blog.kaggle.com/2016/09/08/draper-satellite-image-chronology-damien-soukhavong/)
* [deep-cnn-sat-image-time-series](https://github.com/MickyDowns/deep-cnn-sat-image-time-series) -> uses LSTM

### Kaggle - Dubai segmentation
* https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery
* 72 satellite images of Dubai, the UAE, and is segmented into 6 classes
* [dubai-satellite-imagery-segmentation](https://github.com/ayushdabra/dubai-satellite-imagery-segmentation) -> due to the small dataset, image augmentation was used
* [U-Net for Semantic Segmentation on Unbalanced Aerial Imagery](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56) -> using the Dubai dataset
* [Multiclass-semantic-segmentation-in-satallite-images](https://github.com/tahirjhan/Multiclass-semantic-segmentation-in-satallite-images) -> uses keras
* [Semantic-Segmentation-using-U-Net](https://github.com/Anay21110/Semantic-Segmentation-using-U-Net) -> uses keras
* [unet_satelite_image_segmentation](https://github.com/nassimaliou/unet_satelite_image_segmentation)

### Kaggle - Massachusetts Roads & Buildings Datasets - segmentation
* https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset
* https://www.kaggle.com/datasets/balraj98/massachusetts-buildings-dataset
* [Official published dataset](https://www.cs.toronto.edu/~vmnih/data/)
* [Road_seg_dataset](https://github.com/parth1620/Road_seg_dataset) -> subset of the roads dataset containing only 200 images and masks
* [Road and Building Semantic Segmentation in Satellite Imagery](https://github.com/Paulymorphous/Road-Segmentation) uses U-Net on the Massachusetts Roads Dataset & keras
* [Semantic-segmentation repo by fuweifu-vtoo](https://github.com/fuweifu-vtoo/Semantic-segmentation) -> uses pytorch and the [Massachusetts Buildings & Roads Datasets](https://www.cs.toronto.edu/~vmnih/data/)
* [ssai-cnn](https://github.com/mitmul/ssai-cnn) -> This is an implementation of Volodymyr Mnih's dissertation methods on his Massachusetts road & building dataset
* [building-footprint-segmentation](https://github.com/fuzailpalnak/building-footprint-segmentation) -> pip installable library to train building footprint segmentation on satellite and aerial imagery, applied to Massachusetts Buildings Dataset and Inria Aerial Image Labeling Dataset
* [Road detection using semantic segmentation and albumentations for data augmention](https://towardsdatascience.com/road-detection-using-segmentation-models-and-albumentations-libraries-on-keras-d5434eaf73a8) using the Massachusetts Roads Dataset, U-net & Keras
* [Image-Segmentation)](https://github.com/mschulz/Image-Segmentation) -> using Massachusetts Road dataset and fast.ai

### Kaggle - Deepsat classification challenge
Not satellite but airborne imagery. Each sample image is 28x28 pixels and consists of 4 bands - red, green, blue and near infrared. The training and test labels are one-hot encoded 1x6 vectors. Each image patch is size normalized to 28x28 pixels. Data in `.mat` Matlab format. JPEG?
* [Sat4](https://www.kaggle.com/crawford/deepsat-sat4) 500,000 image patches covering four broad land cover classes - **barren land, trees, grassland and a class that consists of all land cover classes other than the above three**
* [Sat6](https://www.kaggle.com/crawford/deepsat-sat6) 405,000 image patches each of size 28x28 and covering 6 landcover classes - **barren land, trees, grassland, roads, buildings and water bodies.**

### Kaggle - High resolution ship collections 2016 (HRSC2016)
* https://www.kaggle.com/guofeng/hrsc2016
* Ship images harvested from Google Earth
* [HRSC2016_SOTA](https://github.com/ming71/HRSC2016_SOTA) -> Fair comparison of different algorithms on the HRSC2016 dataset

### Kaggle - SWIM-Ship Wake Imagery Mass
* https://www.kaggle.com/datasets/lilitopia/swimship-wake-imagery-mass
* An optical ship wake detection benchmark dataset built for deep learning
* [WakeNet](https://github.com/Lilytopia/WakeNet) -> A CNN-based optical image ship wake detector, code for 2021 paper: Rethinking Automatic Ship Wake Detection: State-of-the-Art CNN-based Wake Detection via Optical Images

### Kaggle - Understanding Clouds from Satellite Images
In this challenge, you will build a model to classify cloud organization patterns from satellite images.
* https://www.kaggle.com/c/understanding_cloud_organization/
* [3rd place solution on Github by naivelamb](https://github.com/naivelamb/kaggle-cloud-organization)
* [15th place solution on Github by Soongja](https://github.com/Soongja/kaggle-clouds)
* [69th place solution on Github by yukkyo](https://github.com/yukkyo/Kaggle-Understanding-Clouds-69th-solution)
* [161st place solution on Github by michal-nahlik](https://github.com/michal-nahlik/kaggle-clouds-2019)
* [Solution by yurayli](https://github.com/yurayli/satellite-cloud-segmentation)
* [Solution by HazelMartindale](https://github.com/HazelMartindale/kaggle_understanding_clouds_learning_project) uses 3 versions of U-net architecture
* [Solution by khornlund](https://github.com/khornlund/understanding-cloud-organization)
* [Solution by Diyago](https://github.com/Diyago/Understanding-Clouds-from-Satellite-Images)
* [Solution by tanishqgautam](https://github.com/tanishqgautam/Multi-Label-Segmentation-With-FastAI)

### Kaggle - 38-Cloud Cloud Segmentation
* https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images
* Contains 38 Landsat 8 images and manually extracted pixel-level ground truths
* [38-Cloud Github repository](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) and follow up [95-Cloud](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset) dataset
* [How to create a custom Dataset / Loader in PyTorch, from Scratch, for multi-band Satellite Images Dataset from Kaggle](https://medium.com/analytics-vidhya/how-to-create-a-custom-dataset-loader-in-pytorch-from-scratch-for-multi-band-satellite-images-c5924e908edf)
* [Cloud-Net: A semantic segmentation CNN for cloud detection](https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection) -> an end-to-end cloud detection algorithm for Landsat 8 imagery, trained on 38-Cloud Training Set
* [Segmentation of Clouds in Satellite Images Using Deep Learning](https://medium.com/swlh/segmentation-of-clouds-in-satellite-images-using-deep-learning-a9f56e0aa83d) -> semantic segmentation using a Unet on the Kaggle 38-Cloud dataset

### Kaggle - Airbus Aircraft Detection Dataset
* https://www.kaggle.com/airbusgeo/airbus-aircrafts-sample-dataset
* One hundred civilian airports and over 3000 annotated commercial aircrafts
* [detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5](https://medium.com/artificialis/detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5-5f3d464b75ad)
* [pytorch-remote-sensing](https://github.com/miko7879/pytorch-remote-sensing) -> Aircraft detection using the 'Airbus Aircraft Detection' dataset and Faster-RCNN with ResNet-50 backbone in pytorch

### Kaggle - Airbus oil storage detection dataset
* https://www.kaggle.com/airbusgeo/airbus-oil-storage-detection-dataset
* [Oil-Storage Tank Instance Segmentation with Mask R-CNN](https://github.com/georgiosouzounis/instance-segmentation-mask-rcnn/blob/main/mask_rcnn_oiltanks_gpu.ipynb) with [accompanying article](https://medium.com/@georgios.ouzounis/oil-storage-tank-instance-segmentation-with-mask-r-cnn-77c94433045f)
* [Oil Storage Detection on Airbus Imagery with YOLOX](https://medium.com/artificialis/oil-storage-detection-on-airbus-imagery-with-yolox-9e38eb6f7e62) -> uses the Kaggle Airbus Oil Storage Detection dataset

### Kaggle - Satellite images of hurricane damage
* https://www.kaggle.com/kmader/satellite-images-of-hurricane-damage
* https://github.com/dbuscombe-usgs/HurricaneHarvey_buildingdamage

### Kaggle - Austin Zoning Satellite Images
* https://www.kaggle.com/franchenstein/austin-zoning-satellite-images
* classify a images of Austin into one of its zones, such as residential, industrial, etc. 3667 satellite images

### Kaggle - Statoil/C-CORE Iceberg Classifier Challenge
* https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
* [Deep Learning for Iceberg detection in Satellite Images](https://towardsdatascience.com/deep-learning-for-iceberg-detection-in-satellite-images-c667acf4bad0)
* [radar-image-recognition](https://github.com/siarez/radar-image-recognition)
* [Iceberg-Classification-Using-Deep-Learning](https://github.com/mankadronit/Iceberg-Classification-Using-Deep-Learning)
* [Deep-Learning-Project](https://github.com/singh-shakti94/Deep-Learning-Project)
* [iceberg-classifier-challenge solution by ShehabSunny](https://github.com/ShehabSunny/iceberg-classifier-challenge)

### Kaggle - Land Cover Classification Dataset from DeepGlobe Challenge - segmentation
* https://www.kaggle.com/balraj98/deepglobe-land-cover-classification-dataset
* [Satellite Imagery Semantic Segmentation with CNN](https://joshting.medium.com/satellite-imagery-segmentation-with-convolutional-neural-networks-f9254de3b907) -> 7 different segmentation classes, DeepGlobe Land Cover Classification Challenge dataset, with [repo](https://github.com/justjoshtings/satellite_image_segmentation)
* [Land Cover Classification with U-Net](https://baratam-tarunkumar.medium.com/land-cover-classification-with-u-net-aa618ea64a1b) -> Satellite Image Multi-Class Semantic Segmentation Task with PyTorch Implementation of U-Net, uses DeepGlobe Land Cover Segmentation dataset, with [code](https://github.com/TarunKumar1995-glitch/land_cover_classification_unet)
* [DeepGlobe Land Cover Classification Challenge solution](https://github.com/GeneralLi95/deepglobe_land_cover_classification_with_deeplabv3plus)

### Kaggle - Next Day Wildfire Spread
A Data Set to Predict Wildfire Spreading from Remote-Sensing Data
* https://www.kaggle.com/fantineh/next-day-wildfire-spread
* https://arxiv.org/abs/2112.02447

### Kaggle - Satellite Next Day Wildfire Spread
Inspired by the above dataset, using different data sources
* https://www.kaggle.com/satellitevu/satellite-next-day-wildfire-spread
* https://github.com/SatelliteVu/SatelliteVu-AWS-Disaster-Response-Hackathon

## Kaggle - Spacenet 7 Multi-Temporal Urban Change Detection
* https://www.kaggle.com/datasets/amerii/spacenet-7-multitemporal-urban-development
* [SatFootprint](https://github.com/PriyanK7n/SatFootprint) -> building segmentation on the Spacenet 7 dataset

## Kaggle - Satellite Images to predict poverty in Africa
* https://www.kaggle.com/datasets/sandeshbhat/satellite-images-to-predict-povertyafrica
* Uses satellite imagery and nightlights data to predict poverty levels at a local level
* [Predicting-Poverty](https://github.com/jmather625/predicting-poverty-replication) -> Combining satellite imagery and machine learning to predict poverty, in PyTorch

## Kaggle - NOAA Fisheries Steller Sea Lion Population Count
* https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count -> count sea lions from aerial images
* [Sealion-counting](https://github.com/babyformula/Sealion-counting)
* [Sealion_Detection_Classification](https://github.com/yyc9268/Sealion_Detection_Classification)

## Kaggle - Arctic Sea Ice Image Masking
* https://www.kaggle.com/datasets/alexandersylvester/arctic-sea-ice-image-masking
* [sea_ice_remote_sensing](https://github.com/sum1lim/sea_ice_remote_sensing)

## Kaggle - Overhead-MNIST
* A Benchmark Satellite Dataset as Drop-In Replacement for MNIST
* https://www.kaggle.com/datamunge/overheadmnist -> kaggle
* https://arxiv.org/abs/2102.04266 -> paper
* https://github.com/reveondivad/ov-mnist -> github

## Kaggle - Satellite Image Classification
* https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification
* [satellite-image-classification-pytorch](https://github.com/dilaraozdemir/satellite-image-classification-pytorch)

## Kaggle - EuroSAT - Sentinel-2 Dataset
* https://www.kaggle.com/datasets/raoofnaushad/eurosat-sentinel2-dataset
* RGB Land Cover and Land Use Classification using Sentinel-2 Satellite
* Used in paper [Image Augmentation for Satellite Images](https://arxiv.org/abs/2207.14580)

### Kaggle - miscellaneous
* https://www.kaggle.com/reubencpereira/spatial-data-repo -> Satellite + loan data
* https://www.kaggle.com/towardsentropy/oil-storage-tanks -> Image data of industrial tanks with bounding box annotations, estimate tank fill % from shadows
* https://www.kaggle.com/airbusgeo/airbus-wind-turbines-patches -> Airbus SPOT satellites images over wind turbines for classification
* https://www.kaggle.com/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes -> CGI planes object detection dataset
* https://www.kaggle.com/atilol/aerialimageryforroofsegmentation -> Aerial Imagery for Roof Segmentation
* https://www.kaggle.com/andrewmvd/ship-detection -> 621 images of boats and ships
* https://www.kaggle.com/alpereniek/vehicle-detection-from-satellite-images-data-set
* https://www.kaggle.com/sergiishchus/maxar-satellite-data -> Example Maxar data at 15 cm resolution
* https://www.kaggle.com/cici118/swimming-pool-detection-algarves-landscape
* https://www.kaggle.com/datasets/donkroco/solar-panel-module -> object detection for solar panels
* https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset -> segment roads
