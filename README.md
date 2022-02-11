# Introduction
This document lists resources for performing deep learning (DL) on satellite imagery. To a lesser extent classical Machine learning (ML, e.g. random forests) are also discussed, as are classical image processing techniques. Note there is a huge volume of academic literature published on these topics, and this repo does not seek to index them all but rather list approachable resources with published code that will benefit both the research *and* developer communities.

# Table of contents
* [Techniques](https://github.com/robmarkcole/satellite-image-deep-learning#techniques)
* [ML best practice](https://github.com/robmarkcole/satellite-image-deep-learning#ml-best-practice)
* [ML metrics](https://github.com/robmarkcole/satellite-image-deep-learning#ml-metrics)
* [Datasets](https://github.com/robmarkcole/satellite-image-deep-learning#datasets)
* [Online platforms for analytics](https://github.com/robmarkcole/satellite-image-deep-learning#online-platforms-for-analytics)
* [Free online compute](https://github.com/robmarkcole/satellite-image-deep-learning#free-online-compute)
* [State of the art engineering](https://github.com/robmarkcole/satellite-image-deep-learning#state-of-the-art-engineering)
* [Cloud providers](https://github.com/robmarkcole/satellite-image-deep-learning#cloud-providers)
* [Deploying models](https://github.com/robmarkcole/satellite-image-deep-learning#deploying-models)
* [Image formats, data management and catalogues](https://github.com/robmarkcole/satellite-image-deep-learning#image-formats-data-management-and-catalogues)
* [Image annotation](https://github.com/robmarkcole/satellite-image-deep-learning#image-annotation)
* [Paid software](https://github.com/robmarkcole/satellite-image-deep-learning#paid-software)
* [Open source software](https://github.com/robmarkcole/satellite-image-deep-learning#open-source-software)
* [Movers and shakers on Github](https://github.com/robmarkcole/satellite-image-deep-learning#movers-and-shakers-on-github)
* [Companies & organisations on Github](https://github.com/robmarkcole/satellite-image-deep-learning#companies--organisations-on-github)
* [Courses](https://github.com/robmarkcole/satellite-image-deep-learning#courses)
* [Online communities](https://github.com/robmarkcole/satellite-image-deep-learning#online-communities)
* [Jobs](https://github.com/robmarkcole/satellite-image-deep-learning#jobs)
* [About the author](https://github.com/robmarkcole/satellite-image-deep-learning#about-the-author)

# Techniques
This section explores the different deep and machine learning (ML) techniques applied to common problems in satellite imagery analysis. Good background reading is [Deep learning in remote sensing applications: A meta-analysis and review](https://www.iges.or.jp/en/publication_documents/pub/peer/en/6898/Ma+et+al+2019.pdf)

## Classification
The classic cats vs dogs image classification task, which in the remote sensing domain is used to assign a label to an image, e.g. this is an image of a forest. The more complex case is applying multiple labels to an image. This approach of image level classification is not to be confused with pixel-level classification which is called semantic segmentation. In general, aerial images cover large geographical areas that include multiple classes of land, so treating this is as a classification problem is less common than using semantic segmentation.
* Land classification on Sentinel 2 data using a [simple sklearn cluster algorithm](https://github.com/acgeospatial/Satellite_Imagery_Python/blob/master/Clustering_KMeans-Sentinel2.ipynb) or [deep learning CNN](https://towardsdatascience.com/land-use-land-cover-classification-with-deep-learning-9a5041095ddb)
* Land Use Classification on Merced dataset using CNN [in Keras](https://github.com/tavgreen/landuse_classification)
or [fastai](https://medium.com/spatial-data-science/deep-learning-for-geospatial-data-applications-multi-label-classification-2b0a1838fcf3). Also checkout [Multi-label Land Cover Classification](https://towardsdatascience.com/multi-label-land-cover-classification-with-deep-learning-d39ce2944a3d) using the redesigned multi-label Merced dataset with 17 land cover classes
* [Multi-Label Classification of Satellite Photos of the Amazon Rainforest using keras](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/) or [FastAI](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95)
* [Detecting Informal Settlements from Satellite Imagery using fine-tuning of ResNet-50 classifier](https://blog.goodaudience.com/detecting-informal-settlements-using-satellite-imagery-and-convolutional-neural-networks-d571a819bf44) with [repo](https://github.com/dymaxionlabs/ap-latam)
* [Vision Transformers Use Case: Satellite Image Classification without CNNs](https://joaootavionf007.medium.com/vision-transformers-use-case-satellite-image-classification-without-cnns-2c4dbeb06f87)
* [Land-Cover-Classification-using-Sentinel-2-Dataset](https://github.com/raoofnaushad/Land-Cover-Classification-using-Sentinel-2-Dataset) -> [well written Medium article](https://raoofnaushad7.medium.com/applying-deep-learning-on-satellite-imagery-classification-5f2588b932c1) accompanying this repo but using the EuroSAT dataset
* [Land Cover Classification of Satellite Imagery using Convolutional Neural Networks](https://towardsdatascience.com/land-cover-classification-of-satellite-imagery-using-convolutional-neural-networks-91b5bb7fe808) using Keras and a multi spectral dataset captured over vineyard fields of Salinas Valley, California
* [Detecting deforestation from satellite images](https://towardsdatascience.com/detecting-deforestation-from-satellite-images-7aa6dfbd9f61) -> using FastAI and ResNet50, with repo [fsdl_deforestation_detection](https://github.com/karthikraja95/fsdl_deforestation_detection)
* [Neural Network for Satellite Data Classification Using Tensorflow in Python](https://towardsdatascience.com/neural-network-for-satellite-data-classification-using-tensorflow-in-python-a13bcf38f3e1) -> A step-by-step guide for Landsat 5 multispectral data classification for binary built-up/non-built-up class prediction, with [repo](https://github.com/PratyushTripathy/Landsat-Classification-Using-Neural-Network)
* [Slums mapping from pretrained CNN network](https://github.com/deepankverma/slums_detection) on VHR (Pleiades: 0.5m) and MR (Sentinel: 10m) imagery
* [Comparing urban environments using satellite imagery and convolutional neural networks](https://github.com/adrianalbert/urban-environments) -> includes interesting study of the image embedding features extracted for each image on the Urban Atlas dataset. Accompanying [paper](https://www.researchgate.net/publication/315882788_Using_convolutional_networks_and_satellite_imagery_to_identify_patterns_in_urban_environments_at_a_large_scale)
* [RSI-CB](https://github.com/lehaifeng/RSI-CB) -> A Large Scale Remote Sensing Image Classification Benchmark via Crowdsource Data
* [NAIP_PoolDetection](https://github.com/annaptasznik/NAIP_PoolDetection) -> modelled as an object recognition problem, a CNN is used to identify images as being swimming pools or something else - specifically a street, rooftop, or lawn
* [Land Use and Land Cover Classification using a ResNet Deep Learning Architecture](https://www.luigiselmi.eu/eo/lulc-classification-deeplearning.html) -> uses fastai and the EuroSAT dataset
* [Vision Transformers Use Case: Satellite Image Classification without CNNs](https://medium.com/nerd-for-tech/vision-transformers-use-case-satellite-image-classification-without-cnns-2c4dbeb06f87)
* [WaterNet](https://github.com/treigerm/WaterNet) -> a CNN that identifies water in satellite images
* [Road-Network-Classification](https://github.com/ualsg/Road-Network-Classification) -> Road network classification model using ResNet-34, road classes organic, gridiron, radial and no pattern
* [Scaling AI to map every school on the planet](https://developmentseed.org/blog/2021-03-18-ai-enabling-school-mapping)
* [Landsat classification CNN tutorial](https://towardsdatascience.com/is-cnn-equally-shiny-on-mid-resolution-satellite-data-9e24e68f0c08) with [repo](https://github.com/PratyushTripathy/Landsat-Classification-Using-Convolution-Neural-Network)
* [satellite-crosswalk-classification](https://github.com/rodrigoberriel/satellite-crosswalk-classification)
* [Understanding the Amazon Rainforest with Multi-Label Classification + VGG-19, Inceptionv3, AlexNet & Transfer Learning](https://towardsdatascience.com/understanding-the-amazon-rainforest-with-multi-label-classification-vgg-19-inceptionv3-5084544fb655)
* [Implementation of the 3D-CNN model for land cover classification](https://medium.com/geekculture/remote-sensing-deep-learning-for-land-cover-classification-of-satellite-imagery-using-python-6a7b4c4f570f) -> uses the Sundarbans dataset, with [repo](https://github.com/syamkakarla98/Satellite_Imagery_Analysis). Also read [Land cover classification of Sundarbans satellite imagery using K-Nearest Neighbor(K-NNC), Support Vector Machine (SVM), and Gradient Boosting classification algorithms](https://towardsdatascience.com/land-cover-classification-in-satellite-imagery-using-python-ae39dbf2929) which is by the same author and shares the repo
* [SSTN](https://github.com/zilongzhong/SSTN) -> PyTorch Implementation of SSTNs for hyperspectral image classifications from the IEEE T-GRS paper "Spectral-Spatial Transformer Network for Hyperspectral Image Classification: A FAS Framework." Demonstrates a novel spectral-spatial transformer network (SSTN), which consists of spatial attention and spectral association modules, to overcome the constraints of convolution kernels
* [SatellitePollutionCNN](https://github.com/arnavbansal1/SatellitePollutionCNN) -> A novel algorithm to predict air pollution levels with state-of-art accuracy using deep learning and GoogleMaps satellite images
* [PropertyClassification](https://github.com/Sardhendu/PropertyClassification) -> Classifying the type of property given Real Estate, satellite and Street view Images

## Segmentation
Segmentation will assign a class label to each **pixel** in an image. Segmentation is typically grouped into semantic or instance segmentation. In semantic segmentation objects of the same class are assigned the same label, whilst in instance segmentation each object is assigned a unique label. Read this [beginner’s guide to segmentation](https://medium.com/gsi-technology/a-beginners-guide-to-segmentation-in-satellite-images-9c00d2028d52). Single class models are often trained for road or building segmentation, with multi class for land use/crop type classification. Image annotation can take long than for classification/object detection since every pixel must be annotated. **Note** that many articles which refer to 'hyperspectral land classification' are actually describing semantic segmentation.

### Semantic segmentation
Almost always performed using U-Net. For multi/hyper-spectral imagery more classical techniques may be used (e.g. k-means).
* [awesome-satellite-images-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation#satellite-images-segmentation)
* [Satellite Image Segmentation: a Workflow with U-Net](https://medium.com/vooban-ai/satellite-image-segmentation-a-workflow-with-u-net-7ff992b2a56e) is a decent intro article
* [nga-deep-learning](https://github.com/jordancaraballo/nga-deep-learning) -> performs semantic segmentation on high resultion GeoTIF data using a modified U-Net & Keras, published by NASA researchers
* [How to create a DataBlock for Multispectral Satellite Image Semantic Segmentation using Fastai](https://towardsdatascience.com/how-to-create-a-datablock-for-multispectral-satellite-image-segmentation-with-the-fastai-v2-bc5e82f4eb5)
* [Using a U-Net for image segmentation, blending predicted patches smoothly is a must to please the human eye](https://github.com/Vooban/Smoothly-Blend-Image-Patches) -> python code to blend predicted patches smoothly
* [Automatic Detection of Landfill Using Deep Learning](https://github.com/AnupamaRajkumar/LandfillDetection_SemanticSegmentation)
* [SpectralNET](https://github.com/tanmay-ty/SpectralNET) -> a 2D wavelet CNN for Hyperspectral Image Classification, uses Salinas Scene dataset & Keras
* [FactSeg](https://github.com/Junjue-Wang/FactSeg) -> Foreground Activation Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery (TGRS), also see [FarSeg](https://github.com/Z-Zheng/FarSeg) and [FreeNet](https://github.com/Z-Zheng/FreeNet), implementations of research paper
* [SCAttNet](https://github.com/lehaifeng/SCAttNet) -> Semantic Segmentation Network with Spatial and Channel Attention Mechanism
* [laika](https://github.com/datasciencecampus/laika) -> The goal of this repo is to research potential sources of satellite image data and to implement various algorithms for satellite image segmentation
* [PEARL](https://www.landcover.io/) -> a human-in-the-loop AI tool to drastically reduce the time required to produce an accurate Land Use/Land Cover (LULC) map, [blog post](http://devseed.com/blog/2021-05-17-pearl-ai-land-cover), uses Microsoft Planetary Computer and ML models run locally in the browser
* [unetseg](https://github.com/dymaxionlabs/unetseg) -> A set of classes and CLI tools for training a semantic segmentation model based on the U-Net architecture, using Tensorflow and Keras. This implementation is tuned specifically for satellite imagery and other geospatial raster data.
* [CropMask_RCNN](https://github.com/ecohydro/CropMask_RCNN) -> Segmenting center pivot agriculture to monitor crop water use in drylands with Mask R-CNN and Landsat satellite imagery
* [Semantic Segmentation of Satellite Imagery using U-Net & fast.ai](https://medium.com/dataseries/image-semantic-segmentation-of-satellite-imagery-using-u-net-e99ae13cf464) -> with [repo](https://github.com/raoofnaushad/Image-Semantic-Segmentation-of-Satellite-Imagery-using-U-Net.)
* [SSRN](https://github.com/zilongzhong/SSRN) -> Implementation of SSRN for Hyperspectral Image Classification
* [Implementing Transfer Learning from RGB to Multi-channel Imagery](https://towardsdatascience.com/implementing-transfer-learning-from-rgb-to-multi-channel-imagery-f87924679166) -> Semantic Segmentation using ResNet50 Backbone plus Pyramid Pooling
* [clusternet_segmentation](https://github.com/zhygallo/clusternet_segmentation) -> Unsupervised Segmentation by applying K-Means clustering to the features generated by Neural Network

### Semantic segmentation - multiclass classification
* [Land Cover Classification with U-Net](https://baratam-tarunkumar.medium.com/land-cover-classification-with-u-net-aa618ea64a1b) -> Satellite Image Multi-Class Semantic Segmentation Task with PyTorch Implementation of U-Net
* [Multi-class semantic segmentation of satellite images using U-Net](https://github.com/rogerxujiang/dstl_unet) using DSTL dataset, tensorflow 1 & python 2.7. Accompanying [article](https://towardsdatascience.com/dstl-satellite-imagery-contest-on-kaggle-2f3ef7b8ac40)
* [Codebase for multi class land cover classification with U-Net](https://github.com/jaeeolma/lulc_ml) accompanying a masters thesis, uses Keras
* [dubai-satellite-imagery-segmentation](https://github.com/ayushdabra/dubai-satellite-imagery-segmentation) -> due to the small dataset, image augmentation was used
* [U-Net for Semantic Segmentation on Unbalanced Aerial Imagery](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56) -> using the Dubai dataset
* [CDL-Segmentation](https://github.com/asimniazi63/CDL-Segmentation) -> code for the paper: "Deep Learning Based Land Cover and Crop Type Classification: A Comparative Study", comparing UNet, SegNet & DeepLabv3+
* [floatingobjects](https://github.com/ESA-PhiLab/floatingobjects) -> code for the paper: TOWARDS DETECTING FLOATING OBJECTS ON A GLOBAL SCALE WITHLEARNED SPATIAL FEATURES USING SENTINEL 2. Uses U-Net & pytorch
* [LoveDA](https://github.com/Junjue-Wang/LoveDA) -> code for the paper "A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation"
* [Satellite Imagery Semantic Segmentation with CNN](https://joshting.medium.com/satellite-imagery-segmentation-with-convolutional-neural-networks-f9254de3b907) -> 7 different segmentation classes, DeepGlobe Land Cover Classification Challenge dataset, with [repo](https://github.com/justjoshtings/satellite_image_segmentation)

### Semantic segmentation - buildings, rooftops & solar panels
* [Semantic Segmentation on Aerial Images using fastai](https://medium.com/swlh/semantic-segmentation-on-aerial-images-using-fastai-a2696e4db127) uses U-Net on the Inria Aerial Image Labeling Dataset of urban settlements in Europe and the United States, and is labelled as a building and not building classes (no repo)
* [Road and Building Semantic Segmentation in Satellite Imagery](https://github.com/Paulymorphous/Road-Segmentation) uses U-Net on the Massachusetts Roads Dataset & keras
* [find-unauthorized-constructions-using-aerial-photography](https://medium.com/towards-artificial-intelligence/find-unauthorized-constructions-using-aerial-photography-and-deep-learning-with-code-part-2-b56ca80c8c99) -> semantic segmentation using U-Net with custom_f1 metric & Keras. The creation of the dataset is described in [this article](https://pub.towardsai.net/find-unauthorized-constructions-using-aerial-photography-and-deep-learning-with-code-part-1-6d3ca7ff6fa0)
* [semantic segmentation model to identify newly developed or flooded land](https://github.com/Azure/pixel_level_land_classification) using NAIP imagery provided by the Chesapeake Conservancy, training on MS Azure
* [Pix2Pix-for-Semantic-Segmentation-of-Satellite-Images](https://github.com/A2Amir/Pix2Pix-for-Semantic-Segmentation-of-Satellite-Images) -> using Pix2Pix GAN network to segment the building footprint from Satellite Images, uses tensorflow
* [SpaceNetUnet](https://github.com/boggis30/SpaceNetUnet) -> Baseline model is U-net like, applied to SpaceNet Vegas data, using Keras
* [Building footprint detection with fastai on the challenging SpaceNet7 dataset](https://deeplearning.berlin/satellite%20imagery/computer%20vision/fastai/2021/02/17/Building-Detection-SpaceNet7.html) uses U-Net
* [automated-building-detection](https://github.com/rodekruis/automated-building-detection) -> Input: very-high-resolution (<= 0.5 m/pixel) RGB satellite images. Output: buildings in vector format (geojson), to be used in digital map products. Built on top of robosat and robosat.pink.
* [project_sunroof_india](https://github.com/AKASH2907/project_sunroof_india) -> Analyzed Google Satellite images to generate a report on individual house rooftop's solar power potential, uses a range of classical computer vision techniques (e.g Canny Edge Detection) to segment the roofs
* [JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction](https://github.com/ThomasWangWeiHong/JointNet-A-Common-Neural-Network-for-Road-and-Building-Extraction)
* [Mapping Africa’s Buildings with Satellite Imagery: Google AI blog post](https://ai.googleblog.com/2021/07/mapping-africas-buildings-with.html)
* [DeepSolar: A Machine Learning Framework to Efficiently Construct a Solar Deployment Database in the United States](https://www.cell.com/joule/fulltext/S2542-4351(18)30570-1) -> with [website](http://web.stanford.edu/group/deepsolar/home), [repo](https://github.com/wangzhecheng/DeepSolar) and [dataset on kaggle](https://www.kaggle.com/tunguz/deep-solar-dataset), actually used a CNN for classification and segmentation is obtained by applying a threshold to the activation map
* [nz_convnet](https://github.com/weiji14/nz_convnet) -> A U-net based ConvNet for New Zealand imagery to classify building outlines
* [polycnn](https://github.com/Lydorn/polycnn) -> End-to-End Learning of Polygons for Remote Sensing Image Classification
* [spacenet_building_detection](https://github.com/motokimura/spacenet_building_detection) solution by [motokimura](https://github.com/motokimura) using Unet
* [How to extract building footprints from satellite images using deep learning](https://azure.microsoft.com/en-gb/blog/how-to-extract-building-footprints-from-satellite-images-using-deep-learning/)
* [Vec2Instance](https://github.com/lakmalnd/Vec2Instance) -> applied to the SpaceNet challenge AOI 2 (Vegas) building footprint dataset, tensorflow v1.12
* [EarthquakeDamageDetection](https://github.com/JaneKravchenko/EarthquakeDamageDetection) -> Buildings segmentation from satellite imagery and damage classification for each build, using Keras
* [Semantic-segmentation repo by fuweifu-vtoo](https://github.com/fuweifu-vtoo/Semantic-segmentation) -> uses pytorch and the [Massachusetts Buildings & Roads Datasets](https://www.cs.toronto.edu/~vmnih/data/)
* [Mask_RCNN-for-Caravans](https://github.com/OrdnanceSurvey/Mask_RCNN-for-Caravans) -> detect caravan footprints from OS imagery
* [remote-sensing-solar-pv](https://github.com/Lkruitwagen/remote-sensing-solar-pv) -> A repository for sharing progress on the automated detection of solar PV arrays in sentinel-2 remote sensing imagery
* [Extracting buildings and roads from AWS Open Data using Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/extracting-buildings-and-roads-from-aws-open-data-using-amazon-sagemaker/) -> uses merged RGB (SpaceNet) and LiDAR (USGS 3DEP) datasets with Unet to reproduce the winning algorithm from SpaceNet challenge 4 by XD_XD. With [repo](https://github.com/aws-samples/aws-open-data-satellite-lidar-tutorial)
* [TF-SegNet](https://github.com/mathildor/TF-SegNet) -> AirNet is a segmentation network based on SegNet, but with some modifications
* [rgb-footprint-extract](https://github.com/aatifjiwani/rgb-footprint-extract) -> a Semantic Segmentation Network for Urban-Scale Building Footprint Extraction Using RGB Satellite Imagery, DeepLavV3+ module with a Dilated ResNet C42 backbone
* [Predicting the Solar Potential of Rooftops using Image Segmentation and Structured Data](https://medium.com/nam-r/predicting-the-solar-potential-of-rooftops-using-image-segmentation-and-structured-data-61198c39d57c) Medium article, using 20cm imagery & Unet
* [Roofpedia](https://github.com/ualsg/Roofpedia) -> an open registry of green roofs and solar roofs across the globe identified by Roofpedia through deep learning
* [solar-pv-global-inventory](https://github.com/Lkruitwagen/solar-pv-global-inventory) -> code from the Nature paper of Kruitwagen et al, used to produce a global inventory of utility-scale solar photvoltaic generating stations
* [SpaceNetExploration](https://github.com/yangsiyu007/SpaceNetExploration) -> A sample project demonstrating how to extract building footprints from satellite images using a semantic segmentation model. Data from the SpaceNet Challenge
* [Rooftop-Instance-Segmentation](https://github.com/MasterSkepticista/Rooftop-Instance-Segmentation) -> VGG-16, Instance Segmentation, uses the Airs dataset
* [Panel-Segmentation](https://github.com/NREL/Panel-Segmentation) -> Determine the presence of a solar array in the satellite image (boolean True/False), using a VGG16 classification model
* [solar-farms-mapping](https://github.com/microsoft/solar-farms-mapping) -> An Artificial Intelligence Dataset for Solar Energy Locations in India
* [poultry-cafos](https://github.com/microsoft/poultry-cafos) -> This repo contains code for detecting poultry barns from high-resolution aerial imagery and an accompanying dataset of predicted barns over the United States
* [ssai-cnn](https://github.com/mitmul/ssai-cnn) -> This is an implementation of Volodymyr Mnih's dissertation methods on his Massachusetts road & building dataset
* [predict_pv_yield](https://github.com/openclimatefix/predict_pv_yield) -> Using optical flow & machine learning to predict PV yield

### Semantic segmentation - roads
* [Semantic segmentation of roads and highways using Sentinel-2 imagery (10m) super-resolved using the SENX4 model up to x4 the initial spatial resolution (2.5m)](https://tracasa.es/innovative-stories/sen2roadlasviastambiensevendesdesentinel-2/) (results, no repo)
* [Semantic Segmentation of roads](https://vihan-tyagi.medium.com/semantic-segmentation-of-satellite-images-based-on-deep-learning-algorithms-ea5ec408ac53) using  U-net Keras, OSM data, project summary article by student, no code
* [Road detection using semantic segmentation and albumentations for data augmention](https://towardsdatascience.com/road-detection-using-segmentation-models-and-albumentations-libraries-on-keras-d5434eaf73a8) using the Massachusetts Roads Dataset, U-net & Keras
* [Winning Solutions from SpaceNet Road Detection and Routing Challenge](https://github.com/SpaceNetChallenge/RoadDetector)
* [RoadVecNet](https://github.com/gismodelling/RoadVecNet) -> Road-Network-Segmentation-and-Vectorization in keras with dataset and [paper](https://www.tandfonline.com/doi/abs/10.1080/15481603.2021.1972713?journalCode=tgrs20&)
* [Detecting road and road types jupyter notebook](https://github.com/taspinar/sidl/blob/master/notebooks/2_Detecting_road_and_roadtypes_in_sattelite_images.ipynb)
* [awesome-deep-map](https://github.com/antran89/awesome-deep-map) -> A curated list of resources dedicated to deep learning / computer vision algorithms for mapping. The mapping problems include road network inference, building footprint extraction, etc.
* [RoadTracer: Automatic Extraction of Road Networks from Aerial Images](https://github.com/mitroadmaps/roadtracer) -> uses an iterative search process guided by a CNN-based decision function to derive the road network graph directly from the output of the CNN

### Semantic segmentation - vegitation & crop boundaries
* [Сrор field boundary detection: approaches overview and main challenges](https://soilmate.medium.com/%D1%81r%D0%BE%D1%80-field-boundary-detection-approaches-overview-and-main-challenges-53736725cb06) - review article, no code
* [kenya-crop-mask](https://github.com/nasaharvest/kenya-crop-mask) -> Annual and in-season crop mapping in Kenya - LSTM classifier to classify pixels as containing crop or not, and a multi-spectral forecaster that provides a 12 month time series given a partial input. Dataset downloaded from GEE and pytorch lightning used for training
* [What’s growing there? Identify crops from multi-spectral remote sensing data (Sentinel 2)](https://towardsdatascience.com/whats-growing-there-a5618a2e6933) using eo-learn for data pre-processing, cloud detection, NDVI calculation, image augmentation & fastai
* [Tree species classification from from airborne LiDAR and hyperspectral data using 3D convolutional neural networks](https://github.com/jaeeolma/tree-detection-evo) accompanies research paper and uses fastai
* [crop-type-classification](https://medium.com/nerd-for-tech/crop-type-classification-cf5cc2593396) -> using Sentinel 1 & 2 data with a U-Net + LSTM, more features (i.e. bands) and higher resolution produced better results (article, no code)
* [Find sports fields using Mask R-CNN and overlay on open-street-map](https://github.com/jremillard/images-to-osm)
* [An LSTM to generate a crop mask for Togo](https://github.com/nasaharvest/togo-crop-mask)
* [DeepSatModels](https://github.com/michaeltrs/DeepSatModels) -> Code for paper "Context-self contrastive pretraining for crop type semantic segmentation"
* [farm-pin-crop-detection-challenge](https://github.com/simongrest/farm-pin-crop-detection-challenge) -> Using eo-learn and fastai to identify crops from multi-spectral remote sensing data
* [Detecting Agricultural Croplands from Sentinel-2 Satellite Imagery](https://medium.com/radiant-earth-insights/detecting-agricultural-croplands-from-sentinel-2-satellite-imagery-a025735d3bd8) -> We developed UNet-Agri, a benchmark machine learning model that classifies croplands using open-access Sentinel-2 imagery at 10m spatial resolution
* [DeepTreeAttention](https://github.com/weecology/DeepTreeAttention) -> Implementation of Hang et al. 2020 "Hyperspectral Image Classification with Attention Aided CNNs" for tree species prediction

### Semantic segmentation - water & floods
* [UNSOAT used fastai to train a Unet to perform semantic segmentation on satellite imageries to detect water](https://forums.fast.ai/t/unosat-used-fastai-ai-for-their-floodai-model-discussion-on-how-to-move-forward/78468) - [paper](https://www.mdpi.com/2072-4292/12/16/2532) + [notebook](https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/Fastai%20training.ipynb), accuracy 0.97, precision 0.91, recall 0.92
* [Semi-Supervised Classification and Segmentation on High Resolution Aerial Images - Solving the FloodNet problem](https://sahilkhose.medium.com/paper-presentation-e9bd0f3fb0bf)
* [Flood Detection and Analysis using UNET with Resnet-34 as the back bone](https://github.com/orion29/Satellite-Image-Segmentation-for-Flood-Damage-Analysis) uses fastai
* [Houston_flooding](https://github.com/Lichtphyz/Houston_flooding) -> labeling each pixel as either flooded or not using data from Hurricane Harvey. Dataset consisted of pre and post flood images, and a ground truth floodwater mask was created using unsupervised clustering (with DBScan) of image pixels with human cluster verification/adjustment
* [ml4floods](https://github.com/spaceml-org/ml4floods) -> An ecosystem of data, models and code pipelines to tackle flooding with ML
* [A comprehensive guide to getting started with the ETCI Flood Detection competition](https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a) -> using Sentinel1 SAR & pytorch
* [Map Floodwater of SAR Imagery with SageMaker](https://github.com/JayThibs/map-floodwater-sar-imagery-on-sagemaker) -> applied to Sentinel-1 dataset
* [1st place solution for STAC Overflow: Map Floodwater from Radar Imagery hosted by Microsoft AI for Earth](https://github.com/sweetlhare/STAC-Overflow) -> combines Unet with Catboostclassifier, taking their maxima, not the average
* [hydra-floods](https://github.com/Servir-Mekong/hydra-floods) -> an open source Python application for downloading, processing, and delivering surface water maps derived from remote sensing data

### Semantic segmentation - fire & burn areas
* [Wild Fire Detection](https://github.com/yueureka/WildFireDetection) using U-Net trained on Databricks & Keras, semantic segmentation
* [A Practical Method for High-Resolution Burned Area Monitoring Using Sentinel-2 and VIIRS](https://www.mdpi.com/2072-4292/13/9/1608) with [code](https://github.com/mnpinto/FireHR). Dataset created on Google Earth Engine, downloaded to local machine for model training using fastai. The BA-Net model used is much smaller than U-Net, resulting in lower memory requirements and a faster computation
* [AI Geospatial Wildfire Risk Prediction](https://towardsdatascience.com/ai-geospatial-wildfire-risk-prediction-8c6b1d415eb4) -> A predictive model using geospatial raster data to asses wildfire hazard potential over the contiguous United States using Unet

### Semantic segmentation - glaciers
* [HED-UNet](https://github.com/khdlr/HED-UNet) -> a model for simultaneous semantic segmentation and edge detection, examples provided are glacier fronts and building footprints using the Inria Aerial Image Labeling dataset
* [glacier_mapping](https://github.com/krisrs1128/glacier_mapping) -> Mapping glaciers in the Hindu Kush Himalaya, Landsat 7 images, Shapefile labels of the glaciers, Unet with dropout
* [glacier-detect-ML](https://github.com/mikeskaug/glacier-detect-ML) -> a simple logistic regression model to identify a glacier in Landsat satellite imagery

### Semantic segmentation - electrical substations
The repos below resulted from the [ICETCI 2021 competition on Machine Learning based feature extraction of Electrical Substations from Satellite Data using Open Source Tools](https://competitions.codalab.org/competitions/32132)
* [Aarsh2001/ML_Challenge_NRSC](https://github.com/Aarsh2001/ML_Challenge_NRSC) -> 3rd place entry
* [Electrical-Substation-detection-by-fine-tunning-Segnet-Architecture](https://github.com/dksharp108/Electrical-Substation-detection-by-fine-tunning-Segnet-Architecture) -> 9th place entry
* [electrical_substation_detection](https://github.com/thisishardik/electrical_substation_detection) -> entry using UNet, Albumentations for image augmentation, and OpenCV for computer vision tasks

### Instance segmentation
In instance segmentation, each individual 'instance' of a segmented area is given a unique lable. For detection of very small objects this may a good approach, but it can struggle seperating individual objects that are closely spaced.
* [Mask_RCNN](https://github.com/matterport/Mask_RCNN) generates bounding boxes and segmentation masks for each instance of an object in the image. It is very commonly used for instance segmentation & object detection
* [Instance segmentation of center pivot irrigation system in Brazil](https://github.com/saraivaufc/instance-segmentation-maskrcnn) using free Landsat images, mask R-CNN & Keras
* [Oil tank instance segmentation with Mask R-CNN](https://github.com/georgiosouzounis/instance-segmentation-mask-rcnn) with [accompanying article](https://medium.com/@georgios.ouzounis/oil-storage-tank-instance-segmentation-with-mask-r-cnn-77c94433045f) using Keras & Airbus Oil Storage Detection Dataset on Kaggle
* [Building-Detection-MaskRCNN](https://github.com/Mstfakts/Building-Detection-MaskRCNN) -> Building detection from the SpaceNet dataset by using Mask RCNN

## Object detection
Several different techniques can be used to count the number of objects in an image. The returned data can be an object count (regression), a bounding box around individual objects in an image (typically using Yolo or Faster R-CNN architectures), a pixel mask for each object (instance segmentation), key points for an an object (such as wing tips, nose and tail of an aircraft), or simply a classification for a sliding tile over an image. A good introduction to the challenge of performing object detection on aerial imagery is given in [this paper](https://arxiv.org/abs/1902.06042v2). In summary, images are large and objects may comprise only a few pixels, easily confused with random features in background. For the same reason, object detection datasets are inherently imbalanced, since the area of background typically dominates over the area of the objects to be detected. In general object detecion performs well on large objects, and gets increasingly difficult as the objects get smaller & more densely packed. Model accuracy falls off rapidly as image resolution degrades, so it is common for object detection to use very high resolution imagery, e.g. 30cm RGB. A particular characteristic of aerial images is that objects can be oriented in any direction, so using rotated bounding boxes which aligning with the object can be crucial for extracting metrics such as the length and width of an object.
* [Object Detection and Image Segmentation with Deep Learning on Earth Observation Data: A Review](https://www.mdpi.com/2072-4292/12/10/1667)
* [awesome-aerial-object-detection](https://github.com/murari023/awesome-aerial-object-detection) and [awesome-tiny-object-detection](https://github.com/kuanhungchen/awesome-tiny-object-detection) list many relevant papers
* [Tackling the Small Object Problem in Object Detection](https://blog.roboflow.com/tackling-the-small-object-problem-in-object-detection)
* [Satellite Imagery Multiscale Rapid Detection with Windowed Networks (SIMRDWN)](https://github.com/avanetten/simrdwn) -> combines some of the leading object detection algorithms into a unified framework designed to detect objects both large and small in overhead imagery. Train models and test on arbitrary image sizes with YOLO (versions 2 and 3), Faster R-CNN, SSD, or R-FCN.
* [YOLTv4](https://github.com/avanetten/yoltv4) -> YOLTv4 is designed to detect objects in aerial or satellite imagery in arbitrarily large images that far exceed the ~600×600 pixel size typically ingested by deep learning object detection frameworks. Read [Announcing YOLTv4: Improved Satellite Imagery Object Detection](https://towardsdatascience.com/announcing-yoltv4-improved-satellite-imagery-object-detection-f5091e913fad)
* [Tensorflow Benchmarks for Object Detection in Aerial Images](https://github.com/yangxue0827/RotationDetection) -> tensorflow-based codebase created to build benchmarks for object detection in aerial images
* [Pytorch Benchmarks for Object Detection in Aerial Images](https://github.com/dingjiansw101/AerialDetection) -> pytorch-based codebase created to build benchmarks for object detection in aerial images using mmdetection
* [ASPDNet](https://github.com/liuqingjie/ASPDNet) -> Counting dense objects in remote sensing images, [arxiv paper](https://arxiv.org/abs/2002.05928)
* [xview-yolov3](https://github.com/ultralytics/xview-yolov3) -> xView 2018 Object Detection Challenge: YOLOv3 Training and Inference
* [Faster RCNN for xView satellite data challenge](https://github.com/samirsen/small-object-detection)
* [How to detect small objects in (very) large images](https://blog.ml6.eu/how-to-detect-small-objects-in-very-large-images-70234bab0f98) -> A practical guide to using Slicing-Aided Hyper Inference (SAHI) for performing inference on the DOTAv1.0 object detection dataset using the mmdetection framework
* [Object detection on Satellite Imagery using RetinaNet](https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-1-training-e589975afbd5) -> using the Kaggle Swimming Pool and Car Detection dataset
* [Object Detection Satellite Imagery Multi-vehicles Dataset (SIMD)](https://github.com/asimniazi63/Object-Detection-on-Satellite-Images) -> RetinaNet,Yolov3 and Faster RCNN for multi object detection on satellite images dataset
* [SNIPER/AutoFocus](https://github.com/mahyarnajibi/SNIPER) -> an efficient multi-scale object detection training/inference algorithm
* [marine_debris_ML](https://github.com/NASA-IMPACT/marine_debris_ML) -> Marine debris detection, uses 3-meter imagery product called Planetscope with bands in the red, green, blue, and near-infrared. Uses Tensorflow Object Detection API with pre-trained resnet 101
* [pool-detection-from-aerial-imagery](https://towardsdatascience.com/pool-detection-from-aerial-imagery-f5b76d0a6093) -> Use Icevision and Detectron2 to detect swimming pools from aerial imagery
* [Electric-Pylon-Detection-in-RSI](https://github.com/qsjxyz/Electric-Pylon-Detection-in-RSI) -> a dataset which contains 1500 remote sensing images of electric pylons used to train ten deep learning models
* [Synthesizing Robustness YOLTv4 Results Part 2: Dataset Size Requirements and Geographic Insights](https://www.iqt.org/synthesizing-robustness-yoltv4-results-part-2-dataset-size-requirements-and-geographic-insights/) -> quantify how much harder rare objects are to localize

#### Object detection enhanced by super resolution
* [Super-Resolution and Object Detection](https://medium.com/the-downlinq/super-resolution-and-object-detection-a-love-story-part-4-8ad971eef81e) -> Super-resolution is a relatively inexpensive enhancement that can improve object detection performance
* [EESRGAN](https://github.com/Jakaria08/EESRGAN) -> Small-Object Detection in Remote Sensing Images with End-to-End Edge-Enhanced GAN and Object Detector Network
* [Mid-Low Resolution Remote Sensing Ship Detection Using Super-Resolved Feature Representation](https://www.preprints.org/manuscript/202108.0337/v1)

#### Object detection with rotated bounding boxes
* [OBBDetection](https://github.com/jbwang1997/OBBDetection) -> an oriented object detection library, which is based on MMdetection
* [rotate-yolov3](https://github.com/ming71/rotate-yolov3) -> Rotation object detection implemented with yolov3. Also see [yolov3-polygon](https://github.com/ming71/yolov3-polygon)
* [DRBox](https://github.com/liulei01/DRBox) -> for detection tasks where the objects are orientated arbitrarily, e.g. vehicles, ships and airplanes
* [s2anet](https://github.com/csuhan/s2anet) -> Official code of the paper 'Align Deep Features for Oriented Object Detection'
* [CFC-Net](https://github.com/ming71/CFC-Net) -> Official implementation of "CFC-Net: A Critical Feature Capturing Network for Arbitrary-Oriented Object Detection in Remote Sensing Images"
* [ReDet](https://github.com/csuhan/ReDet) -> Official code of the paper "ReDet: A Rotation-equivariant Detector for Aerial Object Detection"
* [BBAVectors-Oriented-Object-Detection](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection) -> Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors

#### Object detection - buildings, rooftops & solar panels
* [Machine Learning For Rooftop Detection and Solar Panel Installment](https://omdena.com/blog/machine-learning-rooftops/) discusses tiling large images and generating annotations from OSM data. Features of the roofs were calculated using a combination of contour detection and classification. [Follow up article using semantic segmentation](https://omdena.com/blog/rooftops-classification/)
* [Building Extraction with YOLT2 and SpaceNet Data](https://medium.com/the-downlinq/building-extraction-with-yolt2-and-spacenet-data-a926f9ffac4f)
* [AIcrowd dataset of building outlines](https://www.aicrowd.com/challenges/mapping-challenge-old) -> 300x300 pixel RGB images with annotations in MS-COCO format
* [XBD-hurricanes](https://github.com/dbuscombe-usgs/XBD-hurricanes) -> Models for building (and building damage) detection in high-resolution (<1m) satellite and aerial imagery using a modified RetinaNet model
* [Detecting solar panels from satellite imagery](https://towardsdatascience.com/weekend-project-detecting-solar-panels-from-satellite-imagery-f6f5d5e0da40) using segmentation
* [ssd-spacenet](https://github.com/aurotripathy/ssd-spacenet) -> Detect buildings in the Spacenet dataset using Single Shot MultiBox Detector (SSD)
* [3DBuildingInfoMap](https://github.com/LllC-mmd/3DBuildingInfoMap) -> simultaneous extraction of building height and footprint from Sentinel imagery using ResNet

#### Object detection - ships & boats
* [How hard is it for an AI to detect ships on satellite images?](https://medium.com/earthcube-stories/how-hard-it-is-for-an-ai-to-detect-ships-on-satellite-images-7265e34aadf0)
* [Object Detection in Satellite Imagery, a Low Overhead Approache](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7)
* [Detecting Ships in Satellite Imagery](https://medium.com/dataseries/detecting-ships-in-satellite-imagery-7f0ca04e7964) using the Planet dataset and Keras
* [Planet use non DL felzenszwalb algorithm to detect ships](https://github.com/planetlabs/notebooks/blob/master/jupyter-notebooks/ship-detector/01_ship_detector.ipynb)
* [Ship detection using k-means clustering & CNN classifier on patches](https://towardsdatascience.com/data-science-and-satellite-imagery-985229e1cd2f)
* [sentinel2-xcube-boat-detection](https://github.com/MichelDeudon/sentinel2-xcube-boat-detection) -> detect and count boat traffic in Sentinel-2 imagery using temporal, spectral and spatial features
* [Arbitrary-Oriented Ship Detection through Center-Head Point Extraction](https://arxiv.org/abs/2101.11189) -> arxiv paper. Keypoint estimation is performed to find the center of ships. Then, the size and head point of the ships are regressed. Repo [ASD](https://github.com/JinleiMa/ASD)
* [ship_detection](https://github.com/rugg2/ship_detection) -> using an interesting combination of CNN classifier, Class Activation Mapping (CAM) & UNET segmentation. Accompanying [three part blog post](https://www.vortexa.com/insight/satellite-images-object-detection)
* [Building a complete Ship detection algorithm using YOLOv3 and Planet satellite images](https://medium.com/intel-software-innovators/ship-detection-in-satellite-images-from-scratch-849ccfcc3072) -> covers finding and annotating data (using LabelMe), preprocessing large images into chips, and training Yolov3. [Repo](https://github.com/amanbasu/ship-detection)
* [Ship-detection-in-satellite-images](https://github.com/zmf0507/Ship-detection-in-satellite-images) -> experiments with  UNET, YOLO, Mask R-CNN, SSD, Faster R-CNN, RETINA-NET
* [Ship-Detection-from-Satellite-Images-using-YOLOV4](https://github.com/debasis-dotcom/Ship-Detection-from-Satellite-Images-using-YOLOV4) -> uses Kaggle Airbus Ship Detection dataset
* [kaggle-airbus-ship-detection-challenge](https://github.com/toshi-k/kaggle-airbus-ship-detection-challenge) -> using oriented SSD
* [shipsnet-detector](https://github.com/rhammell/shipsnet-detector) -> Detect container ships in Planet imagery using machine learning
* [Classifying Ships in Satellite Imagery with Neural Networks](https://towardsdatascience.com/classifying-ships-in-satellite-imagery-with-neural-networks-944024879651) -> applied to the Kaggle Ships in Satellite Imagery dataset
* [Mask R-CNN for Ship Detection & Segmentation](https://medium.com/@gabogarza/mask-r-cnn-for-ship-detection-segmentation-a1108b5a083) blog post with [repo](https://github.com/gabrielgarza/Mask_RCNN)
* [contrastive_SSL_ship_detection](https://github.com/alina2204/contrastive_SSL_ship_detection) -> Contrastive self supervised learning for ship detection in Sentinel 2 images
* [Boat detection with multi-region-growing method in satellite images](https://medium.com/@ipmach/boat-detection-with-multi-region-growing-method-in-satellite-images-3339a6c29a8c)

#### Object detection - vehicles & trains
* [Truck Detection with Sentinel-2 during COVID-19 crisis](https://github.com/hfisser/Truck_Detection_Sentinel2_COVID19) -> moving objects in Sentinel-2 data causes a specific reflectance relationship in the RGB, which looks like a rainbow, and serves as a marker for trucks. Improve accuracy by only analysing roads. Not using object detection but relevant
* [cowc_car_counting](https://github.com/motokimura/cowc_car_counting) -> car counting on the [Cars Overhead With Context (COWC) dataset](https://gdo152.llnl.gov/cowc/). Not sctictly object detection but a CNN to predict the car count in a tile
* [Traffic density estimation as a regression problem instead of object detection](https://omdena.com/blog/ai-road-safety/) -> inspired by [this paper](https://ieeexplore.ieee.org/document/8916990)
* [Applying Computer Vision to Railcar Detection](https://orbitalinsight.com/blog/apping-computer-vision-to-railcar-detection) -> useful insights into counting railcars (i.e. train carriages) using Mask-RCNN with rotated bounding boxes output
* [Leveraging Deep Learning for Vehicle Detection And Classification](https://orbitalinsight.com/blog/leveraging-deep-learning-for-vehicle-detection-and-classification)

#### Object detection - planes & aircraft
* [yoltv4](https://github.com/avanetten/yoltv4) includes examples on the [RarePlanes dataset](https://registry.opendata.aws/rareplanes/)
* [aircraft-detection](https://github.com/hakeemtfrank/aircraft-detection) -> experiments to test the performance of a Gaussian process (GP) classifier with various kernels on the UC Merced land use land cover (LULC) dataset
* The [Rareplanes guide](https://www.cosmiqworks.org/rareplanes-public-user-guide/) recommends annotating airplanes in a diamond style, which has several advantages (easily reproducible, convertible to a bounding box etc) and allows extracting the aircraft length and wingspan
* [rareplanes-yolov5](https://github.com/jeffaudi/rareplanes-yolov5) -> using YOLOv5 and the RarePlanes dataset to detect and classify sub-characteristics of aircraft
* [detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5](https://medium.com/artificialis/detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5-5f3d464b75ad)
* [Using Detectron2 to segment aircraft from satellite imagery](https://share.buitrongan.com/using-detectron2-to-segments-aircraft-from-satellite-images-5a8ac1a0d35e) -> pytorch and Rare Planes

#### Object detection - animals
* [cownter_strike](https://github.com/IssamLaradji/cownter_strike) -> counting cows, located with point-annotations, two models: CSRNet (a density-based method) & LCFCN (a detection-based method)

## Counting trees
* [DeepForest](https://deepforest.readthedocs.io/en/latest/index.html) is a python package for training and predicting individual tree crowns from airborne RGB imagery
* [Official repository for the "Identifying trees on satellite images" challenge from Omdena](https://github.com/cienciaydatos/ai-challenge-trees)
* [Counting-Trees-using-Satellite-Images](https://github.com/A2Amir/Counting-Trees-using-Satellite-Images) -> create an inventory of incoming and outgoing trees for an annual tree inspections, uses keras & semantic segmentation
* [2020 Nature paper - An unexpectedly large count of trees in the West African Sahara and Sahel](https://www.nature.com/articles/s41586-020-2824-5) -> tree detection framework based on U-Net & tensorflow 2 with code [here](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel/tree/v1.0.0)

## Oil storage tank detection & oil spills
Oil is stored in tanks at many points between extraction and sale, and the volume of oil in storage is an important economic indicator.
* [A Beginner’s Guide To Calculating Oil Storage Tank Occupancy With Help Of Satellite Imagery](https://medium.com/planet-stories/a-beginners-guide-to-calculating-oil-storage-tank-occupancy-with-help-of-satellite-imagery-e8f387200178)
* [Oil Storage Tank’s Volume Occupancy On Satellite Imagery Using YoloV3](https://towardsdatascience.com/oil-storage-tanks-volume-occupancy-on-satellite-imagery-using-yolov3-3cf251362d9d) with [repo](https://github.com/mdmub0587/Oil-Storage-Tank-s-Volume-Occupancy)
* [Oil-Tank-Volume-Estimation](https://github.com/kheyer/Oil-Tank-Volume-Estimation) -> combines object detection and classical computer vision
* [MCAN-OilSpillDetection](https://github.com/liyongqingupc/MCAN-OilSpillDetection) -> Oil Spill Detection with A Multiscale Conditional Adversarial Network under Small Data Training, with [paper](https://www.mdpi.com/2072-4292/13/12/2378). A multiscale conditional adversarial network (MCAN) trained with four oil spill observation images accurately detects oil spills in new images.
* [Oil tank instance segmentation with Mask R-CNN](https://github.com/georgiosouzounis/instance-segmentation-mask-rcnn) with [accompanying article](https://medium.com/@georgios.ouzounis/oil-storage-tank-instance-segmentation-with-mask-r-cnn-77c94433045f) using Keras & Airbus Oil Storage Detection Dataset on Kaggle

## Cloud detection & removal
Generally treated as a semantic segmentation problem.
* From [this article on sentinelhub](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13) there are three popular classical algorithms that detects thresholds in multiple bands in order to identify clouds. In the same article they propose using semantic segmentation combined with a CNN for a cloud classifier (excellent review paper [here](https://arxiv.org/pdf/1704.06857.pdf)), but state that this requires too much compute resources.
* [This article](https://www.mdpi.com/2072-4292/8/8/666) compares a number of ML algorithms, random forests, stochastic gradient descent, support vector machines, Bayesian method.
* [Segmentation of Clouds in Satellite Images Using Deep Learning](https://medium.com/swlh/segmentation-of-clouds-in-satellite-images-using-deep-learning-a9f56e0aa83d) -> semantic segmentation using a Unet on the Kaggle [38-cloud](https://www.kaggle.com/sorour/38cloud-cloud-segmentation-in-satellite-images) Landsat dataset
* [Cloud Detection in Satellite Imagery](https://www.azavea.com/blog/2021/02/08/cloud-detection-in-satellite-imagery/) compares FPN+ResNet18 and CheapLab architectures on Sentinel-2 L1C and L2A imagery
* [Cloud-Removal-with-GAN-Satellite-Image-Processing](https://github.com/Akash-Ramjyothi/Cloud-Removal-with-GAN-Satellite-Image-Processing)
* [Benchmarking Deep Learning models for Cloud Detection in Landsat-8 and Sentinel-2 images](https://github.com/IPL-UV/DL-L8S2-UV)
* [Landsat-8 to Proba-V Transfer Learning and Domain Adaptation for Cloud detection](https://github.com/IPL-UV/pvl8dagans)
* [Multitemporal Cloud Masking in Google Earth Engine](https://github.com/IPL-UV/ee_ipl_uv)
* [s2cloudmask](https://github.com/daleroberts/s2cloudmask) -> Sentinel-2 Cloud and Shadow Detection using Machine Learning
* [sentinel2-cloud-detector](https://github.com/sentinel-hub/sentinel2-cloud-detector) -> Sentinel Hub Cloud Detector for Sentinel-2 images in Python
* [dsen2-cr](https://github.com/ameraner/dsen2-cr) -> cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion, contains the model code, written in Python/Keras, as well as links to pre-trained checkpoints and the SEN12MS-CR dataset
* [pyatsa](https://github.com/agroimpacts/pyatsa) -> Python package implementing the Automated Time-Series Analysis method for masking clouds in satellite imagery developed by Zhu and Helmer 2018

## Change detection & time-series
Monitor water levels, coast lines, size of urban areas, wildfire damage. Note, clouds change often too..!
* [awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection) lists many datasets and publications
* [Change-Detection-Review](https://github.com/MinZHANG-WHU/Change-Detection-Review) -> A review of change detection methods, including code and open data sets for deep learning
* [Unsupervised Changed Detection in Multi-Temporal Satellite Images using PCA & K-Means](https://appliedmachinelearning.blog/2017/11/25/unsupervised-changed-detection-in-multi-temporal-satellite-images-using-pca-k-means-python-code/) -> python 2
* [LANDSAT Time Series Analysis for Multi-temporal Land Cover Classification using Random Forest](https://github.com/agr-ayush/Landsat-Time-Series-Analysis-for-Multi-Temporal-Land-Cover-Classification)
* [Unstructured-change-detection-using-CNN](https://github.com/vbhavank/Unstructured-change-detection-using-CNN)
* [Siamese neural network to detect changes in aerial images](https://github.com/vbhavank/Siamese-neural-network-for-change-detection) -> uses Keras and VGG16 architecture
* [Change Detection in 3D: Generating Digital Elevation Models from Dove Imagery](https://www.planet.com/pulse/publications/change-detection-in-3d-generating-digital-elevation-models-from-dove-imagery/)
* [QGIS plugin for applying change detection algorithms on high resolution satellite imagery](https://github.com/dymaxionlabs/massive-change-detection)
* [LamboiseNet](https://github.com/hbaudhuin/LamboiseNet) -> Master thesis about change detection in satellite imagery using Deep Learning
* [Fully Convolutional Siamese Networks for Change Detection](https://github.com/rcdaudt/fully_convolutional_change_detection) -> with [paper](https://ieeexplore.ieee.org/abstract/document/8451652)
* [Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks](https://github.com/rcdaudt/patch_based_change_detection) -> with [paper](https://ieeexplore.ieee.org/abstract/document/8518015), used the Onera Satellite Change Detection (OSCD) dataset
* [STANet](https://github.com/justchenhao/STANet) -> official implementation of the spatial-temporal attention neural network (STANet) for remote sensing image change detection
* [BIT_CD](https://github.com/justchenhao/BIT_CD) -> Official Pytorch Implementation of Remote Sensing Image Change Detection with Transformers
* [IAug_CDNet](https://github.com/justchenhao/IAug_CDNet) -> Official Pytorch Implementation of Adversarial Instance Augmentation for Building Change Detection in Remote Sensing Images
* [dpm-rnn-public](https://github.com/olliestephenson/dpm-rnn-public) -> Code implementing a damage mapping method combining satellite data with deep learning
* [SenseEarth2020-ChangeDetection](https://github.com/LiheYoung/SenseEarth2020-ChangeDetection) -> 1st place solution to the Satellite Image Change Detection Challenge hosted by SenseTime; predictions of five HRNet-based segmentation models are ensembled, serving as pseudo labels of unchanged areas
* [KPCAMNet](https://github.com/I-Hope-Peace/KPCAMNet) -> Python implementation of the paper Unsupervised Change Detection in Multi-temporal VHR Images Based on Deep Kernel PCA Convolutional Mapping Network
* [CDLab](https://github.com/Bobholamovic/CDLab) -> benchmarking deep learning-based change detection methods.
* [Siam-NestedUNet](https://github.com/likyoo/Siam-NestedUNet) -> The pytorch implementation for "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images"
* [SUNet-change_detection](https://github.com/ShaoRuizhe/SUNet-change_detection) -> Implementation of paper SUNet: Change Detection for Heterogeneous Remote Sensing Images from Satellite and UAV Using a Dual-Channel Fully Convolution Network
* [temporalCNN](https://github.com/charlotte-pel/temporalCNN) -> Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series
* [Self-supervised Change Detection in Multi-view Remote Sensing Images](https://github.com/cyx669521/self-supervised_change_detetction)
* [MFPNet](https://github.com/wzjialang/MFPNet) -> Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity
* [pytorch-psetae](https://github.com/VSainteuf/pytorch-psetae) -> PyTorch implementation of the model presented in Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention
* [GitHub for the DIUx xView Detection Challenge](https://github.com/DIUx-xView) -> The xView2 Challenge focuses on automating the process of assessing building damage after a natural disaster
* [DASNet](https://github.com/lehaifeng/DASNet) -> Dual attentive fully convolutional siamese networks for change detection of high-resolution satellite images
* [Self-Attention for Raw Optical Satellite Time Series Classification](https://github.com/MarcCoru/crop-type-mapping)
* [satflow](https://github.com/openclimatefix/satflow) -> optical flow models for predicting future satellite images from current and past ones
* [planet-movement](https://github.com/rhammell/planet-movement) -> Find and process Planet image pairs to highlight object movement
* [UNet-based-Unsupervised-Change-Detection](https://github.com/annabosman/UNet-based-Unsupervised-Change-Detection) -> A convolutional neural network (CNN) and semantic segmentation is implemented to detect the changes between the images, as well as classify the changes into the correct semantic class, with [arxiv paper](https://arxiv.org/abs/1812.05815)
* [temporal-cluster-matching](https://github.com/microsoft/temporal-cluster-matching) -> detecting change in structure footprints from time series of remotely sensed imagery
* [autoRIFT](https://github.com/nasa-jpl/autoRIFT) -> fast and intelligent algorithm for finding the pixel displacement between two images
* [esa-superresolution-forecasting](https://github.com/PiSchool/esa-superresolution-forecasting) -> Forecasting air pollution using ESA Sentinel-5p data, and an encoder-decoder convolutional LSTM neural network architecture, implemented in Pytorch
* [Radiant-Earth-Spot-the-Crop-Challenge](https://github.com/DariusTheGeek/Radiant-Earth-Spot-the-Crop-Challenge) -> The main objective of this challenge was to use time-series of Sentinel-2 multi-spectral data to classify crops in the Western Cape of South Africa. The challenge was to build a machine learning model to predict crop type classes for the test dataset.
* [DSAMNet](https://github.com/liumency/DSAMNet) -> Code for “A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection”. The main types of changes in the dataset include: (a) newly built urban buildings; (b) suburban dilation; (c) groundwork before construction; (d) change of vegetation; (e) road expansion; (f) sea construction.
* [SRCDNet](https://github.com/liumency/SRCDNet) -> The pytorch implementation for "Super-resolution-based Change Detection Network with Stacked Attention Module for Images with Different Resolutions ". SRCDNet is designed to learn and predict change maps from bi-temporal images with different resolutions

## Wealth and economic activity
The goal is to predict economic activity from satellite imagery rather than conducting labour intensive ground surveys
* [Using publicly available satellite imagery and deep learning to understand economic well-being in Africa, Nature Comms 22 May 2020](https://www.nature.com/articles/s41467-020-16185-w) -> Used CNN on Ladsat imagery (night & day) to predict asset wealth of African villages
* [Combining Satellite Imagery and machine learning to predict poverty](https://towardsdatascience.com/combining-satellite-imagery-and-machine-learning-to-predict-poverty-884e0e200969) -> review article
* [Measuring Human and Economic Activity from Satellite Imagery to Support City-Scale Decision-Making during COVID-19 Pandemic](https://arxiv.org/abs/2004.07438) -> arxiv article
* [Predicting Food Security Outcomes Using CNNs for Satellite Tasking](https://arxiv.org/pdf/1902.05433.pdf) -> arxiv article
* [Crop yield Prediction with Deep Learning](https://github.com/JiaxuanYou/crop_yield_prediction) -> code for the paper Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data
* [Measuring the Impacts of Poverty Alleviation Programs with Satellite Imagery and Deep Learning](https://github.com/luna983/beyond-nightlight) -> code and paper
* [Crop Yield Prediction Using Deep Neural Networks and LSTM](https://omdena.com/blog/deep-learning-yield-prediction/)
* [Building a Crop Yield Prediction App in Senegal Using Satellite Imagery and Jupyter Voila](https://omdena.com/blog/yield-prediction/)
* [Advanced Deep Learning Techniques for Predicting Maize Crop Yield using Sentinel-2 Satellite Imagery](https://zionayomide.medium.com/advanced-deep-learning-techniques-for-predicting-maize-crop-yield-using-sentinel-2-satellite-1b63ac8b0789)
* [Building a Spatial Model to Classify Global Urbanity Levels](https://towardsdatascience.com/building-a-spatial-model-to-classify-global-urbanity-levels-e2fb9da7252) -> estimage global urbanity levels from population data, nightime lights and road networks
* [deeppop](https://deeppop.github.io/) -> Deep Learning Approach for Population Estimation from Satellite Imagery, also [on Github](https://github.com/deeppop)
* [Estimating telecoms demand in areas of poor data availability](https://github.com/edwardoughton/taddle) -> with papers on [arxiv](https://arxiv.org/abs/2006.07311) and [Science Direct](https://www.sciencedirect.com/science/article/abs/pii/S0736585321000617)
* [satimage](https://github.com/mani-shailesh/satimage) -> Code and models for the manuscript "Predicting Poverty and Developmental Statistics from Satellite Images using Multi-task Deep Learning". Predict the main material of a roof, source of lighting and source of drinking water for properties, from satellite imagery
* [Deep-Transfer-Learning-Crop-Yield-Prediction](https://github.com/sustainlab-group/Deep-Transfer-Learning-Crop-Yield-Prediction)

## Disaster response
* [DisaVu](https://github.com/SrzStephen/DisaVu) -> combines building & damage detection and provides an app for viewing predictions

## Super-resolution
Super-resolution attempts to enhance the resolution of an imaging system, and can be applied as a pre-processing step to improve the detection of small objects. For an introduction to this topic [read this excellent article](https://bleedai.com/super-resolution-going-from-3x-to-8x-resolution-in-opencv/). Note that super resolution techniques are generally grouped into single image super resolution (SISR) **or** a multi image super resolution (MISR) which is typically applied to video frames.
* [Super-Resolution on Satellite Imagery using Deep Learning](https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-1-ec5c5cd3cd2) -> Nov 2016 blog post by CosmiQ Works with a nice introduction to the topic. Proposes and demonstrates a new architecture with perturbation layers with practical guidance on the methodology and [code](https://github.com/CosmiQ/super-resolution). [Three part series](https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-3-2e2f61eee1d3)
* [Awesome-Super-Resolution](https://github.com/ptkin/Awesome-Super-Resolution) -> another 'awesome' repo, getting a little out of date now
* [Super-Resolution (python) Utilities for managing large satellite images](https://github.com/jshermeyer/SR_Utils)
* [pytorch-enhance](https://github.com/isaaccorley/pytorch-enhance) -> Library of Image Super-Resolution Models, Datasets, and Metrics for Benchmarking or Pretrained Use. Also [checkout this implementation in Jax](https://github.com/isaaccorley/jax-enhance)
* [Super Resolution in OpenCV](https://learnopencv.com/super-resolution-in-opencv/)
* [AI-based Super resolution and change detection to enforce Sentinel-2 systematic usage](https://medium.com/@sistema_gmbh/ai-based-super-resolution-and-change-detection-to-enforce-sentinel-2-systematic-usage-65aa37d0365) -> Worldview-2 images (2m) were used to create a reference dataset and increase the spatial resolution of the Copernicus sensor from 10m to 5m
* [SRCDNet](https://github.com/liumency/SRCDNet) -> The pytorch implementation for "Super-resolution-based Change Detection Network with Stacked Attention Module for Images with Different Resolutions ". SRCDNet is designed to learn and predict change maps from bi-temporal images with different resolutions
* [Model-Guided Deep Hyperspectral Image Super-resolution](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution) -> code accompanying the paper [Model-Guided Deep Hyperspectral Image Super-Resolution](https://ieeexplore.ieee.org/document/9429905)

### Single image super resolution (SISR)
* [Super Resolution for Satellite Imagery - srcnn repo](https://github.com/WarrenGreen/srcnn)
* [TensorFlow implementation of "Accurate Image Super-Resolution Using Very Deep Convolutional Networks" adapted for working with geospatial data](https://github.com/CosmiQ/VDSR4Geo)
* [Random Forest Super-Resolution (RFSR repo)](https://github.com/jshermeyer/RFSR) including [sample data](https://github.com/jshermeyer/RFSR/tree/master/SampleImagery)
* [Enhancing Sentinel 2 images by combining Deep Image Prior and Decrappify](https://medium.com/omdena/pushing-the-limits-of-open-source-data-enhancing-satellite-imagery-through-deep-learning-9d8a3bbc0e0a). Repo for [deep-image-prior](https://github.com/DmitryUlyanov/deep-image-prior) and article on [decrappify](https://www.fast.ai/2019/05/03/decrappify/)
* [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/) -> the keras docs have a great tutorial on this light weight but well performing model
* [super-resolution-using-gan](https://github.com/saraivaufc/super-resolution-using-gan) -> Super-Resolution of Sentinel-2 Using Generative Adversarial Networks
* [Super-resolution of Multispectral Satellite Images Using Convolutional Neural Networks](https://up42.com/blog/tech/super-resolution-of-multispectral-satellite-images-using-convolutional) with [paper](https://arxiv.org/abs/2002.00580)
* [Multi-temporal Super-Resolution on Sentinel-2 Imagery](https://medium.com/sentinel-hub/multi-temporal-super-resolution-on-sentinel-2-imagery-6089c2b39ebc) using HighRes-Net, [repo](https://github.com/sentinel-hub/multi-temporal-super-resolution)
* [SSPSR-Pytorch](https://github.com/junjun-jiang/SSPSR) -> A spatial-spectral prior deep network for single hyperspectral image super-resolution
* [Sentinel-2 Super-Resolution: High Resolution For All (Bands)](https://up42.com/blog/tech/sentinel-2-superresolution)
* [CinCGAN](https://github.com/Junshk/CinCGAN-pytorch) -> Unofficial Implementation of [Unsupervised Image Super-Resolution using Cycle-in-Cycle Generative Adversarial Networks](https://arxiv.org/abs/1809.00437)
* [Satellite-image-SRGAN using PyTorch](https://github.com/xjohnxjohn/Satellite-image-SRGAN)
* [EEGAN](https://github.com/kuijiang0802/EEGAN) -> Edge Enhanced GAN For Remote Sensing Image Super-Resolution, TensorFlow 1.1
* [PECNN](https://github.com/kuijiang0802/PECNN) -> A Progressively Enhanced Network for Video Satellite Imagery Super-Resolution, minimal documentation
* [hs-sr-tvtv](https://github.com/marijavella/hs-sr-tvtv) -> Enhanced Hyperspectral Image Super-Resolution via RGB Fusion and TV-TV Minimization
* [sr4rs](https://github.com/remicres/sr4rs) -> Super resolution for remote sensing, with pre-trained model for Sentinel-2, SRGAN-inspired
* [Restoring old aerial images with Deep Learning](https://towardsdatascience.com/restoring-old-aerial-images-with-deep-learning-60f0cfd2658) -> Medium article on Super Resolution with Perceptual Loss function and real images as input

### Multi image super resolution (MISR)
Note that nearly all the MISR publications resulted from the [PROBA-V Super Resolution competition](https://kelvins.esa.int/proba-v-super-resolution/)
* [deepsum](https://github.com/diegovalsesia/deepsum) -> Deep neural network for Super-resolution of Unregistered Multitemporal images (ESA PROBA-V challenge)
* [3DWDSRNet](https://github.com/frandorr/3DWDSRNet) -> code to reproduce Satellite Image Multi-Frame Super Resolution (MISR) Using 3D Wide-Activation Neural Networks
* [RAMS](https://github.com/EscVM/RAMS) -> Official TensorFlow code for paper Multi-Image Super Resolution of Remotely Sensed Images Using Residual Attention Deep Neural Networks
* [TR-MISR](https://github.com/Suanmd/TR-MISR) ->  Transformer-based MISR framework for the the PROBA-V super-resolution challenge
* [HighRes-net](https://github.com/ElementAI/HighRes-net) -> Pytorch implementation of HighRes-net, a neural network for multi-frame super-resolution, trained and tested on the European Space Agency’s Kelvin competition
* [ProbaVref](https://github.com/centreborelli/ProbaVref) -> Repurposing the Proba-V challenge for reference-aware super resolution
* [The missing ingredient in deep multi-temporal satellite image super-resolution](https://towardsdatascience.com/the-missing-ingredient-in-deep-multi-temporal-satellite-image-super-resolution-78cac0f063d9) -> Permutation invariance harnesses the power of ensembles in a single model, with repo [piunet](https://github.com/diegovalsesia/piunet)

## Image-to-image translation
Translate images e.g. from SAR to RGB.
* [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/) -> how to develop a Pix2Pix model for translating satellite photographs to Google map images. A good intro to GANS
* [SAR to RGB Translation using CycleGAN](https://www.esri.com/arcgis-blog/products/api-python/imagery/sar-to-rgb-translation-using-cyclegan/) -> uses a CycleGAN model in the ArcGIS API for Python
* [A growing problem of ‘deepfake geography’: How AI falsifies satellite images](https://www.washington.edu/news/2021/04/21/a-growing-problem-of-deepfake-geography-how-ai-falsifies-satellite-images/)
* [Kaggle Pix2Pix Maps](https://www.kaggle.com/alincijov/pix2pix-maps) -> dataset for pix2pix to take a google map satellite photo and build a street map
* [guided-deep-decoder](https://github.com/tuezato/guided-deep-decoder) -> With guided deep decoder, you can solve different image pair fusion problems, allowing super-resolution, pansharpening or denoising
* [hackathon-ci-2020](https://github.com/paulaharder/hackathon-ci-2020) -> generate nighttime imagery from infrared observations
* [satellite-to-satellite-translation](https://github.com/anonymous-ai-for-earth/satellite-to-satellite-translation) -> VAE-GAN architecture for unsupervised image-to-image translation with shared spectral reconstruction loss. Model is trained on GOES-16/17 and Himawari-8 L1B data
* [Pytorch implementation of UNet for converting aerial satellite images into google maps kinda images](https://github.com/greed2411/unet_pytorch)
* [Seamless-Satellite-image-Synthesis](https://github.com/Misaliet/Seamless-Satellite-image-Synthesis) -> generate abitrarily large RGB images from a map
* [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/) -> article on machinelearningmastery.com
* [Satellite-Imagery-to-Map-Translation-using-Pix2Pix-GAN-framework](https://github.com/anh-nn01/Satellite-Imagery-to-Map-Translation-using-Pix2Pix-GAN-framework)

## GANS
* [Anomaly Detection on Mars using a GAN](https://omdena.com/projects/anomaly-detection-mars/)
* [Using Generative Adversarial Networks to Address Scarcity of Geospatial Training Data](https://medium.com/radiant-earth-insights/using-generative-adversarial-networks-to-address-scarcity-of-geospatial-training-data-e61cacec986e) -> GAN perform better than CNN in segmenting land cover classes outside of the training dataset (article, no code)
* [Building-A-Nets](https://github.com/lixiang-ucas/Building-A-Nets) -> robust building extraction from high-resolution remote sensing images with adversarial networks
* [GANmapper](https://github.com/ualsg/GANmapper) -> a building footprint generator using Generative Adversarial Networks
* [CSA-CDGAN](https://github.com/wangle53/CSA-CDGAN) -> CSA-CDGAN: Channel Self-Attention Based Generative Adversarial Network for Change Detection of Remote Sensing Images
* [DSGAN](https://github.com/lzhengchun/DSGAN) -> a conditinal GAN for dynamic precipitation downscaling

## Autoencoders, dimensionality reduction, image embeddings & similarity search
* [Autoencoders & their Application in Remote Sensing](https://towardsdatascience.com/autoencoders-their-application-in-remote-sensing-95f6e2bc88f) -> intro article and example use case applied to SAR data for land classification
* [LEt-SNE](https://github.com/meghshukla/LEt-SNE) -> Dimensionality Reduction and visualization technique that compensates for the curse of dimensionality
* [AutoEncoders for Land Cover Classification of Hyperspectral Images](https://towardsdatascience.com/autoencoders-for-land-cover-classification-of-hyperspectral-images-part-1-c3c847ebc69b) -> An autoencoder nerual net is used to reduce 103 band data to 60 features (dimensionality reduction), keras. Also read [part 2](https://syamkakarla.medium.com/auto-encoders-for-land-cover-classification-in-hyperspectral-images-part-2-f8978d443d6d) which implements K-NNC, SVM and Gradient Boosting
* [Image-Similarity-Search](https://github.com/spaceml-org/Image-Similarity-Search) -> an app that helps perform super fast image retrieval on PyTorch models for better embedding space interpretability
* [Interactive-TSNE](https://github.com/spaceml-org/Interactive-TSNE) -> a tool that provides a way to visually view a PyTorch model's feature representation for better embedding space interpretability
* [Fine tuning CLIP with Remote Sensing (Satellite) images and captions](https://huggingface.co/blog/fine-tune-clip-rsicd) -> fine tuning CLIP on the [RSICD](https://github.com/201528014227051/RSICD_optimal) image captioning dataset, to enable querying large catalogues in natural language. With [repo](https://github.com/arampacha/CLIP-rsicd)
* [How Airbus Detects Anomalies in ISS Telemetry Data Using TFX](https://blog.tensorflow.org/2020/04/how-airbus-detects-anomalies-iss-telemetry-data-tfx.html) -> uses an autoencoder
* [RoofNet](https://github.com/ultysim/RoofNet) -> identify roof age using historical satellite images to lower the customer acquisition cost for new solar installations. Uses a VAE: Variational Autoencoder
* [Visual search over billions of aerial and satellite images](https://arxiv.org/abs/2002.02624) -> implemented [at Descartes labs](https://blog.descarteslabs.com/geovisual-search-for-rapid-generation-of-annotated-datasets)
* [Demo_AHCL_for_TGRS2022](https://github.com/weiweisong415/Demo_AHCL_for_TGRS2022) -> The code implementation of "Asymmetric Hash Code Learning" (AHCL) for remote sensing image retreival, which was accepted by IEEE Trans. Geosci. Remote Sens. 2022

## Few/one/zero/low shot learning
This is a class of techniques which attempt to make predictions for classes with few, one or even zero examples provided during training. In zero shot learning (ZSL) the model is assisted by the provision of auxiliary information which typically consists of descriptions/semantic attributes/word embeddings for both the seen and unseen classes at train time ([ref](https://learnopencv.com/zero-shot-learning-an-introduction/)). These approaches are particularly relevant to remote sensing, where there may be many examples of common classes, but few or even zero examples for other classes of interest.
* [Unseen Land Cover Classification from High-Resolution Orthophotos Using Integration of Zero-Shot Learning and Convolutional Neural Networks](https://www.mdpi.com/2072-4292/12/10/1676)
* [FSODM](https://github.com/lixiang-ucas/FSODM) -> Official Code for paper "Few-shot Object Detection on Remote Sensing Images" on [arxiv](https://arxiv.org/abs/2006.07826)
* [Few-Shot Classification of Aerial Scene Images via Meta-Learning](https://www.mdpi.com/2072-4292/13/1/108/htm) -> 2020 publication, a classification model that can quickly adapt to unseen categories using only a few labeled samples

## Self-supervised & contrastive learning
The terms self-supervised, semi-supervised, un-supervised, contrastive learning & SSL describe techniques using un-labelled data. In general, the more classical techniques such as k-means classification or PCA are referred to as unsupervised, whilst newer techniques using CNN feature extraction or autoencoders are referred to as self-supervised. [Yann LeCun](https://braindump.jethro.dev/posts/lecun_cake_analogy/) has described self-supervised/unsupervised learning as the 'base of the cake': *If we think of our brain as a cake, then the cake base is unsupervised learning. The machine predicts any part of its input for any observed part, all without the use of labelled data. Supervised learning forms the icing on the cake, and reinforcement learning is the cherry on top.*
* [Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data](https://devblog.pytorchlightning.ai/seasonal-contrast-transferable-visual-representations-for-remote-sensing-73a17863ed07) -> Seasonal Contrast (SeCo) is an effective pipeline to leverage unlabeled data for in-domain pre-training of remote sensing representations. Models trained with SeCo achieve better performance than their ImageNet pre-trained counterparts and state-of-the-art self-supervised learning methods on multiple downstream tasks. [paper](https://arxiv.org/abs/2103.16607) and [repo](https://github.com/ElementAI/seasonal-contrast)
* [Train SimSiam on Satellite Images](https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html) using [Lightly](https://docs.lightly.ai/index.html) to generate embeddings that can be used for data exploration and understanding
* [Unsupervised Learning for Land Cover Classification in Satellite Imagery](https://omdena.com/blog/land-cover-classification/)
* [Tile2Vec: Unsupervised representation learning for spatially distributed data](https://ermongroup.github.io/blog/tile2vec/)
* [Contrastive Sensor Fusion](https://github.com/descarteslabs/contrastive_sensor_fusion) -> Code implementing Contrastive Sensor Fusion, an approach for unsupervised learning of multi-sensor representations targeted at remote sensing imagery.
* [hyperspectral-autoencoders](https://github.com/lloydwindrim/hyperspectral-autoencoders) -> Tools for training and using unsupervised autoencoders and supervised deep learning classifiers for hyperspectral data, built on tensorflow. Autoencoders are unsupervised neural networks that are useful for a range of applications such as unsupervised feature learning and dimensionality reduction.
* [Sentinel-2 image clustering in python](https://towardsdatascience.com/sentinel-2-image-clustering-in-python-58f7f2c8a7f6)
* [MARTA GANs: Unsupervised Representation Learning for Remote Sensing Image Classification](https://arxiv.org/abs/1612.08879) and [code](https://github.com/BUPTLdy/MARTA-GAN)
* [A generalizable and accessible approach to machine learning with global satellite imagery](https://www.nature.com/articles/s41467-021-24638-z) nature publication -> MOSAIKS is designed to solve an unlimited number of tasks at planet-scale quickly using feature vectors, [with repo](https://github.com/Global-Policy-Lab/mosaiks-paper). Also see [mosaiks-api](https://github.com/calebrob6/mosaiks-api)
* [contrastive-satellite](https://github.com/hakeemtfrank/contrastive-satellite) -> Using contrastive learning to create embeddings from optical EuroSAT Satellite-2 imagery
* [Self-Supervised Learning of Remote Sensing Scene Representations Using Contrastive Multiview Coding](https://arxiv.org/abs/2104.07070) -> arxiv paper and [code](https://github.com/vladan-stojnic/CMC-RSSR)
* [Self-Supervised-Learner by spaceml-org](https://github.com/spaceml-org/Self-Supervised-Learner) -> train a classifier with fewer labeled examples needed using self-supervised learning, example applied to UC Merced land use dataset
* [Tensorflow similarity](https://github.com/tensorflow/similarity) -> offers state-of-the-art algorithms for metric learning and all the necessary components to research, train, evaluate, and serve similarity-based models
* [deepsentinel](https://github.com/Lkruitwagen/deepsentinel) -> a sentinel-1 and -2 self-supervised sensor fusion model for general purpose semantic embedding
* [Semi-supervised learning in satellite image classification](https://medium.com/sentinel-hub/semi-supervised-learning-in-satellite-image-classification-e0874a76fc61) -> experimenting with MixMatch and the EuroSAT data set
* [contrastive_SSL_ship_detection](https://github.com/alina2204/contrastive_SSL_ship_detection) -> Contrastive self supervised learning for ship detection in Sentinel 2 images
* [Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning](https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection) with [arxiv paper](https://arxiv.org/abs/2107.08369)
* [geography-aware-ssl](https://github.com/sustainlab-group/geography-aware-ssl) -> uses spatially aligned images over time to construct temporal positive pairs in contrastive learning and geo-location to design pre-text tasks
* [CNN-Supervised Classification](https://github.com/geojames/CNN-Supervised-Classification) -> Python code for self-supervised classification of remotely sensed imagery - part of the Deep Riverscapes project

## Active learning
Supervised deep learning techniques typically require a huge number of annotated/labelled examples to provide a training dataset. However labelling at scale take significant time, expertise and resources. Active learning techniques aim to reduce the total amount of annotation that needs to be performed by selecting the most useful images to label from a large pool of unlabelled examples, thus reducing the time to generate training datasets. These processes may be referred to as [Human-in-the-Loop Machine Learning](https://medium.com/pytorch/https-medium-com-robert-munro-active-learning-with-pytorch-2f3ee8ebec)
* [Active learning for object detection in high-resolution satellite images](https://arxiv.org/abs/2101.02480) -> arxiv paper
* [AIDE V2 - Tools for detecting wildlife in aerial images using active learning](https://github.com/microsoft/aerial_wildlife_detection)
* [AstronomicAL](https://github.com/grant-m-s/AstronomicAL) -> An interactive dashboard for visualisation, integration and classification of data using Active Learning
* Read about [active learning on the lightly platform](https://docs.lightly.ai/getting_started/active_learning.html) and [in label-studio](https://labelstud.io/guide/ml.html#Active-Learning)
* [Active-Labeler by spaceml-org](https://github.com/spaceml-org/Active-Labeler) -> a CLI Tool that facilitates labeling datasets with just a SINGLE line of code
* [Labelling platform for Mapping Africa active learning project](https://github.com/agroimpacts/labeller)

## Mixed data learning
These techniques combine multiple data types, e.g. imagery and text data.
* [Predicting the locations of traffic accidents with satellite imagery and convolutional neural networks](https://towardsdatascience.com/teaching-a-neural-network-to-see-roads-74bff240c3e5) -> Combining satellite imagery and structured data to predict the location of traffic accidents with a neural network of neural networks, with [repo](https://github.com/L-Lewis/Predicting-traffic-accidents-CNN)
* [Multi-Input Deep Neural Networks with PyTorch-Lightning - Combine Image and Tabular Data](https://rosenfelder.ai/multi-input-neural-network-pytorch/) -> excellent intro article using pytorch, not actually applied to satellite data but to real estate data
* [Joint Learning from Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps](https://arxiv.org/abs/1705.06057) -> fusion based architectures and coarse-to-fine segmentation to include the OpenStreetMap layer into multispectral-based deep fully convolutional networks, arxiv paper
* [Inferring High-Resolution Traffic Accident Risk Maps Based on Satellite Imagery and GPS Trajectories](https://openaccess.thecvf.com/content/ICCV2021/html/He_Inferring_High-Resolution_Traffic_Accident_Risk_Maps_Based_on_Satellite_Imagery_ICCV_2021_paper.html) -> input satellite imagery, GPS trajectories, road maps and the history of accidents to generate high-resolution (5 meters) accident risk maps
* [Composing Decision Forest and Neural Network models](https://www.tensorflow.org/decision_forests/tutorials/model_composition_colab) tensorflow documentation
* [pyimagesearch article on mixed-data](https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/)
* [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep) -> A flexible package for multimodal-deep-learning to combine tabular data with text and images using Wide and Deep models in Pytorch

## Image Captioning
* [remote-sensing-image-caption](https://github.com/TalentBoy2333/remote-sensing-image-caption) -> image classification and image caption by PyTorch

## Pansharpening
Image fusion of low res multispectral with high res pan band.
* Several algorithms described [in the ArcGIS docs](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/fundamentals-of-panchromatic-sharpening.htm), with the simplest being taking the mean of the pan and RGB pixel value.
* Does not require DL, classical algos suffice, [see this notebook](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/panchromatic_sharpening.ipynb) and [this kaggle kernel](https://www.kaggle.com/resolut/panchromatic-sharpening)
* [rio-pansharpen](https://github.com/mapbox/rio-pansharpen) -> pansharpening Landsat scenes
* [PSGAN](https://github.com/liuqingjie/PSGAN) -> A Generative Adversarial Network for Remote Sensing Image Pan-sharpening, [arxiv paper](https://arxiv.org/abs/1805.03371)
* [Pansharpening-by-Convolutional-Neural-Network](https://github.com/ThomasWangWeiHong/Pansharpening-by-Convolutional-Neural-Network)
* [PBR_filter](https://github.com/dbuscombe-usgs/PBR_filter) -> {P}ansharpening by {B}ackground {R}emoval algorithm for sharpening RGB images
* [py_pansharpening](https://github.com/codegaj/py_pansharpening) -> multiple algorithms implemented in python

## NVDI - vegetation index
* Simple band math `ndvi = np.true_divide((ir - r), (ir + r))` but challenging due to the size of the imagery.
* [Example notebook local](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/ndvi_calculation.ipynb)
* [Landsat data in cloud optimised (COG) format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article here](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”
* [Seeing Through the Clouds - Predicting Vegetation Indices Using SAR](https://medium.com/descarteslabs-team/seeing-through-the-clouds-34a24f84b599)
* [A walkthrough on calculating NDWI water index for flooded areas](https://towardsdatascience.com/how-to-compute-satellite-image-statistics-and-use-it-in-pandas-81864a489144) -> Derive zonal statistics from Sentinel 2 images using Rasterio and Geopandas

## General image quality
* Convolutional autoencoder network can be employed to image denoising, [read about this on the Keras blog](https://blog.keras.io/building-autoencoders-in-keras.html)
* [jitter-compensation](https://github.com/caiya55/jitter-compensation) -> Remote Sensing Image Jitter Detection and Compensation Using CNN
* [DeblurGANv2](https://github.com/VITA-Group/DeblurGANv2) -> Deblurring (Orders-of-Magnitude) Faster and Better
* [image-quality-assessment](https://github.com/idealo/image-quality-assessment) -> CNN to predict the aesthetic and technical quality of images
* [Convolutional autoencoder for image denoising](https://keras.io/examples/vision/autoencoder/) -> keras guide
* [piq](https://github.com/photosynthesis-team/piq) -> a collection of measures and metrics for image quality assessment
* [FFA-Net](https://github.com/zhilin007/FFA-Net) -> Feature Fusion Attention Network for Single Image Dehazing
* [DeepCalib](https://github.com/alexvbogdan/DeepCalib) -> A Deep Learning Approach for Automatic Intrinsic Calibration of Wide Field-of-View Cameras
* [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity) -> LPIPS is a perceptual metric which aims to overcome the limitations of traditional metrics such as PSNR & SSIM, to better represent the features the human eye picks up on

## Image registration
Image registration is the process of transforming different sets of data into one coordinate system. Typical use is overlapping images taken at different times or with different cameras.
* [Wikipedia article on registration](https://en.wikipedia.org/wiki/Image_registration) -> register for change detection or [image stitching](https://mono.software/2018/03/14/Image-stitching/)
* [Phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) is used to estimate the XY translation between two images with sub-pixel accuracy. Can be used for accurate registration of low resolution imagery onto high resolution imagery, or to register a [sub-image on a full image](https://www.mathworks.com/help/images/registering-an-image-using-normalized-cross-correlation.html) -> Unlike many spatial-domain algorithms, the phase correlation method is resilient to noise, occlusions, and other defects. With [additional pre-processing](https://scikit-image.org/docs/dev/auto_examples/registration/plot_register_rotation.html) image rotation and scale changes can also be calculated.
* [imreg_dft](https://github.com/matejak/imreg_dft) -> Image registration using discrete Fourier transform. Given two images it can calculate the difference between scale, rotation and position of imaged features. Used by the [up42 co-registration service](https://up42.com/marketplace/blocks/processing/up42-coregistration)
* [arosics](https://danschef.git-pages.gfz-potsdam.de/arosics/doc/about.html) -> Perform automatic subpixel co-registration of two satellite image datasets using phase-correlation, XY translations only.
* [cnn-registration](https://github.com/yzhq97/cnn-registration) -> A image registration method using convolutional neural network features written in Python2, Tensorflow 1.5
* [Detecting Ground Control Points via Convolutional Neural Network for Stereo Matching](https://arxiv.org/abs/1605.02289) -> code?
* [Image Registration: From SIFT to Deep Learning](https://www.sicara.ai/blog/2019-07-16-image-registration-deep-learning) -> background reading on has the state of the art has evolved from OpenCV to Neural Networks
* [ImageCoregistration](https://github.com/ily-R/ImageCoregistration) -> Image registration with openCV using sift and RANSAC
* [mapalignment](https://github.com/Lydorn/mapalignment) -> Aligning and Updating Cadaster Maps with Remote Sensing Images
* [CVPR21-Deep-Lucas-Kanade-Homography](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography) -> deep learning pipeline to accurately align challenging multimodality images. The method is based on traditional Lucas-Kanade algorithm with feature maps extracted by deep neural networks.
* [eolearn](https://eo-learn.readthedocs.io/en/latest/_modules/eolearn/coregistration/coregistration.html) implements phase correlation, feature matching and [ECC](https://learnopencv.com/image-alignment-ecc-in-opencv-c-python/)
* RStoolbox supports [Image to Image Co-Registration based on Mutual Information](https://bleutner.github.io/RStoolbox/rstbx-docu/coregisterImages.html)
* [Reprojecting the Perseverance landing footage onto satellite imagery](https://matthewearl.github.io/2021/03/06/mars2020-reproject/)
* Kornia provides [image registration by gradient decent](https://kornia-tutorials.readthedocs.io/en/latest/image_registration.html)
* [LoFTR](https://github.com/zju3dv/LoFTR) -> Detector-Free Local Feature Matching with Transformers. Good performance matching satellite image pairs, tryout the web demo on your data
* [Multi-Scale-Unsupervised-Framework-MSUF](https://github.com/yeyuanxin110/Multi-Scale-Unsupervised-Framework-MSUF) -> This is an implementation of the paper: 'A Multi-Scale Framework with Unsupervised Learning for Remote Sensing Image Registration'

## Multi-sensor/multi-modal fusion
Overlaps with image registration, above
* [CropTypeMapping](https://github.com/ellaampy/CropTypeMapping) -> Crop type mapping from optical and radar (Sentinel-1&2) time series using attention-based deep learning
* [Multimodal-Remote-Sensing-Toolkit](https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit) -> uses Hyperspectral and LiDAR Data

## Object tracking
* [Object Tracking in Satellite Videos Based on a Multi-Frame Optical Flow Tracker](https://arxiv.org/abs/1804.09323) arxiv paper

## Terrain mapping, Lidar & DEMs
Measure surface contours.
* [Wikipedia DEM article](https://en.wikipedia.org/wiki/Digital_elevation_model) and [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) article
* [Intro to depth from stereo](https://github.com/IntelRealSense/librealsense/blob/master/doc/depth-from-stereo.md)
* Map terrain from stereo images to produce a digital elevation model (DEM) -> high resolution & paired images required, typically 0.3 m, e.g. [Worldview](https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/37/DG-WV2ELEVACCRCY-WP.pdf)
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
* [ResDepth](https://github.com/stuckerc/ResDepth) -> A Deep Prior For 3D Reconstruction From High-resolution Satellite Images
* [overhead-geopose-challenge](https://www.drivendata.org/competitions/78/overhead-geopose-challenge/) -> competition to build computer vision algorithms that can effectively model the height and pose of ground objects for monocular satellite images taken from oblique angles. Blog post [MEET THE WINNERS OF THE OVERHEAD GEOPOSE CHALLENGE](https://www.drivendata.co/blog/overhead-geopose-challenge-winners/)
* [cars](https://github.com/CNES/cars) -> a dedicated and open source 3D tool to produce Digital Surface Models from satellite imaging by photogrammetry. This Multiview stereo pipeline is intended for massive DSM production with a robust and performant design
* [ImageToDEM](https://github.com/Panagiotou/ImageToDEM) -> Generating Elevation Surface from a Single RGB Remotely Sensed Image Using a U-Net for generator and a PatchGAN for the discriminator
* [IMELE](https://github.com/speed8928/IMELE) -> Building Height Estimation from Single-View Aerial Imagery
* [ridges](https://github.com/mikeskaug/ridges) -> deep semantic segmentation model for identifying ridges in topography
* [planet_tools](https://github.com/disbr007/planet_tools) -> Selection of imagery from Planet API for creation of stereo elevation models

## Thermal Infrared
* [The World Needs (a lot) More Thermal Infrared Data from Space](https://towardsdatascience.com/the-world-needs-a-lot-more-thermal-infrared-data-from-space-dbbba389be8a)
* [IR2VI thermal-to-visible image translation framework based on GANs](https://arxiv.org/abs/1806.09565) with [code](https://github.com/linty5/IR2VI_CycleGAN)
* [The finest resolution urban outdoor heat exposure maps in major US cities](https://xiaojianggis.github.io/heatexpo/) -> urban microclimate modeling based on high resolution 3D urban models and meteorological data makes it possible to examine how people are exposed to heat stress at a fine spatio-temporal level.
* [Object_Classification_in_Thermal_Images](https://www.researchgate.net/publication/328400392_Object_Classification_in_Thermal_Images_using_Convolutional_Neural_Networks_for_Search_and_Rescue_Missions_with_Unmanned_Aerial_Systems) -> classification accuracy was improved by adding the object size as a feature directly within the CNN
* [Thermal imaging with satellites](https://chrieke.medium.com/thermal-imaging-with-satellites-34f381856dd1) blog post by Christoph Rieke
* [Object Detection on Thermal Images](https://medium.com/@joehoeller/object-detection-on-thermal-images-f9526237686a) -> using YOLO-v3 and applied to a terrestrial dataset from FLIR, the article offers some usful insights into the model training
* [Fire alerts service by Descartes Labs](https://www.businessinsider.com/how-descartes-labs-leveraging-artificial-intelligence-fight-wildfires-2019-12)
* [Background Invariant Classification on Infrared Imagery by Data Efficient Training and Reducing Bias in CNNs](https://arxiv.org/abs/2201.09144)

## SAR
* [awesome-sar](https://github.com/RadarCODE/awesome-sar) -> A curated list of awesome Synthetic Aperture Radar (SAR) software, libraries, and resources
* [Removing speckle noise from Sentinel-1 SAR using a CNN](https://medium.com/upstream/denoising-sentinel-1-radar-images-5f764faffb3e)
* A dataset which is specifically made for deep learning on SAR and optical imagery is the SEN1-2 dataset, which contains corresponding patch pairs of Sentinel 1 (VV) and 2 (RGB) data. It is the largest manually curated dataset of S1 and S2 products, with corresponding labels for land use/land cover mapping, SAR-optical fusion, segmentation and classification tasks. Data: https://mediatum.ub.tum.de/1474000
* [so2sat on Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/so2sat) -> So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and the corresponding local climate zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions of the world.
* [You do not need clean images for SAR despeckling with deep learning](https://towardsdatascience.com/you-do-not-need-clean-images-for-sar-despeckling-with-deep-learning-fe9c44350b69) -> How Speckle2Void learned to stop worrying and love the noise
* [PySAR - InSAR (Interferometric Synthetic Aperture Radar) timeseries analysis in python](https://github.com/hfattahi/PySAR)
* [Synthetic Aperture Radar (SAR) Analysis With Clarifai](https://www.clarifai.com/blog/synthetic-aperture-radar-sar-analysis-with-clarifai)
* [Labeled SAR imagery dataset of ten geophysical phenomena from Sentinel-1 wave mode](https://www.seanoe.org/data/00456/56796/) consists of more than 37,000 SAR vignettes divided into ten defined geophysical categories
* [Deep Learning and SAR Applications](https://towardsdatascience.com/deep-learning-and-sar-applications-81ba1a319def)
* [Implementing an Ensemble Convolutional Neural Network on Sentinel-1 Synthetic Aperture Radar data and Sentinel-3 Radiometric data for the detecting of forest fires](https://github.com/aalling93/ECNN-on-SAR-data-and-Radiometry-data)
* [s1_parking_occupancy](https://github.com/sdrdis/s1_parking_occupancy) -> Source code for PARKING OCCUPANCY ESTIMATION ON SENTINEL-1 IMAGES, ISPRS 2020
* [Experiments on Flood Segmentation on Sentinel-1 SAR Imagery with Cyclical Pseudo Labeling and Noisy Student Training](https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection)
* [SpaceNet_SAR_Buildings_Solutions](https://github.com/SpaceNetChallenge/SpaceNet_SAR_Buildings_Solutions) -> The winning solutions for the SpaceNet 6 Challenge
* [Mapping and monitoring of infrastructure in desert regions with Sentinel-1](https://github.com/ESA-PhiLab/infrastructure)
* [xView3](https://iuu.xview.us/) is a competition to detect dark vessels using computer vision and global SAR satellite imagery. [The-First-Place-Solution](https://github.com/BloodAxe/xView3-The-First-Place-Solution) and [second place solution](https://github.com/selimsef/xview3_solution). [sar_vessel_detect](https://github.com/allenai/sar_vessel_detect)
* [Winners of the STAC Overflow: Map Floodwater from Radar Imagery competition](https://github.com/drivendataorg/stac-overflow)
* [deSpeckNet-TF-GEE](https://github.com/adugnag/deSpeckNet-TF-GEE) -> implementation of the paper 'deSpeckNet: Generalizing Deep Learning Based SAR Image Despeckling'
* [cnn_sar_image_classification](https://github.com/diogosens/cnn_sar_image_classification) -> CNN for classifying SAR images of the Amazon Rainforest

## Neural nets in space
Processing on board a satellite allows less data to be downlinked. e.g. super-resolution image might take 8 images to generate, then a single image is downlinked. Other applications include cloud detection and collision avoidance.
* [Lockheed Martin and USC to Launch Jetson-Based Nanosatellite for Scientific Research Into Orbit - Aug 2020](https://news.developer.nvidia.com/lockheed-martin-usc-jetson-nanosatellite/) - One app that will run on the GPU-accelerated satellite is SuperRes, an AI-based application developed by Lockheed Martin, that can automatically enhance the quality of an image.
* [Intel to place movidius in orbit to filter images of clouds at source - Oct 2020](https://techcrunch.com/2020/10/20/intel-is-providing-the-smarts-for-the-first-satellite-with-local-ai-processing-on-board/) - Getting rid of these images before they’re even transmitted means that the satellite can actually realize a bandwidth savings of up to 30%
* Whilst not involving neural nets the [PyCubed](https://www.notion.so/PyCubed-4cbfac7e9b684852a2ab2193bd485c4d) project gets a mention here as it is putting python on space hardware such as the [V-R3x](https://www.nasa.gov/ames/v-r3x)
* [WorldFloods](https://watchers.news/2021/07/11/worldfloods-ai-pioneered-at-oxford-for-global-flood-mapping-launches-into-space/) will pioneer the detection of global flood events from space, launched on June 30, 2021. [This paper](https://arxiv.org/pdf/1910.03019.pdf) describes the model which is run on Intel Movidius Myriad2 hardware capable of processing a 12 MP image in less than a minute
* [How AI and machine learning can support spacecraft docking](https://towardsdatascience.com/deep-learning-in-space-964566f09dcd) with [repo](https://github.com/nevers/space-dl) uwing Yolov3
* [exo-space](https://www.exo-space.com/) -> startup with plans to release an AI hardware addon for satellites
* [Sony’s Spresense microcontroller board is going to space](https://developer.sony.com/posts/the-spresense-microcontroller-board-launched-in-space/) -> vision applications include cloud detection, [more details here](https://www.hackster.io/dhruvsheth_/to-space-and-beyond-with-edgeimpulse-and-sony-s-spresense-d87a70)
* [Ororatech Early Detection of Wildfires From Space](https://blogs.nvidia.com/blog/2021/09/30/ororatech-wildfires-from-space/) -> OroraTech is launching its own AI nanosatellites with the NVIDIA Jetson Xavier NX system onboard

# ML best practice
This section includes tips and ideas I have picked up from other practitioners including [ai-fast-track](https://github.com/ai-fast-track), [FraPochetti](https://github.com/FraPochetti) & the IceVision community
* Almost all imagery data on the internet is in RGB format, and common techniques designed for working with this 3 band imagery may fail or need significant adaptation to work with multiband data (e.g. 13-band Sentinel 2)
* In general, classification and object detection models are created using transfer learning, where the majority of the weights are not updated in training but have been pre computed using standard vision datasets such as ImageNet
* Since satellite images are typically very large, it is common to tile them before processing. Alternatively checkout [Fully Convolutional Image Classification on Arbitrary Sized Image](https://learnopencv.com/fully-convolutional-image-classification-on-arbitrary-sized-image/) -> TLDR replace the fully-connected layer with a convolution-layer
* Where you have small sample sizes, e.g. for a small object class which may be under represented in your training dataset, use image augmentation
* In general, larger models will outperform smaller models, particularly on challenging tasks such as detecting small objetcs
* If model performance in unsatisfactory, try to increase your dataset size before switching to another model architecture
* In training, whenever possible increase the batch size, as small batch sizes produce poor normalization statistics
* The vast majority of the literature uses supervised learning with the requirement for large volumes of annotated data, which is a bottleneck to development and deployment. We are just starting to see self-supervised approaches applied to remote sensing data
* [4-ways-to-improve-class-imbalance](https://towardsdatascience.com/4-ways-to-improve-class-imbalance-for-image-data-9adec8f390f1) discusses the pros and cons of several rebalancing techniques, applied to an aerial dataset. Reason to read: models can reach an accuracy ceiling where majority classes are easily predicted but minority classes poorly predicted. Overall model accuracy may not improve until steps are taken to account for class imbalance.
* For general guidance on dataset size see [this issue](https://github.com/ultralytics/yolov5/issues/3306)
* Read [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) by Andrej Karpathy
* [Seven steps towards a satellite imagery dataset](https://omdena.com/blog/satellite-imagery-dataset/)
* [Implementing Transfer Learning from RGB to Multi-channel Imagery](https://towardsdatascience.com/implementing-transfer-learning-from-rgb-to-multi-channel-imagery-f87924679166) -> takes a resnet50 model pre-trained on an input of 224x224 pixels with 3 channels (RGB) and updates it for a new input of 480x400 pixels and 15 channels (12 new + RGB) using keras
* [How to implement augmentations for Multispectral Satellite Images Segmentation using Fastai-v2 and Albumentations](https://towardsdatascience.com/how-to-implement-augmentations-for-multispectral-satellite-images-segmentation-using-fastai-v2-and-ea3965736d1)
* [Principal Component Analysis: In-depth understanding through image visualization](https://towardsdatascience.com/principal-component-analysis-in-depth-understanding-through-image-visualization-892922f77d9f) applied to Landsat TM images, [with repo](https://github.com/Skumarr53/Principal-Component-Analysis-testing-on-Image-data)
* [Leveraging Geolocation Data for Machine Learning: Essential Techniques](https://towardsdatascience.com/leveraging-geolocation-data-for-machine-learning-essential-techniques-192ce3a969bc) -> A Gentle Guide to Feature Engineering and Visualization with Geospatial data, in Plain English
* [3 Tips to Optimize Your Machine Learning Project for Data Labeling](https://www.azavea.com/blog/2020/07/21/3-tips-to-optimize-your-machine-learning-project-for-data-labeling/)
* [Image Classification Labeling: Single Class versus Multiple Class Projects](https://www.azavea.com/blog/2020/06/08/image-classification-labeling-single-class-versus-multiple-class-projects/)
* [Labeling Satellite Imagery for Machine Learning](https://www.azavea.com/blog/2020/03/24/labeling-satellite-imagery-for-machine-learning/)
* [Image Augmentations for Aerial Datasets](https://blog.roboflow.com/image-augmentations-for-aerial-datasets/)
* [Leveraging satellite imagery for machine learning computer vision applications](https://medium.com/artefact-engineering-and-data-science/leveraging-satellite-imagery-for-machine-learning-computer-vision-applications-d22143f72d94)
* [Best Practices for Preparing and Augmenting Image Data for CNNs](https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/)
* [Using TensorBoard While Training Land Cover Models with Satellite Imagery](https://up42.com/blog/tech/using-tensorboard-while-training-land-cover-models-with-satellite-imagery)
* [An Overview of Model Compression Techniques for Deep Learning in Space](https://medium.com/gsi-technology/an-overview-of-model-compression-techniques-for-deep-learning-in-space-3fd8d4ce84e5)
* [Visualise Embeddings with Tensorboard](https://medium.com/gsi-technology/visualising-embeddings-using-t-sne-8fd4e31b56e2) -> also checkout the [Tensorflow Embedding Projector](https://projector.tensorflow.org/)
* [Introduction to Satellite Image Augmentation with Generative Adversarial Networks - video](https://geoawesomeness.com/introduction-to-satellite-image-augmentation-with-generative-adversarial-networks/)
* [Use Gradio and W&B together to monitor training and view predictions](https://wandb.ai/abidlabs/your-test-project/reports/How-Gradio-and-W-B-Work-Beautifully-Together---Vmlldzo4MTk0MzI)
* [Every important satellite imagery analysis project is challenging, but here are ten straightforward steps to get started](https://medium.com/futuring-peace/how-to-change-the-world-from-space-d4186e76da43)
* [Challenges with SpaceNet 4 off-nadir satellite imagery: Look angle and target azimuth angle](https://medium.com/the-downlinq/challenges-with-spacenet-4-off-nadir-satellite-imagery-look-angle-and-target-azimuth-angle-2402bc4c3cf6) -> building prediction in images taken at nearly identical look angles — for example, 29 and 30 degrees — produced radically different performance scores.
* [How not to test your deep learning algorithm?](https://medium.com/earthcube-stories/how-not-to-test-your-deep-learning-algorithm-c435993f873b) - bad ideas to avoid
* [AI products and remote sensing: yes, it is hard and yes, you need a good infra](https://medium.com/earthcube-stories/ai-products-and-remote-sensing-yes-it-is-hard-and-yes-you-need-a-good-infra-4b5d6cf822f1) -> advice on building an in-house data annotation service
* [Boosting object detection performance through ensembling on satellite imagery](https://medium.com/earthcube-stories/boosting-object-detection-performance-through-ensembling-on-satellite-imagery-949e891dfb28)
* [How to use deep learning on satellite imagery — Playing with the loss function](https://medium.com/earthcube-stories/techsecret-how-to-use-deep-learning-on-satellite-imagery-episode-1-playing-with-the-loss-8fc05c90a63a)
* [On the importance of proper data handling](https://medium.com/picterra/on-the-importance-of-proper-data-handling-part-1-b78e4bfd9a7c)
* [Generate SSD anchor box aspect ratios using k-means clustering](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/generate_ssd_anchor_box_aspect_ratios_using_k_means_clustering.ipynb) -> tutorial showing how to discover a set of aspect ratios that are custom-fit for your dataset, applied to tensorflow object detection

# ML metrics
A number of metrics are common to all model types (but can have slightly different meanings in contexts such as object detection), whilst other metrics are very specific to particular classes of model. The correct choice of metric is particularly critical for imbalanced dataset problems, e.g. object detection
* TP = true positive, FP = false positive, TN = true negative, FN = false negative
* `Precision` is the % of correct positive predictions, calculated as `precision = TP/(TP+FP)`
* `Recall` or **true positive rate** (TPR), is the % of true positives captured by the model, calculated as `recall = TP/(TP+FN)`. Note that FN is not possible in object detection, so recall is not appropriate.
* The `F1 score` (also called the F-score or the F-measure) is the harmonic mean of precision and recall, calculated as `F1 = 2*(precision * recall)/(precision + recall)`. It conveys the balance between the precision and the recall. [Ref](https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can-use/)
* The **false positive rate** (FPR), calculated as `FPR = FP/(FP+TN)` is often plotted against recall/TPR in an [ROC curve](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc) which shows how the TPR/FPR tradeoff varies with classification threshold. Lowering the classification threshold returns more true positives, but also more false positives. Note that since FN is not possible in object detection, ROC curves are not appropriate.
* Precision-vs-recall curves visualise the tradeoff between making false positives and false negatives
* [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification) is the most commonly used metric in 'real life' but can be a highly misleading metric for imbalanced data sets.
* `IoU` is an object detection specific metric, being the average intersect over union of prediction and ground truth bounding boxes for a given confidence threshold
* `mAP@0.5` is another object detection specific metric, being the mean value of the average precision for each class. `@0.5` sets a threshold for how much of the predicted bounding box overlaps the ground truth bounding box, i.e. "minimum 50% overlap"
* For more comprehensive definitions checkout [Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

# Datasets
This section contains a short list of datasets relevant to deep learning, particularly those which come up regularly in the literature. For a more comprehensive list of datasets checkout [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets) and review the [long list of satellite missions with example imagery](https://www.satimagingcorp.com/satellite-sensors/)

**Warning** satellite image files can be LARGE, even a small data set may comprise 50 GB of imagery

## Sentinel
* As part of the [EU Copernicus program](https://en.wikipedia.org/wiki/Copernicus_Programme), multiple Sentinel satellites are capturing imagery -> see [wikipedia](https://en.wikipedia.org/wiki/Copernicus_Programme#Sentinel_missions).
* 13 bands, Spatial resolution of 10 m, 20 m and 60 m, 290 km swath, the temporal resolution is 5 days
* [awesome-sentinel](https://github.com/Fernerkundung/awesome-sentinel) - a curated list of awesome tools, tutorials and APIs related to data from the Copernicus Sentinel Satellites.
* [Sentinel-2 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs/) and [Sentinel-2 L2A 120m Mosaic](https://registry.opendata.aws/sentinel-s2-l2a-mosaic-120/)
* [Open access data on GCP](https://console.cloud.google.com/storage/browser/gcp-public-data-sentinel-2?prefix=tiles%2F31%2FT%2FCJ%2F) 
* Paid access to Sentinel & Landsat data via [sentinel-hub](https://www.sentinel-hub.com/) and [python-api](https://github.com/sentinel-hub/sentinelhub-py)
* [Example loading sentinel data in a notebook](https://github.com/binder-examples/getting-data/blob/master/Sentinel2.ipynb)
* [so2sat on Tensorflow datasets](https://www.tensorflow.org/datasets/catalog/so2sat) - So2Sat LCZ42 is a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by the Sentinel-1 and Sentinel-2 remote sensing satellites, and the corresponding local climate zones (LCZ) label. The dataset is distributed over 42 cities across different continents and cultural regions of the world.
* [bigearthnet](https://www.tensorflow.org/datasets/catalog/bigearthnet) - The BigEarthNet is a new large-scale Sentinel-2 benchmark archive, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels.
* [Jupyter Notebooks for working with Sentinel-5P Level 2 data stored on S3](https://github.com/Sentinel-5P/data-on-s3). The data can be browsed [here](https://meeo-s5p.s3.amazonaws.com/index.html#/?t=catalogs)
* [Sentinel NetCDF data](https://github.com/acgeospatial/Sentinel-5P/blob/master/Sentinel_5P.ipynb)
* [Analyzing Sentinel-2 satellite data in Python with Keras](https://github.com/jensleitloff/CNN-Sentinel)
* [Xarray backend to Copernicus Sentinel-1 satellite data products](https://github.com/bopen/xarray-sentinel)

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
* [eurosat](https://www.tensorflow.org/datasets/catalog/eurosat) - EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples.
* [EuroSAT: Land Use and Land Cover Classification with Sentinel-2](https://github.com/phelber/EuroSAT) -> publication where a CNN achieves a classification accuracy 98.57%
* Repos using fastai [here](https://github.com/shakasom/Deep-Learning-for-Satellite-Imagery) and [here](https://www.luigiselmi.eu/eo/lulc-classification-deeplearning.html)

## PatternNet
* Land use classification dataset with 38 classes and 800 RGB JPG images for each class
* https://sites.google.com/view/zhouwx/dataset?authuser=0
* Publication: [PatternNet: A Benchmark Dataset for Performance Evaluation of Remote Sensing Image Retrieval](https://arxiv.org/abs/1706.03424)

## FAIR1M object detection dataset
* [FAIR1M: A Benchmark Dataset for Fine-grained Object Recognition in High-Resolution Remote Sensing Imagery](https://arxiv.org/abs/2103.05569)
* Download at [gaofen-challenge.com](http://gaofen-challenge.com/)

## DOTA object detection dataset
* https://captain-whu.github.io/DOTA/dataset.html
* A Large-Scale Benchmark and Challenges for Object Detection in Aerial Images
* [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) for loading dataset

## xView object detection dataset
* http://xviewdataset.org/
* One million annotated objects on 30cm imagery
* [The XView Dataset and Baseline Results](https://medium.com/picterra/the-xview-dataset-and-baseline-results-5ab4a1d0f47f) blog post by Picterra
* [Databricks tutorial](https://databricks.com/notebooks/1_data_eng_xview_object_detection.html) demonstrating inference of xView images and using SQL to generate meaningful insights

## AIRS (Aerial Imagery for Roof Segmentation)
* https://www.airs-dataset.com
* Public dataset for roof segmentation from very-high-resolution aerial imagery (7.5cm)
* AIRS dataset covers almost the full area of Christchurch, the largest city in the South Island of New Zealand.
* [Also on Kaggle](https://www.kaggle.com/atilol/aerialimageryforroofsegmentation/version/1)
* [Rooftop-Instance-Segmentation](https://github.com/MasterSkepticista/Rooftop-Instance-Segmentation) -> VGG-16, Instance Segmentation, uses the Airs dataset

## Inria building/not building segmentation dataset
* https://project.inria.fr/aerialimagelabeling/contest/
* RGB GeoTIFF at spatial resolution of 0.3 m
* Data covering Austin, Chicago, Kitsap County, Western & Easter Tyrol, Innsbruck, San Francisco & Vienna

## AICrowd building segmentation dataset
* Dataset release as part of the [mapping-challenge](https://www.aicrowd.com/challenges/mapping-challenge)
* 300x300 pixel RGB images with annotations in COCO format
* Imagery appears to be global but with significant fraction from North America
* Winning solution published by neptune.ai [here](https://github.com/neptune-ai/open-solution-mapping-challenge), achieved precision 0.943 and recall 0.954 using Unet with Resnet.

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

### Kaggle - DSTL segmentation challenge
* https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
* Rating - medium, many good examples (see the Discussion as well as kernels), but as this competition was run a couple of years ago many examples use python 2
* WorldView 3 - 45 satellite images covering 1km x 1km in both 3 (i.e. RGB) and 16-band (400nm - SWIR) images
* 10 Labelled classes include - **Buildings, Road, Trees, Crops, Waterway, Vehicles**
* [Interview with 1st place winner who used segmentation networks](http://blog.kaggle.com/2017/04/26/dstl-satellite-imagery-competition-1st-place-winners-interview-kyle-lee/) - 40+ models, each tweaked for particular target (e.g. roads, trees)
* [Deepsense 4th place solution](https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/)
* [Entry by lopuhin](https://github.com/lopuhin/kaggle-dstl) using UNet with batch-normalization
* [ZF_UNET_224_Pretrained_Model](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model) -> 2nd place solution

### Kaggle - Airbus ship detection Challenge
* https://www.kaggle.com/c/airbus-ship-detection/overview
* Rating - medium, most solutions using deep-learning, many kernels, [good example kernel](https://www.kaggle.com/kmader/baseline-u-net-model-part-1)
* I believe there was a problem with this dataset, which led to many complaints that the competition was ruined
* [Deep Learning for Ship Detection and Segmentation](https://towardsdatascience.com/deep-learning-for-ship-detection-and-segmentation-71d223aca649) -> treated as instance segmentation problem, with [notebook](https://github.com/abhinavsagar/kaggle-notebooks/blob/master/ship_segmentation.ipynb)
* [Lessons Learned from Kaggle’s Airbus Challenge](https://towardsdatascience.com/lessons-learned-from-kaggles-airbus-challenge-252e25c5efac)

### Kaggle - Shipsnet classification dataset
* https://www.kaggle.com/rhammell/ships-in-satellite-imagery -> Classify ships in San Franciso Bay using Planet satellite imagery
* 4000 80x80 RGB images labeled with either a "ship" or "no-ship" classification, 3 meter pixel size
* [shipsnet-detector](https://github.com/rhammell/shipsnet-detector) -> Detect container ships in Planet imagery using machine learning

### Kaggle - Ships in Google Earth
* https://www.kaggle.com/tomluther/ships-in-google-earth
* 794 jpegs showing various sized ships in satellite imagery, annotations in Pascal VOC format for object detection models
* [kaggle-ships-in-Google-Earth-yolov5](https://github.com/robmarkcole/kaggle-ships-in-Google-Earth-yolov5)

### Kaggle - Swimming pool and car detection using satellite imagery
* https://www.kaggle.com/kbhartiya83/swimming-pool-and-car-detection
* 3750 satellite images of residential areas with annotation data for swimming pools and cars
* [Object detection on Satellite Imagery using RetinaNet](https://medium.com/@ije_good/object-detection-on-satellite-imagery-using-retinanet-part-1-training-e589975afbd5)

### Kaggle - Planesnet classification dataset
* https://www.kaggle.com/rhammell/planesnet -> Detect aircraft in Planet satellite image chips
* 20x20 RGB images, the "plane" class includes 8000 images and the "no-plane" class includes 24000 images
* [Dataset repo](https://github.com/rhammell/planesnet) and [planesnet-detector](https://github.com/rhammell/planesnet-detector) demonstrates a small CNN classifier on this dataset

### Kaggle - Draper challenge to place images in order of time
* https://www.kaggle.com/c/draper-satellite-image-chronology/data
* Rating - hard. Not many useful kernels.
* Images are grouped into sets of five, each of which have the same setId. Each image in a set was taken on a different day (but not necessarily at the same time each day). The images for each set cover approximately the same area but are not exactly aligned.
* Kaggle interviews for entrants who [used XGBOOST](http://blog.kaggle.com/2016/09/15/draper-satellite-image-chronology-machine-learning-solution-vicens-gaitan/) and a [hybrid human/ML approach](http://blog.kaggle.com/2016/09/08/draper-satellite-image-chronology-damien-soukhavong/)

### Kaggle - Dubai segmentation
* https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery
* 72 satellite images of Dubai, the UAE, and is segmented into 6 classes
* [dubai-satellite-imagery-segmentation](https://github.com/ayushdabra/dubai-satellite-imagery-segmentation) -> due to the small dataset, image augmentation was used
* [U-Net for Semantic Segmentation on Unbalanced Aerial Imagery](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56) -> using the Dubai dataset

### Kaggle - Deepsat classification challenge
Not satellite but airborne imagery. Each sample image is 28x28 pixels and consists of 4 bands - red, green, blue and near infrared. The training and test labels are one-hot encoded 1x6 vectors. Each image patch is size normalized to 28x28 pixels. Data in `.mat` Matlab format. JPEG?
* [Sat4](https://www.kaggle.com/crawford/deepsat-sat4) 500,000 image patches covering four broad land cover classes - **barren land, trees, grassland and a class that consists of all land cover classes other than the above three**
* [Sat6](https://www.kaggle.com/crawford/deepsat-sat6) 405,000 image patches each of size 28x28 and covering 6 landcover classes - **barren land, trees, grassland, roads, buildings and water bodies.**
* [Deep Gradient Boosted Learning article](https://alan.do/deep-gradient-boosted-learning-4e33adaf2969)

### Kaggle - High resolution ship collections 2016 (HRSC2016)
* https://www.kaggle.com/guofeng/hrsc2016
* Ship images harvested from Google Earth
* [HRSC2016_SOTA](https://github.com/ming71/HRSC2016_SOTA) -> Fair comparison of different algorithms on the HRSC2016 dataset

### Kaggle - Understanding Clouds from Satellite Images
In this challenge, you will build a model to classify cloud organization patterns from satellite images.
* https://www.kaggle.com/c/understanding_cloud_organization/
* [3rd place solution on Github by naivelamb](https://github.com/naivelamb/kaggle-cloud-organization)

## Kaggle - Airbus Aircraft Detection Dataset
* https://www.kaggle.com/airbusgeo/airbus-aircrafts-sample-dataset
* One hundred civilian airports and over 3000 annotated commercial aircrafts
* [detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5](https://medium.com/artificialis/detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5-5f3d464b75ad)

### Kaggle - Airbus oil storage detection dataset
* https://www.kaggle.com/airbusgeo/airbus-oil-storage-detection-dataset
* [Oil-Storage Tank Instance Segmentation with Mask R-CNN](https://github.com/georgiosouzounis/instance-segmentation-mask-rcnn/blob/main/mask_rcnn_oiltanks_gpu.ipynb) with [accompanying article](https://medium.com/@georgios.ouzounis/oil-storage-tank-instance-segmentation-with-mask-r-cnn-77c94433045f)

### Kaggle - Satellite images of hurricane damage
* https://www.kaggle.com/kmader/satellite-images-of-hurricane-damage
* https://github.com/dbuscombe-usgs/HurricaneHarvey_buildingdamage

### Kaggle - Austin Zoning Satellite Images
* https://www.kaggle.com/franchenstein/austin-zoning-satellite-images
* classify a images of Austin into one of its zones, such as residential, industrial, etc. 3667 satellite images

### Kaggle - Statoil/C-CORE Iceberg Classifier Challenge
* https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/data
* [Deep Learning for Iceberg detection in Satellite Images](https://towardsdatascience.com/deep-learning-for-iceberg-detection-in-satellite-images-c667acf4bad0)

## Kaggle - Land Cover Classification Dataset from DeepGlobe Challenge - segmentation
* https://www.kaggle.com/balraj98/deepglobe-land-cover-classification-dataset
* [Satellite Imagery Semantic Segmentation with CNN](https://joshting.medium.com/satellite-imagery-segmentation-with-convolutional-neural-networks-f9254de3b907) -> 7 different segmentation classes, DeepGlobe Land Cover Classification Challenge dataset, with [repo](https://github.com/justjoshtings/satellite_image_segmentation)

## Kaggle - Next Day Wildfire Spread
A Data Set to Predict Wildfire Spreading from Remote-Sensing Data
* https://www.kaggle.com/fantineh/next-day-wildfire-spread
* https://arxiv.org/abs/2112.02447

## Kaggle - Satellite Next Day Wildfire Spread
Inspired by the above dataset, using different data sources
* https://www.kaggle.com/satellitevu/satellite-next-day-wildfire-spread
* https://github.com/SatelliteVu/SatelliteVu-AWS-Disaster-Response-Hackathon

## Kaggle - xView2 challenge - assessing building damage after a natural disaster
The xView2 Challenge focuses on automating the process of assessing building damage after a natural disaster
* https://www.kaggle.com/tunguz/xview2-challenge-dataset-train-and-test
* https://xview2.org/
* [GitHub for the challenge](https://github.com/DIUx-xView)
* [A look at IBM's approach](https://www.ibm.com/cloud/blog/the-xview2-ai-challenge) -> used a single U-Net model

### Kaggle - miscellaneous
* https://www.kaggle.com/reubencpereira/spatial-data-repo -> Satellite + loan data
* https://www.kaggle.com/towardsentropy/oil-storage-tanks -> Image data of industrial tanks with bounding box annotations, estimate tank fill % from shadows
* https://www.kaggle.com/datamunge/overheadmnist -> A Benchmark Satellite Dataset as Drop-In Replacement for MNIST
* https://www.kaggle.com/airbusgeo/airbus-wind-turbines-patches -> Airbus SPOT satellites images over wind turbines for classification
* https://www.kaggle.com/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes -> CGI planes object detection dataset
* https://www.kaggle.com/atilol/aerialimageryforroofsegmentation -> Aerial Imagery for Roof Segmentation
* https://www.kaggle.com/andrewmvd/ship-detection -> 621 images of boats and ships

## SpaceNet
* [spacenet.ai](https://spacenet.ai/) is an online hub for data, challenges, algorithms, and tools. Note that [CosmiQ Ended its Leadership of SpaceNet](https://medium.com/the-downlinq/closing-time-cosmiq-works-is-closing-down-and-ending-its-leadership-of-spacenet-a53ba239745b), handing over the reigns to Maxar
* SpaceNet ran a series consisting of seven challenges with datasets and utilities provided. Challenges covered (1&2) building segmentation, (3) road segmentation, (4) off-nadir buildings, (5) road network extraction, (6)multi-senor mapping, (7) multi-temporal urban change
* Building datasets covered a number of cities including: Rio, Paris, Vegas, Shanghai, Khartoum, Atlana, Moscow, Mumbai & Rotterdam
* [The SpaceNet 7 Multi-Temporal Urban Development Challenge: Dataset Release](https://medium.com/the-downlinq/the-spacenet-7-multi-temporal-urban-development-challenge-dataset-release-9e6e5f65c8d5)
* SpaceNet - WorldView-3 [article here](https://spark-in.me/post/spacenet-three-challenge), and [semantic segmentation using Raster Vision](https://docs.rastervision.io/en/0.8/quickstart.html)
* [spacenet-three-topcoder](https://github.com/snakers4/spacenet-three-topcoder) solution
* [official utilities](https://github.com/SpaceNetChallenge/utilities) -> Packages intended to assist in the preprocessing of SpaceNet satellite imagery data corpus to a format that is consumable by machine learning algorithms
* [andraugust spacenet-utils](https://github.com/andraugust/spacenet-utils) -> Display geotiff image with building-polygon overlay & label buildings using kNN on the pixel spectra

## Tensorflow datasets
* [resisc45](https://www.tensorflow.org/datasets/catalog/resisc45) -> RESISC45 dataset is a publicly available benchmark for Remote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class.
* [eurosat](https://www.tensorflow.org/datasets/catalog/eurosat) -> EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting of 10 classes with 27000 labeled and geo-referenced samples.
* [BigEarthNet](https://www.tensorflow.org/datasets/catalog/bigearthnet) -> a large-scale Sentinel-2 land use classification dataset, consisting of 590,326 Sentinel-2 image patches. The image patch size on the ground is 1.2 x 1.2 km with variable image size depending on the channel resolution. This is a multi-label dataset with 43 imbalanced labels. Official website includes version of the dataset with Sentinel 1 & 2 chips
* [so2sat](https://www.tensorflow.org/datasets/catalog/so2sat) -> a dataset consisting of co-registered synthetic aperture radar and multispectral optical image patches acquired by Sentinel 1 & 2

## AWS datasets
* [Earth on AWS](https://aws.amazon.com/earth/) is the AWS equivalent of Google Earth Engine
* Currently 36 satellite datasets on the [Registry of Open Data on AWS](https://registry.opendata.aws)

## Microsoft
* [US Building Footprints](https://github.com/Microsoft/USBuildingFootprints) -> building footprints in all 50 US states, GeoJSON format, generated using semantic segmentation. Also [Australia](https://github.com/microsoft/AustraliaBuildingFootprints), [Canadian](https://github.com/Microsoft/CanadianBuildingFootprints), [Uganda-Tanzania](https://github.com/microsoft/Uganda-Tanzania-Building-Footprints) and [Kenya-Nigeria](https://github.com/microsoft/KenyaNigeriaBuildingFootprints) are available. Checkout [RasterizingBuildingFootprints](https://github.com/mehdiheris/RasterizingBuildingFootprints) to convert vector shapefiles to raster layers
* [Microsoft Planetary Computer](https://innovation.microsoft.com/en-us/planetary-computer) is a Dask-Gateway enabled JupyterHub deployment focused on supporting scalable geospatial analysis, [source repo](https://github.com/microsoft/planetary-computer-hub)
* [Ai for Earth program](https://www.microsoft.com/en-us/ai/ai-for-earth)
* [landcover-orinoquia](https://github.com/microsoft/landcover-orinoquia) -> Land cover mapping of the Orinoquía region in Colombia, in collaboration with Wildlife Conservation Society Colombia. An #AIforEarth project

## Google Earth Engine (GEE)
Since there is a whole community around GEE I will not reproduce it here but list very select references. Get started at https://developers.google.com/earth-engine/
* Various imagery and climate datasets, including Landsat & Sentinel imagery
* [awesome-google-earth-engine](https://github.com/gee-community/awesome-google-earth-engine) & [awesome-earth-engine-apps](https://github.com/philippgaertner/awesome-earth-engine-apps)
* [How to Use Google Earth Engine and Python API to Export Images to Roboflow](https://blog.roboflow.com/how-to-use-google-earth-engine-with-roboflow/) -> to acquire training data
* [ee-fastapi](https://github.com/csaybar/ee-fastapi) is a simple FastAPI web application for performing flood detection using Google Earth Engine in the backend.
* [How to Download High-Resolution Satellite Data for Anywhere on Earth](https://towardsdatascience.com/how-to-download-high-resolution-satellite-data-for-anywhere-on-earth-5e6dddee2803)

## Radiant Earth
* https://www.radiant.earth/
* Datasets and also models on https://mlhub.earth/

### DEM (digital elevation maps)
* Shuttle Radar Topography Mission, search online at usgs.gov
* Copernicus Digital Elevation Model (DEM) on S3, represents the surface of the Earth including buildings, infrastructure and vegetation. Data is provided as Cloud Optimized GeoTIFFs. [link](https://registry.opendata.aws/copernicus-dem/)

## Weather Datasets
* UK metoffice -> https://www.metoffice.gov.uk/datapoint
* NASA (make request and emailed when ready) -> https://search.earthdata.nasa.gov
* NOAA (requires BigQuery) -> https://www.kaggle.com/noaa/goes16/home
* Time series weather data for several US cities -> https://www.kaggle.com/selfishgene/historical-hourly-weather-data

## Forest datasets
* [awesome-forests](https://github.com/blutjens/awesome-forests) -> A curated list of ground-truth forest datasets for the machine learning and forestry community

## Time series & change detection datasets
* [BreizhCrops](https://github.com/dl4sits/BreizhCrops) -> A Time Series Dataset for Crop Type Mapping
* The SeCo dataset contains image patches from Sentinel-2 tiles captured at different timestamps at each geographical location. [Download SeCo here](https://github.com/ElementAI/seasonal-contrast)
* [Onera Satellite Change Detection Dataset](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) comprises 24 pairs of multispectral images taken from the Sentinel-2 satellites between 2015 and 2018
* [SYSU-CD](https://github.com/liumency/SYSU-CD) -> The dataset contains 20000 pairs of 0.5-m aerial images of size 256×256 taken between the years 2007 and 2014 in Hong Kong

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

## Synthetic data
* [The Synthinel-1 dataset: a collection of high resolution synthetic overhead imagery for building segmentation](https://arxiv.org/ftp/arxiv/papers/2001/2001.05130.pdf)
* [RarePlanes](https://www.cosmiqworks.org/RarePlanes/) ->  incorporates both real and synthetically generated satellite imagery including aircraft. Read the [arxiv paper](https://arxiv.org/abs/2006.02963) and checkout [the repo](https://github.com/aireveries/RarePlanes). Note the dataset is available through the AWS Open-Data Program for free download
* Read [this article from NVIDIA](https://developer.nvidia.com/blog/preparing-models-for-object-detection-with-real-and-synthetic-data-and-tao-toolkit/) which discusses fine tuning a model pre-trained on synthetic data (Rareplanes) with 10% real data, then pruning the model to reduce its size, before quantizing the model to improve inference speed
* Checkout Microsoft [AirSim](https://microsoft.github.io/AirSim/), which is a simulator for drones, cars and more, built on Unreal Engine
* [Combining Synthetic Data with Real Data to Improve Detection Results in Satellite Imagery](https://one-view.ai/combining-synthetic-data-with-real-data-to-improve-detection-results-in-satellite-imagery-case-study/)
* [Synthinel](https://github.com/timqqt/Synthinel) -> synthetic overhead imagery with full pixel-wise building labels, created using ESRI CityEngine
* [BlenderGIS](https://github.com/domlysz/BlenderGIS) could be used for synthetic data generation
* [bifrost.ai](https://www.bifrost.ai/) -> simulated data service with geospatial output data formats
* [oktal-se](https://www.oktal-se.fr/deep-learning/) -> software for generating simulated data across a wide range of bands including optical and SAR
* [The Nuances of Extracting Utility from Synthetic Data](https://www.iqt.org/synthesizing-robustness-yoltv4-results-part-1/) -> We find that strategically augmenting the real dataset is nearly as effective as adding synthetic data in the quest to improve the detection or rare object classes, and that fully extracting the utility of synthetic data is a nuanced process
* [Synthesizing Robustness](https://www.iqt.org/synthesizing-robustness/) -> explores how to best leverage and enhance synthetic data

# Online platforms for analytics
* [This article discusses some of the available platforms](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2)
* [Pangeo](http://pangeo.io/index.html) -> There is no single software package called “pangeo”; rather, the Pangeo project serves as a coordination point between scientists, software, and computing infrastructure. Includes open source resources for parallel processing using Dask and Xarray. Pangeo recently announced their 2.0 goals: pivoting away from directly operating cloud-based JupyterHubs, and towards eductaion and research
* [Airbus Sandbox](https://sandbox.intelligence-airbusds.com/web/) -> will provide access to imagery
* [Descartes Labs](https://www.descarteslabs.com/) -> access to EO imagery from a variety of providers via python API
* Planet have a [Jupyter notebook platform](https://developers.planet.com/) which can be deployed locally.
* [eurodatacube.com](https://eurodatacube.com/) -> data & platform for EO analytics in Jupyter env, paid
* [up42](https://up42.com/) is a developer platform and marketplace, offering all the building blocks for powerful, scalable geospatial products
* [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) -> direct Google Earth Engine competitor in the making?
* [eofactory.ai](https://eofactory.ai/) -> supports multi public and private data sources that can be used to analyse and extract information
* [mapflow.ai](https://mapflow.ai/) -> imagery analysis platform with its instant access to the major satellite imagery providers, models for extract building footprints etc & [QGIS plugin](https://www.gislounge.com/run-ai-mapping-in-qgis-over-high-resolution-satellite-imagery/)
* [openeo](https://openeo.cloud/) by ESA data platform
* [Adam platform](https://adamplatform.eu/) -> the Advanced geospatial Data Management platform (ADAM) is a tool to access a large variety and volume of global environmental data

# Free online compute
A GPU is required for training deep learning models (but not necessarily for inferencing), and this section lists a couple of free Jupyter environments with GPU available. There is a good overview of online Jupyter development environments [on the fastai site](https://course19.fast.ai). I personally use Colab Pro with data hosted on Google Drive, or Sagemaker if I have very long running training jobs.

## Google Colab
* Collaboratory [notebooks](https://colab.research.google.com) with GPU as a backend for free for 12 hours at a time. Note that the GPU may be shared with other users, so if you aren't getting good performance try reloading.
* Also a pro tier for $10 a month -> https://colab.research.google.com/signup
* Tensorflow, pytorch & fastai available but you may need to update them
* [Colab Alive](https://chrome.google.com/webstore/detail/colab-alive/eookkckfbbgnhdgcbfbicoahejkdoele?hl=en) is a chrome extension that keeps Colab notebooks alive.
* [colab-ssh](https://github.com/WassimBenzarti/colab-ssh) -> lets you ssh to a colab instance like it’s an EC2 machine and install packages that require full linux functionality

## Kaggle - also Google!
* Free to use
* GPU Kernels - may run for 1 hour
* Tensorflow, pytorch & fastai available but you may need to update them
* Advantage that many datasets are already available

## AWS SageMaker Studio Lab
* [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) is a recent release which competes with Google colab being free to use with no credit card or AWS account required
* [Github profile](https://github.com/topics/amazon-sagemaker-lab)

## Others
* [Paperspace gradient](https://gradient.run/notebooks) -> free tier includes GPU usage
* [Deepnote](https://deepnote.com/) -> many features for collaboration, GPU use is paid

# State of the art engineering
* Compute and data storage are moving to the cloud. Read how [Planet](https://cloud.google.com/customers/planet) and [Airbus](https://cloud.google.com/customers/airbus) use the cloud
* Google Earth Engine and Microsoft Planetary Computer are democratising access to massive compute platforms
* No-code platforms and auto-ml are making ML techniques more accessible than ever
* Custom hardware is being developed for rapid training and inferencing with deep learning models, both in the datacenter and at the edge
* Supervised ML methods typically require large annotated datasets, but approaches such as self-supervised and active learning are offering alternatives pathways
* Traditional data formats aren't designed for processing on the cloud, so new standards are evolving such as COGS and STAC
* Computer vision traditionally delivered high performance image processing on a CPU by using compiled languages like C++, as used by OpenCV for example. The advent of GPUs are changing the paradigm, with alternatives optimised for GPU being created, such as [Kornia](https://kornia.github.io/)
* Whilst the combo of python and keras/tensorflow/pytorch are currently preeminent, new python libraries such as [Jax](https://github.com/google/jax) and alternative languages such as [Julia](https://julialang.org/) are showing serious promise

# Cloud providers
An overview of the most relevant services provided by AWS, Google and Microsoft. Also consider one of the many smaller but more specialised platorms such as [paperspace](https://www.paperspace.com/)

## AWS
* Host your data on [S3](https://aws.amazon.com/s3/) and metadata in a db such as [postgres](https://aws.amazon.com/rds/postgresql/)
* For batch processing use [Batch](https://aws.amazon.com/batch/). GPU instances are available for [batch deep learning](https://aws.amazon.com/blogs/compute/deep-learning-on-aws-batch/) inferencing. See how Rastervision implement this [here](https://docs.rastervision.io/en/0.13/cloudformation.html)
* If processing can be performed in 15 minutes or less, serverless [Lambda](https://aws.amazon.com/lambda/) functions are an attractive option owing to their ability to scale. Note that lambda may not be a particularly quick solution for deep learning applications, since you do not have the option to batch inference on a GPU. Creating a docker container with all the required dependencies can be a challenge. To get started read [Using container images to run PyTorch models in AWS Lambda](https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/) and for an image classification example [checkout this repo](https://github.com/aws-samples/aws-lambda-docker-serverless-inference). Also read [Processing satellite imagery with serverless architecture](https://aws.amazon.com/blogs/compute/processing-satellite-imagery-with-serverless-architecture/) which discusses queuing & lambda. For managing a serverless infrastructure composed of multiple lambda functions use [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/index.html)
* Use [Glue](https://aws.amazon.com/glue) for data preprocessing - or use Sagemaker
* To orchestrate basic data pipelines use [Step functions](https://aws.amazon.com/step-functions/). Use the [AWS Step Functions Workflow Studio](https://aws.amazon.com/blogs/aws/new-aws-step-functions-workflow-studio-a-low-code-visual-tool-for-building-state-machines/) to get started. Read [Orchestrating and Monitoring Complex, Long-running Workflows Using AWS Step Functions](https://aws.amazon.com/blogs/architecture/field-notes-orchestrating-and-monitoring-complex-long-running-workflows-using-aws-step-functions/). Note that step functions are defined in JSON
* If step functions are too limited or you want to write pipelines in python and use Directed Acyclic Graphs (DAGs) for workflow management, checkout hosted [AWS managed Airflow](https://aws.amazon.com/managed-workflows-for-apache-airflow/). Read [Orchestrate XGBoost ML Pipelines with Amazon Managed Workflows for Apache Airflow](https://aws.amazon.com/blogs/machine-learning/orchestrate-xgboost-ml-pipelines-with-amazon-managed-workflows-for-apache-airflow/) and checkout [amazon-mwaa-examples](https://github.com/aws-samples/amazon-mwaa-examples)
* [Sagemaker](https://aws.amazon.com/sagemaker/) is a whole ecosystem of ML tools that includes a hosted Jupyter environment for training of ML models. There are also tools for deployment of models using docker. [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) is a recent release which competes with Google colab being free to use with no credit card or AWS account required
* [Deep learning AMIs](https://aws.amazon.com/machine-learning/amis/) are EC2 instances with deep learning frameworks preinstalled. They do require more setup from the user than Sagemaker but in return allow access to the underlying hardware, which makes debugging issues more straightforward. There is a [good guide to setting up your AMI instance on the Keras blog](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html)
* Specifically created for deep learning inferencing is [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/)
* [Rekognition](https://aws.amazon.com/rekognition/custom-labels-features/) custom labels is a 'no code' annotation, training and inferencing service. Read [Training models using Satellite (Sentinel-2) imagery on Amazon Rekognition Custom Labels](https://ryfeus.medium.com/training-models-using-satellite-imagery-on-amazon-rekognition-custom-labels-dd44ac6a3812). For a comparison with Azure and Google alternatives [read this article](https://blog.roboflow.com/automl-vs-rekognition-vs-custom-vision/)
* When developing you will definitely want to use [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) and probably [aws-data-wrangler](https://github.com/awslabs/aws-data-wrangler)
* For managing infrastructure use [Terraform](https://www.terraform.io/). Alternatively if you wish to use TypeScript, JavaScript, Python, Java, or C# checkout [AWS CDK](https://aws.amazon.com/cdk/), although I found relatively few examples to get going using python
* [AWS Ground Station now supports data delivery to Amazon S3](https://aws.amazon.com/about-aws/whats-new/2021/04/aws-ground-station-now-supports-data-delivery-to-amazon-s3/)
* [Redshift](https://aws.amazon.com/redshift/) is a fast, scalable data warehouse that can extend queries to S3. Redshift is based on PostgreSQL but [has some differences](https://docs.aws.amazon.com/redshift/latest/dg/c_redshift-and-postgres-sql.html). Redshift supports geospatial data.
* [AWS App Runner](https://aws.amazon.com/blogs/containers/introducing-aws-app-runner/) enables quick deployment of containers as apps
* [AWS Athena](https://aws.amazon.com/athena/) allows running SQL queries against CSV files stored on S3. Serverless so pay only for the queries you run
* If you are using pytorch checkout [the S3 plugin for pytorch](https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/) which provides streaming data access
* [aws-gdal-robot](https://github.com/mblackgeo/aws-gdal-robot) -> A proof of concept implementation of running GDAL based jobs using AWS S3/Lambda/Batch

## Google cloud
* For storage use [Cloud Storage](https://cloud.google.com/storage) (AWS S3 equivalent)
* For data warehousing use [BigQuery](https://cloud.google.com/bigquery) (AWS Redshift equivalent). Visualize massive spatial datasets directly in BigQuery using [CARTO](https://carto.com/bigquery-tiler/)
* For model training use [Vertex](https://cloud.google.com/vertex-ai) (AWS Sagemaker equivalent)
* For containerised apps use [Cloud Run](https://cloud.google.com/run) (AWS App Runner equivalent but can scale to zero)

## Microsoft Azure
* [Azure Orbital](https://azure.microsoft.com/en-us/services/orbital/) -> Satellite ground station and scheduling services for fast downlinking of data
* [ShipDetection](https://github.com/microsoft/ShipDetection) -> use the Azure Custom Vision service to train an object detection model that can detect and locate ships in a satellite image

# Deploying models
This section discusses how to get a trained machine learning & specifically deep learning model into production. For an overview on serving deep learning models checkout [Practical-Deep-Learning-on-the-Cloud](https://github.com/PacktPublishing/-Practical-Deep-Learning-on-the-Cloud). There are many options if you are happy to dedicate a server, although you may want a GPU for batch processing. For serverless use AWS lambda.

## Rest API on dedicated server
A common approach to serving up deep learning model inference code is to wrap it in a rest API. The API can be implemented in python (flask or FastAPI), and hosted on a dedicated server e.g. EC2 instance. Note that making this a scalable solution will require significant experience.
* Basic API: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html with code [here](https://github.com/jrosebr1/simple-keras-rest-api)
* Advanced API with request queuing: https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/

## Framework/provider specific model serving options
If you are happy to live with some lock-in, these are good options:
* [Tensorflow serving](https://www.tensorflow.org/tfx/guide/serving) is limited to Tensorflow models
* [Pytorch serve](https://github.com/pytorch/serve) is easy to use, limited to Pytorch models, can be deployed via AWS Sagemaker
* [sagemaker-inference-toolkit](https://github.com/aws/sagemaker-inference-toolkit) -> Serve machine learning models within a Docker container using AWS SageMaker

## NVIDIA Triton server
* The [Triton Inference Server](https://github.com/triton-inference-server/server) provides an optimized cloud and edge inferencing solution
* Supports TensorFlow, ONNX, PyTorch TorchScript and OpenVINO model formats. Both TensorFlow 1.x and TensorFlow 2.x are supported.
* Read [CAPE Analytics Uses Computer Vision to Put Geospatial Data and Risk Information in Hands of Property Insurance Companies](https://blogs.nvidia.com/blog/2021/05/21/cape-analytics-computer-vision/)
* Available on the [AWS Marketplace](https://aws.amazon.com/marketplace/pp/B08NHXW4MN)

## Using lambda functions - i.e. serverless
Using lambda functions allows inference without having to configure or manage the underlying infrastructure
* On AWS either use regular lambdas from AWS or [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
* [Object detection inference with AWS Lambda and IceVision (PyTorch)](https://laurenzstrothmann.com/object-detection-inference-aws-lambda-icevision) with [repo](https://github.com/2649/laurenzstrothmann)
* [Example deployment behind an API Gateway Proxy](https://github.com/philschmid/cdk-samples/tree/master/sagemaker-serverless-huggingface-endpoint)

## Models in the browser
The model is run in the browser itself on live images, ensuring processing is always with the latest model available and removing the requirement for dedicated server side inferencing
* [Classifying satellite imagery - Made with TensorFlow.js YoutTube video](https://www.youtube.com/watch?v=9zqjgeqc-ew)

## Model optimisation for deployment
The general approaches are outlined in [this article from NVIDIA](https://developer.nvidia.com/blog/preparing-models-for-object-detection-with-real-and-synthetic-data-and-tao-toolkit/) which discusses fine tuning a model pre-trained on synthetic data (Rareplanes) with 10% real data, then pruning the model to reduce its size, before quantizing the model to improve inference speed. Training notebook [here](https://github.com/aireveries/rareplanes-tlt)

## Model monitoring
Once your model is deployed you will want to monitor for data errors, broken pipelines, and model performance degradation/drift [ref](https://towardsdatascience.com/deploy-and-monitor-your-ml-application-with-flask-and-whylabs-4cd1e757c94b)
* [Blog post by Neptune: Doing ML Model Performance Monitoring The Right Way](https://neptune.ai/blog/ml-model-performance-monitoring)
* [whylogs](https://github.com/whylabs/whylogs) -> Profile and monitor your ML data pipeline end-to-end

# Image formats, data management and catalogues
* [GeoServer](http://geoserver.org/) -> an open source server for sharing geospatial data
* Open Data Cube - serve up cubes of data https://www.opendatacube.org/
* https://terria.io/ for pretty catalogues
* Large datasets may come in HDF5 format, can view with -> https://www.hdfgroup.org/downloads/hdfview/
* Climate data is often in netcdf format, which can be opened using xarray
* The xarray docs list a number of ways that data [can be stored and loaded](http://xarray.pydata.org/en/latest/io.html#).
* [TileDB](https://tiledb.com/) -> a 'Universal Data Engine' to store, analyze and share any data (beyond tables), with any API or tool (beyond SQL) at planet-scale (beyond clusters), open source and managed options. [Recently hiring](https://discourse.pangeo.io/t/job-openings-at-tiledb-inc/787) to work with xarray, dask, netCDF and cloud native storage
* [BigVector database](https://deepai.org/bigvector) -> A fully-managed, highly-scalable, and cost-effective database for vectors. Vectorize structured data or orbital imagery and discover new insights
* Read about [Serverless PostGIS on AWS Aurora](https://blog.addresscloud.com/serverless-postgis/)
* [Hub](https://github.com/activeloopai/Hub) -> The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Works locally or on any cloud. Read [Faster Machine Learning Using Hub by Activeloop: A code walkthrough of using the hub package for satellite imagery](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005)
* [A Comparison of Spatial Functions: PostGIS, Athena, PrestoDB, BigQuery vs RedShift](https://ual.sg/post/2020/07/03/a-comparison-of-spatial-functions-postgis-athena-prestodb-bigquery-vs-redshift/)
* [Unfolded Studio](https://studio.unfolded.ai/) -> visualization platform building on open source geospatial technologies including kepler.gl, deck.gl and H3. Processing is performed browser side enabling very responsive visualisations.
* [DroneDB](https://github.com/DroneDB/DroneDB) -> can index and extract useful information from the EXIF/XMP tags of aerial images to display things like image footprint, flight path and image GPS location
* [embeddinghub](https://github.com/featureform/embeddinghub) -> A vector database for machine learning embeddings
* [Resonant GeoData](https://github.com/ResonantGeoData/ResonantGeoData/) -> a Django application well suited for catalogging and searching annotated geospatial imagery, shapefiles, and full motion video datasets

## Cloud Optimised GeoTiff (COG)
A Cloud Optimized GeoTIFF (COG) is a regular GeoTIFF that supports HTTP range requests, enabling downloading of specific tiles rather than the full file. COG generally work normally in GIS software such as QGIS, but are larger than regular GeoTIFFs
* https://www.cogeo.org/
* [cog-best-practices](https://github.com/pangeo-data/cog-best-practices)
* [COGs in production](https://sean-rennie.medium.com/cogs-in-production-e9a42c7f54e4)
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
* [cog_worker](https://github.com/Vizzuality/cog_worker) -> Scalable arbitrary analysis on COGs

## SpatioTemporal Asset Catalog specification (STAC)
The STAC specification provides a common metadata specification, API, and catalog format to describe geospatial assets, so they can more easily indexed and discovered.
* Spec at https://github.com/radiantearth/stac-spec
* [STAC 1.0.0: The State of the STAC Software Ecosystem](https://medium.com/radiant-earth-insights/stac-1-0-0-software-ecosystem-updates-da4e800a4973)
* [Planet Disaster Data catalogue](https://planet.stac.cloud/) has the [catalogue source on Github](https://github.com/cholmes/pdd-stac) and uses the [stac-browser](https://github.com/radiantearth/stac-browser)
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
* [stac-fastapi](https://github.com/stac-utils/stac-fastapi) -> STAC API implementation with FastAPI
* [ml-aoi](https://github.com/stac-extensions/ml-aoi) -> An Item and Collection extension to provide labeled training data for machine learning models
* Discoverable and Reusable ML Workflows for Earth Observation -> [part 1](https://medium.com/radiant-earth-insights/discoverable-and-reusable-ml-workflows-for-earth-observation-part-1-e198507b5eaa) and [part 2](https://medium.com/radiant-earth-insights/discoverable-and-reusable-ml-workflows-for-earth-observation-part-2-ebe2b4812d5a) with the Geospatial Machine Learning Model Catalog (GMLMC)
* [eoAPI](https://github.com/developmentseed/eoAPI) -> Earth Observation API with STAC + dynamic Raster/Vector Tiler
* [stac-nb](https://github.com/darrenwiens/stac-nb) -> STAC in Jupyter Notebooks
* [xstac](https://github.com/TomAugspurger/xstac) -> Generate STAC Collections from xarray datasets
* [qgis-stac-plugin](https://github.com/stac-utils/qgis-stac-plugin) -> QGIS plugin for reading STAC APIs
* [cirrus-geo](https://github.com/cirrus-geo/cirrus-geo) -> a STAC-based processing pipeline

# Image annotation
For supervised machine learning, you will require annotated images. For example if you are performing object detection you will need to annotate images with bounding boxes. Check that your annotation tool of choice supports large image (likely geotiff) files, as not all will. Note that GeoJSON is widely used by remote sensing researchers but this annotation format is not commonly supported in general computer vision frameworks, and in practice you may have to convert the annotation format to use the data with your chosen framework. There are both closed and open source tools for creating and converting annotation formats. Some of these tools are simply for performing annotation, whilst others add features such as dataset management and versioning. Note that self-supervised and active learning approaches might circumvent the need to perform a large scale annotation exercise.

## General purpose annotation tools
* [awesome-data-labeling](https://github.com/heartexlabs/awesome-data-labeling) -> long list of annotation tools
* [Roboflow](https://roboflow.com/robincole). In addition to annotation the platform makes it easy to convert between annotation formats & manage datasets, as well as train and deploy custom models to private API endpoints. Read [How to Train Computer Vision Models on Aerial Imagery](https://blog.roboflow.com/how-to-use-roboflow-with-aerial-imagery/)
* [labelImg](https://github.com/tzutalin/labelImg) is the classic desktop tool, limited to bounding boxes for object detection. Also checkout [roLabelImg](https://github.com/cgvict/roLabelImg) which supports ROTATED rectangle regions, as often occurs in aerial imagery.
* [Labelme](https://github.com/wkentaro/labelme) is a simple dektop app for polygonal annotation, but note it outputs annotations in a custom LabelMe JSON format which you will need to convert. Read [Labelme Image Annotation for Geotiffs](https://medium.com/@wvsharber/labelme-image-annotation-for-geotiffs-b460ba83804f)
* [Label Studio](https://labelstud.io/) is a multi-type data labeling and annotation tool with standardized output format, syncing to buckets, and supports importing pre-annotations (create with a model). Checkout [label-studio-converter](https://github.com/heartexlabs/label-studio-converter) for converting Label Studio annotations into common dataset formats
* [CVAT](https://github.com/openvinotoolkit/cvat) suports object detection, segmentation and classification via a local web app. There is an [open issue](https://github.com/openvinotoolkit/cvat/issues/531) to support large TIFF files. [This article on Roboflow](https://blog.roboflow.com/cvat/) gives a good intro to CVAT.
* [Create your own annotation tool using Bokeh Holoviews](https://examples.pyviz.org/ml_annotators/ml_annotators.html#ml-annotators-gallery-ml-annotators)
* [VoTT](https://github.com/Microsoft/VoTT) -> an electron app for building end to end Object Detection Models from Images and Videos, by Microsoft
* [Deeplabel](https://github.com/jveitchmichaelis/deeplabel) is a cross-platform tool for annotating images with labelled bounding boxes. Deeplabel also supports running inference using state-of-the-art object detection models like Faster-RCNN and YOLOv4. With support out-of-the-box for CUDA, you can quickly label an entire dataset using an existing model.
* [Alturos.ImageAnnotation](https://github.com/AlturosDestinations/Alturos.ImageAnnotation) is a collaborative tool for labeling image data on S3 for yolo
* [rectlabel](https://rectlabel.com/) is a desktop app for MacOS to annotate images for bounding box object detection and segmentation, paid and free (rectlabel-lite) versions
* [pigeonXT](https://github.com/dennisbakhuis/pigeonXT) can be used to create custom image classification annotators within Jupyter notebooks
* [ipyannotations](https://github.com/janfreyberg/ipyannotations) -> Image annotations in python using jupyter notebooks
* [Label-Detect](https://github.com/Jakaria08/Label-Detect) -> is a graphical image annotation tool and using this tool a user can also train and test large satellite images, fork of the popular labelImg tool
* [Swipe-Labeler](https://github.com/spaceml-org/Swipe-Labeler) -> Swipe Labeler is a Graphical User Interface based tool that allows rapid labeling of image data
* SuperAnnotate can be run [locally](https://github.com/opencv-ai/superannotate) or used via a [cloud service](https://superannotate.com/)
* [dash_doodler](https://github.com/dbuscombe-usgs/dash_doodler) -> A web application built with plotly/dash for image segmentation with minimal supervision
* [remo](https://remo.ai) -> A webapp and Python library that lets you explore and control your image datasets
* [supervise.ly](https://supervise.ly) is one of the more fully featured platforms, decent free tier
* AWS supports image annotation via the Rekognition Custom Labels console
* [diffgram](https://github.com/diffgram/diffgram) describes itself as a complete training data platform for machine learning delivered as a single application, supports [streaming data to pytorch & tensorflow](https://medium.com/diffgram/stream-training-data-to-your-models-with-diffgram-f0f25f6688c5)
* [hasty.ai](https://hasty.ai/) -> supports model assisted annotation & inferencing
* TensorFlow Object Detection API provides a [handy utility](https://github.com/tensorflow/models/blob/6a55ecdea7afda51f9dc42dc17104bd6444395d9/research/object_detection/utils/colab_utils.py#L384) for object annotation within Google Colab notebooks. See usage [here](https://github.com/yasserius/tf2-object-detection-api#label-images-in-colab)
* [coco-annotator](https://github.com/jsbroks/coco-annotator)
* [pylabel](https://github.com/pylabel-project/pylabel) -> Python library for computer vision labeling tasks. The core functionality is to translate bounding box annotations between different formats-for example, from coco to yolo. PyLabel also includes an image labeling tool that runs in a Jupyter notebook that can annotate images manually or perform automatic labeling using a pre-trained model

## Annotation tools with GEO features
Also check the section **Image handling, manipulation & dataset creation**
* [GroundWork](https://groundwork.azavea.com/) is designed for annotating and labeling geospatial data like satellite imagery, from Azavea
* [labelbox.com](https://labelbox.com/) -> free tier is quite generous, [supports annotating Geotiffs](https://docs.labelbox.com/docs/tiled-imagery-editor) & returning annotations with geospatial coordinates. Watch [this webcast](https://www.arturo.ai/webcastbuilding-ai-products-from-the-ground-up/)
* [iris](https://github.com/ESA-PhiLab/iris) -> Tool for manual image segmentation and classification of satellite imagery
* If you are considering building an in house annotation platform [read this article](https://medium.com/earthcube-stories/ai-products-and-remote-sensing-yes-it-is-hard-and-yes-you-need-a-good-infra-4b5d6cf822f1). Used PostGis database, GeoJson format and GIS standard in a stateless architecture

## Annotation formats
Note there are many annotation formats, although PASCAL VOC and coco-json are the most commonly used.
* PASCAL VOC format: XML files in the format used by ImageNet
* coco-json format: JSON in the format used by the 2015 COCO dataset
* YOLO Darknet TXT format: contains one text file per image, used by YOLO
* Tensorflow TFRecord: a proprietary binary file format used by the Tensorflow Object Detection API
* Many more formats listed [here](https://roboflow.com/formats)

# Paid software
Many of these companies & products predate the open source software boom, and offer functionality which can be found in open source alternatives. However it is important to consider the licensing and support aspects before adopting an open source stack.
* [ENVI](https://www.l3harrisgeospatial.com/Software-Technology/ENVI) -> image processing and analysis
* [ERDAS IMAGINE](https://www.hexagongeospatial.com/products/power-portfolio/erdas-imagine) -> remote sensing, photogrammetry, LiDAR analysis, basic vector analysis, and radar processing into a single product
* [Spacemetric Keystone](http://spacemetric.com/) -> transform unprocessed sensor data into quality geospatial imagery ready for analysis
* [microimages TNTgis](https://www.microimages.com/) -> advanced GIS, image processing, and geospatial analysis at an affordable price
* [socet-gxp](https://www.geospatialexploitationproducts.com/content/socet-gxp/) -> Advanced geospatial exploitation and customized product creation combined into one comprehensive solution
* [video-inform](https://www.video-inform.com/) -> image and video analysis

## ArcGIS
Arguably the most significant paid software for working with maps and geographic information
* [ArcGIS](https://www.arcgis.com/index.html) -> mapping and analytics software, with both local and cloud hosted options. 
* [Geospatial deep learning with arcgis.learn](https://developers.arcgis.com/python/guide/geospatial-deep-learning/) -> the docs
* [Integrating Deep Learning with GIS](https://medium.com/geoai/integrating-deep-learning-with-gis-70e7c5aa9dfe) is a 2019 article by Rohit Singh which introduces the deep learning capabilities in ArcGIS
* [deep-learning-frameworks](https://github.com/Esri/deep-learning-frameworks) -> lists frameworks supported by the ArcGIS backend
* [sar-to-rgb-translation-using-cyclegan](https://www.esri.com/arcgis-blog/products/api-python/imagery/sar-to-rgb-translation-using-cyclegan/) using fastai
* [ArcGIS Jupyter Notebooks](https://www.esri.com/arcgis-blog/products/arcgis-enterprise/analytics/introducing-arcgis-notebooks/) are built to run big data analysis, deep learning models, and dynamic visualization tools.

# Open source software
[A note on licensing](https://www.gislounge.com/businesses-using-open-source-gis/): The two general types of licenses for open source are copyleft and permissive. Copyleft requires that subsequent derived software products also carry the license forward, e.g. the GNU Public License (GNU GPLv3). For permissive, options to modify and use the code as one please are more open, e.g. MIT & Apache 2. Checkout [choosealicense.com/](https://choosealicense.com/)
* [awesome-earthobservation-code](https://github.com/acgeospatial/awesome-earthobservation-code) -> lists many useful tools and resources
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
* [Datasette](https://datasette.io/) is a tool for exploring and publishing data as an interactive website and accompanying API, with SQLite backend. Various plugins extend its functionality, for example to allow displaying geospatial info, render images (useful for thumbnails), and add user authentication. Available as a [desktop app](https://datasette.io/desktop). Read [Drawing shapes on a map to query a SpatiaLite database](https://simonwillison.net/2021/Jan/24/drawing-shapes-spatialite/)
* [Photoprism](https://github.com/photoprism/photoprism) is a privately hosted app for browsing, organizing, and sharing your photo collection, with support for tiffs
* [dbeaver](https://github.com/dbeaver/dbeaver) is a free universal database tool and SQL client with [geospatial features](https://github.com/dbeaver/dbeaver/wiki/Working-with-Spatial-GIS-data)
* [Grafana](https://grafana.com/) can be used to make interactive dashboards, checkout [this example showing Point data](https://blog.timescale.com/blog/grafana-variables-101/). Note there is an [AWS managed service for Grafana](https://aws.amazon.com/grafana/)
* [litestream](https://litestream.io/) -> Continuously stream SQLite changes to S3-compatible storage
* [ImageFusion)](https://github.com/JohMast/ImageFusion) -> Temporal fusion of raster image time-Series
* [nvtop](https://github.com/Syllo/nvtop) -> NVIDIA GPUs htop like monitoring tool
* [rgis](https://github.com/frewsxcv/rgis) -> Geospatial data viewer written in Rust
* [aerialbot](https://github.com/doersino/aerialbot) -> A simple yet highly configurable bot that tweets geotagged aerial imagery of a random location in the world

## QGIS
A popular open source alternative to ArcGIS, desktop appication written in python and extended with plugins
* [QGIS](https://qgis.org/en/site/)
* Create, edit, visualise, analyse and publish geospatial information. Open source alternative to ArcGIS.
* [Python scripting](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/intro.html#scripting-in-the-python-console)
* [Plugins](https://plugins.qgis.org/plugins/) extend the functionality of QGIS - note these are essentially python scripts
* Create your own plugins using the [QGIS Plugin Builder](http://g-sherman.github.io/Qgis-Plugin-Builder/)
* [DeepLearningTools plugin](https://plugins.qgis.org/plugins/DeepLearningTools/) -> aid training Deep Learning Models
* [Mapflow.ai plugin](https://www.gislounge.com/run-ai-mapping-in-qgis-over-high-resolution-satellite-imagery/) -> various models to extract building footprints etc from Maxar imagery
* [dzetsaka plugin](https://github.com/nkarasiak/dzetsaka) -> classify different kind of vegetation
* [Coregistration-Qgis-processing](https://github.com/SMByC/Coregistration-Qgis-processing) -> Qgis processing plugin for image co-registration; projection and pixel alignment based on a target image, uses Arosics
* [qgis-stac-plugin](https://github.com/stac-utils/qgis-stac-plugin) -> QGIS plugin for reading STAC APIs

## GDAL & Rasterio
So improtant this pair gets their own section. GDAL is THE command line tool for reading and writing raster and vector geospatial data formats. If you are using python you will probably want to use Rasterio which provides a pythonic wrapper for GDAL
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
* [Short list of useful GDAL commands](https://github.com/MaxLenormand/Data-Science-for-Remote-Sensing) while working in data science for remote sensing
* [gdal-segment](https://github.com/cbalint13/gdal-segment) -> implements various segmentation algorithms over raster images
* [aws-gdal-robot](https://github.com/mblackgeo/aws-gdal-robot) -> A proof of concept implementation of running GDAL based jobs using AWS S3/Lambda/Batch

## General utilities
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
* [rsgislib](https://github.com/remotesensinginfo/rsgislib) -> Remote Sensing and GIS Software Library; python module tools for processing spatial and image data.
* [eo-learn](https://eo-learn.readthedocs.io/en/latest/index.html) is a collection of open source Python packages that have been developed to seamlessly access and process spatio-temporal image sequences acquired by any satellite fleet in a timely and automatic manner
* [RStoolbox: Tools for Remote Sensing Data Analysis in R](https://bleutner.github.io/RStoolbox/)
* [nd](https://github.com/jnhansen/nd) -> Framework for the analysis of n-dimensional, multivariate Earth Observation data, built on xarray
* [reverse-geocoder](https://github.com/thampiman/reverse-geocoder) -> a fast, offline reverse geocoder in Python
* [MuseoToolBox](https://github.com/nkarasiak/MuseoToolBox) -> a python library to simplify the use of raster/vector, especially for machine learning and remote sensing
* [py6s](https://py6s.readthedocs.io/en/latest/) -> an interface to the Second Simulation of the Satellite Signal in the Solar Spectrum (6S) atmospheric Radiative Transfer Model
* [timvt](https://github.com/developmentseed/timvt) -> PostGIS based Vector Tile server built on top of the modern and fast FastAPI framework
* [titiler](https://github.com/developmentseed/titiler) -> A dynamic Web Map tile server using FastAPI
* [BRAILS](https://github.com/NHERI-SimCenter/BRAILS) -> an AI-based pipeline for city-scale building information modelling (BIM)
* [color-thief-py](https://github.com/fengsp/color-thief-py) -> Grabs the dominant color or a representative color palette from an image
* [force](https://github.com/davidfrantz/force) -> an all-in-one processing engine for medium-resolution Earth Observation image archives
* [mapwarper](https://github.com/timwaters/mapwarper) -> an open source map geo-rectification, warping and georeferencing application
* [sarpy](https://github.com/ngageoint/sarpy) -> A basic Python library to demonstrate reading, writing, display, and simple processing of complex SAR data using the NGA SICD standard

## Low level numerical & data formats
* [xarray](http://xarray.pydata.org/en/stable/) -> N-D labeled arrays and datasets. Read [Handling multi-temporal satellite images with Xarray](https://medium.com/@bonnefond.virginie/handling-multi-temporal-satellite-images-with-xarray-30d142d3391). Checkout [xarray_leaflet](https://github.com/davidbrochart/xarray_leaflet) for tiled map plotting
* [xarray-spatial](https://github.com/makepath/xarray-spatial) -> Fast, Accurate Python library for Raster Operations. Implements algorithms using Numba and Dask, free of GDAL
* [xarray-beam](https://github.com/google/xarray-beam) -> Distributed Xarray with Apache Beam by Google
* [Geowombat](https://geowombat.readthedocs.io/) -> geo-utilities applied to air- and space-borne imagery, uses Rasterio, Xarray and Dask for I/O and distributed computing with named coordinates
* [NumpyTiles](https://github.com/planetlabs/numpytiles-spec) -> a specification for providing multiband full-bit depth raster data in the browser
* [Zarr](https://zarr.readthedocs.io/en/stable/) -> Zarr is a format for the storage of chunked, compressed, N-dimensional arrays. Zarr depends on NumPy

## Image processing, handling, manipulation & dataset creation
* [Pillow is the Python Imaging Library](https://pillow.readthedocs.io/en/stable/) -> this will be your go-to package for image manipulation in python
* [opencv-python](https://github.com/opencv/opencv-python) is pre-built CPU-only OpenCV packages for Python
* [kornia](https://github.com/kornia/kornia) is a differentiable computer vision library for PyTorch, like openCV but on the GPU. Perform image transformations, epipolar geometry, depth estimation, and low-level image processing such as filtering and edge detection that operate directly on tensors.
* [tifffile](https://github.com/cgohlke/tifffile) -> Read and write TIFF files
* [xtiff](https://github.com/BodenmillerGroup/xtiff) -> A small Python 3 library for writing multi-channel TIFF stacks
* [geotiff](https://github.com/Open-Source-Agriculture/geotiff) -> A noGDAL tool for reading and writing geotiff files
* [image_slicer](https://github.com/samdobson/image_slicer) -> Split images into tiles. Join the tiles back together.
* [tiler](https://github.com/nuno-faria/tiler) -> split images into tiles and merge tiles into a large image
* [geolabel-maker](https://github.com/makinacorpus/geolabel-maker) -> combine satellite or aerial imagery with vector spatial data to create your own ground-truth dataset in the COCO format for deep-learning models
* [felicette](https://github.com/plant99/felicette) -> Satellite imagery for dummies. Generate JPEG earth imagery from coordinates/location name with publicly available satellite data.
* [imagehash](https://github.com/JohannesBuchner/imagehash) -> Image hashes tell whether two images look nearly identical.
* [xbatcher](https://github.com/pangeo-data/xbatcher) -> Xbatcher is a small library for iterating xarray DataArrays in batches. The goal is to make it easy to feed xarray datasets to machine learning libraries such as Keras.
* [fake-geo-images](https://github.com/up42/fake-geo-images) -> A module to programmatically create geotiff images which can be used for unit tests
* [imagededup](https://github.com/idealo/imagededup) -> Finding duplicate images made easy! Uses perceptual hashing
* [rmstripes](https://github.com/DHI-GRAS/rmstripes) -> Remove stripes from images with a combined wavelet/FFT approach
* [activeloopai Hub](https://github.com/activeloopai/hub) -> The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Works locally or on any cloud. Scalable data pipelines.
* [sewar](https://github.com/andrewekhalel/sewar) -> All image quality metrics you need in one package
* [fiftyone](https://github.com/voxel51/fiftyone) -> open-source tool for building high quality datasets and computer vision models. Visualise labels, evaluate model predictions, explore scenarios of interest, identify failure modes, find annotation mistakes, and much more!
* [GeoTagged_ImageChip](https://github.com/Hejarshahabi/GeoTagged_ImageChip) -> A simple script to create geo tagged image chips from high resolution RS iamges for training deep learning models such as Unet.
* [Satellite imagery label tool](https://github.com/calebrob6/labeling-tool) -> provides an easy way to collect a random sample of labels over a given scene of satellite imagery
* [DeepSatData](https://github.com/michaeltrs/DeepSatData) -> Automatically create machine learning datasets from satellite images
* [image-reconstructor-patches](https://github.com/marijavella/image-reconstructor-patches) -> Reconstruct Image from Patches with a Variable Stride
* [geotiff-crop-dataset](https://github.com/tayden/geotiff-crop-dataset) -> A Pytorch Dataloader for tif image files that dynamically crops the image
* [Missing-Pixel-Filler](https://github.com/spaceml-org/Missing-Pixel-Filler) -> given images that may contain missing data regions (like satellite imagery with swath gaps), returns these images with the regions filled
* [deepsentinel-osm](https://github.com/Lkruitwagen/deepsentinel-osm) -> A repository to generate land cover labels from OpenStreetMap
* [img2dataset](https://github.com/rom1504/img2dataset) -> Easily turn large sets of image urls to an image dataset. Can download, resize and package 100M urls in 20h on one machine
* [satproc](https://github.com/dymaxionlabs/satproc) -> Python library and CLI tools for processing geospatial imagery for ML
* [Sliding Window](https://github.com/adamrehn/slidingwindow) ->  break large images into a series of smaller chunks
* [color_range_filter](https://github.com/developmentseed/color_range_filter) -> a script that allows us to find range of colors in images using openCV, and then convert them into geo vectors
* [eo4ai](https://github.com/ESA-PhiLab/eo4ai) -> easy-to-use tools for preprocessing datasets for image segmentation tasks in Earth Observation
* [Train-Test-Validation-Dataset-Generation](https://github.com/salarghaffarian/Train-Test-Validation-Dataset-Generation) ->  app to crop images and create small patches of a large image e.g. Satellite/Aerial Images, which will then be used for training and testing Deep Learning models specifically semantic segmentation models
* [rasterix](https://github.com/mogasw/rasterix) -> a cross-platform utility built around the GDAL library and the Qt framework designed to process geospatial raster data
* [jimutmap](https://github.com/Jimut123/jimutmap) -> get enormous amount of high resolution satellite images from apple / google maps quickly through multi-threading
* [Export thumbnails from Earth Engine](https://gorelick.medium.com/fast-er-downloads-a2abd512aa26)
* [datumaro](https://github.com/openvinotoolkit/datumaro) -> Dataset Management Framework, a Python library and a CLI tool to build, analyze and manage Computer Vision datasets
* [patchify](https://github.com/dovahcrow/patchify.py) -> A library that helps you split image into small, overlappable patches, and merge patches into original image
* [ohsome2label](https://github.com/GIScience/ohsome2label) -> Historical OpenStreetMap (OSM) Objects to Machine Learning Training Samples
* [Label Maker](https://github.com/developmentseed/label-maker) -> downloads OpenStreetMap QA Tile information and satellite imagery tiles and saves them as an `.npz` file for use in machine learning training. This should be used instead of the deprecated [skynet-data](https://github.com/developmentseed/skynet-data)
* [sentinelPot](https://github.com/LLeiSong/sentinelPot) -> a python package to preprocess sentinel-1&2 imagery
* [ImageAnalysis](https://github.com/UASLab/ImageAnalysis) -> Aerial imagery analysis, processing, and presentation scripts.
* [rebox](https://github.com/tensorturtle/rebox) -> Easily convert between bounding box annotation formats
* [rastertodataframe](https://github.com/mblackgeo/rastertodataframe) -> Convert any GDAL compatible raster to a Pandas DataFrame

## Image augmentation packages
Image augmentation is a technique used to expand a training dataset in order to improve ability of the model to generalise
* [AugLy](https://github.com/facebookresearch/AugLy) -> A data augmentations library for audio, image, text, and video. By Facebook
* [albumentations](https://github.com/albumentations-team/albumentations) -> Fast image augmentation library and an easy-to-use wrapper around other libraries
* [FoHIS](https://github.com/noahzn/FoHIS) -> Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
* [Kornia](https://kornia.readthedocs.io/en/latest/augmentation.html) provides augmentation on the GPU
* [toolbox by ming71](https://github.com/ming71/toolbox) -> various cv tools, such as label tools, data augmentation, label conversion, etc.

## Model tracking, versioning, specification & compilation
* [dvc](https://dvc.org/) -> a git extension to keep track of changes in data, source code, and ML models together
* [Weights and Biases](https://wandb.ai/) -> keep track of your ML projects. Log hyperparameters and output metrics from your runs, then visualize and compare results and quickly share findings with your colleagues
* [geo-ml-model-catalog](https://github.com/radiantearth/geo-ml-model-catalog) -> provides a common metadata definition for ML models that operate on geospatial data
* [hummingbird](https://github.com/microsoft/hummingbird) ->  a library for compiling trained traditional ML models into tensor computations, e.g. scikit learn model to pytorch for fast inference on a GPU
* [deepchecks](https://github.com/deepchecks/deepchecks) -> Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort

## Deep learning packages, frameworks & projects
* [rastervision](https://docs.rastervision.io/) -> An open source Python framework for building computer vision models on aerial, satellite, and other large imagery sets
* [torchrs](https://github.com/isaaccorley/torchrs) -> PyTorch implementation of popular datasets and models in remote sensing tasksenhance) -> Enhance PyTorch vision for semantic segmentation, multi-channel images and TIF file
[torchgeo](https://github.com/microsoft/torchgeo) -> popular datasets, model architectures
* [DeepHyperX](https://github.com/eecn/Hyperspectral-Classification) -> A Python/pytorch tool to perform deep learning experiments on various hyperspectral datasets
* [DELTA](https://github.com/nasa/delta) -> Deep Earth Learning, Tools, and Analysis, by NASA is a framework for deep learning on satellite imagery, based on Tensorflow & using MLflow for tracking experiments
* [Lightly](https://docs.lightly.ai/index.html) is a computer vision framework for training deep learning models using self-supervised learning
* [Icevision](https://airctic.com/) offers a curated collection of hundreds of high-quality pre-trained models within an easy to use framework
* [pytorch_eo](https://github.com/earthpulse/pytorch_eo) -> aims to make Deep Learning for Earth Observation data easy and accessible to real-world cases and research alike
* [NGVEO](https://github.com/ESA-PhiLab/NGVEO) -> applying convolutional neural networks (CNN) to Earth Observation (EO) data from Sentinel 1 and 2 using python and PyTorch
* [chip-n-scale-queue-arranger by developmentseed](https://github.com/developmentseed/chip-n-scale-queue-arranger) -> an orchestration pipeline for running machine learning inference at scale. [Supports fastai models](https://github.com/developmentseed/fastai-serving)
* http://spaceml.org/ -> A Machine Learning toolbox and developer community building the next generation AI applications for space science and exploration
* [TorchSat](https://github.com/sshuair/torchsat) is an open-source deep learning framework for satellite imagery analysis based on PyTorch (no activity since June 2020)
* [DeepNetsForEO](https://github.com/nshaud/DeepNetsForEO) -> Uses SegNET for working on remote sensing images using deep learning (no activity since 2019)
* [RoboSat](https://github.com/mapbox/robosat) -> semantic segmentation on aerial and satellite imagery. Extracts features such as: buildings, parking lots, roads, water, clouds (no longer maintained)
* [DeepOSM](https://github.com/trailbehind/DeepOSM) -> Train a deep learning net with OpenStreetMap features and satellite imagery (no activity since 2017)
* [mapwith.ai](https://mapwith.ai/) -> AI assisted mapping of roads with OpenStreetMap. Part of [Open-Mapping-At-Facebook](https://github.com/facebookmicrosites/Open-Mapping-At-Facebook)
* [sahi](https://github.com/obss/sahi) -> A vision library for performing sliced inference on large images/small objects
* [terragpu](https://github.com/nasa-cisto-ai/terragpu) -> Python library to process and classify remote sensing imagery by means of GPUs and AI/ML

## Data discovery and ingestion
* [landsat_ingestor](https://github.com/landsat-pds/landsat_ingestor) -> Scripts and other artifacts for landsat data ingestion into Amazon public hosting
* [satpy](https://github.com/pytroll/satpy) -> a python library for reading and manipulating meteorological remote sensing data and writing it to various image and data file formats
* [GIBS-Downloader](https://github.com/spaceml-org/GIBS-Downloader) -> a command-line tool which facilitates the downloading of NASA satellite imagery and offers different functionalities in order to prepare the images for training in a machine learning pipeline
* [eodag](https://github.com/CS-SI/eodag) -> Earth Observation Data Access Gateway
* [pylandsat](https://github.com/yannforget/pylandsat) -> Search, download, and preprocess Landsat imagery
* [sentinelsat](https://github.com/sentinelsat/sentinelsat) -> Search and download Copernicus Sentinel satellite images
* [landsatxplore](https://github.com/yannforget/landsatxplore) -> Search and download Landsat scenes from EarthExplorer
* [OpenSarToolkit](https://github.com/ESA-PhiLab/OpenSarToolkit) -> High-level functionality for the inventory, download and pre-processing of Sentinel-1 data in the python language
* [lsru](https://github.com/loicdtx/lsru) -> Query and Order Landsat Surface Reflectance data via ESPA

## OpenStreetMap
[OpenStreetMap](https://www.openstreetmap.org/) (OSM) is a map of the world, created by people like you and free to use under an open license. Quite a few publications use OSM data for annotations & ground truth. Note that the data is created by volunteers and the quality can be variable
* [osmnx](https://github.com/gboeing/osmnx) -> Retrieve, model, analyze, and visualize data from OpenStreetMap
* [ohsome2label](https://github.com/GIScience/ohsome2label) -> Historical OpenStreetMap Objects to Machine Learning Training Samples
* [Label Maker](https://github.com/developmentseed/label-maker) -> downloads OpenStreetMap QA Tile information and satellite imagery tiles and saves them as an `.npz` file for use in machine learning training. This should be used instead of the deprecated [skynet-data](https://github.com/developmentseed/skynet-data)
* [prettymaps](https://github.com/marceloprates/prettymaps) -> A small set of Python functions to draw pretty maps from OpenStreetMap data
* [Joint Learning from Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps](https://arxiv.org/abs/1705.06057) -> fusion based architectures and coarse-to-fine segmentation to include the OpenStreetMap layer into multispectral-based deep fully convolutional networks, arxiv paper
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”

## Graphing and visualisation
* [hvplot](https://hvplot.holoviz.org/) -> A high-level plotting API for the PyData ecosystem built on HoloViews. Allows overlaying data on map tiles, see [Exploring USGS Terrain Data in COG format using hvPlot](https://discourse.holoviz.org/t/exploring-usgs-terrain-data-in-cog-format-using-hvplot/1727)
* [Pyviz](https://examples.pyviz.org/) examples include several interesting geospatial visualisations
* [napari](https://napari.org) -> napari is a fast, interactive, multi-dimensional image viewer for Python. It’s designed for browsing, annotating, and analyzing large multi-dimensional images. By integrating closely with the Python ecosystem, napari can be easily coupled to leading machine learning and image analysis tools. Note that to view a 3GB COG I had to install the [napari-tifffile-reader](https://github.com/GenevieveBuckley/napari-tifffile-reader) plugin.
* [pixel-adjust](https://github.com/cisaacstern/pixel-adjust) -> Interactively select and adjust specific pixels or regions within a single-band raster. Built with rasterio, matplotlib, and panel.
* [Plotly Dash](https://plotly.com/dash/) can be used for making interactive dashboards
* [folium](https://python-visualization.github.io/folium/) -> a python wrapper to the excellent [leaflet.js](https://leafletjs.com/) which makes it easy to visualize data that’s been manipulated in Python on an interactive leaflet map. Also checkout the [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) component for adding folium maps to your streamlit apps
* [ipyearth](https://github.com/davidbrochart/ipyearth) -> An IPython Widget for Earth Maps
* [geopandas-view](https://github.com/martinfleis/geopandas-view) -> Interactive exploration of GeoPandas GeoDataFrames
* [geogif](https://github.com/gjoseph92/geogif) -> Turn xarray timestacks into GIFs
* [leafmap](https://github.com/giswqs/leafmap) -> geospatial analysis and interactive mapping with minimal coding in a Jupyter environment
* [xmovie](https://github.com/jbusecke/xmovie) -> A simple way of creating movies from xarray objects
* [acquisition-time](https://github.com/charlotte-pel/acquisition-time) -> Drawing (Satellite) acquisition dates in a timeline
* [splot](https://github.com/pysal/splot) -> Lightweight plotting for geospatial analysis in PySAL
* [prettymaps](https://github.com/marceloprates/prettymaps) -> A small set of Python functions to draw pretty maps from OpenStreetMap data
* [Tools to Design or Visualize Architecture of Neural Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network)
* [AstronomicAL](https://github.com/grant-m-s/AstronomicAL) -> An interactive dashboard for visualisation, integration and classification of data using Active Learning
* [pyodi](https://github.com/Gradiant/pyodi) -> A simple tool for explore your object detection dataset
* [Interactive-TSNE](https://github.com/spaceml-org/Interactive-TSNE) -> a tool that provides a way to visually view a PyTorch model's feature representation for better embedding space interpretability
* [fastgradio](https://github.com/aliabd/fastgradio) -> Build fast gradio demos of fastai learners
* [pysheds](https://github.com/mdbartos/pysheds) -> Simple and fast watershed delineation in python
* [mapboxgl-jupyter](https://github.com/mapbox/mapboxgl-jupyter) -> Use Mapbox GL JS to visualize data in a Python Jupyter notebook
* [cartoframes](https://github.com/CartoDB/cartoframes) -> integrate CARTO maps, analysis, and data services into data science workflows
* [datashader](https://datashader.org/) -> create meaningful representations of large datasets quickly and flexibly. Read [Creating Visual Narratives from Geospatial Data Using Open-Source Technology Maxar blog post](https://blog.maxar.com/tech-and-tradecraft/2021/creating-visual-narratives-from-geospatial-data-using-open-source-technology)
* [Kaleido](https://github.com/plotly/Kaleido) -> Fast static image export for web-based visualization libraries with zero dependencies
* [flask-vector-tiles](https://github.com/mblackgeo/flask-vector-tiles) -> A simple Flask/leaflet based webapp for rendering vector tiles from PostGIS
* [Embedding Projector in Wandb](https://docs.wandb.ai/ref/app/features/panels/weave/embedding-projector) -> allows users to plot multi-dimensional embeddings on a 2D plane using common dimension reduction algorithms like PCA, UMAP, and t-SNE
* [flask-geocoding-webapp](https://github.com/mblackgeo/flask-geocoding-webapp) -> A quick example Flask application for geocoding and rendering a webmap using Folium/Leaflet
* [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) -> Latex code for making neural networks diagrams
* [Damage Assessment Visualizer](https://github.com/microsoft/Nonprofits/tree/master/Damage%20Assessment%20Visualizer) -> leverages satellite imagery from a disaster region to visualize conditions of building and structures before and after a disaster
* [NN-SVG](https://github.com/alexlenail/NN-SVG) -> is a tool for creating Neural Network (NN) architecture drawings parametrically rather than manually
* [localtileserver](https://github.com/banesullivan/localtileserver) -> image tile server for viewing geospatial rasters with ipyleaflet, folium, or CesiumJS locally in Jupyter or remotely in Flask applications

## Streamlit
[Streamlit](https://streamlit.io/) is an awesome python framework for creating apps with python. Additionally they will host the apps free of charge. Here I list resources which are EO related. Note that a component is an addon which extends Streamlits basic functionality
* [cogviewer](https://github.com/mykolakozyr/cogviewer) -> Simple Cloud Optimized GeoTIFF viewer
* [cogcreator](https://github.com/mykolakozyr/cogcreator) -> Simple Cloud Optimized GeoTIFF Creator. Generates COG from GeoTIFF files.
* [cogvalidator](https://github.com/mykolakozyr/cogvalidator) -> Simple Cloud Optimized GeoTIFF validator
* [streamlit-image-juxtapose](https://github.com/robmarkcole/streamlit-image-juxtapose) -> A simple Streamlit component to compare images in Streamlit apps
* [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) -> Streamlit Component for rendering Folium maps
* [streamlit-keplergl](https://github.com/chrieke/streamlit-keplergl) -> Streamlit component for rendering kepler.gl maps
* [streamlit-light-leaflet](https://github.com/andfanilo/streamlit-light-leaflet) -> Streamlit quick & dirty Leaflet component that sends back coordinates on map click
* [leafmap-streamlit](https://github.com/giswqs/leafmap-streamlit) -> various examples showing how to use streamlit to: create a 3D map using Kepler.gl, create a heat map, display a GeoJSON file on a map, and add a colorbar or change the basemap on a map
* [geemap-apps](https://github.com/giswqs/geemap-apps) -> build a multi-page Earth Engine App using streamlit and geemap
* [streamlit-geospatial](https://github.com/giswqs/streamlit-geospatial) -> A multi-page streamlit app for geospatial
* [geospatial-apps](https://github.com/giswqs/geospatial-apps) -> A collection of streamlit web apps for geospatial applications
* [BirdsPyView](https://github.com/rjtavares/BirdsPyView) -> convert images to top-down view and get coordinates of objects
* [Build a useful web application in Python: Geolocating Photos](https://medium.com/spatial-data-science/build-a-useful-web-application-in-python-geolocating-photos-186122de1968) -> Step by Step tutorial using Streamlit, Exif, and Pandas
* [Wild fire detection app](https://github.com/yueureka/WildFireDetection)
* [dvc-streamlit-example](https://github.com/sicara/dvc-streamlit-example) -> how dvc and streamlit can help track model performance during R&D exploration
* [stacdiscovery](https://github.com/mykolakozyr/stacdiscovery) -> Simple STAC Catalogs discovery tool
* [SARveillance](https://github.com/MJCruickshank/SARveillance) -> Sentinel-1 SAR time series analysis for OSINT use
* [streamlit-template](https://github.com/giswqs/streamlit-template) -> A streamlit app template for geospatial applications
* [streamlit-labelstudio](https://github.com/deneland/streamlit-labelstudio) -> A Streamlit component that provides an annotation interface using the LabelStudio Frontend
* [streamlit-img-label](https://github.com/lit26/streamlit-img-label) -> a graphical image annotation tool using streamlit. Annotations are saved as XML files in PASCAL VOC format

## Cluster computing with Dask
* [Dask](https://docs.dask.org/en/latest/) works with your favorite PyData libraries to provide performance at scale for the tools you love -> checkout [Read and manipulate tiled GeoTIFF datasets](https://examples.dask.org/applications/satellite-imagery-geotiff.html#)
* [Coiled](https://coiled.io) is a managed Dask service. Get started by reading [Democratizing Satellite Imagery Analysis with Dask](https://coiled.io/blog/democratizing-satellite-imagery-analysis-with-dask/)
* [Dask with PyTorch for large scale image analysis](https://blog.dask.org/2021/03/29/apply-pretrained-pytorch-model)
* [stackstac](https://github.com/gjoseph92/stackstac) -> Turn a STAC catalog into a dask-based xarray
* [dask-geopandas](https://github.com/jsignell/dask-geopandas) -> Parallel GeoPandas with Dask
* [dask-image](https://github.com/dask/dask-image) -> many SciPy ndimage functions implemented

## Algorithms
* [WaterDetect](https://github.com/cordmaur/WaterDetect) -> an end-to-end algorithm to generate open water cover mask, specially conceived for L2A Sentinel 2 imagery. It can also be used for Landsat 8 images and for other multispectral clustering/segmentation tasks.
* [GatorSense Hyperspectral Image Analysis Toolkit](https://github.com/GatorSense/hsi_toolkit_py) -> This repo contains algorithms for Anomaly Detectors, Classifiers, Dimensionality Reduction, Endmember Extraction, Signature Detectors, Spectral Indices
* [detectree](https://github.com/martibosch/detectree) -> Tree detection from aerial imagery
* [pylandstats](https://github.com/martibosch/pylandstats) -> compute landscape metrics
* [dg-calibration](https://github.com/DHI-GRAS/dg-calibration) -> Coefficients and functions for calibrating DigitalGlobe imagery
* [python-fmask](https://github.com/ubarsc/python-fmask) -> Implementation in Python of the cloud and shadow algorithms known collectively as Fmask
* [pyshepseg](https://github.com/ubarsc/pyshepseg) -> Python implementation of image segmentation algorithm of Shepherd et al (2019) Operational Large-Scale Segmentation of Imagery Based on Iterative Elimination.
* [Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images](https://github.com/ThomasWangWeiHong/Shadow-Detection-Algorithm-for-Aerial-and-Satellite-Images) -> shadow detection and correction algorithm
* [faiss](https://github.com/facebookresearch/faiss) -> A library for efficient similarity search and clustering of dense vectors, e.g. image embeddings
* [awesome-spectral-indices](https://github.com/davemlz/awesome-spectral-indices) -> A ready-to-use curated list of Spectral Indices for Remote Sensing applications
* [urban-footprinter](https://github.com/martibosch/urban-footprinter) -> A convolution-based approach to detect urban extents from raster datasets
* [ocean_color](https://github.com/marrs-lab/ocean_color) -> Tools and algorithms for drone and satellite based ocean color science
* [poliastro](https://github.com/poliastro/poliastro) -> pure Python library for interactive Astrodynamics and Orbital Mechanics, with a focus on ease of use, speed, and quick visualization
* [acolite](https://github.com/acolite/acolite) -> generic atmospheric correction module
* [pmapper](https://github.com/nasa-jpl/pmapper) -> a super-resolution and deconvolution toolkit for python. PMAP stands for Poisson Maximum A-Posteriori, a highly flexible and adaptable algorithm for these problems
* [pylandtemp](https://github.com/pylandtemp/pylandtemp) -> Algorithms for computing global land surface temperature and emissivity from NASA's Landsat satellite images with Python

## Julia language
[Julia](https://julialang.org/) looks and feels a lot like Python, but can be much faster. Julia can call Python, C, and Fortran libraries and is capabale of C/Fortran speeds. Julia can be used in the familiar Jupyterlab notebook environment
* [Why you should invest in Julia now, as a Data Scientist](https://medium.com/@logankilpatrick/why-you-should-invest-in-julia-now-as-a-data-scientist-30dc346d62e4)
* [eBook: Introduction to Datascience with Julia](https://datascience-book.gitlab.io/)
* [FastAI.jl](https://github.com/FluxML/FastAI.jl) -> Repository of best practices for deep learning in Julia, inspired by fastai
* [Flux.jl](https://github.com/FluxML/Flux.jl) -> the ML library that doesn't make you tensor. Checkout [The Deep Learning with Julia book](https://github.com/logankilpatrick/DeepLearningWithJulia)
* [GDAL.jl](https://github.com/JuliaGeo/GDAL.jl) -> Thin Julia wrapper for GDAL
* [GeoInterface.jl](https://github.com/JuliaGeo/GeoInterface.jl) -> A Julia Protocol for Geospatial Data
* [GeoJSON.jl](https://github.com/JuliaGeo/GeoJSON.jl) -> Utilities for working with GeoJSON data
* [JuliaImages: image processing and machine vision for Julia](https://juliaimages.org/stable/)
* [Julia_Geospatial](https://github.com/acgeospatial/Julia_Geospatial) -> Examples for a blog series on Geospatial Julia using ArchGDAL
* [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl) -> A Julia machine learning framework
* [Proj4.jl](https://github.com/JuliaGeo/Proj4.jl) -> Julia wrapper for the PROJ cartographic projections library
* [RemoteS.jl](https://github.com/GenericMappingTools/RemoteS.jl) -> Remote sensing data processing
* [SatelliteToolbox.jl](https://github.com/JuliaSpace/SatelliteToolbox.jl) -> This package contains several functions to build simulations related with satellites
* [SatelliteDynamics.jl](https://github.com/sisl/SatelliteDynamics.jl) -> a satellite dynamics modeling package
* [Sentinel.jl](https://github.com/mhudecheck/Sentinel.jl) -> library for processing ESA Sentinel 2 satellite data

# Movers and shakers on Github
* [Adam Van Etten](https://github.com/avanetten) is doing interesting things in object detection and segmentation
* [Andrew Cutts](https://github.com/acgeospatial) cohosts the [Scene From Above podcast](https://scenefromabove.podbean.com) and has many interesting repos
* [Ankit Kariryaa](https://github.com/ankitkariryaa) published a recent nature paper on tree detection
* [Chris Holmes](https://github.com/cholmes) is doing great things at Planet
* [Christoph Rieke](https://github.com/chrieke) maintains a very popular imagery repo and has published his thesis on segmentation
* [Daniel J Dufour](https://github.com/DanielJDufour) builds [geotiff.io](https://geotiff.io/) and more
* [Daniel Moraite](https://daniel-moraite.medium.com/) is publishing some excellent articles
* [Even Rouault](https://github.com/rouault) maintains several of the most critical tools in this domain such as GDAL
* [Fatih Cagatay Akyon](https://github.com/fcakyon) aka fcakyon is maintaining sahi and many other useful projects
* [Gonzalo Mateo García](https://github.com/gonzmg88) is working on clouds and Water segmentation with CNNs
* [Isaac Corley](https://github.com/isaaccorley) is working on super-resolution and torchrs
* [Jake Shermeyer](https://github.com/jshermeyer) many interesting repos
* [Martin Black](https://github.com/mblackgeo) is tech lead at Sparkgeo in the UK
* [Maxime Lenormand](https://github.com/maximelenormand) authors the Minds Behind Maps podcast
* [Mike Skaug](https://github.com/mikeskaug) is a Data scientist at Aurora Insight
* [Mort Canty](https://github.com/mortcanty) is an expert in change detection
* [Mykola Kozyr](https://github.com/mykolakozyr) is working on streamlit apps
* [Nicholas Murray](https://www.murrayensis.org/) is an Australia-based scientist with a focus on delivering the science necessary to inform large scale environmental management and conservation
* [Oscar Mañas](https://oscmansan.github.io/) is advancing the state of the art in SSL
* [Qiusheng Wu](https://github.com/giswqs) is an Assistant Professor in the Department of Geography at the University of Tennessee, checkout his [YouTube channel](https://www.youtube.com/c/QiushengWu)
* [Rodrigo Caye Daudt](https://rcdaudt.github.io/oscd/) is doing great work on change detection
* [Robin Wilson](https://github.com/robintw) is a former academic who is very active in the satellite imagery space
* [Rohit Singh](https://github.com/rohitgeo) is Director of Esri R&D Center and has some great [Medium articles](https://medium.com/@rohitgeo)

# Companies & organisations on Github
For a full list of companies, on and off Github, checkout [awesome-geospatial-companies](https://github.com/chrieke/awesome-geospatial-companies). The following lists companies with interesting Github profiles.
* [AI. Reverie](https://github.com/aireveries) -> synthetic data
* [Airbus Defence And Space](https://github.com/AirbusDefenceAndSpace)
* [Agricultural Impacts Research Group](https://github.com/agroimpacts)
* [Applied-GeoSolutions](https://github.com/Applied-GeoSolutions)
* [Azavea](https://github.com/azavea) -> lots of interesting repos around STAC
* [CARTO](https://github.com/CartoDB) -> "The leading platform for Location Intelligence and Spatial Data Science"
* [Citymapper](https://github.com/citymapper)
* [Defense Innovation Unit (DIU)](https://github.com/DIUx-xView) -> run the xView challenges
* [Development Seed](https://github.com/developmentseed)
* [Descartes Labs](https://github.com/descarteslabs)
* [Dymaxion Labs](https://github.com/dymaxionlabs)
* [DHI GRAS](https://github.com/DHI-GRAS)
* [ElementAI](https://github.com/ElementAI)
* [Element 84](https://github.com/element84)
* [ESA-PhiLab](https://github.com/ESA-PhiLab)
* [Esri](https://github.com/Esri)
* [Geoalert](https://github.com/Geoalert) -> checkout their [Medium articles](https://medium.com/geoalert-platform-urban-monitoring)
* [Global Environmental Remote Sensing Laboratory](https://github.com/GERSL)
* [GIC-AIT](https://github.com/gicait) -> Aisan Institute of Technology
* [Hummingbird Technologies Ltd](https://github.com/HummingbirdTechGroup) -> sustainability and optimised food production
* [ICEYE](https://github.com/iceye-ltd)
* [indigo-ag](https://github.com/indigo-ag)
* [L3Harris Geospatial Solutions (HGSI)](https://github.com/hgsolutions)
* [Mapbox](https://github.com/mapbox) -> thanks for Rasterio!
* [Maxar-Analytics](https://github.com/maxar-analytics)
* [ml6team](https://github.com/ml6team)
* [NASA](https://github.com/nasa)
* [NASA-JPL](https://github.com/nasa-jpl) -> THE Jet Propulsion Laboratory
* [National Geospatial-Intelligence Agency USA](https://github.com/ngageoint)
* [Near Space Labs](https://github.com/nearspacelabs)
* [OpenGeoScales](https://github.com/OpenGeoScales)
* [Ordnance Survey](https://github.com/OrdnanceSurvey)
* [OroraTech](https://github.com/OroraTech)
* [Planet Labs](https://github.com/planetlabs) -> thanks for COGS!
* [Preligens](https://github.com/earthcube-lab) -> formerly Earthcube Lab
* [SatelliteVu](https://github.com/SatelliteVu) -> thermal imagery from space!
* [Sinergise](https://github.com/sentinel-hub) -> maintaining Sentinel-hub
* [SkyTruth](https://github.com/SkyTruth)
* [SpaceKnow](https://github.com/SpaceKnow)
* [Sparkgeo](https://github.com/sparkgeo)
* [up42](https://github.com/up42) -> Airbus spinout providing 'The easiest way to build geospatial solutions'
* [Vortexa](https://github.com/VorTECHsa) -> energy & shipping insights

# Courses
* [Introduction to Geospatial Raster and Vector Data with Python](https://carpentries-incubator.github.io/geospatial-python/aio/index.html) -> an intro course on a single page
* [Manning: Monitoring Changes in Surface Water Using Satellite Image Data](https://liveproject.manning.com/course/106/monitoring-changes-in-surface-water-using-satellite-image-data?)
* [Automating GIS processes](https://automating-gis-processes.github.io/2016/index.html) includes a lesson on automating raster data processing
* For deep learning checkout the [fastai course](https://course.fast.ai/) which uses the fastai library & pytorch
* [pyimagesearch.com](https://www.pyimagesearch.com/) hosts courses and plenty of material using opencv and keras
* Official opencv courses on opencv.org
* [TensorFlow Developer Professional Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice)
* [Geospatial_Python_CourseV1](https://github.com/acgeospatial/Geospatial_Python_CourseV1) -> a collection of blog posts turned into a course format
* [Satellite Machine Learning Training](http://devseed.com/sat-ml-training/) -> lessons on how to apply Machine Learning analysis to satellite data
* [DL-for-satellite-image-analysis](https://github.com/gicait/DL-for-satellite-image-analysis) -> short and minimalistic examples covering fundamentals of Deep Learning for Satellite Image Analysis using Jupyter notebooks, created by [lakmalnd](https://github.com/lakmalnd)
* [ESRI MOOC: Spatial Data Science: The New Frontier in Analytics](https://www.esri.com/training/catalog/5d76dcf7e9ccda09bef61294/spatial-data-science:-the-new-frontier-in-analytics/#!) -> Online OCT 27 - DEC 8, 2021
* [Machine Learning on Earth Observation: ML4EO Bootcamp](https://online.atingi.org/course/view.php?id=1107%27)
* [Artificial Intelligence (AI) for Earth Monitoring](https://www.futurelearn.com/courses/artificial-intelligence-for-earth-monitoring)
* [Deep Learning DIY](https://dataflowr.github.io/website/)

# Books
* [Image Analysis, Classification and Change Detection in Remote Sensing With Algorithms for Python, Fourth Edition, By Morton John Canty](https://www.routledge.com/Image-Analysis-Classification-and-Change-Detection-in-Remote-Sensing-With/Canty/p/book/9781138613225) -> code [here](https://github.com/mortcanty/CRC4Docker)
* I highly recommend [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) by François Chollet
* [Practical Deep Learning for Cloud, Mobile & Edge](https://github.com/PracticalDL/Practical-Deep-Learning-Book)
* [d2l.ai](https://www.d2l.ai/) -> Interactive deep learning book with code, math, and discussions
* [interpretable-ml-book](https://github.com/christophM/interpretable-ml-book)
* [pyGIS](https://github.com/mmann1123/pyGIS) -> an online textbook covering all the core geospatial functionality available in Python

# Podcasts
* [Scene From Above Podcast](https://scenefromabove.podbean.com/)
* [Mapscaping podcast](https://mapscaping.com/blogs/the-mapscaping-podcast)
* [Minds Behind Maps](https://minds-behind-maps.simplecast.com/)

# Online communities
* [fast AI geospatial study group](https://forums.fast.ai/t/geospatial-deep-learning-resources-study-group/31044)
* [Kaggle Intro to Satellite imagery Analysis group](https://www.kaggle.com/getting-started/131455)
* [Omdena](https://omdena.com/) brings together small teams of engineers to work on AI projects

# Jobs
Signup for the [geospatial-jobs-newsletter](https://geospatial.substack.com/p/welcome-to-geospatial-jobs-newsletter) and [Pangeo discourse](https://discourse.pangeo.io/c/news/jobs) lists multiple jobs, global. List of companies job portals below:
* [Capella Space](https://apply.workable.com/capellaspace/)
* [Development SEED](http://devseed.com/careers)
* [Planet](https://boards.greenhouse.io/planetlabs)
* [Sparkgeo](https://sparkgeo.com/jobs/)

# About the author
My background is in optical physics, and I hold a PhD from Cambridge on the topic of [localised surface Plasmons](https://pubs.acs.org/doi/abs/10.1021/nl0710506). Since academia I have held a variety of roles, including doing research at [Sharp Labs Europe](https://www.sle.sharp.co.uk/), developing optical systems at [Surrey Satellites](https://www.sstl.co.uk/) (SSTL), and working at an IOT startup. It was whilst at SSTL that I started this repository as a personal resource. Over time I have steadily gravitated towards data analytics and software engineering with python, and I now work as a senior data scientist at [Satellite Vu](https://www.satellitevu.com/). Please feel free to connect with me on Twitter & LinkedIn, and please do let me know if this repository is useful to your work.

<!-- markdown-link-check-disable -->
[![Linkedin: robmarkcole](https://img.shields.io/badge/-Robin%20Cole-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/robmarkcole/)](https://www.linkedin.com/in/robmarkcole/)
[![Twitter Follow](https://img.shields.io/twitter/follow/robmarkcole?label=Follow)](https://twitter.com/robmarkcole)
<!-- markdown-link-check-enable -->
