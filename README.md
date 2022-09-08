<p align="center">
<img src="logo.png" width="700">
</p>

This page lists resources for performing deep learning on satellite imagery. To a lesser extent classical Machine learning (e.g. random forests) are also discussed, as are classical image processing techniques. Note there is a huge volume of academic literature published on these topics, and this repository does not seek to index them all but rather list approachable resources with published code that will benefit both the research *and* developer communities. If you find this work useful please give it a star and consider sponsoring it. You can also follow me on Twitter and LinkedIn where I aim to post frequent updates on my new discoveries, and I have created a dedicated [group on LinkedIn](https://www.linkedin.com/groups/12698393/). I have also started a blog [here](https://robmarkcole.com/) and have published a post on the history of this repository called [Dissecting the satellite-image-deep-learning repo](https://robmarkcole.com/markdown/2022/07/24/satellite-image-deep-learning.html) If you use this work in your research please cite using the citation information on the right. Thanks!

<!-- markdown-link-check-disable -->
[![Twitter Follow](https://img.shields.io/twitter/follow/robmarkcole?label=Follow)](https://twitter.com/robmarkcole)
[![Linkedin: robmarkcole](https://img.shields.io/badge/-Robin%20Cole-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/robmarkcole/)](https://www.linkedin.com/in/robmarkcole/)
[![Google Scholar Badge](https://img.shields.io/badge/Google-Scholar-red)](https://scholar.google.com/citations?user=oHe5ozwAAAAJ&hl=en)
<!-- markdown-link-check-enable -->

# Table of contents
* [Techniques](https://github.com/robmarkcole/satellite-image-deep-learning#techniques)
  * [Classification](https://github.com/robmarkcole/satellite-image-deep-learning#Classification)
  * [Segmentation](https://github.com/robmarkcole/satellite-image-deep-learning#Segmentation)
    * [Land use & land cover](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---land-use--land-cover)
    * [Vegetation, crops & crop boundaries](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---vegetation-crops--crop-boundaries)
    * [Water, coastlines & floods](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---water-coastlines--floods)
    * [Fire, smoke & burn areas](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---fire-smoke--burn-areas)
    * [Landslides](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---landslides)
    * [Glaciers](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---glaciers)
    * [Other environmental](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---other-environmental)
    * [Roads](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---roads)
    * [Buildings & rooftops](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---buildings--rooftops)
    * [Solar panels](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---solar-panels)
    * [Electrical substations](https://github.com/robmarkcole/satellite-image-deep-learning#segmentation---electrical-substations)
    * [Instance segmentation](https://github.com/robmarkcole/satellite-image-deep-learning#instance-segmentation)
    * [Panoptic segmentation](https://github.com/robmarkcole/satellite-image-deep-learning#panoptic-segmentation)
  * [Object detection](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection)
    * [Object counting](https://github.com/robmarkcole/satellite-image-deep-learning#object-counting)
    * [Object detection with rotated bounding boxes](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection-with-rotated-bounding-boxes)
    * [Object detection enhanced by super resolution](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection-enhanced-by-super-resolution)
    * [Salient object detection](https://github.com/robmarkcole/satellite-image-deep-learning#salient-object-detection)
    * [Buildings, rooftops & solar panels](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection---buildings-rooftops--solar-panels)
    * [Ships & boats](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection---ships--boats)
    * [Cars, vehicles & trains](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection---cars-vehicles--trains)
    * [Planes & aircraft](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection---planes--aircraft)
    * [Infrastructure & utilities](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection---infrastructure--utilities)
    * [Animals](https://github.com/robmarkcole/satellite-image-deep-learning#object-detection---animals)
  * [Counting trees](https://github.com/robmarkcole/satellite-image-deep-learning#counting-trees)
  * [Oil storage tank detection & oil spills](https://github.com/robmarkcole/satellite-image-deep-learning#oil-storage-tank-detection--oil-spills)
  * [Cloud detection & removal](https://github.com/robmarkcole/satellite-image-deep-learning#cloud-detection--removal)
  * [Change detection](https://github.com/robmarkcole/satellite-image-deep-learning#change-detection)
  * [Time series](https://github.com/robmarkcole/satellite-image-deep-learning#time-series)
  * [Crop yield](https://github.com/robmarkcole/satellite-image-deep-learning#crop-yield)
  * [Wealth and economic activity](https://github.com/robmarkcole/satellite-image-deep-learning#wealth-and-economic-activity)
  * [Disaster response](https://github.com/robmarkcole/satellite-image-deep-learning#disaster-response)
  * [Weather phenomena](https://github.com/robmarkcole/satellite-image-deep-learning#weather-phenomena)
  * [Super-resolution](https://github.com/robmarkcole/satellite-image-deep-learning#super-resolution)
  * [Pansharpening](https://github.com/robmarkcole/satellite-image-deep-learning#pansharpening)
  * [Image-to-image translation](https://github.com/robmarkcole/satellite-image-deep-learning#image-to-image-translation)
  * [GANS](https://github.com/robmarkcole/satellite-image-deep-learning#gans)
  * [Transformers](https://github.com/robmarkcole/satellite-image-deep-learning#transformers)
  * [Adversarial ML](https://github.com/robmarkcole/satellite-image-deep-learning#adversarial-ml)
  * [Autoencoders, dimensionality reduction, image embeddings & similarity search](https://github.com/robmarkcole/satellite-image-deep-learning#autoencoders-dimensionality-reduction-image-embeddings--similarity-search)
  * [Image retreival](https://github.com/robmarkcole/satellite-image-deep-learning#image-retreival)
  * [Image Captioning & Visual Question Answering](https://github.com/robmarkcole/satellite-image-deep-learning#image-captioning--visual-question-answering)
  * [Mixed data learning](https://github.com/robmarkcole/satellite-image-deep-learning#mixed-data-learning)
  * [Few-shot learning](https://github.com/robmarkcole/satellite-image-deep-learning#few-shot-learning)
  * [Self-supervised, unsupervised & contrastive learning](https://github.com/robmarkcole/satellite-image-deep-learning#self-supervised-unsupervised--contrastive-learning)
  * [Weakly & semi-supervised learning](https://github.com/robmarkcole/satellite-image-deep-learning#weakly--semi-supervised-learning)
  * [Active learning](https://github.com/robmarkcole/satellite-image-deep-learning#active-learning)
  * [Federated learning](https://github.com/robmarkcole/satellite-image-deep-learning#federated-learning)
  * [Image registration](https://github.com/robmarkcole/satellite-image-deep-learning#image-registration)
  * [Data fusion](https://github.com/robmarkcole/satellite-image-deep-learning#data-fusion)
  * [Terrain mapping, Disparity Estimation, Lidar, DEMs & NeRF](https://github.com/robmarkcole/satellite-image-deep-learning#terrain-mapping-disparity-estimation-lidar-dems--nerf)
  * [Thermal Infrared](https://github.com/robmarkcole/satellite-image-deep-learning#thermal-infrared)
  * [SAR](https://github.com/robmarkcole/satellite-image-deep-learning#sar)
  * [NVDI - vegetation index](https://github.com/robmarkcole/satellite-image-deep-learning#nvdi---vegetation-index)
  * [General image quality](https://github.com/robmarkcole/satellite-image-deep-learning#general-image-quality)
* [Neural nets in space](https://github.com/robmarkcole/satellite-image-deep-learning#neural-nets-in-space)
* [ML best practice](https://github.com/robmarkcole/satellite-image-deep-learning#ml-best-practice)
* [Metrics](https://github.com/robmarkcole/satellite-image-deep-learning#metrics)
* [Datasets](https://github.com/robmarkcole/satellite-image-deep-learning#datasets)
  * [Lists of datasets](https://github.com/robmarkcole/satellite-image-deep-learning#lists-of-datasets)
  * [Sentinel](https://github.com/robmarkcole/satellite-image-deep-learning#sentinel)
  * [Landsat](https://github.com/robmarkcole/satellite-image-deep-learning#landsat)
  * [Maxar](https://github.com/robmarkcole/satellite-image-deep-learning#maxar)
  * [Planet](https://github.com/robmarkcole/satellite-image-deep-learning#planet)
  * [EuroSAT](https://github.com/robmarkcole/satellite-image-deep-learning#eurosat)
  * [UAV & Drone datasets](https://github.com/robmarkcole/satellite-image-deep-learning#uav--drone-datasets)
  * [Other datasets](https://github.com/robmarkcole/satellite-image-deep-learning#other-datasets)
  * [Kaggle](https://github.com/robmarkcole/satellite-image-deep-learning#kaggle)
* [Synthetic data](https://github.com/robmarkcole/satellite-image-deep-learning#synthetic-data)
* [Online platforms for analytics](https://github.com/robmarkcole/satellite-image-deep-learning#online-platforms-for-analytics)
* [Free online compute](https://github.com/robmarkcole/satellite-image-deep-learning#free-online-compute)
* [State of the art engineering](https://github.com/robmarkcole/satellite-image-deep-learning#state-of-the-art-engineering)
* [Cloud providers](https://github.com/robmarkcole/satellite-image-deep-learning#cloud-providers)
  * [AWS](https://github.com/robmarkcole/satellite-image-deep-learning#aws)
  * [Google Cloud](https://github.com/robmarkcole/satellite-image-deep-learning#google-cloud)
  * [Microsoft Azure](https://github.com/robmarkcole/satellite-image-deep-learning#microsoft-azure)
* [Deploying models](https://github.com/robmarkcole/satellite-image-deep-learning#deploying-models)
* [Image annotation](https://github.com/robmarkcole/satellite-image-deep-learning#image-annotation)
* [Open source software](https://github.com/robmarkcole/satellite-image-deep-learning#open-source-software)
  * [General utilities](https://github.com/robmarkcole/satellite-image-deep-learning#general-utilities)
  * [Low level numerical & data formats](https://github.com/robmarkcole/satellite-image-deep-learning#low-level-numerical--data-formats)
  * [Image processing, handling, manipulation](https://github.com/robmarkcole/satellite-image-deep-learning#image-processing-handling-manipulation)
  * [Image chipping/tiling & merging](https://github.com/robmarkcole/satellite-image-deep-learning#image-chippingtiling--merging)
  * [Image dataset creation](https://github.com/robmarkcole/satellite-image-deep-learning#image-dataset-creation)
  * [Image augmentation packages](https://github.com/robmarkcole/satellite-image-deep-learning#image-augmentation-packages)
  * [Image formats, data management and catalogues](https://github.com/robmarkcole/satellite-image-deep-learning#image-formats-data-management-and-catalogues)
  * [Deep learning packages, frameworks & projects](https://github.com/robmarkcole/satellite-image-deep-learning#deep-learning-packages-frameworks--projects)
  * [Model tracking, versioning, specification & compilation](https://github.com/robmarkcole/satellite-image-deep-learning#model-tracking-versioning-specification--compilation)
  * [Graphing and visualisation](https://github.com/robmarkcole/satellite-image-deep-learning#graphing-and-visualisation)
  * [Algorithms](https://github.com/robmarkcole/satellite-image-deep-learning#algorithms)
  * [GDAL & Rasterio](https://github.com/robmarkcole/satellite-image-deep-learning#gdal--rasterio)
  * [Cloud Optimised GeoTiff (COG)](https://github.com/robmarkcole/satellite-image-deep-learning#cloud-optimised-geotiff-cog)
  * [SpatioTemporal Asset Catalog specification (STAC)](https://github.com/robmarkcole/satellite-image-deep-learning#spatiotemporal-asset-catalog-specification-stac)
  * [OpenStreetMap](https://github.com/robmarkcole/satellite-image-deep-learning#openstreetmap)
  * [QGIS](https://github.com/robmarkcole/satellite-image-deep-learning#qgis)
  * [Jupyter](https://github.com/robmarkcole/satellite-image-deep-learning#jupyter)
  * [Streamlit](https://github.com/robmarkcole/satellite-image-deep-learning#streamlit)
  * [Julia language](https://github.com/robmarkcole/satellite-image-deep-learning#julia-language)
* [Movers and shakers on Github](https://github.com/robmarkcole/satellite-image-deep-learning#movers-and-shakers-on-github)
* [Companies & organisations on Github](https://github.com/robmarkcole/satellite-image-deep-learning#companies--organisations-on-github)
* [Courses](https://github.com/robmarkcole/satellite-image-deep-learning#courses)
* [Books](https://github.com/robmarkcole/satellite-image-deep-learning#books)
* [Bedtime reading](https://github.com/robmarkcole/satellite-image-deep-learning#bedtime-reading)
* [Podcasts](https://github.com/robmarkcole/satellite-image-deep-learning#podcasts)
* [Newsletters](https://github.com/robmarkcole/satellite-image-deep-learning#newsletters)
* [Online communities](https://github.com/robmarkcole/satellite-image-deep-learning#online-communities)
* [Jobs](https://github.com/robmarkcole/satellite-image-deep-learning#jobs)

# Techniques
This section explores the different deep and machine learning (ML) techniques applied to common problems in satellite imagery analysis. Good background reading is [Deep learning in remote sensing applications: A meta-analysis and review](https://www.iges.or.jp/en/publication_documents/pub/peer/en/6898/Ma+et+al+2019.pdf)

## Classification
The classic cats vs dogs image classification task, which in the remote sensing domain is used to assign a label to an image, e.g. this is an image of a forest. The more complex case is applying multiple labels to an image. This approach of image level classification is not to be confused with pixel-level classification which is called semantic segmentation. In general, aerial images cover large geographical areas that include multiple classes of land, so treating this is as a classification problem is less common than using semantic segmentation. I recommend to get started with the EuroSAT dataset.
* Land classification on Sentinel 2 data using a [simple sklearn cluster algorithm](https://github.com/acgeospatial/Satellite_Imagery_Python/blob/master/Clustering_KMeans-Sentinel2.ipynb) or [deep learning CNN](https://towardsdatascience.com/land-use-land-cover-classification-with-deep-learning-9a5041095ddb)
* Land Use Classification on Merced dataset using CNN [in Keras](https://github.com/tavgreen/landuse_classification)
or [fastai](https://medium.com/spatial-data-science/deep-learning-for-geospatial-data-applications-multi-label-classification-2b0a1838fcf3). Also checkout [Multi-label Land Cover Classification](https://towardsdatascience.com/multi-label-land-cover-classification-with-deep-learning-d39ce2944a3d) using the redesigned multi-label Merced dataset with 17 land cover classes
* [Multi-Label Classification of Satellite Photos of the Amazon Rainforest using keras](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/) or [FastAI](https://towardsdatascience.com/fastai-multi-label-image-classification-8034be646e95)
* [Detecting Informal Settlements from Satellite Imagery using fine-tuning of ResNet-50 classifier](https://blog.goodaudience.com/detecting-informal-settlements-using-satellite-imagery-and-convolutional-neural-networks-d571a819bf44) with [repo](https://github.com/dymaxionlabs/ap-latam)
* [Land-Cover-Classification-using-Sentinel-2-Dataset](https://github.com/raoofnaushad/Land-Cover-Classification-using-Sentinel-2-Dataset) -> [well written Medium article](https://raoofnaushad7.medium.com/applying-deep-learning-on-satellite-imagery-classification-5f2588b932c1) accompanying this repo but using the EuroSAT dataset
* [Land Cover Classification of Satellite Imagery using Convolutional Neural Networks](https://towardsdatascience.com/land-cover-classification-of-satellite-imagery-using-convolutional-neural-networks-91b5bb7fe808) using Keras and a multi spectral dataset captured over vineyard fields of Salinas Valley, California
* [Detecting deforestation from satellite images](https://towardsdatascience.com/detecting-deforestation-from-satellite-images-7aa6dfbd9f61) -> using FastAI and ResNet50, with repo [fsdl_deforestation_detection](https://github.com/karthikraja95/fsdl_deforestation_detection)
* [Neural Network for Satellite Data Classification Using Tensorflow in Python](https://towardsdatascience.com/neural-network-for-satellite-data-classification-using-tensorflow-in-python-a13bcf38f3e1) -> A step-by-step guide for Landsat 5 multispectral data classification for binary built-up/non-built-up class prediction, with [repo](https://github.com/PratyushTripathy/Landsat-Classification-Using-Neural-Network)
* [Slums mapping from pretrained CNN network](https://github.com/deepankverma/slums_detection) on VHR (Pleiades: 0.5m) and MR (Sentinel: 10m) imagery
* [Comparing urban environments using satellite imagery and convolutional neural networks](https://github.com/adrianalbert/urban-environments) -> includes interesting study of the image embedding features extracted for each image on the Urban Atlas dataset. Accompanying [paper](https://www.researchgate.net/publication/315882788_Using_convolutional_networks_and_satellite_imagery_to_identify_patterns_in_urban_environments_at_a_large_scale)
* [RSI-CB](https://github.com/lehaifeng/RSI-CB) -> A Large Scale Remote Sensing Image Classification Benchmark via Crowdsource Data. See also [Remote-sensing-image-classification](https://github.com/aashishrai3799/Remote-sensing-image-classification)
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
* [remote-sense-quickstart](https://github.com/CarryHJR/remote-sense-quickstart) -> classification on a number of datasets, including with attention visualization
* [Satellite image classification using multiple machine learning algorithms](https://github.com/tanmay-delhikar/satellite-image-analysis-ml)
* [satsense](https://github.com/DynaSlum/satsense) -> a Python library for land use/cover classification using classical features including HoG & NDVI
* [PyTorch_UCMerced_LandUse](https://github.com/GeneralLi95/PyTorch_UCMerced_LandUse) -> simple pytorch implementation fine tuned on ResNet and basic augmentations
* [EuroSAT-image-classification](https://github.com/artemisart/EuroSAT-image-classification) -> simple pytorch implementation fine tuned on ResNet
* [landcover_classification](https://github.com/reidfalconer/landcover_classification) -> using fast.ai on EuroSAT
* [IGARSS2020_BWMS](https://github.com/jiankang1991/IGARSS2020_BWMS) -> Band-Wise Multi-Scale CNN Architecture for Remote Sensing Image Scene Classification with a novel CNN architecture for the feature embedding of high-dimensional RS images
* [image.classification.on.EuroSAT](https://github.com/canturan10/image.classification.on.EuroSAT) -> solution in pure pytorch
* [hurricane_damage](https://github.com/allankapoor/hurricane_damage) -> Post-hurricane structure damage assessment based on aerial imagery with CNN
* [openai-drivendata-challenge](https://github.com/buildwithcycy/openai-drivendata-challenge) -> Using deep learning to classify the building material of rooftops (aerial imagery from South America)
* [is-it-abandoned](https://github.com/zach-brown-18/is-it-abandoned) -> Can we tell if a house is abandoned based on aerial LIDAR imagery?
* [BoulderAreaDetector](https://github.com/pszemraj/BoulderAreaDetector) -> CNN to classify whether a satellite image shows an area would be a good rock climbing spot or not
* [ISPRS_S2FL](https://github.com/danfenghong/ISPRS_S2FL) -> code for paper: Multimodal Remote Sensing Benchmark Datasets for Land Cover Classification with A Shared and Specific Feature Learning Model. S2FL is capable of decomposing multimodal RS data into modality-shared and modality-specific components, enabling the information blending of multi-modalities more effectively
* [Brazilian-Coffee-Detection](https://github.com/MrSquidward/Brazilian-Coffee-Detection) -> uses Keras with public dataset
* [tf-crash-severity](https://github.com/SoySauceNZ/tf-crash-severity) -> predict the crash severity for given road features contained within satellite images
* [ensemble_LCLU](https://github.com/burakekim/ensemble_LCLU) -> code for 2021 [paper](https://www.tandfonline.com/doi/full/10.1080/17538947.2021.1980125): Deep neural network ensembles for remote sensing land cover and land use classification
* [cerraNet](https://github.com/MirandaMat/cerraNet-v2) -> contextually classify the types of use and coverage in the Brazilian Cerrado
* [Urban-Analysis-Using-Satellite-Imagery](https://github.com/mominali12/Urban-Analysis-Using-Satellite-Imagery) -> classify urban area as planned or unplanned using a combination of segmentation and classification
* [ChipClassification](https://github.com/yurithefury/ChipClassification) -> code for 2019 [paper](https://www.sciencedirect.com/science/article/pii/S0924271619302023): Deep learning for multi-modal classification of cloud, shadow and land cover scenes in PlanetScope and Sentinel-2 imagery
* [DeeplearningClassficationLandsat-tImages](https://github.com/VinayarajPoliyapram/DeeplearningClassficationLandsat-tImages) -> Water/Ice/Land Classification Using Large-Scale Medium Resolution Landsat Satellite Images
* [wildfire-detection-from-satellite-images-ml](https://github.com/shrey24/wildfire-detection-from-satellite-images-ml) -> detect whether an image contains a wildfire, with example flask web app
* [mining-discovery-with-deep-learning](https://github.com/remis/mining-discovery-with-deep-learning) -> code for the 2020 paper: Mining and Tailings Dam Detection in Satellite Imagery Using Deep Learning
* [e-Farmerce-platform](https://github.com/efarmerce/e-Farmerce-platform) -> classify crop type
* [sentinel2-deep-learning](https://github.com/d-smit/sentinel2-deep-learning) -> Novel Training Methodologies for Land Classification of Sentinel-2 Imagery
* [RSSC-transfer](https://github.com/risojevicv/RSSC-transfer) -> code for 2021 [paper](https://arxiv.org/abs/2111.03690): The Role of Pre-Training in High-Resolution Remote Sensing Scene Classification
* [Classifying Geo-Referenced Photos and Satellite Images for Supporting Terrain Classification](https://github.com/jorgemspereira/Classifying-Geo-Referenced-Photos) -> detect floods
* [Pay-More-Attention](https://github.com/williamzhao95/Pay-More-Attention) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9157951): Remote Sensing Image Scene Classification Based on an Enhanced Attention Module
* [Remote-Sensing-Image-Classification-via-Improved-Cross-Entropy-Loss-and-Transfer-Learning-Strategy](https://github.com/AliBahri94/Remote-Sensing-Image-Classification-via-Improved-Cross-Entropy-Loss-and-Transfer-Learning-Strategy) -> code for 2019 [paper](https://ieeexplore.ieee.org/abstract/document/8844264): Remote Sensing Image Classification via Improved Cross-Entropy Loss and Transfer Learning Strategy Based on Deep Convolutional Neural Networks
* [DenseNet40-for-HRRSISC](https://github.com/BiQiWHU/DenseNet40-for-HRRSISC) -> DenseNet40 for remote sensing image scene classification, uses UC Merced Dataset
* [SKAL](https://github.com/hw2hwei/SKAL) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9298485): Looking Closer at the Scene: Multiscale Representation Learning for Remote Sensing Image Scene Classification
* [potsdam-tensorflow-practice](https://github.com/medicinely/potsdam-tensorflow-practice) -> image classification of Potsdam dataset using tensorflow
* [SAFF](https://github.com/zh-hike/SAFF) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/8982033): Self-Attention-Based Deep Feature Fusion for Remote Sensing Scene Classification
* [GLNET](https://github.com/wuchangsheng951/GLNET) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9607791): Convolutional Neural Networks Based Remote Sensing Scene Classification under Clear and Cloudy Environments
* [Remote-sensing-image-classification](https://github.com/hiteshK03/Remote-sensing-image-classification) -> transfer learning using pytorch to classify remote sensing data into three classes: aircrafts, ships, none
* [remote_sensing_pretrained_models](https://github.com/lsh1994/remote_sensing_pretrained_models) -> as an alternative to fine tuning on models pretrained on ImageNet, here some CNN are pretrained on the RSD46-WHU & AID datasets
* [CNN_AircraftDetection](https://github.com/UKMIITB/CNN_AircraftDetection) -> CNN for aircraft detection in satellite images using keras
* [OBIC-GCN](https://github.com/CVEO/OBIC-GCN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9411513): Object-based Classification Framework of Remote Sensing Images with Graph Convolutional Networks
* [aitlas-arena](https://github.com/biasvariancelabs/aitlas-arena) -> An open-source benchmark framework for evaluating state-of-the-art deep learning approaches for image classification in Earth Observation (EO)
* [droughtwatch](https://github.com/wandb/droughtwatch) -> code for 2020 [paper](https://arxiv.org/abs/2004.04081): Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya
* [JSTARS_2020_DPN-HRA](https://github.com/B-Xi/JSTARS_2020_DPN-HRA) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9126161): Deep Prototypical Networks With Hybrid Residual Attention for Hyperspectral Image Classification
* [SIGNA](https://github.com/kyle-one/SIGNA) -> code for 2022 [paper](https://arxiv.org/abs/2208.02613): Semantic Interleaving Global Channel Attention for Multilabel Remote Sensing Image Classification

## Segmentation
 Segmentation will assign a class label to each **pixel** in an image. Segmentation is typically grouped into semantic, instance or panoptic segmentation. In semantic segmentation objects of the same class are assigned the same label, whilst in instance segmentation each object is assigned a unique label. Panoptic segmentation combines instance and semantic predictions. Read this [beginner’s guide to segmentation](https://medium.com/gsi-technology/a-beginners-guide-to-segmentation-in-satellite-images-9c00d2028d52). Single class models are often trained for road or building segmentation, with multi class for land use/crop type classification. Image annotation can take longer than for object detection since every pixel must be annotated. **Note** that many articles which refer to 'hyperspectral land classification' are actually describing semantic segmentation. Note that cloud detection can be addressed with semantic segmentation and has its own section [Cloud detection & removal](https://github.com/robmarkcole/satellite-image-deep-learning#cloud-detection--removal)
* [awesome-satellite-images-segmentation](https://github.com/mrgloom/awesome-semantic-segmentation#satellite-images-segmentation)
* [Satellite Image Segmentation: a Workflow with U-Net](https://medium.com/vooban-ai/satellite-image-segmentation-a-workflow-with-u-net-7ff992b2a56e) is a decent intro article
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) -> Semantic Segmentation Toolbox with support for many remote sensing datasets including LoveDA
, Potsdam, Vaihingen & iSAID
* [How to create a DataBlock for Multispectral Satellite Image Semantic Segmentation using Fastai](https://towardsdatascience.com/how-to-create-a-datablock-for-multispectral-satellite-image-segmentation-with-the-fastai-v2-bc5e82f4eb5)
* [Using a U-Net for image segmentation, blending predicted patches smoothly is a must to please the human eye](https://github.com/Vooban/Smoothly-Blend-Image-Patches) -> python code to blend predicted patches smoothly. See [Satellite-Image-Segmentation-with-Smooth-Blending](https://github.com/MaitrySinha21/Satellite-Image-Segmentation-with-Smooth-Blending)
* [DCA](https://github.com/Luffy03/DCA) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9745130): Deep Covariance Alignment for Domain Adaptive Remote Sensing Image Segmentation
* [SCAttNet](https://github.com/lehaifeng/SCAttNet) -> Semantic Segmentation Network with Spatial and Channel Attention Mechanism
* [unetseg](https://github.com/dymaxionlabs/unetseg) -> A set of classes and CLI tools for training a semantic segmentation model based on the U-Net architecture, using Tensorflow and Keras. This implementation is tuned specifically for satellite imagery and other geospatial raster data
* [Semantic Segmentation of Satellite Imagery using U-Net & fast.ai](https://medium.com/dataseries/image-semantic-segmentation-of-satellite-imagery-using-u-net-e99ae13cf464) -> with [repo](https://github.com/raoofnaushad/Image-Semantic-Segmentation-of-Satellite-Imagery-using-U-Net.)
* [clusternet_segmentation](https://github.com/zhygallo/clusternet_segmentation) -> Unsupervised Segmentation by applying K-Means clustering to the features generated by Neural Network
* [Collection of different Unet Variant](https://github.com/ashishpatel26/satellite-Image-Semantic-Segmentation-Unet-Tensorflow-keras) -> demonstrates VggUnet, ResUnet, DenseUnet, Unet. AttUnet, MobileNetUnet, NestedUNet, R2AttUNet, R2UNet, SEUnet, scSEUnet, Unet_Xception_ResNetBlock, in keras
* [Efficient-Transformer](https://github.com/zyxu1996/Efficient-Transformer) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/18/3585): Efficient Transformer for Remote Sensing Image Segmentation
* [weakly_supervised](https://github.com/LobellLab/weakly_supervised) -> code for the 2020 [paper](https://www.mdpi.com/2072-4292/12/2/207): Weakly Supervised Deep Learning for Segmentation of Remote Sensing Imagery
* [HRCNet-High-Resolution-Context-Extraction-Network](https://github.com/zyxu1996/HRCNet-High-Resolution-Context-Extraction-Network) -> code to 2021 [paper](https://www.mdpi.com/2072-4292/13/1/71): High-Resolution Context Extraction Network for Semantic Segmentation of Remote Sensing Images
* [Semantic segmentation of SAR images using a self supervised technique](https://github.com/cattale93/pytorch_self_supervised_learning)
* [satellite-segmentation-pytorch](https://github.com/obravo7/satellite-segmentation-pytorch) -> explores a wide variety of image augmentations to increase training dataset size
* [IEEE_TGRS_SpectralFormer](https://github.com/danfenghong/IEEE_TGRS_SpectralFormer) -> code for 2021 [paper](https://arxiv.org/abs/2107.02988): Spectralformer: Rethinking hyperspectral image classification with transformers
* [Unsupervised Segmentation of Hyperspectral Remote Sensing Images with Superpixels](https://github.com/mpBarbato/Unsupervised-Segmentation-of-Hyperspectral-Remote-Sensing-Images-with-Superpixels) -> code for 2022 [paper](https://arxiv.org/abs/2204.12296)
* [Semantic-Segmentation-with-Sparse-Labels](https://github.com/Hua-YS/Semantic-Segmentation-with-Sparse-Labels) -> codes and data for learning from sparse annotations
* [SNDF](https://github.com/mi18/SNDF) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271619302606): Superpixel-enhanced deep neural forest for remote sensing image semantic segmentation
* [Satellite-Image-Classification](https://github.com/yxian29/Satellite-Image-Classification) -> using random forest or support vector machines (SVM) and sklearn
* [dynamic-rs-segmentation](https://github.com/keillernogueira/dynamic-rs-segmentation) -> code for 2019 [paper](https://arxiv.org/abs/1804.04020): Dynamic Multi-Context Segmentation of Remote Sensing Images based on Convolutional Networks
* [Remote-sensing-image-semantic-segmentation-tf2](https://github.com/TachibanaYoshino/Remote-sensing-image-semantic-segmentation-tf2) -> remote sensing image semantic segmentation repository based on tf.keras includes backbone networks such as resnet, densenet, mobilenet, and segmentation networks such as deeplabv3+, pspnet, panet, and refinenet
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) -> Segmentation models with pretrained backbones, has been used in multiple winning solutions to remote sensing competitions
* [SSRN](https://github.com/zilongzhong/SSRN) -> code for 2017 [paper](https://ieeexplore.ieee.org/document/8061020): Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework
* [SO-DNN](https://github.com/PanXinZebra/SO-DNN) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002525): Simplified object-based deep neural network for very high resolution remote sensing image classification
* [SANet](https://github.com/mrluin/SANet-PyTorch) -> code for 2019 [paper](https://arxiv.org/abs/1907.03089): Scale-Aware Network for Semantic Segmentation of High-Resolution Aerial Images
* [aerial-segmentation](https://github.com/alpemek/aerial-segmentation) -> code for 2017 [paper](https://arxiv.org/abs/1707.06879): Learning Aerial Image Segmentation from Online Maps
* [IterativeSegmentation](https://github.com/gaudetcj/IterativeSegmentation) -> code for 2016 [paper](https://arxiv.org/abs/1608.03440): Recurrent Neural Networks to Correct Satellite Image Classification Maps
* [Detectron2 FPN + PointRend Model for amazing Satellite Image Segmentation](https://affine.medium.com/detectron2-fpn-pointrend-model-for-amazing-satellite-image-segmentation-183456063e15) -> 15% increase in accuracy when compared to the U-Net model
* [HybridSN](https://github.com/gokriznastic/HybridSN) -> code for 2019 [paper](https://arxiv.org/abs/1902.06701): HybridSN: Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification. Also a [pytorch implementation here](https://github.com/purbayankar/HybridSN-pytorch)
* [TNNLS_2022_X-GPN](https://github.com/B-Xi/TNNLS_2022_X-GPN) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9740412): Semisupervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification
* [singleSceneSemSegTgrs2022](https://github.com/sudipansaha/singleSceneSemSegTgrs2022) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9773162): Unsupervised Single-Scene Semantic Segmentation for Earth Observation
* [A-Fast-and-Compact-3-D-CNN-for-HSIC](https://github.com/mahmad00/A-Fast-and-Compact-3-D-CNN-for-HSIC) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9307220): A Fast and Compact 3-D CNN for Hyperspectral Image Classification
* [HSNRS](https://github.com/Walkerlikesfish/HSNRS) -> code for 2017 [paper](https://www.mdpi.com/2072-4292/9/6/522): Hourglass-ShapeNetwork Based Semantic Segmentation for High Resolution Aerial Imagery
* [GiGCN](https://github.com/ShuGuoJ/GiGCN) -> code for 2022 [paper](https://pubmed.ncbi.nlm.nih.gov/35724277/): Graph-in-Graph Convolutional Network for Hyperspectral Image Classification
* [SSAN](https://github.com/EtPan/SSAN) -> code for 2019 [paper](https://www.mdpi.com/2072-4292/11/8/963): Spectral-Spatial Attention Networks for Hyperspectral Image Classification
* [drone-images-semantic-segmentation](https://github.com/ayushdabra/drone-images-semantic-segmentation) -> Multiclass Semantic Segmentation of Aerial Drone Images Using Deep Learning
* [Satellite-Image-Segmentation-with-Smooth-Blending](https://github.com/MaitrySinha21/Satellite-Image-Segmentation-with-Smooth-Blending) -> uses [Smoothly-Blend-Image-Patches](https://github.com/Vooban/Smoothly-Blend-Image-Patches)
* [BayesianUNet](https://github.com/tha-santacruz/BayesianUNet) -> Pytorch Bayesian UNet model for segmentation and uncertainty prediction, applied to the Potsdam Dataset
* [RAANet](https://github.com/Lrr0213/RAANet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/13/3109): RAANet: A Residual ASPP with Attention Framework for Semantic Segmentation of High-Resolution Remote Sensing Images
* [wheelRuts_semanticSegmentation](https://github.com/SmartForest-no/wheelRuts_semanticSegmentation) -> code for 2022 [paper](https://academic.oup.com/forestry/advance-article/doi/10.1093/forestry/cpac023/6627280): Mapping wheel-ruts from timber harvesting operations using deep learning techniques in drone imagery
* [LWN-for-UAVRSI](https://github.com/syliudf/LWN-for-UAVRSI) -> Light-Weight Semantic Segmentation Network for UAV Remote Sensing Images, applied to Vaihingen, UAVid and UDD6 datasets
* [hypernet](https://github.com/ESA-PhiLab/hypernet) -> library which implements; accurate hyperspectral image (HSI) segmentation and analysis using deep neural networks, optimization of deep neural network architectures for hyperspectral data segmentation, hyperspectral data augmentation, validation of existent and emerging HSI segmentation algorithms, simulation of multispectral data using HSI
* [ST-UNet](https://github.com/XinnHe/ST-UNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9686686): Swin Transformer Embedding UNet for Remote Sensing Image Semantic Segmentation
* [EDFT](https://github.com/h1063135843/EDFT) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/5/1294): Efficient Depth Fusion Transformer for Aerial Image Semantic Segmentation
* [WiCoNet](https://github.com/ggsDing/WiCoNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9759447): Looking Outside the Window: Wide-Context Transformer for the Semantic Segmentation of High-Resolution Remote Sensing Images
* [CRGNet](https://github.com/YonghaoXu/CRGNet) -> code for 2022 [paper](https://arxiv.org/abs/2202.03740): Consistency-Regularized Region-Growing Network for Semantic Segmentation of Urban Scenes with Point-Level Annotations
* [SA-UNet](https://github.com/Yancccccc/SA-UNet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/15/3591): Improved U-Net Remote Sensing Classification Algorithm Fusing Attention and Multiscale Features
* [MANet](https://github.com/lironui/Multi-Attention-Network) -> code for 2020 [paper](https://arxiv.org/abs/2009.02130): Multi-Attention-Network for Semantic Segmentation of Fine Resolution Remote Sensing Images
* [BANet](https://github.com/lironui/BANet) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/16/3065): Transformer Meets Convolution: A Bilateral Awareness Network for Semantic Segmentation of Very Fine Resolution Urban Scene Images
* [MACU-Net](https://github.com/lironui/MACU-Net) -> code for 2022 [paper](https://arxiv.org/abs/2007.13083): MACU-Net for Semantic Segmentation of Fine-Resolution Remotely Sensed Images
* [DNAS](https://github.com/faye0078/DNAS) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/16/3864): DNAS: Decoupling Neural Architecture Search for High-Resolution Remote Sensing Image Semantic Segmentation
* [A2-FPN](https://github.com/lironui/A2-FPN) -> code for 2021 [paper](https://arxiv.org/abs/2102.07997): A2-FPN for Semantic Segmentation of Fine-Resolution Remotely Sensed Images
* [MAResU-Net](https://github.com/lironui/MAResU-Net) -> code for 2020 [paper](https://arxiv.org/abs/2011.14302): Multi-stage Attention ResU-Net for Semantic Segmentation of Fine-Resolution Remote Sensing Images
* [ml_segmentation](https://github.com/dgriffiths3/ml_segmentation) -> semantic segmentation of buildings using Random Forest, Support Vector Machine (SVM) & Gradient Boosting Classifier (GBC)

### Segmentation - Land use & land cover
* [nga-deep-learning](https://github.com/jordancaraballo/nga-deep-learning) -> performs semantic segmentation on high resultion GeoTIF data using a modified U-Net & Keras, published by NASA researchers
* [Automatic Detection of Landfill Using Deep Learning](https://github.com/AnupamaRajkumar/LandfillDetection_SemanticSegmentation)
* [SpectralNET](https://github.com/tanmay-ty/SpectralNET) -> a 2D wavelet CNN for Hyperspectral Image Classification, uses Salinas Scene dataset & Keras
* [laika](https://github.com/datasciencecampus/laika) -> The goal of this repo is to research potential sources of satellite image data and to implement various algorithms for satellite image segmentation
* [PEARL](https://www.landcover.io/) -> a human-in-the-loop AI tool to drastically reduce the time required to produce an accurate Land Use/Land Cover (LULC) map, [blog post](http://devseed.com/blog/2021-05-17-pearl-ai-land-cover), uses Microsoft Planetary Computer and ML models run locally in the browser. Code for [backelnd](https://github.com/developmentseed/pearl-backend) and [frontend](https://github.com/developmentseed/pearl-frontend)
* [Land Cover Classification with U-Net](https://baratam-tarunkumar.medium.com/land-cover-classification-with-u-net-aa618ea64a1b) -> Satellite Image Multi-Class Semantic Segmentation Task with PyTorch Implementation of U-Net, uses DeepGlobe Land Cover Segmentation dataset, with [code](https://github.com/TarunKumar1995-glitch/land_cover_classification_unet)
* [Multi-class semantic segmentation of satellite images using U-Net](https://github.com/rogerxujiang/dstl_unet) using DSTL dataset, tensorflow 1 & python 2.7. Accompanying [article](https://towardsdatascience.com/dstl-satellite-imagery-contest-on-kaggle-2f3ef7b8ac40)
* [Codebase for multi class land cover classification with U-Net](https://github.com/jaeeolma/lulc_ml) accompanying a masters thesis, uses Keras
* [dubai-satellite-imagery-segmentation](https://github.com/ayushdabra/dubai-satellite-imagery-segmentation) -> due to the small dataset, image augmentation was used
* [U-Net for Semantic Segmentation on Unbalanced Aerial Imagery](https://towardsdatascience.com/u-net-for-semantic-segmentation-on-unbalanced-aerial-imagery-3474fa1d3e56) -> using the Dubai dataset
* [CDL-Segmentation](https://github.com/asimniazi63/CDL-Segmentation) -> code for the 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9441483): Deep Learning Based Land Cover and Crop Type Classification: A Comparative Study. Compares UNet, SegNet & DeepLabv3+
* [LoveDA](https://github.com/Junjue-Wang/LoveDA) -> code for the 2021 [paper](https://arxiv.org/abs/2110.08733): A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation
* [Satellite Imagery Semantic Segmentation with CNN](https://joshting.medium.com/satellite-imagery-segmentation-with-convolutional-neural-networks-f9254de3b907) -> 7 different segmentation classes, DeepGlobe Land Cover Classification Challenge dataset, with [repo](https://github.com/justjoshtings/satellite_image_segmentation)
* [Aerial Semantic Segmentation using U-Net Deep Learning Model](https://medium.com/@rehman.aimal/aerial-semantic-segmentation-using-u-net-deep-learning-model-3356a53c915f) medium article, with [repo](https://github.com/aimalrehman92/Multiclass-Semantic-Segmentation-with-U-NET)
* [UNet-Satellite-Image-Segmentation](https://github.com/YudeWang/UNet-Satellite-Image-Segmentation) -> A Tensorflow implentation of light UNet semantic segmentation framework
* [DeepGlobe Land Cover Classification Challenge solution](https://github.com/GeneralLi95/deepglobe_land_cover_classification_with_deeplabv3plus)
* [Semantic-segmentation-with-PyTorch-Satellite-Imagery](https://github.com/JenAlchimowicz/Semantic-segmentation-with-PyTorch-Satellite-Imagery) -> predict 25 classes on RGB imagery taken to assess the damage after Hurricane Harvey
* [Semantic Segmentation With Sentinel-2 Imagery](https://github.com/pavlo-seimskyi/semantic-segmentation-satellite-imagery) -> uses LandCoverNet dataset and fast.ai
* [CNN_Enhanced_GCN](https://github.com/qichaoliu/CNN_Enhanced_GCN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9268479): CNN-Enhanced Graph Convolutional Network With Pixel- and Superpixel-Level Feature Fusion for Hyperspectral Image Classification
* [LULCMapping-WV3images-CORINE-DLMethods](https://github.com/esertel/LULCMapping-WV3images-CORINE-DLMethods) -> Land Use and Land Cover Mapping Using Deep Learning Based Segmentation Approaches and VHR Worldview-3 Images
* [SOLC](https://github.com/yisun98/SOLC) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S0303243421003457): MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification. Uses [WHU-OPT-SAR-dataset](https://github.com/AmberHen/WHU-OPT-SAR-dataset)
* [MUnet-LUC](https://github.com/abhi170599/MUnet-LUC) -> Land Use with mUnet
* [land-cover](https://github.com/lucashu1/land-cover) -> code for 2021 [paper](https://arxiv.org/abs/2008.10351): Model Generalization in Deep Learning Applications for Land Cover Mapping
* [generalizablersc](https://github.com/dgominski/generalizablersc) -> code for 2022 paper: Cross-dataset Learning for Generalizable Land Use Scene Classification
* [Large-scale-Automatic-Identification-of-Urban-Vacant-Land](https://github.com/SkydustZ/Large-scale-Automatic-Identification-of-Urban-Vacant-Land) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0169204622000330): Large-scale automatic identification of urban vacant land using semantic segmentation of high-resolution remote sensing images
* [SSLTransformerRS](https://github.com/HSG-AIML/SSLTransformerRS) -> code for 2022 paper: Self-supervised Vision Transformers for Land-cover Segmentation and
Classification
* [aerial-tile-segmentation](https://github.com/mrsebai/aerial-tile-segmentation) -> Large satellite image semantic segmentation into 6 classes using Tensorflow 2.0 and ISPRS benchmark dataset

### Segmentation - Vegetation, crops & crop boundaries
* [Сrор field boundary detection: approaches overview and main challenges](https://soilmate.medium.com/%D1%81r%D0%BE%D1%80-field-boundary-detection-approaches-overview-and-main-challenges-53736725cb06) - review article, no code
* [Сrор field boundary detection: approaches and main challenges](https://medium.com/geekculture/%D1%81r%D0%BE%D1%80-field-boundary-detection-approaches-and-main-challenges-46e37dd276bc) -> Medium article, covering historical approaches to present dat
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
* [Crop-Classification](https://github.com/bhavesh907/Crop-Classification) -> crop classification using multi temporal satellite images
* [ParcelDelineation](https://github.com/sustainlab-group/ParcelDelineation) -> using a French polygons dataset and unet in keras
* [crop-mask](https://github.com/nasaharvest/crop-mask) -> End-to-end workflow for generating high resolution cropland maps, uses GEE & LSTM model
* [DeepCropMapping](https://github.com/Lab-IDEAS/DeepCropMapping) -> A multi-temporal deep learning approach with improved spatial generalizability for dynamic corn and soybean mapping, uses LSTM
* [Segment Canopy Cover and Soil using NDVI and Rasterio](https://towardsdatascience.com/segment-satellite-imagery-using-ndvi-and-rasterio-6dcae02a044b)
* [Use KMeans clustering to segment satellite imagery by land cover/land use](https://towardsdatascience.com/segment-satellite-images-using-rasterio-and-scikit-learn-fc048f465874)
* [ResUnet-a](https://github.com/Akhilesh64/ResUnet-a) -> Implementation of the paper "ResUNet-a: a deep learning framework for semantic segmentation of remotely sensed data" in TensorFlow
* [DSD_paper_2020](https://github.com/JacobJeppesen/DSD_paper_2020) -> The code for the paper: Crop Type Classification based on Machine Learning with Multitemporal Sentinel-1 Data
* [MR-DNN](https://github.com/yasir2afaq/Multi-resolution-deep-neural-network) -> extract rice field from Landsat 8 satellite imagery
* [deep_learning_forest_monitoring](https://github.com/waldeland/deep_learning_forest_monitoring) -> Estimate vegetation height, code for paper: Forest mapping and monitoring of the African continent using Sentinel-2 data and deep learning
* [global-cropland-mapping](https://github.com/Charly-tian/global-cropland-mapping) -> global multi-temporal cropland mapping
* [U-Net for Semantic Segmentation of Soyabean Crop Fields with SAR images](https://joaootavionf007.medium.com/u-net-for-semantic-segmentation-of-soyabeans-crop-fields-with-sar-images-604232e49315)
* [UNet-RemoteSensing](https://github.com/aryanVijaywargia/UNet-RemoteSensing) -> uses 7 bands of Landsat and keras
* [Landuse_DL](https://github.com/yghlc/Landuse_DL) -> delineate landforms due to the thawing of ice-rich permafrost
* [canopy](https://github.com/jonathanventura/canopy) -> code for 2019 [paper](https://www.mdpi.com/2072-4292/11/19/2326): A Convolutional Neural Network Classifier Identifies Tree Species in Mixed-Conifer Forest from Hyperspectral Imagery
* [RandomForest-Classification](https://github.com/florianbeyer/RandomForest-Classification) -> script is for random forest classification of remote sensing multi-band images, used in 2019 [paper](https://www.tandfonline.com/doi/abs/10.1080/01431161.2019.1580825): Multisensor data to derive peatland vegetation communities using a fixed-wing unmanned aerial vehicle
* [forest_change_detection](https://github.com/QuantuMobileSoftware/forest_change_detection) -> forest change segmentation with time-dependent models, including Siamese, UNet-LSTM, UNet-diff, UNet3D models. Code for 2021 [paper](https://ieeexplore.ieee.org/document/9241044): Deep Learning for Regular Change Detection in Ukrainian Forest Ecosystem With Sentinel-2
* [cultionet](https://github.com/jgrss/cultionet) -> segmentation of cultivated land, built on PyTorch Geometric and PyTorch Lightning
* [sentinel-tree-cover](https://github.com/wri/sentinel-tree-cover) -> code for 2020 [paper](https://arxiv.org/abs/2005.08702): A global method to identify trees outside of closed-canopy forests with medium-resolution satellite imagery
* [crop-type-detection-ICLR-2020](https://github.com/RadiantMLHub/crop-type-detection-ICLR-2020) -> Winning Solutions from Crop Type Detection Competition at CV4A workshop, ICLR 2020
* [Crop identification using satellite imagery](https://write.agrevolution.in/crop-identification-using-satellite-imagery-introduction-83d79344f9ee) -> Medium article, introduction to crop identification

### Segmentation - Water, coastlines & floods
* [UNSOAT used fastai to train a Unet to perform semantic segmentation on satellite imageries to detect water](https://forums.fast.ai/t/unosat-used-fastai-ai-for-their-floodai-model-discussion-on-how-to-move-forward/78468) - [paper](https://www.mdpi.com/2072-4292/12/16/2532) + [notebook](https://github.com/UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service/blob/master/Fastai%20training.ipynb), accuracy 0.97, precision 0.91, recall 0.92
* [Semi-Supervised Classification and Segmentation on High Resolution Aerial Images - Solving the FloodNet problem](https://sahilkhose.medium.com/paper-presentation-e9bd0f3fb0bf)
* [Flood Detection and Analysis using UNET with Resnet-34 as the back bone](https://github.com/orion29/Satellite-Image-Segmentation-for-Flood-Damage-Analysis) uses fastai
* [Houston_flooding](https://github.com/Lichtphyz/Houston_flooding) -> labeling each pixel as either flooded or not using data from Hurricane Harvey. Dataset consisted of pre and post flood images, and a ground truth floodwater mask was created using unsupervised clustering (with DBScan) of image pixels with human cluster verification/adjustment
* [ml4floods](https://github.com/spaceml-org/ml4floods) -> An ecosystem of data, models and code pipelines to tackle flooding with ML
* [A comprehensive guide to getting started with the ETCI Flood Detection competition](https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a) -> using Sentinel1 SAR & pytorch
* [Map Floodwater of SAR Imagery with SageMaker](https://github.com/JayThibs/map-floodwater-sar-imagery-on-sagemaker) -> applied to Sentinel-1 dataset
* [1st place solution for STAC Overflow: Map Floodwater from Radar Imagery hosted by Microsoft AI for Earth](https://github.com/sweetlhare/STAC-Overflow) -> combines Unet with Catboostclassifier, taking their maxima, not the average
* [hydra-floods](https://github.com/Servir-Mekong/hydra-floods) -> an open source Python application for downloading, processing, and delivering surface water maps derived from remote sensing data
* [CoastSat](https://github.com/kvos/CoastSat) -> tool for mapping coastlines which has an extension [CoastSeg](https://github.com/dbuscombe-usgs/CoastSeg) using  segmentation models
* [Satellite_Flood_Segmentation_of_Harvey](https://github.com/morgan-tam/Satellite_Flood_Segmentation_of_Harvey) -> explores both deep learning and traditional kmeans
* [Flood Event Detection Utilizing Satellite Images](https://github.com/KonstantinosF/Flood-Detection---Satellite-Images)
* [ETCI-2021-Competition-on-Flood-Detection](https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection) -> Experiments on Flood Segmentation on Sentinel-1 SAR Imagery with Cyclical Pseudo Labeling and Noisy Student Training, with [arxiv paper](https://arxiv.org/abs/2107.08369)
* [FDSI](https://github.com/keillernogueira/FDSI) -> Flood Detection in Satellite Images - 2017 Multimedia Satellite Task
* [deepwatermap](https://github.com/isikdogan/deepwatermap) -> a deep model that segments water on multispectral images
* [rivamap](https://github.com/isikdogan/rivamap) -> an automated river analysis and mapping engine
* [deep-water](https://github.com/maxbeber/deep-water) -> track changes in water level
* [WatNet](https://github.com/xinluo2018/WatNet) -> A deep ConvNet for surface water mapping based on Sentinel-2 image, uses the [Earth Surface Water Dataset](https://zenodo.org/record/5205674#.YoMjyZPMK3I)
* [A-U-Net-for-Flood-Extent-Mapping](https://github.com/jorgemspereira/A-U-Net-for-Flood-Extent-Mapping) -> in keras
* [floatingobjects](https://github.com/ESA-PhiLab/floatingobjects) -> code for the paper: TOWARDS DETECTING FLOATING OBJECTS ON A GLOBAL SCALE WITHLEARNED SPATIAL FEATURES USING SENTINEL 2. Uses U-Net & pytorch
* [River-Network-Extraction-from-Satellite-Image-using-UNet-and-Tensorflow](https://github.com/Diwas524/River-Network-Extraction-from-Satellite-Image-using-UNet-and-Tensorflow) -> uses Sentinel-2 imagery
* [SpaceNet8](https://github.com/SpaceNetChallenge/SpaceNet8) -> baseline Unet solution to detect flooded roads and buildings
* [dlsim](https://github.com/nyokoya/dlsim) -> code for 2020 [paper](https://arxiv.org/abs/2006.05180): Breaking the Limits of Remote Sensing by Simulation and Deep Learning for Flood and Debris Flow Mapping
* [Water-HRNet](https://github.com/faye0078/Water-Extraction) -> HRNet trained on Sentinel 2

### Segmentation - Fire, smoke & burn areas
* [Wild Fire Detection](https://github.com/yueureka/WildFireDetection) using U-Net trained on Databricks & Keras, semantic segmentation
* [A Practical Method for High-Resolution Burned Area Monitoring Using Sentinel-2 and VIIRS](https://www.mdpi.com/2072-4292/13/9/1608) with [code](https://github.com/mnpinto/FireHR). Dataset created on Google Earth Engine, downloaded to local machine for model training using fastai. The BA-Net model used is much smaller than U-Net, resulting in lower memory requirements and a faster computation
* [AI Geospatial Wildfire Risk Prediction](https://towardsdatascience.com/ai-geospatial-wildfire-risk-prediction-8c6b1d415eb4) -> A predictive model using geospatial raster data to asses wildfire hazard potential over the contiguous United States using Unet
* [IndustrialSmokePlumeDetection](https://github.com/HSG-AIML/IndustrialSmokePlumeDetection) -> using Sentinel-2 & a modified ResNet-50
* [burned-area-detection](https://github.com/dymaxionlabs/burned-area-detection) -> uses Sentinel-2
* [rescue](https://github.com/dbdmg/rescue) -> code of the paper: Attention to fires: multi-channel deep-learning models forwildfire severity prediction
* [smoke_segmentation](https://github.com/jeffwen/smoke_segmentation) -> Segmenting smoke plumes and predicting density from GOES imagery
* [wildfire-detection](https://github.com/amanbasu/wildfire-detection) -> Using Vision Transformers for enhanced wildfire detection in satellite images
* [Burned_Area_Detection](https://github.com/prhuppertz/Burned_Area_Detection) -> Detecting Burned Areas with Sentinel-2 data

### Segmentation - Landslides
* [landslide4sense](https://www.iarai.ac.at/landslide4sense/) -> a competition focused on landslide detection using globally distributed multi-source satellite imagery. [Baseline solution unet](https://github.com/isaaccorley/landslide4sense)
* [landslide-mapping-with-cnn](https://github.com/nprksh/landslide-mapping-with-cnn) -> code for 2021 [paper](https://www.nature.com/articles/s41598-021-89015-8): A new strategy to map landslides with a generalized convolutional neural network
* [Relict_landslides_CNN_kmeans](https://github.com/SPAMLab/data_sharing/tree/main/Relict_landslides_CNN_kmeans) -> code for 2022 [paper](https://arxiv.org/abs/2208.02693): Relict landslide detection in rainforest areas using a combination of k-means clustering algorithm and Deep-Learning semantic segmentation models

### Segmentation - Glaciers
* [HED-UNet](https://github.com/khdlr/HED-UNet) -> a model for simultaneous semantic segmentation and edge detection, examples provided are glacier fronts and building footprints using the Inria Aerial Image Labeling dataset
* [glacier_mapping](https://github.com/krisrs1128/glacier_mapping) -> Mapping glaciers in the Hindu Kush Himalaya, Landsat 7 images, Shapefile labels of the glaciers, Unet with dropout
* [glacier-detect-ML](https://github.com/mikeskaug/glacier-detect-ML) -> a simple logistic regression model to identify a glacier in Landsat satellite imagery
* [GlacierSemanticSegmentation](https://github.com/n9Mtq4/GlacierSemanticSegmentation) -> uses unet

### Segmentation - Other environmental
* [Detection of Open Landfills](https://github.com/dymaxionlabs/basurales) -> uses Sentinel-2 to detect large changes in the Normalized Burn Ratio (NBR)
* [sea_ice_remote_sensing](https://github.com/sum1lim/sea_ice_remote_sensing) -> Sea Ice Concentration classification
* [Methane-detection-from-hyperspectral-imagery](https://github.com/satish1901/Methane-detection-from-hyperspectral-imagery) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9093600): Deep Remote Sensing Methods for Methane Detection in Overhead Hyperspectral Imagery
* [EddyNet](https://github.com/redouanelg/EddyNet) -> A Deep Neural Network For Pixel-Wise Classification of Oceanic Eddies
* [schisto-vegetation](https://github.com/deleo-lab/schisto-vegetation) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/6/1345): Deep Learning Segmentation of Satellite Imagery Identifies Aquatic Vegetation Associated with Snail Intermediate Hosts of Schistosomiasis in Senegal, Africa

### Segmentation - Roads
Extracting roads is challenging due to the occlusions caused by other objects and the complex traffic environment
* [Semantic segmentation of roads and highways using Sentinel-2 imagery (10m) super-resolved using the SENX4 model up to x4 the initial spatial resolution (2.5m)](https://tracasa.es/innovative-stories/sen2roadlasviastambiensevendesdesentinel-2/) (results, no repo)
* [Semantic Segmentation of roads](https://vihan-tyagi.medium.com/semantic-segmentation-of-satellite-images-based-on-deep-learning-algorithms-ea5ec408ac53) using  U-net Keras, OSM data, project summary article by student, no code
* [Road detection using semantic segmentation and albumentations for data augmention](https://towardsdatascience.com/road-detection-using-segmentation-models-and-albumentations-libraries-on-keras-d5434eaf73a8) using the Massachusetts Roads Dataset, U-net & Keras. With [code](https://github.com/Diyago/ML-DL-scripts/tree/master/DEEP%20LEARNING/segmentation/Segmentation%20pipeline)
* [Winning Solutions from SpaceNet Road Detection and Routing Challenge](https://github.com/SpaceNetChallenge/RoadDetector)
* [RoadVecNet](https://github.com/gismodelling/RoadVecNet) -> Road-Network-Segmentation-and-Vectorization in keras with dataset and [paper](https://www.tandfonline.com/doi/abs/10.1080/15481603.2021.1972713?journalCode=tgrs20&)
* [Detecting road and road types jupyter notebook](https://github.com/taspinar/sidl/blob/master/notebooks/2_Detecting_road_and_roadtypes_in_sattelite_images.ipynb)
* [awesome-deep-map](https://github.com/antran89/awesome-deep-map) -> A curated list of resources dedicated to deep learning / computer vision algorithms for mapping. The mapping problems include road network inference, building footprint extraction, etc.
* [RoadTracer: Automatic Extraction of Road Networks from Aerial Images](https://github.com/mitroadmaps/roadtracer) -> uses an iterative search process guided by a CNN-based decision function to derive the road network graph directly from the output of the CNN
* [road_detection_mtl](https://github.com/ntelo007/road_detection_mtl) -> Road Detection using a multi-task Learning technique to improve the performance of the road detection task by incorporating prior knowledge constraints, uses the SpaceNet Roads Dataset
* [road_connectivity](https://github.com/anilbatra2185/road_connectivity) -> Improved Road Connectivity by Joint Learning of Orientation and Segmentation (CVPR2019)
* [Road-Network-Extraction using classical Image processing](https://github.com/abhaykes1/Road-Network-Extraction) -> blur & canny edge detection
* [SPIN_RoadMapper](https://github.com/wgcban/SPIN_RoadMapper) -> Extracting Roads from Aerial Images via Spatial and Interaction Space Graph Reasoning for Autonomous Driving
* [road_extraction_remote_sensing](https://github.com/jiankang1991/road_extraction_remote_sensing) -> pytorch implementation, CVPR2018 DeepGlobe Road Extraction Challenge submission. See also [DeepGlobe-Road-Extraction-Challenge](https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge)
* [RoadDetections dataset by Microsoft](https://github.com/microsoft/RoadDetections)
* [CoANet](https://github.com/mj129/CoANet) -> Connectivity Attention Network for Road Extraction From Satellite Imagery. The CoA module incorporates graphical information to ensure the connectivity of roads are better preserved. With [paper](https://ieeexplore.ieee.org/document/9563125)
* [ML_EPFL_Project_2](https://github.com/LucasBrazCappelo/ML_EPFL_Project_2) -> U-Net in Pytorch to perform semantic segmentation of roads on satellite images
* [Satellite Imagery Road Segmentation](https://medium.com/@nithishmailme/satellite-imagery-road-segmentation-ad2964dc3812) -> intro articule on Medium using the kaggle [Massachusetts Roads Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset)
* [Label-Pixels](https://github.com/venkanna37/Label-Pixels) -> for semantic segmentation of roads and other features
* [Satellite-image-road-extraction](https://github.com/amanhari-projects/Satellite-image-road-extraction) -> code for 2018 paper: Road Extraction by Deep Residual U-Net
* [road_building_extraction](https://github.com/jeffwen/road_building_extraction) -> Pytorch implementation of U-Net architecture for road and building extraction
* [Satellite-Imagery-Road-Extraction](https://github.com/Akash-Ramjyothi/Satellite-Imagery-Road-Extraction) -> research project in keras
* [SGCN](https://github.com/tist0bsc/SGCN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9614130): Split Depth-Wise Separable Graph-Convolution Network for Road Extraction in Complex Environments From High-Resolution Remote-Sensing Images
* [ASPN](https://github.com/pshams55/ASPN) -> code for 2020 [paper](https://arxiv.org/abs/2008.04021): Road Segmentation for Remote Sensing Images using Adversarial Spatial Pyramid Networks
* [FCNs-for-road-extraction-keras](https://github.com/zetrun-liu/FCNs-for-road-extraction-keras) -> Road extraction of high-resolution remote sensing images based on various semantic segmentation networks
* [cresi](https://github.com/avanetten/cresi) -> Road network extraction from satellite imagery, with speed and travel time estimates
* [road-extraction-d-linknet](https://github.com/NekoApocalypse/road-extraction-d-linknet) -> code for 2018 [paper](https://ieeexplore.ieee.org/document/8575492): D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction
* [Sat2Graph](https://github.com/songtaohe/Sat2Graph) -> code for 2020 paper: Road Graph Extraction through Graph-Tensor Encoding
* [Image-Segmentation)](https://github.com/mschulz/Image-Segmentation) -> using Massachusetts Road dataset and fast.ai
* [RoadTracer-M](https://github.com/astro-ck/RoadTracer-M) -> code for 2019 [paper](https://ieeexplore.ieee.org/abstract/document/8898565): Road Network Extraction from Satellite Images Using CNN Based Segmentation and Tracing
* [ScRoadExtractor](https://github.com/weiyao1996/ScRoadExtractor) -> code for 2020 [paper](https://arxiv.org/abs/2010.13106): Scribble-based Weakly Supervised Deep Learning for Road Surface Extraction from Remote Sensing Images
* [RoadDA](https://github.com/LANMNG/RoadDA) -> code for 2021 [paper](https://arxiv.org/abs/2108.12611): Stagewise Unsupervised Domain Adaptation with Adversarial Self-Training for Road Segmentation of Remote Sensing Images
* [DeepSegmentor](https://github.com/yhlleo/DeepSegmentor) -> A Pytorch implementation of DeepCrack and RoadNet projects
* [Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction](https://github.com/liaochengcsu/Cascade_Residual_Attention_Enhanced_for_Refinement_Road_Extraction) -> code for 2021 [paper](https://www.mdpi.com/2220-9964/11/1/9): Cascaded Residual Attention Enhanced Road Extraction from Remote Sensing Images
* [nia-road-baseline](https://github.com/SIAnalytics/nia-road-baseline) -> code for 2020 [paper](https://arxiv.org/abs/1908.08223): NL-LinkNet: Toward Lighter but More Accurate Road Extraction with Non-Local Operations
* [IRSR-net](https://github.com/yangzhen1252/IRSR-net) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9785827): Lightweight Remote Sensing Road Detection Network
* [hironex](https://github.com/johannesuhl/hironex) -> A python tool for automatic, fully unsupervised extraction of historical road networks from historical maps
* [Road_detection_model](https://github.com/JonasImazon/Road_detection_model) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/15/3625): Mapping Roads in the Brazilian Amazon with Artificial Intelligence and Sentinel-2
* [DTnet](https://github.com/huzican695/DTnet) -> code for 2022 [paper](https://arxiv.org/abs/2208.08116): Road detection via a dual-task network based on cross-layer graph fusion modules

### Segmentation - Buildings & rooftops
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
* [Mapping Africa’s Buildings with Satellite Imagery: Google AI blog post](https://ai.googleblog.com/2021/07/mapping-africas-buildings-with.html). See the [open-buildings](https://sites.research.google/open-buildings/) dataset
* [nz_convnet](https://github.com/weiji14/nz_convnet) -> A U-net based ConvNet for New Zealand imagery to classify building outlines
* [polycnn](https://github.com/Lydorn/polycnn) -> End-to-End Learning of Polygons for Remote Sensing Image Classification
* [spacenet_building_detection](https://github.com/motokimura/spacenet_building_detection) solution by [motokimura](https://github.com/motokimura) using Unet
* [How to extract building footprints from satellite images using deep learning](https://azure.microsoft.com/en-gb/blog/how-to-extract-building-footprints-from-satellite-images-using-deep-learning/)
* [Vec2Instance](https://github.com/lakmalnd/Vec2Instance) -> applied to the SpaceNet challenge AOI 2 (Vegas) building footprint dataset, tensorflow v1.12
* [EarthquakeDamageDetection](https://github.com/JaneKravchenko/EarthquakeDamageDetection) -> Buildings segmentation from satellite imagery and damage classification for each build, using Keras
* [Semantic-segmentation repo by fuweifu-vtoo](https://github.com/fuweifu-vtoo/Semantic-segmentation) -> uses pytorch and the [Massachusetts Buildings & Roads Datasets](https://www.cs.toronto.edu/~vmnih/data/)
* [Extracting buildings and roads from AWS Open Data using Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/extracting-buildings-and-roads-from-aws-open-data-using-amazon-sagemaker/) -> uses merged RGB (SpaceNet) and LiDAR (USGS 3DEP) datasets with Unet to reproduce the winning algorithm from SpaceNet challenge 4 by XD_XD. With [repo](https://github.com/aws-samples/aws-open-data-satellite-lidar-tutorial)
* [TF-SegNet](https://github.com/mathildor/TF-SegNet) -> AirNet is a segmentation network based on SegNet, but with some modifications
* [rgb-footprint-extract](https://github.com/aatifjiwani/rgb-footprint-extract) -> a Semantic Segmentation Network for Urban-Scale Building Footprint Extraction Using RGB Satellite Imagery, DeepLavV3+ module with a Dilated ResNet C42 backbone
* [SpaceNetExploration](https://github.com/yangsiyu007/SpaceNetExploration) -> A sample project demonstrating how to extract building footprints from satellite images using a semantic segmentation model. Data from the SpaceNet Challenge
* [Rooftop-Instance-Segmentation](https://github.com/MasterSkepticista/Rooftop-Instance-Segmentation) -> VGG-16, Instance Segmentation, uses the Airs dataset
* [solar-farms-mapping](https://github.com/microsoft/solar-farms-mapping) -> An Artificial Intelligence Dataset for Solar Energy Locations in India
* [poultry-cafos](https://github.com/microsoft/poultry-cafos) -> This repo contains code for detecting poultry barns from high-resolution aerial imagery and an accompanying dataset of predicted barns over the United States
* [ssai-cnn](https://github.com/mitmul/ssai-cnn) -> This is an implementation of Volodymyr Mnih's dissertation methods on his Massachusetts road & building dataset
* [Remote-sensing-building-extraction-to-3D-model-using-Paddle-and-Grasshopper](https://github.com/Youssef-Harby/Remote-sensing-building-extraction-to-3D-model-using-Paddle-and-Grasshopper)
* [segmentation-enhanced-resunet](https://github.com/tranleanh/segmentation-enhanced-resunet) -> Urban building extraction in Daejeon region using Modified Residual U-Net (Modified ResUnet) and applying post-processing
* [Mask RCNN for Spacenet Off Nadir Building Detection](https://github.com/ashnair1/Mask-RCNN-for-Off-Nadir-Building-Detection)
* [GRSL_BFE_MA](https://github.com/jiankang1991/GRSL_BFE_MA) -> Deep Learning-based Building Footprint Extraction with Missing Annotations using a novel loss function
* [FER-CNN](https://github.com/runnergirl13/FER-CNN) -> Detection, Classification and Boundary Regularization of Buildings in Satellite Imagery Using Faster Edge Region Convolutional Neural Networks, with [paper](https://www.mdpi.com/2072-4292/12/14/2240/htm)
* [UNET-Image-Segmentation-Satellite-Picture](https://github.com/rwie1and/UNET-Image-Segmentation-Satellite-Pictures) -> Unet to predict roof tops on Crowed AI Mapping dataset, uses keras
* [Vector-Map-Generation-from-Aerial-Imagery-using-Deep-Learning-GeoSpatial-UNET](https://github.com/ManishSahu53/Vector-Map-Generation-from-Aerial-Imagery-using-Deep-Learning-GeoSpatial-UNET) -> applied to geo-referenced images which are very large size > 10k x 10k pixels
* [building-footprint-segmentation](https://github.com/fuzailpalnak/building-footprint-segmentation) -> pip installable library to train building footprint segmentation on satellite and aerial imagery, applied to Massachusetts Buildings Dataset and Inria Aerial Image Labeling Dataset
* [SemSegBuildings](https://github.com/SharpestProjects/SemSegBuildings) -> Project using fast.ai framework for semantic segmentation on Inria building segmentation dataset
* [FCNN-example](https://github.com/emredog/FCNN-example) -> overfit to a given single image to detect houses
* [SAT2LOD2](https://github.com/gdaosu/lod2buildingmodel) -> an open-source, python-based GUI-enabled software that takes the satellite images as inputs and returns LoD2 building models as outputs, with [paper](https://arxiv.org/abs/2204.04139)
* [SatFootprint](https://github.com/PriyanK7n/SatFootprint) -> building segmentation on the Spacenet 7 dataset
* [Building-Detection](https://github.com/EL-BID/Building-Detection) -> code for running a Raster Vision experiment to train a model to detect buildings from satellite imagery in three cities in Latin America
* [Multi-building-tracker](https://github.com/sebasmos/Multi-building-tracker) -> code for paper: Multi-target building tracker for satellite images using deep learning
* [Boundary Enhancement Semantic Segmentation for Building Extraction](https://github.com/hin1115/BEmodule-Satellite-Building-Segmentation)
* [UNet_keras_for_RSimage](https://github.com/loveswine/UNet_keras_for_RSimage) -> keras code for binary semantic segmentation
* [Spacenet-Building-Detection](https://github.com/IdanC1s2/Spacenet-Building-Detection) -> uses keras
* [LGPNet-BCD](https://github.com/TongfeiLiu/LGPNet-BCD) -> code for 2021 paper: Building Change Detection for VHR Remote Sensing Images via Local-Global Pyramid Network and Cross-Task Transfer Learning Strategy
* [MTL_homoscedastic_SRB](https://github.com/burakekim/MTL_homoscedastic_SRB) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9554766): A Multi-Task Deep Learning Framework for Building Footprint Segmentation
* [UNet_CNN](https://github.com/Inamdarpushkar/UNet_CNN) -> UNet model to segment building coverage in Boston using Remote sensing data, uses keras
* [FDANet](https://github.com/daifeng2016/FDANet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9481881): Full-Level Domain Adaptation for Building Extraction in Very-High-Resolution Optical Remote-Sensing Images
* [CBRNet](https://github.com/HaonanGuo/CBRNet) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002975): A Coarse-to-fine Boundary Refinement Network for Building Extraction from Remote Sensing Imagery
* [ASLNet](https://github.com/ggsDing/ASLNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9653801): Adversarial Shape Learning for Building Extraction in VHR Remote Sensing Images
* [BRRNet](https://github.com/wangyi111/Building-Extraction) -> implementation of Modified U-Net from 2020 [paper](https://www.mdpi.com/2072-4292/12/6/1050): BRRNet: A Fully Convolutional Neural Network for Automatic Building Extraction From High-Resolution Remote Sensing Images
* [Multi-Scale-Filtering-Building-Index](https://github.com/ThomasWangWeiHong/Multi-Scale-Filtering-Building-Index) -> Python implementation of building extraction index proposed in 2019 [paper](https://www.mdpi.com/2072-4292/11/5/482): A Multi - Scale Filtering Building Index for Building Extraction in Very High - Resolution Satellite Imagery
* [Models for Remote Sensing](https://github.com/bohaohuang/mrs) -> long list of unets etc applied to building detection
* [boundary_loss_for_remote_sensing](https://github.com/yiskw713/boundary_loss_for_remote_sensing) -> code for 2019 paper: Boundary Loss for Remote Sensing Imagery
Semantic Segmentation
* [Open Cities AI Challenge](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/) -> Segmenting Buildings for Disaster Resilience. Winning solutions [on Github](https://github.com/drivendataorg/open-cities-ai-challenge/)
* [MAPNet](https://github.com/lehaifeng/MAPNet) -> code for 2020 [paper](https://arxiv.org/abs/1910.12060): Multi Attending Path Neural Network for Building Footprint Extraction from Remote Sensed Imagery
* [dual-hrnet](https://github.com/SIAnalytics/dual-hrnet) -> localizing buildings and classifying their damage level
* [ESFNet](https://github.com/mrluin/ESFNet-Pytorch) -> code for 2019 [paper](https://arxiv.org/abs/1903.12337): Efficient Network for Building Extraction from High-Resolution Aerial Images
* [rooftop-detection-python](https://github.com/sayonpalit/rooftop-detection-python) -> Detect Rooftops from low resolution satellite images and calculate area for cultivation and solar panel installment using classical computer vision techniques
* [keras_segmentation_models](https://github.com/sajmonogy/keras_segmentation_models) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/12/2745): Using Open Vector-Based Spatial Data to Create Semantic Datasets for Building Segmentation for Raster Data
* [CVCMFFNet](https://github.com/Jiankun-chen/CVCMFFNet-master) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9397870): Complex-Valued Convolutional and Multifeature Fusion Network for Building Semantic Segmentation of InSAR Images
* [STEB-UNet](https://github.com/BrightGuo048/STEB-UNet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/11/2611): A Swin Transformer-Based Encoding Booster Integrated in U-Shaped Network for Building Extraction
* [dfc2020_baseline](https://github.com/lukasliebel/dfc2020_baseline) -> Baseline solution for the IEEE GRSS Data Fusion Contest 2020. Predict land cover labels from Sentinel-1 and Sentinel-2 imagery. Code for 2020 [paper](https://arxiv.org/abs/2002.08254): Weakly Supervised Semantic Segmentation of Satellite Images for Land Cover Mapping
* [Fusing multiple segmentation models based on different datasets into a single edge-deployable model](https://github.com/markusmeingast/Satellite-Classifier) -> roof, car & road segmentation
* [ground-truth-gan-segmentation](https://github.com/zakariamejdoul/ground-truth-gan-segmentation) -> use Pix2Pix to segment the footprint of a building. The dataset used is AIRS
* [UNICEF-Giga_Sudan](https://github.com/Kamal-Eldin/UNICEF-Giga_Sudan) -> Detecting school lots from satellite imagery in Southern Sudan using a UNET segmentation model
* [building_footprint_extraction](https://github.com/shubhamgoel27/building_footprint_extraction) -> The project retrieves satellite imagery from Google and performs building footprint extraction using a U-Net.
* [projectRegularization](https://github.com/zorzi-s/projectRegularization) -> code for 2019 [paper](https://arxiv.org/abs/2007.11840): Regularization of building boundaries in satellite images using adversarial and regularized losses
* [PolyWorldPretrainedNetwork](https://github.com/zorzi-s/PolyWorldPretrainedNetwork) -> code for 2021 [paper](https://arxiv.org/abs/2111.15491): Polygonal Building Extraction with Graph Neural Networks in Satellite Images
* [dl_image_segmentation](https://github.com/harry-gibson/dl_image_segmentation) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/13/3072): Uncertainty-Aware Interpretable Deep Learning for Slum Mapping and Monitoring. Uses SHAP
* [UBC-dataset](https://github.com/AICyberTeam/UBC-dataset) -> a dataset for building detection and classification from very high-resolution satellite imagery with the focus on object-level interpretation of individual buildings
* [GeoSeg](https://github.com/WangLibo1995/GeoSeg) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S0924271622001654): UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery
* [BESNet](https://github.com/FlyC235/BESNet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/7/1638): BES-Net: Boundary Enhancing Semantic Context Network for High-Resolution Image Semantic Segmentation. Applied to Vaihingen and Potsdam datasets
* [CVNet](https://github.com/xzq-njust/CVNet) -> code for 2022 paper: CVNet: Contour Vibration Network for Building Extraction
* [CFENet](https://github.com/djzgroup/CFENet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/9/2276): A Context Feature Enhancement Network for Building Extraction from High-Resolution Remote Sensing Imagery
* [HiSup](https://github.com/SarahwXU/HiSup) -> code for 2022 [paper](https://arxiv.org/abs/2208.00609): Accurate Polygonal Mapping of Buildings in Satellite Imagery
* [BuildingExtraction](https://github.com/KyanChen/BuildingExtraction) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/21/4441): Building Extraction from Remote Sensing Images with Sparse Token Transformers
* [coseg_building](https://github.com/lqycrystal/coseg_building) -> code for the 2022 [paper](https://www.sciencedirect.com/science/article/pii/S1569843222000267): CrossGeoNet: A Framework for Building Footprint Generation of Label-Scarce Geographical Regions
* [AFM_building](https://github.com/lqycrystal/AFM_building) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9538384): Building Footprint Generation Through Convolutional Neural Networks With Attraction Field Representation

### Segmentation - Solar panels
* [DeepSolar](https://github.com/wangzhecheng/DeepSolar) -> A Machine Learning Framework to Efficiently Construct a Solar Deployment Database in the United States. [Dataset on kaggle](https://www.kaggle.com/tunguz/deep-solar-dataset), actually used a CNN for classification and segmentation is obtained by applying a threshold to the activation map. Original code is tf1 but [tf2/kers](https://github.com/aidan-fitz/deepsolar-v2) and a [pytorch implementation](https://github.com/wangzhecheng/deepsolar_pytorch) are available. Also checkout [Visualizations and in-depth analysis .. of the factors that can explain the adoption of solar energy in ..  Virginia](https://github.com/bessammehenni/DeepSolar_adoption_Virginia) and [DeepSolar tracker: towards unsupervised assessment with open-source data of the accuracy of deep learning-based distributed PV mapping](https://github.com/gabrielkasmi/dsfrance)
* [hyperion_solar_net](https://github.com/fvergaracontesse/hyperion_solar_net) -> trained classificaton & segmentation models on RGB imagery from Google Maps. Provides app for viewing predictions, and has [arxiv paper](https://arxiv.org/abs/2201.02107) and a [nice webpage](https://groups.ischool.berkeley.edu/HyperionSolarNet/)
* [3D-PV-Locator](https://github.com/kdmayer/3D-PV-Locator) -> Large-scale detection of rooftop-mounted photovoltaic systems in 3D
* [PV_Pipeline](https://github.com/kdmayer/PV_Pipeline) -> PyTorch models and pipeline developed for "DeepSolar for Germany"
* [solar-panels-detection](https://github.com/dbaofd/solar-panels-detection) -> using SegNet, Fast SCNN & ResNet
* [predict_pv_yield](https://github.com/openclimatefix/predict_pv_yield) -> Using optical flow & machine learning to predict PV yield
* [Large-scale-solar-plant-monitoring](https://github.com/osmarluiz/Large-scale-solar-plant-monitoring) -> code for the paper "Remote Sensing for Monitoring of Photovoltaic Power Plants in Brazil Using Deep Semantic Segmentation"
* [Panel-Segmentation](https://github.com/NREL/Panel-Segmentation) -> Determine the presence of a solar array in the satellite image (boolean True/False), using a VGG16 classification model
* [Roofpedia](https://github.com/ualsg/Roofpedia) -> an open registry of green roofs and solar roofs across the globe identified by Roofpedia through deep learning
* [Predicting the Solar Potential of Rooftops using Image Segmentation and Structured Data](https://medium.com/nam-r/predicting-the-solar-potential-of-rooftops-using-image-segmentation-and-structured-data-61198c39d57c) Medium article, using 20cm imagery & Unet
* [solar-pv-global-inventory](https://github.com/Lkruitwagen/solar-pv-global-inventory) -> code from the Nature paper of Kruitwagen et al, used to produce a global inventory of utility-scale solar photvoltaic generating stations
* [remote-sensing-solar-pv](https://github.com/Lkruitwagen/remote-sensing-solar-pv) -> A repository for sharing progress on the automated detection of solar PV arrays in sentinel-2 remote sensing imagery
* [solar-panel-segmentation)](https://github.com/gabrieltseng/solar-panel-segmentation) -> Finding solar panels using USGS satellite imagery
* [solar_seg](https://github.com/tcapelle/solar_seg) -> Solar segmentation of PV modules (sub elements of panels) using drone images and fast.ai
* [solar_plant_detection](https://github.com/Amirmoradi94/solar_plant_detection) -> boundary extraction of Photovoltaic (PV) plants using Mask RCNN and Amir dataset
* [SolarDetection](https://github.com/A-Stangeland/SolarDetection) -> unet on satellite image from the USA and France
* [adopptrs](https://github.com/francois-rozet/adopptrs) -> Automatic Detection Of Photovoltaic Panels Through Remote Sensing using unet & pytorch
* [solar-panel-locator](https://github.com/TorrBorr/solar-panel-locator) -> the number of solar panel pixels was only ~0.2% of the total pixels in the dataset, so solar panel data was upsampled to account for the class imbalance
* [projects-solar-panel-detection](https://github.com/top-on/projects-solar-panel-detection) -> List of project to detect solar panels from aerial/satellite images
* [Satellite_ComputerVision](https://github.com/mjevans26/Satellite_ComputerVision) -> UNET to detect solar arrays from Sentinel-2 data, using Google Earth Engine and Tensorflow. Also covers parking lot detection
* [ood_instances_sophia](https://github.com/gabrielkasmi/ood_instances_sophia) -> Material for the poster "The effect of PV array instances and images backgrounds on OOD generalization"

### Segmentation - Electrical substations
The repos below resulted from the ICETCI 2021 competition on Machine Learning based feature extraction of Electrical Substations from Satellite Data using Open Source Tools
* [Aarsh2001/ML_Challenge_NRSC](https://github.com/Aarsh2001/ML_Challenge_NRSC) -> 3rd place entry
* [electrical_substation_detection](https://github.com/thisishardik/electrical_substation_detection) -> entry using UNet, Albumentations for image augmentation, and OpenCV for computer vision tasks

### Instance segmentation
In instance segmentation, each individual 'instance' of a segmented area is given a unique lable. For detection of very small objects this may a good approach, but it can struggle seperating individual objects that are closely spaced.
* [Mask_RCNN](https://github.com/matterport/Mask_RCNN) generates bounding boxes and segmentation masks for each instance of an object in the image. It is very commonly used for instance segmentation & object detection
* [Instance segmentation of center pivot irrigation system in Brazil](https://github.com/saraivaufc/instance-segmentation-maskrcnn) using free Landsat images, mask R-CNN & Keras
* [Oil tank instance segmentation with Mask R-CNN](https://github.com/georgiosouzounis/instance-segmentation-mask-rcnn) with [accompanying article](https://medium.com/@georgios.ouzounis/oil-storage-tank-instance-segmentation-with-mask-r-cnn-77c94433045f) using Keras & Airbus Oil Storage Detection Dataset on Kaggle
* [Building-Detection-MaskRCNN](https://github.com/Mstfakts/Building-Detection-MaskRCNN) -> Building detection from the SpaceNet dataset by using Mask RCNN
* [Mask_RCNN-for-Caravans](https://github.com/OrdnanceSurvey/Mask_RCNN-for-Caravans) -> detect caravan footprints from OS imagery
* [parking_bays_detectron2](https://github.com/spiyer99/parking_bays_detectron2) -> Detecting parking bays with satellite imagery. Used Detectron2 and synthetic data with Unreal, superior performance to using Mask RCNN
* [Locate buildings with a dark roof that feed heat island phenomenon using Mask RCNN](https://towardsdatascience.com/my-rooftop-project-a-satellite-imagery-computer-vision-example-e45a296129a0) -> with [repo](https://github.com/vintel38/RoofTop-Project), used INRIA dataset & labelme for annotation
* [Circle_Finder](https://github.com/zinsmatt/Circle_Finder) -> Circular Shapes Detection in Satellite Imagery, 2nd place solution to the Circle Finder Challenge
* [Lawn_maskRCNN](https://github.com/matthewnaples/Lawn_maskRCNN) -> Detecting lawns from satellite images of properties in the Cedar Rapids area using Mask-R-CNN
* [CropMask_RCNN](https://github.com/ecohydro/CropMask_RCNN) -> Segmenting center pivot agriculture to monitor crop water use in drylands with Mask R-CNN and Landsat satellite imagery
* [Mask RCNN for Spacenet Off Nadir Building Detection](https://github.com/ashnair1/Mask-RCNN-for-Off-Nadir-Building-Detection)
* [CATNet](https://github.com/yeliudev/CATNet) -> code for 2021 [paper](https://arxiv.org/abs/2111.11057): Learning to Aggregate Multi-Scale Context for Instance Segmentation in Remote Sensing Images
* [Object-Detection-on-Satellite-Images-using-Mask-R-CNN](https://github.com/ThayN15/Object-Detection-on-Satellite-Images-using-Mask-R-CNN) -> detect ships
* [FactSeg](https://github.com/Junjue-Wang/FactSeg) -> Foreground Activation Driven Small Object Semantic Segmentation in Large-Scale Remote Sensing Imagery (TGRS), also see [FarSeg](https://github.com/Z-Zheng/FarSeg) and [FreeNet](https://github.com/Z-Zheng/FreeNet), implementations of research paper
* [aqua_python](https://github.com/tclavelle/aqua_python) -> detecting aquaculture farms using Mask R-CNN

### Panoptic segmentation
* [Things and stuff or how remote sensing could benefit from panoptic segmentation](https://softwaremill.com/things-and-stuff-or-how-remote-sensing-could-benefit-from-panoptic-segmentation/)
* [Panoptic Segmentation Meets Remote Sensing (paper)](https://www.mdpi.com/2072-4292/14/4/965)
* [pastis-benchmark](https://github.com/VSainteuf/pastis-benchmark)
* [Panoptic-Generator](https://github.com/abilius-app/Panoptic-Generator) -> This module converts GIS data into panoptic segmentation tiles
* [BSB-Aerial-Dataset](https://github.com/osmarluiz/BSB-Aerial-Dataset) -> an example on how to use Detectron2's Panoptic-FPN in the BSB Aerial Dataset
* [utae-paps](https://github.com/VSainteuf/utae-paps) -> PyTorch implementation of U-TAE and PaPs for satellite image time series panoptic segmentation

## Object detection
Several different techniques can be used to count the number of objects in an image. The returned data can be an object count (regression), a bounding box around individual objects in an image (typically using Yolo or Faster R-CNN architectures), a pixel mask for each object (instance segmentation), key points for an an object (such as wing tips, nose and tail of an aircraft), or simply a classification for a sliding tile over an image. A good introduction to the challenge of performing object detection on aerial imagery is given in [this paper](https://arxiv.org/abs/1902.06042v2). In summary, images are large and objects may comprise only a few pixels, easily confused with random features in background. For the same reason, object detection datasets are inherently imbalanced, since the area of background typically dominates over the area of the objects to be detected. In general object detection performs well on large objects, and gets increasingly difficult as the objects get smaller & more densely packed. Model accuracy falls off rapidly as image resolution degrades, so it is common for object detection to use very high resolution imagery, e.g. 30cm RGB. A particular characteristic of aerial images is that objects can be oriented in any direction, so using rotated bounding boxes which aligning with the object can be crucial for extracting metrics such as the length and width of an object. Note that newer models such as Yolov5 may not achieve the same performance as 'older' models like Faster RCNN or Retinanet since they are no longer pre-trained on datasets such as ImageNet, but smaller datasets such as COCO dataset.
* Background reading: [Anchor Boxes for Object Detection](https://uk.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html)
* [Object Detection and Image Segmentation with Deep Learning on Earth Observation Data: A Review](https://www.mdpi.com/2072-4292/12/10/1667)
* [awesome-aerial-object-detection bu murari023](https://github.com/murari023/awesome-aerial-object-detection), [another by visionxiang](https://github.com/visionxiang/awesome-object-detection-in-aerial-images) and [awesome-tiny-object-detection](https://github.com/kuanhungchen/awesome-tiny-object-detection) list many relevant papers
* [Tackling the Small Object Problem in Object Detection](https://blog.roboflow.com/tackling-the-small-object-problem-in-object-detection)
* [Object Detection Accuracy as a Function of Image Resolution](https://medium.com/the-downlinq/the-satellite-utility-manifold-object-detection-accuracy-as-a-function-of-image-resolution-ebb982310e8c) -> Medium article using COWC dataset, performance rapidly degrades below 30cm imagery
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
* [IS-Count](https://github.com/sustainlab-group/IS-Count) -> IS-Count is a sampling-based and learnable method for estimating the total object count in a region. 
* [Object Detection On Aerial Imagery Using RetinaNet](https://towardsdatascience.com/object-detection-on-aerial-imagery-using-retinanet-626130ba2203)
* [Clustered-Object-Detection-in-Aerial-Image](https://github.com/fyangneil/Clustered-Object-Detection-in-Aerial-Image)
* [yolov5s_for_satellite_imagery](https://github.com/KevinMuyaoGuo/yolov5s_for_satellite_imagery) -> yolov5s applied to the DOTA dataset
* [RetinaNet-PyTorch](https://github.com/HsLOL/RetinaNet-PyTorch) -> RetinaNet implementation on remote sensing ship dataset (SSDD)
* [Detecting-Cyclone-Centers-Custom-YOLOv3](https://github.com/ShubhayanS/Detecting-Cyclone-Centers-Custom-YOLOv3) -> tropical cyclones (TCs) are intense warm-corded cyclonic vortices, developed from low-pressure systems over the tropical oceans and driven by complex air-sea interaction
* [Object-Detection-YoloV3-RetinaNet-FasterRCNN](https://github.com/bostankhan6/Object-Detection-YoloV3-RetinaNet-FasterRCNN) -> trained on a private datset
* [Google-earth-Object-Recognition](https://github.com/InnovAIco/Google-earth-Object-Recognition) -> Code for training and evaluating on Dior Dataset (Google Earth Images) using RetinaNet and YOLOV5
* [HIECTOR: Hierarchical object detector at scale](https://medium.com/sentinel-hub/hiector-hierarchical-object-detector-at-scale-5a61753b51a3) -> HIECTOR facilitates multiple satellite data collections of increasingly detailed spatial resolution for a cost-efficient and accurate object detection over large areas
* [Detection of Multiclass Objects in Optical Remote Sensing Images](https://github.com/WenchaoliuMUC/Detection-of-Multiclass-Objects-in-Optical-Remote-Sensing-Images) -> code for 2018 [paper](https://ieeexplore.ieee.org/document/8573851): Detection of Multiclass Objects in Optical Remote Sensing Images
* [SB-MSN](https://github.com/weihancug/Sampling-Balance_Multi-stage_Network) -> Sampling-Balance based Multi-stage Network (SB-MSN) for aerial image object detection. Code for 2021 [paper](https://ieeexplore.ieee.org/document/9281082): Improving Training Instance Quality in Aerial Image Object Detection With a Sampling-Balance-Based Multistage Network
* [yoltv5](https://github.com/avanetten/yoltv5) -> detects objects in arbitrarily large aerial or satellite images that far exceed the ~600×600 pixel size typically ingested by deep learning object detection frameworks. Uses YOLOv5 & pytorch
* [AIR](https://github.com/Accenture/AIR) -> A deep learning object detector framework written in Python for supporting Land Search and Rescue Missions
* [dior_detect](https://github.com/hm-better/dior_detect) -> benchmarks for object detection on DIOR dataset
* [Panchromatic to Multispectral: Object Detection Performance as a Function of Imaging Bands](https://medium.com/the-downlinq/panchromatic-to-multispectral-object-detection-performance-as-a-function-of-imaging-bands-51ecaaa3dc56) -> Medium article, concludes that more bands are not always beneficial, but likely varies by use case
* [OPLD-Pytorch](https://github.com/yf19970118/OPLD-Pytorch) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9252176): Learning Point-Guided Localization for Detection in Remote Sensing Images
* [F3Net](https://github.com/yxhnjust/F3Net) -> code for 2020 [paper](https://www.mdpi.com/2072-4292/12/24/4027): Feature Fusion and Filtration Network for Object Detection in Optical Remote Sensing Images
* [GLNet](https://github.com/Zhu1Teng/GLNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9386208): Global to Local: Clip-LSTM-Based Object Detection From Remote Sensing Images
* [SRAF-Net](https://github.com/Complicateddd/SRAF-Net) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9598916): A Scene-Relevant Anchor-Free Object Detection Network in Remote Sensing Images
* [object_detection_in_remote_sensing_images](https://github.com/EEexplorer001/object_detection_in_remote_sensing_images) -> using CNN and attention mechanism
* [SHAPObjectDetection](https://github.com/hiroki-kawauchi/SHAPObjectDetection) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/9/1970): SHAP-Based Interpretable Object Detection Method for Satellite Imagery
* [NWD](https://github.com/jwwangchn/NWD) -> code for 2021 [paper](https://arxiv.org/abs/2110.13389): A Normalized Gaussian Wasserstein Distance for Tiny Object Detection. Uses AI-TOD dataset
* [MSFC-Net](https://github.com/ZhAnGToNG1/MSFC-Net) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9535169): Multiscale Semantic Fusion-Guided Fractal Convolutional Object Detection Network for Optical Remote Sensing Imagery
* [LO-Det](https://github.com/Shank2358/LO-Det) -> code for the 2021 [paper](https://ieeexplore.ieee.org/document/9390310): LO-Det: Lightweight Oriented Object Detection in Remote Sensing Images
* [R2IPoints](https://github.com/shnew/R2IPoints) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9770816): R²IPoints: Pursuing Rotation-Insensitive Point Representation for Aerial Object Detection
* [Object-Detection](https://github.com/xiaojs18/Object-Detection) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/16/3969): Multi-Scale Object Detection with the Pixel Attention Mechanism in a Complex Background
* [mmdet-rfla](https://github.com/Chasel-Tsui/mmdet-rfla) -> code for 2022 [paper](https://arxiv.org/abs/2208.08738): RFLA: Gaussian Receptive based Label Assignment for Tiny Object Detection
* [Interactive-Multi-Class-Tiny-Object-Detection](https://github.com/ChungYi347/Interactive-Multi-Class-Tiny-Object-Detection) -> code for 2022 [paper](https://arxiv.org/abs/2203.15266): Interactive Multi-Class Tiny-Object Detection
* [small-object-detection-benchmark](https://github.com/fcakyon/small-object-detection-benchmark) -> code for ICIP 2022 [paper](https://arxiv.org/abs/2202.06934): Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection

#### Object counting
When the object count, but not its shape is required, U-net can be used to treat this as an image-to-image translation problem.
* [centroid-unet](https://github.com/gicait/centroid-unet) -> Centroid-UNet is deep neural network model to detect centroids from satellite images, with [paper](https://arxiv.org/abs/2112.06530)
* [cownter_strike](https://github.com/IssamLaradji/cownter_strike) -> counting cows, located with point-annotations, two models: CSRNet (a density-based method) & LCFCN (a detection-based method)
* [DO-U-Net](https://github.com/ToyahJade/DO-U-Net) -> an effective approach for when the size of an object needs to be known, as well as the number of objects in the image, initially created to segment and count Internally Displaced People (IDP) camps in Afghanistan
* [Cassava Crop Counting](https://medium.com/@wongsirikuln/cassava-standing-crop-counting-869cca486ce3)
* [Counting from Sky](https://github.com/gaoguangshuai/Counting-from-Sky-A-Large-scale-Dataset-for-Remote-Sensing-Object-Counting-and-A-Benchmark-Method) -> A Large-scale Dataset for Remote Sensing Object Counting and A Benchmark Method
* [PSGCNet](https://github.com/gaoguangshuai/PSGCNet) -> code for 2022 [paper](https://arxiv.org/abs/2012.03597): PSGCNet: A Pyramidal Scale and Global Context Guided Network for Dense Object Counting in Remote Sensing Images
* [psgcnet](https://github.com/gaoguangshuai/psgcnet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9720990): PSGCNet: A Pyramidal Scale and Global Context Guided Network for Dense Object Counting in Remote-Sensing Images

#### Object detection with rotated bounding boxes
* OBB: orinted bounding boxes are polygons representing rotated rectangles
* For datasets checkout [DOTA](https://github.com/robmarkcole/satellite-image-deep-learning#dota-object-detection-dataset) & [HRSC2016](https://github.com/robmarkcole/satellite-image-deep-learning#hrsc2016-ship-object-detection-dataset)
* [mmrotate](https://github.com/open-mmlab/mmrotate) -> Rotated Object Detection Benchmark, with pretrained models and function for inferencing on very large images
* [OBBDetection](https://github.com/jbwang1997/OBBDetection) -> an oriented object detection library, which is based on MMdetection
* [rotate-yolov3](https://github.com/ming71/rotate-yolov3) -> Rotation object detection implemented with yolov3. Also see [yolov3-polygon](https://github.com/ming71/yolov3-polygon)
* [DRBox](https://github.com/liulei01/DRBox) -> for detection tasks where the objects are orientated arbitrarily, e.g. vehicles, ships and airplanes
* [s2anet](https://github.com/csuhan/s2anet) -> Official code of the paper 'Align Deep Features for Oriented Object Detection'
* [CFC-Net](https://github.com/ming71/CFC-Net) -> Official implementation of "CFC-Net: A Critical Feature Capturing Network for Arbitrary-Oriented Object Detection in Remote Sensing Images"
* [ReDet](https://github.com/csuhan/ReDet) -> Official code of the paper "ReDet: A Rotation-equivariant Detector for Aerial Object Detection"
* [BBAVectors-Oriented-Object-Detection](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection) -> Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors
* [CSL_RetinaNet_Tensorflow](https://github.com/Thinklab-SJTU/CSL_RetinaNet_Tensorflow) -> Code for ECCV 2020 paper: Arbitrary-Oriented Object Detection with Circular Smooth Label
* [r3det-on-mmdetection](https://github.com/SJTU-Thinklab-Det/r3det-on-mmdetection) -> R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object
* [R-DFPN_FPN_Tensorflow](https://github.com/yangxue0827/R-DFPN_FPN_Tensorflow) -> Rotation Dense Feature Pyramid Networks (Tensorflow)
* [R2CNN_Faster-RCNN_Tensorflow](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow) -> Rotational region detection based on Faster-RCNN
* [Rotated-RetinaNet](https://github.com/ming71/Rotated-RetinaNet) -> implemented in pytorch, it supports the following datasets: DOTA, HRSC2016, ICDAR2013, ICDAR2015, UCAS-AOD, NWPU VHR-10, VOC2007
* [OBBDet_Swin](https://github.com/ming71/OBBDet_Swin) -> The sixth place winning solution in 2021 Gaofen Challenge
* [CG-Net](https://github.com/WeiZongqi/CG-Net) -> Learning Calibrated-Guidance for Object Detection in Aerial Images. With [paper](https://ieeexplore.ieee.org/abstract/document/9735375)
* [OrientedRepPoints_DOTA](https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA) -> Oriented RepPoints + Swin Transformer/ReResNet
* [yolov5_obb](https://github.com/hukaixuan19970627/yolov5_obb) -> yolov5 + Oriented Object Detection
* [How to Train YOLOv5 OBB](https://blog.roboflow.com/yolov5-for-oriented-object-detection/) -> YOLOv5 OBB tutorial and [YOLOv5 OBB noteboook](https://colab.research.google.com/drive/16nRwsioEYqWFLBF5VpT_NvELeOeupURM#scrollTo=1NZxhXTMWvek)
* [OHDet_Tensorflow](https://github.com/SJTU-Thinklab-Det/OHDet_Tensorflow) -> can be applied to rotation detection and object heading detection
* [Seodore](https://github.com/nijkah/Seodore) -> framework maintaining recent updates of mmdetection
* [Rotation-RetinaNet-PyTorch](https://github.com/HsLOL/Rotation-RetinaNet-PyTorch) -> oriented detector Rotation-RetinaNet implementation on Optical and SAR ship dataset
* [AIDet](https://github.com/jwwangchn/aidet) -> an open source object detection in aerial image toolbox based on MMDetection
* [rotation-yolov5](https://github.com/BossZard/rotation-yolov5) -> rotation detection based on yolov5
* [ShipDetection](https://github.com/lilinhao/ShipDetection) -> Ship Detection in HR Optical Remote Sensing Images via Rotated Bounding Box, based on Faster R-CNN and ORN, uses caffe
* [SLRDet](https://github.com/LUCKMOONLIGHT/SLRDet) -> project based on mmdetection to reimplement RRPN and use the model Faster R-CNN OBB
* [AxisLearning](https://github.com/RSIA-LIESMARS-WHU/AxisLearning) -> code for 2020 [paper](https://www.mdpi.com/2072-4292/12/6/908): Axis Learning for Orientated Objects Detection in Aerial Images
* [Detection_and_Recognition_in_Remote_Sensing_Image](https://github.com/whywhs/Detection_and_Recognition_in_Remote_Sensing_Image) -> This work uses PaNet to realize Detection and Recognition in Remote Sensing Image by MXNet
* [DrBox-v2-tensorflow](https://github.com/ZongxuPan/DrBox-v2-tensorflow) -> tensorflow implementation of DrBox-v2 which is an improved detector with rotatable boxes for target detection in remote sensing images
* [Rotation-EfficientDet-D0](https://github.com/HsLOL/Rotation-EfficientDet-D0) -> A PyTorch Implementation Rotation Detector based EfficientDet Detector, applied to custom rotation vehicle datasets
* [DODet](https://github.com/yanqingyao1994/DODet) -> Dual alignment for oriented object detection, uses DOTA dataset. With [paper](https://ieeexplore.ieee.org/abstract/document/9706434)
* [GF-CSL](https://github.com/WangJian981002/GF-CSL) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9776580): Gaussian Focal Loss: Learning Distribution Polarized Angle Prediction for Rotated Object Detection in Aerial Images
* [simplified_rbox_cnn](https://github.com/SIAnalytics/simplified_rbox_cnn) -> code for 2018 [paper](https://dl.acm.org/doi/10.1145/3274895.3274915): RBox-CNN: rotated bounding box based CNN for ship detection in remote sensing image. Uses Tensorflow object detection API
* [Polar-Encodings](https://github.com/flyingshan/Learning-Polar-Encodings-For-Arbitrary-Oriented-Ship-Detection-In-SAR-Images) -> code for 2021 [paper](Learning Polar Encodings for Arbitrary-Oriented Ship Detection in SAR Images)
* [R-CenterNet](https://github.com/ZeroE04/R-CenterNet) -> detector for rotated-object based on CenterNet
* [piou](https://github.com/clobotics/piou) -> Orientated Object Detection; IoU Loss, applied to DOTA dataset
* [DAFNe](https://github.com/steven-lang/DAFNe) -> code for 2021 [paper](https://arxiv.org/abs/2109.06148): DAFNe: A One-Stage Anchor-Free Approach for Oriented Object Detection
* [AProNet](https://github.com/geovsion/AProNet) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S092427162100229X): AProNet: Detecting objects with precise orientation from aerial images. Applied to datasets DOTA and HRSC2016
* [UCAS-AOD-benchmark](https://github.com/ming71/UCAS-AOD-benchmark) -> A benchmark of UCAS-AOD dataset
* [RotateObjectDetection](https://github.com/XinzeLee/RotateObjectDetection) -> based on Ultralytics/yolov5, with adjustments to enable rotate prediction boxes. Also see [PolygonObjectDetection](https://github.com/XinzeLee/PolygonObjectDetection)
* [AD-Toolbox](https://github.com/liuyanyi/AD-Toolbox) -> Aerial Detection Toolbox based on MMDetection and MMRotate, with support for more datasets
* [GGHL](https://github.com/Shank2358/GGHL) -> code for 2022 [paper](https://arxiv.org/abs/2109.12848): A General Gaussian Heatmap Label Assignment for Arbitrary-Oriented Object Detection
* [NPMMR-Det](https://github.com/Shank2358/NPMMR-Det) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9364888): A Novel Nonlocal-Aware Pyramid and Multiscale Multitask Refinement Detector for Object Detection in Remote Sensing Images
* [AOPG](https://github.com/jbwang1997/AOPG) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9795321): Anchor-Free Oriented Proposal Generator for Object Detection
* [SE2-Det](https://github.com/Virusxxxxxxx/SE2-Det) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/15/3637): Semantic-Edge-Supervised Single-Stage Detector for Oriented Object Detection in Remote Sensing Imagery

#### Object detection enhanced by super resolution
* [Super-Resolution and Object Detection](https://medium.com/the-downlinq/super-resolution-and-object-detection-a-love-story-part-4-8ad971eef81e) -> Super-resolution is a relatively inexpensive enhancement that can improve object detection performance
* [EESRGAN](https://github.com/Jakaria08/EESRGAN) -> Small-Object Detection in Remote Sensing Images with End-to-End Edge-Enhanced GAN and Object Detector Network
* [Mid-Low Resolution Remote Sensing Ship Detection Using Super-Resolved Feature Representation](https://www.preprints.org/manuscript/202108.0337/v1)
* [EESRGAN](https://github.com/divyam96/EESRGAN) -> code for 2020 [paper](https://www.mdpi.com/2072-4292/12/9/1432): Small-Object Detection in Remote Sensing Images with End-to-End Edge-Enhanced GAN and Object Detector Network. Applied to COWC & [OGST](https://data.mendeley.com/datasets/bkxj8z84m9/3) datasets
* [FBNet](https://github.com/wdzhao123/FBNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9739789): Feature Balance for Fine-Grained Object Classification in Aerial Images

#### Salient object detection
Detecting the most noticeable or important object in a scene
* [ACCoNet](https://github.com/MathLee/ACCoNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9756652): Adjacent Context Coordination Network for Salient Object Detection in Optical Remote Sensing Images
* [MCCNet](https://github.com/MathLee/MCCNet) -> Multi-Content Complementation Network for Salient Object Detection in Optical Remote Sensing Images
* [CorrNet](https://github.com/MathLee/CorrNet) -> Lightweight Salient Object Detection in Optical Remote Sensing Images via Feature Correlation. With [paper](https://arxiv.org/abs/2201.08049)
* [Reading list for deep learning based Salient Object Detection in Optical Remote Sensing Images](https://github.com/MathLee/ORSI-SOD_Summary)
* [ORSSD-dataset](https://github.com/rmcong/ORSSD-dataset) -> salient object detection dataset
* [EORSSD-dataset](https://github.com/rmcong/EORSSD-dataset) -> Extended Optical Remote Sensing Saliency Detection (EORSSD) Dataset
* [DAFNet_TIP20](https://github.com/rmcong/DAFNet_TIP20) -> code for 2020 [paper](https://arxiv.org/abs/2011.13144): Dense Attention Fluid Network for Salient Object Detection in Optical Remote Sensing Images
* [EMFINet](https://github.com/Kunye-Shen/EMFINet) -> code for 2021 paper: Edge-Aware Multiscale Feature Integration Network for Salient Object Detection in Optical Remote Sensing Images
* [ERPNet](https://github.com/zxforchid/ERPNet) -> code for 2022 paper: Edge-guided Recurrent Positioning Network for Salient Object Detection in Optical Remote Sensing Images
* [FSMINet](https://github.com/zxforchid/FSMINet) -> code for 2022 paper: Fully Squeezed Multi-Scale Inference Network for Fast and Accurate Saliency Detection in Optical Remote Sensing Images
* [AGNet](https://github.com/NuaaYH/AGNet) -> code for 2022 paper: AGNet: Attention Guided Network for Salient Object Detection in Optical Remote Sensing Images
* [MSCNet](https://github.com/NuaaYH/MSCNet) -> code for 2022 [paper](https://arxiv.org/abs/2205.08959): A lightweight multi-scale context network for salient object detection in optical remote sensing images
* [GPnet](https://github.com/liuyu1002/GPnet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9687549): Global Perception Network for Salient Object Detection in Remote Sensing Images

#### Object detection - Buildings, rooftops & solar panels
* [Machine Learning For Rooftop Detection and Solar Panel Installment](https://omdena.com/blog/machine-learning-rooftops/) discusses tiling large images and generating annotations from OSM data. Features of the roofs were calculated using a combination of contour detection and classification. [Follow up article using semantic segmentation](https://omdena.com/blog/rooftops-classification/)
* [Building Extraction with YOLT2 and SpaceNet Data](https://medium.com/the-downlinq/building-extraction-with-yolt2-and-spacenet-data-a926f9ffac4f)
* [AIcrowd dataset of building outlines](https://www.aicrowd.com/challenges/mapping-challenge-old) -> 300x300 pixel RGB images with annotations in MS-COCO format
* [XBD-hurricanes](https://github.com/dbuscombe-usgs/XBD-hurricanes) -> Models for building (and building damage) detection in high-resolution (<1m) satellite and aerial imagery using a modified RetinaNet model
* [Detecting solar panels from satellite imagery](https://towardsdatascience.com/weekend-project-detecting-solar-panels-from-satellite-imagery-f6f5d5e0da40) using segmentation
* [ssd-spacenet](https://github.com/aurotripathy/ssd-spacenet) -> Detect buildings in the Spacenet dataset using Single Shot MultiBox Detector (SSD)
* [3DBuildingInfoMap](https://github.com/LllC-mmd/3DBuildingInfoMap) -> simultaneous extraction of building height and footprint from Sentinel imagery using ResNet
* [Solar Panel Detection](https://medium.com/analytics-vidhya/solar-panel-detection-from-aerial-view-or-satellite-images-648c22c260ba) -> using Faster R-CNN & Tensorflow object detection API. With [repo](https://github.com/shiva2410/Solar_Panel-Detection-in-Aerial-Images) 
* [DeepSolaris](https://github.com/thinkpractice/DeepSolaris) -> a EuroStat project to detect solar panels in aerial images, further material [here](https://github.com/FHNW-IVGI/workshop_geopython2019/tree/master/Ex.02_SolarPanels)
* [ML_ObjectDetection_CAFO](https://github.com/Qberto/ML_ObjectDetection_CAFO) -> Detect Concentrated Animal Feeding Operations (CAFO) in Satellite Imagery
* [Multi-level-Building-Detection-Framework](https://github.com/luoxiaoliaolan/Multi-level-Building-Detection-Framework) -> code for 2018 [paper](https://ieeexplore.ieee.org/document/8458225): Multilevel Building Detection Framework in Remote Sensing Images Based on Convolutional Neural Networks
* [satellite_image_tinhouse_detector](https://github.com/yasserius/satellite_image_tinhouse_detector) -> Detection of tin houses from satellite/aerial images using the Tensorflow Object Detection API
* [Automatic Damage Annotation on Post-Hurricane Satellite Imagery](https://dds-lab.github.io/disaster-damage-detection/) -> detect damaged buildings using tensorflow object detection API. With repos [here](https://github.com/DDS-Lab/disaster-image-processing) and [here](https://github.com/annieyan/PreprocessSatelliteImagery-ObjectDetection)
* [mappingchallenge](https://github.com/krishanr/mappingchallenge) -> YOLOv5 applied to the AICrowd Mapping Challenge dataset

#### Object detection - Ships & boats
* Also see the [kaggle competition run by Airbus](https://github.com/robmarkcole/satellite-image-deep-learning#kaggle---airbus-ship-detection-challenge)
* [How hard is it for an AI to detect ships on satellite images?](https://medium.com/earthcube-stories/how-hard-it-is-for-an-ai-to-detect-ships-on-satellite-images-7265e34aadf0)
* [Object Detection in Satellite Imagery, a Low Overhead Approach](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7)
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
* [small-boat-detector](https://github.com/swricci/small-boat-detector) -> Trained yolo v3 model weights and configuration file to detect small boats in satellite imagery
* [Satellite-Imagery-Datasets-Containing-Ships](https://github.com/JasonManesis/Satellite-Imagery-Datasets-Containing-Ships) -> A list of optical and radar satellite datasets for ship detection, classification, semantic segmentation and instance segmentation tasks
* [Ship-Classification-in-Satellite-Images](https://github.com/JasonManesis/Ship-Classification-in-Satellite-Images) -> Convolutional neural network model for ship classification in satellite images
* [Ship-Detection](https://github.com/gouravbarkle/Ship-Detection) -> CNN approach for ship detection in the ocean using a satellite image
* [vesselTracker](https://github.com/carlossantamarizq/vesselTracker) -> Project based on reduced model of Yolov5 architecture using Pytorch. Custom dataset based on SAR imagery provided by Sentinel-1 through Earth Engine API
* [marine-debris-ml-model](https://github.com/danieltyukov/marine-debris-ml-model) -> Marine Debris Detection using tensorflow object detection API
* [SDGH-Net](https://github.com/WangZhenqing-RS/SDGH-Net-Ship-Detection-in-Optical-Remote-Sensing-Images-Based-on-Gaussian-Heatmap-Regression) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/3/499): SDGH-Net: Ship Detection in Optical Remote Sensing Images Based on Gaussian Heatmap Regression
* [LR-TSDet](https://github.com/Lausen-Ng/LR-TSDet) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/19/3890): LR-TSDet: Towards Tiny Ship Detection in Low-Resolution Remote Sensing Images
* [FGSCR-42](https://github.com/DYH666/FGSCR-42) -> A public Dataset for Fine-Grained Ship Classification in Remote sensing images
* [ShipDetection](https://github.com/lilinhao/ShipDetection) -> Ship Detection in HR Optical Remote Sensing Images via Rotated Bounding Box, based on Faster R-CNN and ORN, uses caffe
* [WakeNet](https://github.com/Lilytopia/WakeNet) -> A CNN-based optical image ship wake detector, code for 2021 paper: Rethinking Automatic Ship Wake Detection: State-of-the-Art CNN-based Wake Detection via Optical Images
* [Histogram of Oriented Gradients (HOG) Boat Heading Classification](https://medium.com/the-downlinq/histogram-of-oriented-gradients-hog-heading-classification-a92d1cf5b3cc) -> Medium article
* [Object Detection in Satellite Imagery, a Low Overhead Approach](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7) -> Medium article which demonstrates how to combine Canny edge detector pre-filters with HOG feature descriptors, random forest classifiers, and sliding windows to perform ship detection
* [simplified_rbox_cnn](https://github.com/SIAnalytics/simplified_rbox_cnn) -> code for 2018 [paper](https://dl.acm.org/doi/10.1145/3274895.3274915): RBox-CNN: rotated bounding box based CNN for ship detection in remote sensing image. Uses Tensorflow object detection API
* [Ship-Detection-based-on-YOLOv3-and-KV260](https://github.com/xlsjdjdk/Ship-Detection-based-on-YOLOv3-and-KV260) -> entry project of the Xilinx Adaptive Computing Challenge 2021. It uses YOLOv3 for ship target detection in optical remote sensing images, and deploys DPU on the KV260 platform to achieve hardware acceleration
* [LEVIR-Ship](https://github.com/WindVChen/LEVIR-Ship) -> a dataset for tiny ship detection under medium-resolution remote sensing images
* [Push-and-Pull-Network](https://github.com/WindVChen/Push-and-Pull-Network) -> code for 2022 paper: Contrastive Learning for Fine-grained Ship Classification in Remote Sensing Images
* [DRENet](https://github.com/WindVChen/DRENet) -> code for 2022 [paper])(https://ieeexplore.ieee.org/abstract/document/9791363): A Degraded Reconstruction Enhancement-Based Method for Tiny Ship Detection in Remote Sensing Images With a New Large-Scale Dataset
* [xView3-The-First-Place-Solution](https://github.com/BloodAxe/xView3-The-First-Place-Solution) - A winning solution for [xView 3](https://iuu.xview.us/) challenge (Vessel detection, classification and length estimation on Sentinetl-1 images). Contains trained models, inference pipeline and training code & configs to reproduce the results.
* [SARfish](https://github.com/MJCruickshank/SARfish) -> Ship detection in Sentinel 1 Synthetic Aperture Radar (SAR) imagery

#### Object detection - Cars, vehicles & trains
* [Truck Detection with Sentinel-2 during COVID-19 crisis](https://github.com/hfisser/Truck_Detection_Sentinel2_COVID19) -> moving objects in Sentinel-2 data causes a specific reflectance relationship in the RGB, which looks like a rainbow, and serves as a marker for trucks. Improve accuracy by only analysing roads. Not using object detection but relevant. Also see [S2TD](https://github.com/hfisser/S2TD)
* [cowc_car_counting](https://github.com/motokimura/cowc_car_counting) -> car counting on the [Cars Overhead With Context (COWC) dataset](https://gdo152.llnl.gov/cowc/). Not sctictly object detection but a CNN to predict the car count in a tile
* [CarCounting](https://github.com/JacksonPeoples/CarCounting) -> using Yolov3 & COWC dataset
* [Traffic density estimation as a regression problem instead of object detection](https://omdena.com/blog/ai-road-safety/) -> inspired by [this paper](https://ieeexplore.ieee.org/document/8916990)
* [Applying Computer Vision to Railcar Detection](https://orbitalinsight.com/blog/apping-computer-vision-to-railcar-detection) -> useful insights into counting railcars (i.e. train carriages) using Mask-RCNN with rotated bounding boxes output
* [Leveraging Deep Learning for Vehicle Detection And Classification](https://orbitalinsight.com/blog/leveraging-deep-learning-for-vehicle-detection-and-classification)
* [Detection of parkinglots and driveways with retinanet](https://github.com/spiyer99/retinanet)
* [pytorch-vedai](https://github.com/MichelHalmes/pytorch-vedai) -> object detection on the VEDAI dataset: Vehicle Detection in Aerial Imagery
* [Rotation-EfficientDet-D0](https://github.com/HsLOL/Rotation-EfficientDet-D0) -> PyTorch implementation of Rotated EfficientDet, applied to a custom rotation vehicle dataset (car counting)
* [RSVC2021-Dataset](https://github.com/YinongGuo/RSVC2021-Dataset) -> A dataset for Vehicle Counting in Remote Sensing images, created from the DOTA & [ITCVD](https://research.utwente.nl/en/datasets/itcvd-dataset) Datasets
* [Car Localization and Counting with Overhead Imagery, an Interactive Exploration](https://medium.com/the-downlinq/car-localization-and-counting-with-overhead-imagery-an-interactive-exploration-9d5a029a596b) -> Medium article by Adam Van Etten
* [Vehicle-Counting-in-Very-Low-Resolution-Aerial-Images](https://github.com/hbsszq/Vehicle-Counting-in-Very-Low-Resolution-Aerial-Images) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9775767): Vehicle Counting in Very Low-Resolution Aerial Images via Cross-Resolution Spatial Consistency and Intraresolution Time Continuity

#### Object detection - Planes & aircraft
* [yoltv4](https://github.com/avanetten/yoltv4) includes examples on the [RarePlanes dataset](https://registry.opendata.aws/rareplanes/)
* [aircraft-detection](https://github.com/hakeemtfrank/aircraft-detection) -> experiments to test the performance of a Gaussian process (GP) classifier with various kernels on the UC Merced land use land cover (LULC) dataset
* The [Rareplanes guide](https://www.cosmiqworks.org/rareplanes-public-user-guide/) recommends annotating airplanes in a diamond style, which has several advantages (easily reproducible, convertible to a bounding box etc) and allows extracting the aircraft length and wingspan
* [rareplanes-yolov5](https://github.com/jeffaudi/rareplanes-yolov5) -> using YOLOv5 and the RarePlanes dataset to detect and classify sub-characteristics of aircraft
* [detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5](https://medium.com/artificialis/detecting-aircrafts-on-airbus-pleiades-imagery-with-yolov5-5f3d464b75ad)
* [Using Detectron2 to segment aircraft from satellite imagery](https://share.buitrongan.com/using-detectron2-to-segments-aircraft-from-satellite-images-5a8ac1a0d35e) -> pytorch and Rare Planes
* [Faster RCNN to detect airplanes](https://github.com/ShubhankarRawat/Airplane-Detection-for-Satellites)
* [aircraft-detection-from-satellite-images-yolov3](https://github.com/emrekrtorun/aircraft-detection-from-satellite-images-yolov3) -> trained on kaggle cgi-planes-in-satellite-imagery-w-bboxes dataset
* [HRPlanesv2-Data-Set](https://github.com/dilsadunsal/HRPlanesv2-Data-Set) -> YOLOv4 and YOLOv5 weights trained on the HRPlanesv2 dataset
* [Deep-Learning-for-Aircraft-Recognition](https://github.com/Shayan-Bravo/Deep-Learning-for-Aircraft-Recognition) -> A CNN model trained to classify and identify various military aircraft through satellite imagery
* [FRCNN-for-Aircraft-Detection](https://github.com/Huatsing-Lau/FRCNN-for-Aircraft-Detection) -> faster-rcnn & keras
* [ergo-planes-detector](https://github.com/evilsocket/ergo-planes-detector) -> An ergo based project that relies on a convolutional neural network to detect airplanes from satellite imagery, uses the PlanesNet dataset
* [pytorch-remote-sensing](https://github.com/miko7879/pytorch-remote-sensing) -> Aircraft detection using the 'Airbus Aircraft Detection' dataset and Faster-RCNN with ResNet-50 backbone using pytorch
* [FasterRCNN_ObjectDetection](https://github.com/UKMIITB/FasterRCNN_ObjectDetection) -> faster RCNN model for aircraft detection and localisation in satellite images and creating a webpage with live server for public usage

#### Object detection - Infrastructure & utilities
* [wind-turbine-detector](https://github.com/lbborkowski/wind-turbine-detector) -> Wind Turbine Object Detection from Aerial Imagery Using TensorFlow Object Detection API
* [Water Tanks and Swimming Pools Detection](https://github.com/EduardoFernandes1410/PATREO-Dengue) -> uses Faster R-CNN
* [PCAN](https://www.mdpi.com/2072-4292/13/7/1243) -> Part-Based Context Attention Network for Thermal Power Plant Detection in Remote Sensing Imagery, with [dataset](https://github.com/wenxinYin/AIR-TPPDD)

#### Object detection - Animals
A variety of techniques can be used to count animals, including object detection and instance segmentation. For convenience they are all listed here:
* [cownter_strike](https://github.com/IssamLaradji/cownter_strike) -> counting cows, located with point-annotations, two models: CSRNet (a density-based method) & LCFCN (a detection-based method)
* [elephant_detection](https://github.com/akharina/elephant_detection) -> Using Keras-Retinanet to detect elephants from aerial images
* [CNN-Mosquito-Detection](https://github.com/sriramelango/CNN-Mosquito-Detection) -> determining the locations of potentially dangerous breeding grounds, compared YOLOv4, YOLOR & YOLOv5
* [Borowicz_etal_Spacewhale](https://github.com/lynch-lab/Borowicz_etal_Spacewhale) -> locate whales using ResNet
* [walrus-detection-and-count](https://github.com/sweetlhare/walrus-detection-and-count) -> uses Mask R-CNN instance segmentation
* [MarineMammalsDetection](https://github.com/Pangoraw/MarineMammalsDetection) -> Weakly Supervised Detection of Marine Animals in High Resolution Aerial Images

#### Object tracking in videos
* [Object Tracking in Satellite Videos Based on a Multi-Frame Optical Flow Tracker](https://arxiv.org/abs/1804.09323) arxiv paper
* [CFME](https://github.com/SY-Xuan/CFME) -> Object Tracking in Satellite Videos by Improved Correlation Filters With Motion Estimations
* [TGraM](https://github.com/HeQibin/TGraM) -> code and dataset for 2022 [paper](https://ieeexplore.ieee.org/document/9715124): Multi-Object Tracking in Satellite Videos with Graph-Based Multi-Task Modeling
* [satellite_video_mod_groundtruth](https://github.com/zhangjunpeng9354/satellite_video_mod_groundtruth) -> groundtruth on satellite video for evaluating moving object detection algorithm
* [Moving-object-detection-DSFNet](https://github.com/ChaoXiao12/Moving-object-detection-DSFNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9594855): DSFNet: Dynamic and Static Fusion Network for Moving Object Detection in Satellite Videos
* [HiFT](https://github.com/vision4robotics/HiFT) -> code for 2021 [paper](https://arxiv.org/abs/2108.00202): HiFT: Hierarchical Feature Transformer for Aerial Tracking
* [TCTrack](https://github.com/vision4robotics/TCTrack) -> code for 2022 [paper](https://arxiv.org/abs/2203.01885): TCTrack: Temporal Contexts for Aerial Tracking

## Counting trees
* [DeepForest](https://deepforest.readthedocs.io/en/latest/index.html) is a python package for training and predicting individual tree crowns from airborne RGB imagery
* [Official repository for the "Identifying trees on satellite images" challenge from Omdena](https://github.com/cienciaydatos/ai-challenge-trees)
* [Counting-Trees-using-Satellite-Images](https://github.com/A2Amir/Counting-Trees-using-Satellite-Images) -> create an inventory of incoming and outgoing trees for an annual tree inspections, uses keras & semantic segmentation
* [2020 Nature paper - An unexpectedly large count of trees in the West African Sahara and Sahel](https://www.nature.com/articles/s41586-020-2824-5) -> tree detection framework based on U-Net & tensorflow 2 with code [here](https://github.com/ankitkariryaa/An-unexpectedly-large-count-of-trees-in-the-western-Sahara-and-Sahel/tree/v1.0.0)
* [TreeDetection](https://github.com/AmirNiaraki/TreeDetection) -> A color-based classifier to detect the trees in google image data along with tree visual localization and crown size calculations via OpenCV
* [PTDM](https://github.com/hr8yhtzb/PTDM) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/16/3902): Pomelo Tree Detection Method Based on Attention Mechanism and Cross-Layer Feature Fusion
* [urban-tree-detection](https://github.com/jonathanventura/urban-tree-detection) -> code for 2022 [paper](https://arxiv.org/abs/2208.10607): Individual Tree Detection in Large-Scale Urban Environments using High-Resolution Multispectral Imagery. With [dataset](https://github.com/jonathanventura/urban-tree-detection-data)

## Oil storage tank detection & oil spills
Oil is stored in tanks at many points between extraction and sale, and the volume of oil in storage is an important economic indicator.
* [A Beginner’s Guide To Calculating Oil Storage Tank Occupancy With Help Of Satellite Imagery](https://medium.com/planet-stories/a-beginners-guide-to-calculating-oil-storage-tank-occupancy-with-help-of-satellite-imagery-e8f387200178)
* [Oil Storage Tank’s Volume Occupancy On Satellite Imagery Using YoloV3](https://towardsdatascience.com/oil-storage-tanks-volume-occupancy-on-satellite-imagery-using-yolov3-3cf251362d9d) with [repo](https://github.com/mdmub0587/Oil-Storage-Tank-s-Volume-Occupancy)
* [Oil-Tank-Volume-Estimation](https://github.com/kheyer/Oil-Tank-Volume-Estimation) -> combines object detection and classical computer vision
* [MCAN-OilSpillDetection](https://github.com/liyongqingupc/MCAN-OilSpillDetection) -> Oil Spill Detection with A Multiscale Conditional Adversarial Network under Small Data Training, with [paper](https://www.mdpi.com/2072-4292/13/12/2378). A multiscale conditional adversarial network (MCAN) trained with four oil spill observation images accurately detects oil spills in new images.
* [Oil tank instance segmentation with Mask R-CNN](https://github.com/georgiosouzounis/instance-segmentation-mask-rcnn) with [accompanying article](https://medium.com/@georgios.ouzounis/oil-storage-tank-instance-segmentation-with-mask-r-cnn-77c94433045f) using Keras & Airbus Oil Storage Detection Dataset on Kaggle
* https://www.kaggle.com/towardsentropy/oil-storage-tanks -> large kaggle dataset, note however that approx 85% of images contain no tanks
* https://www.kaggle.com/airbusgeo/airbus-oil-storage-detection-dataset -> smaller kaggle dataset
* [ognet](https://stanfordmlgroup.github.io/projects/ognet/) -> a Global Oil and Gas Infrastructure Database using Deep Learning on Remotely Sensed Imagery
* [RSOD-Dataset](https://github.com/RSIA-LIESMARS-WHU/RSOD-Dataset-) -> dataset for object detection in PASCAL VOC format. Aircraft, playgrounds, overpasses & oiltanks. Used in the 2022 [paper](https://link.springer.com/article/10.1007/s00500-022-07106-8): Improved YOLOv5 network method for remote sensing image-based ground objects recognition
* [oil_storage-detector](https://github.com/TheodorEmanuelsson/oil_storage-detector) -> using yolov5 and the Airbus Oil Storage Detection dataset
* [oil_well_detector](https://github.com/dzubke/oil_well_detector) -> detect oil wells in the Bakken oil field based on satellite imagery
* [OGST](https://data.mendeley.com/datasets/bkxj8z84m9/3) -> Oil and Gas Tank Dataset
* [AContrarioTankDetection](https://github.com/anttad/AContrarioTankDetection) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9323249): Oil Tank Detection in Satellite Images via a Contrario Clustering
* [SubpixelCircleDetection](https://github.com/anttad/SubpixelCircleDetection) -> code for 2020 [paper](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/901/2020/): CIRCULAR-SHAPED OBJECT DETECTION IN LOW RESOLUTION SATELLITE IMAGES
* [Oil Storage Detection on Airbus Imagery with YOLOX](https://medium.com/artificialis/oil-storage-detection-on-airbus-imagery-with-yolox-9e38eb6f7e62) -> uses the Kaggle Airbus Oil Storage Detection dataset

## Cloud detection & removal
Generally treated as a semantic segmentation problem or custom features created using band math
* See section [Kaggle - Understanding Clouds from Satellite Images](https://github.com/robmarkcole/satellite-image-deep-learning#kaggle---understanding-clouds-from-satellite-images)
* From [this article on sentinelhub](https://medium.com/sentinel-hub/improving-cloud-detection-with-machine-learning-c09dc5d7cf13) there are three popular classical algorithms that detects thresholds in multiple bands in order to identify clouds. In the same article they propose using semantic segmentation combined with a CNN for a cloud classifier (excellent review paper [here](https://arxiv.org/pdf/1704.06857.pdf)), but state that this requires too much compute resources.
* [This article](https://www.mdpi.com/2072-4292/8/8/666) compares a number of ML algorithms, random forests, stochastic gradient descent, support vector machines, Bayesian method.
* [Segmentation of Clouds in Satellite Images Using Deep Learning](https://medium.com/swlh/segmentation-of-clouds-in-satellite-images-using-deep-learning-a9f56e0aa83d) -> semantic segmentation using a Unet on the Kaggle 38-Cloud dataset
* [Cloud Detection in Satellite Imagery](https://www.azavea.com/blog/2021/02/08/cloud-detection-in-satellite-imagery/) compares FPN+ResNet18 and CheapLab architectures on Sentinel-2 L1C and L2A imagery
* [Cloud-Removal-with-GAN-Satellite-Image-Processing](https://github.com/Akash-Ramjyothi/Cloud-Removal-with-GAN-Satellite-Image-Processing)
* [Benchmarking Deep Learning models for Cloud Detection in Landsat-8 and Sentinel-2 images](https://github.com/IPL-UV/DL-L8S2-UV)
* [Landsat-8 to Proba-V Transfer Learning and Domain Adaptation for Cloud detection](https://github.com/IPL-UV/pvl8dagans)
* [Multitemporal Cloud Masking in Google Earth Engine](https://github.com/IPL-UV/ee_ipl_uv)
* [s2cloudmask](https://github.com/daleroberts/s2cloudmask) -> Sentinel-2 Cloud and Shadow Detection using Machine Learning
* [sentinel2-cloud-detector](https://github.com/sentinel-hub/sentinel2-cloud-detector) -> Sentinel Hub Cloud Detector for Sentinel-2 images in Python
* [dsen2-cr](https://github.com/ameraner/dsen2-cr) -> cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion, contains the model code, written in Python/Keras, as well as links to pre-trained checkpoints and the SEN12MS-CR dataset
* [pyatsa](https://github.com/agroimpacts/pyatsa) -> Python package implementing the Automated Time-Series Analysis method for masking clouds in satellite imagery developed by Zhu and Helmer 2018
* [decloud](https://github.com/CNES/decloud) -> Decloud enables the training of various deep nets to remove clouds in optical image, using e.g. Sentinel 1 & 2
* [cloudless](https://github.com/BradNeuberg/cloudless) -> Deep learning pipeline for orbital satellite data for detecting clouds
* [Deep-Gapfill](https://github.com/remicres/Deep-Gapfill) -> Official implementation of Optical image gap filling using deep convolutional autoencoder from optical and radar images
* [satellite-cloud-removal-dip](https://github.com/cidcom/satellite-cloud-removal-dip) -> Satellite cloud removal with Deep Image Prior, with [paper](https://www.mdpi.com/2072-4292/14/6/1342)
* [cloudFCN](https://github.com/aliFrancis/cloudFCN) -> Python 3 package for Fully Convolutional Network development, specifically for cloud masking
* [Fmask](https://github.com/GERSL/Fmask) -> Fmask (Function of mask) is used for automated clouds, cloud shadows, snow, and water masking for Landsats 4-9 and Sentinel 2 images, in Matlab. Also see [PyFmask](https://github.com/akalenda/PyFmask)
* [HOW TO USE DEEP LEARNING, PYTORCH LIGHTNING, AND THE PLANETARY COMPUTER TO PREDICT CLOUD COVER IN SATELLITE IMAGERY](https://www.drivendata.co/blog/cloud-cover-benchmark/)
* [cloud-cover-winners](https://github.com/drivendataorg/cloud-cover-winners) -> Code from the winning submissions for the On Cloud N: Cloud Cover Detection Challenge
* [On-Cloud-N: Cloud Cover Detection Challenge - 19th Place Solution](https://github.com/max-schaefer-dev/on-cloud-n-19th-place-solution)
* [ukis-csmask](https://github.com/dlr-eoc/ukis-csmask) -> package to masks clouds in Sentinel-2, Landsat-8, Landsat-7 and Landsat-5 images
* [OpenSICDR](https://github.com/dr-lizhiwei/OpenSICDR) -> long list of satellite image cloud detection resources
* [RS-Net](https://github.com/JacobJeppesen/RS-Net) -> code for the paper: A cloud detection algorithm for satellite imagery based on deep learning
* [Clouds-Segmentation-Project](https://github.com/TamirShalev/Clouds-Segmentation-Project) -> treats as a 3 class problem; Open clouds, Closed clouds and no clouds, uses pytorch on a dataset that consists of IR & Visual Grayscale images
* [STGAN](https://github.com/ermongroup/STGAN) -> PyTorch Implementation of STGAN for Cloud Removal in Satellite Images, with [paper](https://arxiv.org/abs/1912.06838)
* [mcgan-cvprw2017-pytorch](https://github.com/enomotokenji/mcgan-cvprw2017-pytorch) -> code for 2017 paper: Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets
* [Cloud-Net: A semantic segmentation CNN for cloud detection](https://github.com/SorourMo/Cloud-Net-A-semantic-segmentation-CNN-for-cloud-detection) -> an end-to-end cloud detection algorithm for Landsat 8 imagery, trained on 38-Cloud Training Set
* [fcd](https://github.com/jnyborg/fcd) -> code for 2021 paper: Fixed-Point GAN for Cloud Detection. A weakly-supervised approach, training with only image-level labels
* [CloudX-Net](https://github.com/sumitkanu/CloudX-Net) -> an efficient and robust architecture used for detection of clouds from satellite images
* [A simple cloud-detection walk-through using Convolutional Neural Network (CNN and U-Net) and fast.ai library](https://medium.com/analytics-vidhya/a-simple-cloud-detection-walk-through-using-convolutional-neural-network-cnn-and-u-net-and-bc745dda4b04)
* [38Cloud-Medium](https://github.com/cordmaur/38Cloud-Medium) -> Walk-through using u-net to detect clouds in satellite images with fast.ai
* [cloud_detection_using_satellite_data](https://github.com/ZhouPeng-NIMST/cloud_detection_using_satellite_data) -> performed on Sentinel 2 data
* [Luojia1-Cloud-Detection](https://github.com/dedztbh/Luojia1-Cloud-Detection) -> Luojia-1 Satellite Visible Band Nighttime Imagery Cloud Detection
* [SEN12MS-CR-TS](https://github.com/PatrickTUM/SEN12MS-CR-TS) -> code for 2022 paper: A Remote Sensing Data Set for Multi-modal Multi-temporal Cloud Removal
* [ES-CCGAN](https://github.com/AnnaCUG/ES-CCGAN) -> This is a dehazed method for remote sensing image, which based on CycleGAN
* [Cloud_Classification_DL](https://github.com/nishp763/Cloud_Classification_DL) -> Classifying cloud organization patterns from satellite images using Deep Learning techniques (Mask R-CNN)
* [CNN-based-Cloud-Detection-Methods](https://github.com/LK-Peng/CNN-based-Cloud-Detection-Methods) -> Understanding the Role of Receptive Field of Convolutional Neural Network for Cloud Detection in Landsat 8 OLI Imagery
* [cloud-removal-deploy](https://github.com/XavierJiezou/cloud-removal-deploy) -> flask app for cloud removal
* [CloudMattingGAN](https://github.com/flyakon/CloudMattingGAN) -> code for 2019 [paper](https://ieeexplore.ieee.org/document/9009465): Generative Adversarial Training for Weakly Supervised Cloud Matting
* [atrain-cloudseg](https://github.com/seanremy/atrain-cloudseg) -> Official repository for the A-Train Cloud Segmentation Dataset
* [CDnet](https://github.com/nkszjx/CDnet-pytorch-master) -> code for 2019 paper: CNN-Based Cloud Detection for Remote Sensing Imager
* [GLNET](https://github.com/wuchangsheng951/GLNET) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9607791): Convolutional Neural Networks Based Remote Sensing Scene Classification under Clear and Cloudy Environments
* [CDnetV2](https://github.com/nkszjx/CDnetV2-pytorch-master) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9094671): CNN-Based Cloud Detection for Remote Sensing Imagery With Cloud-Snow Coexistence
* [grouped-features-alignment](https://github.com/nkszjx/grouped-features-alignment) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9387459): Unsupervised Domain Adaptation for Cloud Detection Based on Grouped Features Alignment and Entropy Minimization
* [Detecting Cloud Cover Via Sentinel-2 Satellite Data](https://benjaminwarner.dev/2022/03/11/detecting-cloud-cover-via-satellite) -> blog post on Benjamin Warners Top-10 Percent Solution to DrivenData’s On CloudN Competition using fast.ai & customized version of XResNeXt50. [Repo](https://github.com/warner-benjamin/code_for_blog_posts/tree/main/2022/drivendata_cloudn)
* [AISD](https://github.com/RSrscoder/AISD) -> code (Matlab) and dataset for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271620302045): Deeply supervised convolutional neural network for shadow detection based on a novel aerial shadow imagery dataset
* [CloudGAN](https://github.com/JerrySchonenberg/CloudGAN) -> Detecting and Removing Clouds from RGB-images using Image Inpainting
* [Using GANs to Augment Data for Cloud Image Segmentation Task](https://github.com/jain15mayank/GAN-augmentation-cloud-image-segmentation) -> code for 2021 [paper](https://arxiv.org/abs/2106.03064)
* [Cloud-Segmentation-from-Satellite-Imagery](https://github.com/vedantk-b/Cloud-Segmentation-from-Satellite-Imagery) -> applied to Sentinel-2 dataset
* [HRC_WHU](https://github.com/dr-lizhiwei/HRC_WHU) -> High-Resolution Cloud Detection Dataset comprising 150 RGB images and a resolution varying from 0.5 to 15 m in different global regions
* [MEcGANs](https://github.com/andrzejmizera/MEcGANs) -> Cloud Removal from Satellite Imagery using Multispectral Edge-filtered Conditional Generative Adversarial Networks
* [CloudXNet](https://github.com/shyamfec/CloudXNet) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S2352938520303803): CloudX-net: A robust encoder-decoder architecture for cloud detection from satellite remote sensing images
* [refined-unet-lite](https://github.com/92xianshen/refined-unet-lite) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S1877050922005361): Refined UNet Lite: End-to-End Lightweight Network for Edge-precise Cloud Detection
* [cloud-buster](https://github.com/azavea/cloud-buster) -> Sentinel-2 L1C and L2A Imagery with Fewer Clouds
* [SatelliteCloudGenerator](https://github.com/cidcom/SatelliteCloudGenerator) -> A PyTorch-based tool to generate clouds for satellite images
* [SEnSeI](https://github.com/aliFrancis/SEnSeI) -> A python 3 package for developing sensor independent deep learning models for cloud masking in satellite imagery

## Change detection
Generally speaking, change detection methods are applied to a pair of images to generate a mask of change, e.g. of buildings damaged in a disaster. Note, clouds & shadows change often too..!
* [awesome-remote-sensing-change-detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection) lists many datasets and publications
* [Change-Detection-Review](https://github.com/MinZHANG-WHU/Change-Detection-Review) -> A review of change detection methods, including code and open data sets for deep learning
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
* [Self-supervised Change Detection in Multi-view Remote Sensing Images](https://github.com/cyx669521/self-supervised_change_detetction)
* [MFPNet](https://github.com/wzjialang/MFPNet) -> Remote Sensing Change Detection Based on Multidirectional Adaptive Feature Fusion and Perceptual Similarity
* [GitHub for the DIUx xView Detection Challenge](https://github.com/DIUx-xView) -> The xView2 Challenge focuses on automating the process of assessing building damage after a natural disaster
* [DASNet](https://github.com/lehaifeng/DASNet) -> Dual attentive fully convolutional siamese networks for change detection of high-resolution satellite images
* [Self-Attention for Raw Optical Satellite Time Series Classification](https://github.com/MarcCoru/crop-type-mapping)
* [planet-movement](https://github.com/rhammell/planet-movement) -> Find and process Planet image pairs to highlight object movement
* [UNet-based-Unsupervised-Change-Detection](https://github.com/annabosman/UNet-based-Unsupervised-Change-Detection) -> A convolutional neural network (CNN) and semantic segmentation is implemented to detect the changes between the images, as well as classify the changes into the correct semantic class, with [arxiv paper](https://arxiv.org/abs/1812.05815)
* [temporal-cluster-matching](https://github.com/microsoft/temporal-cluster-matching) -> detecting change in structure footprints from time series of remotely sensed imagery
* [autoRIFT](https://github.com/nasa-jpl/autoRIFT) -> fast and intelligent algorithm for finding the pixel displacement between two images
* [DSAMNet](https://github.com/liumency/DSAMNet) -> Code for “A Deeply Supervised Attention Metric-Based Network and an Open Aerial Image Dataset for Remote Sensing Change Detection”. The main types of changes in the dataset include: (a) newly built urban buildings; (b) suburban dilation; (c) groundwork before construction; (d) change of vegetation; (e) road expansion; (f) sea construction.
* [SRCDNet](https://github.com/liumency/SRCDNet) -> The pytorch implementation for "Super-resolution-based Change Detection Network with Stacked Attention Module for Images with Different Resolutions ". SRCDNet is designed to learn and predict change maps from bi-temporal images with different resolutions
* [Land-Cover-Analysis](https://github.com/Kalit31/Land-Cover-Analysis) -> Land Cover Change Detection using Satellite Image Segmentation
* [A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sening images](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images)
* [Satellite-Image-Alignment-Differencing-and-Segmentation](https://github.com/rishi5kesh/Satellite-Image-Alignment-Differencing-and-Segmentation) -> thesis on change detection
* [Change Detection in Multi-temporal Satellite Images](https://github.com/IhebeddineRyahi/Change-detection-in-multitemporal-satellite-images) -> uses Principal Component Analysis (PCA) and K-means clustering
* [Unsupervised Change Detection Algorithm using PCA and K-Means Clustering](https://github.com/leduckhai/Change-Detection-PCA-KMeans) -> in Matlab but has paper
* [ChangeFormer](https://github.com/wgcban/ChangeFormer) -> A Transformer-Based Siamese Network for Change Detection. Uses transformer architecture to address the limitations of CNN in handling multi-scale long-range details. Demonstrates that ChangeFormer captures much finer details compared to the other SOTA methods, achieving better performance on benchmark datasets
* [Heterogeneous_CD](https://github.com/llu025/Heterogeneous_CD) -> Heterogeneous Change Detection in Remote Sensing Images. Accompanies [Code-Aligned Autoencoders for Unsupervised Change Detection in Multimodal Remote Sensing Images](https://arxiv.org/abs/2004.07011)
* [ChangeDetectionProject](https://github.com/previtus/ChangeDetectionProject) -> Trying out Active Learning in with deep CNNs for Change detection on remote sensing data
* [DSFANet](https://github.com/rulixiang/DSFANet) -> Unsupervised Deep Slow Feature Analysis for Change Detection in Multi-Temporal Remote Sensing Images
* [siamese-change-detection](https://github.com/mvkolos/siamese-change-detection) -> Targeted synthesis of multi-temporal remote sensing images for change detection using siamese neural networks
* [Bi-SRNet](https://github.com/ggsDing/Bi-SRNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9721305): Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images
* [RaVAEn](https://github.com/spaceml-org/RaVAEn) -> RaVAEn is a lightweight, unsupervised approach for change detection in satellite data based on Variational Auto-Encoders (VAEs) with the specific purpose of on-board deployment
* [SiROC](https://github.com/lukaskondmann/SiROC) -> Implementation of the [paper](https://ieeexplore.ieee.org/document/9627707) Spatial Context Awareness for Unsupervised Change Detection in Optical Satellite Images. Applied to Sentinel-2 and high-resolution Planetscope imagery on four datasets
* [DSMSCN](https://github.com/I-Hope-Peace/DSMSCN) -> Tensorflow implementation for Change Detection in Multi-temporal VHR Images Based on Deep Siamese Multi-scale Convolutional Neural Networks
* [RaVAEn](https://github.com/spaceml-org/RaVAEn) -> a lightweight, unsupervised approach for change detection in satellite data based on Variational Auto-Encoders (VAEs) with the specific purpose of on-board deployment.  It flags changed areas to prioritise for downlink, shortening the response time
* [SemiCD](https://github.com/wgcban/SemiCD) -> Code for [paper](https://arxiv.org/abs/2204.08454): Revisiting Consistency Regularization for Semi-supervised Change Detection in Remote Sensing Images. Achieves the performance of supervised CD even with access to as little as 10% of the annotated training data
* [FCCDN_pytorch](https://github.com/chenpan0615/FCCDN_pytorch) -> code for [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000636): FCCDN: Feature Constraint Network for VHR Image Change Detection. Uses the [LEVIR-CD](https://justchenhao.github.io/LEVIR/) building change detection dataset
* [INLPG_Python](https://github.com/zcsisiyao/INLPG_Python) -> code for paper: Structure Consistency based Graph for Unsupervised Change Detection with Homogeneous and Heterogeneous Remote Sensing Images
* [NSPG_Python](https://github.com/zcsisiyao/NSPG_Python) -> code for paper: Nonlocal patch similarity based heterogeneous remote sensing change detection
* [LGPNet-BCD](https://github.com/TongfeiLiu/LGPNet-BCD) -> code for 2021 paper: Building Change Detection for VHR Remote Sensing Images via Local-Global Pyramid Network and Cross-Task Transfer Learning Strategy
* [DS_UNet](https://github.com/SebastianHafner/DS_UNet) -> code for 2021 paper: Sentinel-1 and Sentinel-2 Data Fusion for Urban Change Detection using a Dual Stream U-Net, uses Onera Satellite Change Detection dataset
* [SiameseSSL](https://github.com/SebastianHafner/SiameseSSL) -> code for 2022 [paper](https://arxiv.org/abs/2204.12202): Urban change detection with a Dual-Task Siamese network and semi-supervised learning. Uses SpaceNet 7 dataset
* [CD-SOTA-methods](https://github.com/wgcban/CD-SOTA-methods) -> Remote sensing change detection: State-of-the-art methods and available datasets
* [multimodalCD_ISPRS21](https://github.com/PatrickTUM/multimodalCD_ISPRS21) -> code for 2021 paper: Fusing Multi-modal Data for Supervised Change Detection
* [Unsupervised-CD-in-SITS-using-DL-and-Graphs](https://github.com/ekalinicheva/Unsupervised-CD-in-SITS-using-DL-and-Graphs) -> code for article: Unsupervised Change Detection Analysis in Satellite Image Time Series using Deep Learning Combined with Graph-Based Approaches
* [LSNet](https://github.com/qaz670756/LSNet) -> code for 2022 [paper](https://arxiv.org/abs/2201.09156): Extremely Light-Weight Siamese Network For Change Detection in Remote Sensing Image
* [Change-Detection-in-Remote-Sensing-Images](https://github.com/themrityunjay/Change-Detection-in-Remote-Sensing-Images) ->  using PCA & K-means
* [End-to-end-CD-for-VHR-satellite-image](https://github.com/daifeng2016/End-to-end-CD-for-VHR-satellite-image) -> code for 2019 [paper](https://www.mdpi.com/2072-4292/11/11/1382): End-to-End Change Detection for High Resolution Satellite Images Using Improved UNet++
* [Semantic-Change-Detection](https://github.com/daifeng2016/Semantic-Change-Detection) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/pii/S0303243421001720): SCDNET: A novel convolutional network for semantic change detection in high resolution optical remote sensing imagery
* [ERCNN-DRS_urban_change_monitoring](https://github.com/It4innovations/ERCNN-DRS_urban_change_monitoring) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/15/3000): Neural Network-Based Urban Change Monitoring with Deep-Temporal Multispectral and SAR Remote Sensing Data
* [EGRCNN](https://github.com/luting-hnu/EGRCNN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9524849): Edge-guided Recurrent Convolutional Neural Network for Multi-temporal Remote Sensing Image Building Change Detection
* [Unsupervised-Remote-Sensing-Change-Detection](https://github.com/TangXu-Group/Unsupervised-Remote-Sensing-Change-Detection) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9526855): An Unsupervised Remote Sensing Change Detection Method Based on Multiscale Graph Convolutional Network and Metric Learning
* [CropLand-CD](https://github.com/liumency/CropLand-CD) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9780164): A CNN-transformer Network with Multi-scale Context Aggregation for Fine-grained Cropland Change Detection
* [contrastive-surface-image-pretraining](https://github.com/isaaccorley/contrastive-surface-image-pretraining) -> code for 2022 [paper](https://arxiv.org/abs/2202.13251): Supervising Remote Sensing Change Detection Models with 3D Surface Semantics
* [dcvaVHROptical](https://github.com/sudipansaha/dcvaVHROptical) -> Deep Change Vector Analysis (DCVA) change detection. Code for 2019 [paper](https://ieeexplore.ieee.org/document/8608001): Unsupervised Deep Change Vector Analysis for Multiple-Change Detection in VHR Images
* [hyperdimensionalCD](https://github.com/sudipansaha/hyperdimensionalCD) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9582825): Change Detection in Hyperdimensional Images Using Untrained Models
* [DSFANet](https://github.com/wwdAlger/DSFANet) -> code for 2018 [paper](https://arxiv.org/abs/1812.00645): Unsupervised Deep Slow Feature Analysis for Change Detection in Multi-Temporal Remote Sensing Images
* [FCD-GAN-pytorch](https://github.com/Cwuwhu/FCD-GAN-pytorch) -> Fully Convolutional Change Detection Framework with Generative Adversarial Network (FCD-GAN) is a framework for change detection in multi-temporal remote sensing images
* [DARNet-CD](https://github.com/jimmyli08/DARNet-CD) -> code for 2022 paper: A Densely Attentive Refinement Network for Change Detection Based on Very-High-Resolution Bitemporal Remote Sensing Images
* [xView2_Vulcan](https://github.com/RitwikGupta/xView2-Vulcan) -> Damage assessment using pre and post orthoimagery. Modified + productionized model based off the first-place model from the xView2 challenge.
* [ESCNet](https://github.com/Bobholamovic/ESCNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9474911): An End-to-End Superpixel-Enhanced Change Detection Network for Very-High-Resolution Remote Sensing Images
* [ForestCoverChange](https://github.com/annusgit/ForestCoverChange) -> Detecting and Predicting Forest Cover Change in Pakistani Areas Using Remote Sensing Imagery
* [deforestation-detection](https://github.com/vldkhramtsov/deforestation-detection) -> code for 2020 paper: DEEP LEARNING FOR HIGH-FREQUENCY CHANGE DETECTION IN UKRAINIAN FOREST ECOSYSTEM WITH SENTINEL-2
* [forest_change_detection](https://github.com/QuantuMobileSoftware/forest_change_detection) -> forest change segmentation with time-dependent models, including Siamese, UNet-LSTM, UNet-diff, UNet3D models. Code for 2021 [paper](https://ieeexplore.ieee.org/document/9241044): Deep Learning for Regular Change Detection in Ukrainian Forest Ecosystem With Sentinel-2
* [SentinelClearcutDetection](https://github.com/vldkhramtsov/SentinelClearcutDetection) -> Scripts for deforestation detection on the Sentinel-2 Level-A images
* [clearcut_detection](https://github.com/QuantuMobileSoftware/clearcut_detection) -> research & web-service for clearcut detection
* [CDRL](https://github.com/cjf8899/CDRL) -> code for 2022 [paper](https://arxiv.org/abs/2204.01200): Unsupervised Change Detection Based on Image Reconstruction Loss
* [ddpm-cd](https://github.com/wgcban/ddpm-cd) -> code for 2022 [paper](https://arxiv.org/abs/2206.11892): Remote Sensing Change Detection (Segmentation) using Denoising Diffusion Probabilistic Models
* [Remote-sensing-time-series-change-detection](https://github.com/liulianni1688/Remote-sensing-time-series-change-detection) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0034425722001079): Graph-based block-level urban change detection using Sentinel-2 time series
* [austin-ml-change-detection-demo](https://github.com/makepath/austin-ml-change-detection-demo) -> A change detection demo for the Austin area using a pre-trained PyTorch model scaled with Dask on Planet imagery
* [dfc2021-msd-baseline](https://github.com/calebrob6/dfc2021-msd-baseline) -> A baseline for the "Multitemporal Semantic Change Detection" track of the 2021 IEEE GRSS Data Fusion Competition
* [CorrFusionNet](https://github.com/rulixiang/CorrFusionNet) -> code for 2020 [paper](https://arxiv.org/abs/2006.02176): Multi-Temporal Scene Classification and Scene Change Detection with Correlation based Fusion
* [ChangeDetectionPCAKmeans](https://github.com/rulixiang/ChangeDetectionPCAKmeans) -> MATLAB implementation for Unsupervised Change Detection in Satellite Images Using Principal Component Analysis and k-Means Clustering.
* [IRCNN](https://github.com/thebinyang/IRCNN) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9721897): IRCNN: An Irregular-Time-Distanced Recurrent Convolutional Neural Network for Change Detection in Satellite Time Series
* [UTRNet](https://github.com/thebinyang/UTRNet) -> An Unsupervised Time-Distance-Guided Convolutional Recurrent Network for Change Detection in Irregularly Collected Images
* [open-cd](https://github.com/likyoo/open-cd) -> an open source change detection toolbox based on a series of open source general vision task tools
* [Tiny_model_4_CD](https://github.com/AndreaCodegoni/Tiny_model_4_CD) -> code for 2022 [paper](https://arxiv.org/abs/2207.13159): TINYCD: A (Not So) Deep Learning Model For Change Detection. Uses LEVIR-CD & WHU-CD datasets
* [FHD](https://github.com/ZSVOS/FHD) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9837915): Feature Hierarchical Differentiation for Remote Sensing Image Change Detection
* [Change detection with Raster Vision](https://www.azavea.com/blog/2022/04/18/change-detection-with-raster-vision/) -> blog post with Colab notebook
* [building-expansion](https://github.com/reglab/building-expansion) -> code for 2021 [paper](https://arxiv.org/abs/2105.14159): Enhancing Environmental Enforcement with Near Real-Time Monitoring: Likelihood-Based Detection of Structural Expansion of Intensive Livestock Farms

## Time series
More general than change detection, time series observations can be used for applications including improving the accuracy of crop classification, or predicting future patterns & events. Crop yield is very typically application and has its own section below
* [CropDetectionDL](https://github.com/karimmamer/CropDetectionDL) -> using GRU-net, First place solution for Crop Detection from Satellite Imagery competition organized by CV4A workshop at ICLR 2020
* [LANDSAT Time Series Analysis for Multi-temporal Land Cover Classification using Random Forest](https://github.com/agr-ayush/Landsat-Time-Series-Analysis-for-Multi-Temporal-Land-Cover-Classification)
* [temporalCNN](https://github.com/charlotte-pel/temporalCNN) -> Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series
* [pytorch-psetae](https://github.com/VSainteuf/pytorch-psetae) -> PyTorch implementation of the model presented in Satellite Image Time Series Classification with Pixel-Set Encoders and Temporal Self-Attention
* [satflow](https://github.com/openclimatefix/satflow) -> optical flow models for predicting future satellite images from current and past ones
* [esa-superresolution-forecasting](https://github.com/PiSchool/esa-superresolution-forecasting) -> Forecasting air pollution using ESA Sentinel-5p data, and an encoder-decoder convolutional LSTM neural network architecture, implemented in Pytorch
* [Radiant-Earth-Spot-the-Crop-Challenge](https://github.com/DariusTheGeek/Radiant-Earth-Spot-the-Crop-Challenge) -> The main objective of this challenge was to use time-series of Sentinel-2 multi-spectral data to classify crops in the Western Cape of South Africa. The challenge was to build a machine learning model to predict crop type classes for the test dataset
* [lightweight-temporal-attention-pytorch](https://github.com/VSainteuf/lightweight-temporal-attention-pytorch) -> A PyTorch implementation of the Light Temporal Attention Encoder (L-TAE) for satellite image time series. classification
* [Crop-Classification](https://github.com/bhavesh907/Crop-Classification) -> crop classification using multi temporal satellite images
* [dtwSat](https://github.com/vwmaus/dtwSat) -> Time-Weighted Dynamic Time Warping for satellite image time series analysis
* [DeepCropMapping](https://github.com/Lab-IDEAS/DeepCropMapping) -> A multi-temporal deep learning approach with improved spatial generalizability for dynamic corn and soybean mapping, uses LSTM
* [CropMappingInterpretation](https://github.com/Lab-IDEAS/CropMappingInterpretation) -> An interpretation pipeline towards understanding multi-temporal deep learning approaches for crop mapping
* [MTLCC](https://github.com/MarcCoru/MTLCC) -> code for paper: Multitemporal Land Cover Classification Network. A recurrent neural network approach to encode multi-temporal data for land cover classification
* [timematch](https://github.com/jnyborg/timematch) -> code for 2022 paper: A method to perform unsupervised cross-region adaptation of crop classifiers trained with satellite image time series. We also introduce an open-access dataset for cross-region adaptation with SITS from four different regions in Europe
* [PWWB](https://github.com/PannuMuthu/PWWB) -> Code for the 2021 [paper](https://link.springer.com/chapter/10.1007/978-3-030-71704-9_20): Real-Time Spatiotemporal Air Pollution Prediction with Deep Convolutional LSTM through Satellite Image Analysis
* [Classification of Crop Fields through Satellite Image Time Series](https://medium.com/dida-machine-learning/classification-of-crop-fields-through-satellite-image-time-serie-dida-machine-learning-9b64ce2b8c10) -> using a [pytorch-psetae](https://github.com/VSainteuf/pytorch-psetae) & Sentinel-2 data
* [spaceweather](https://github.com/sarttiso/spaceweather) -> predicting geomagnetic storms from satellite measurements of the solar wind and solar corona, uses LSTMs
* [Forest_wildfire_spreading_convLSTM](https://github.com/bessammehenni/Forest_wildfire_spreading_convLSTM) -> Modeling of the spreading of forest wildfire using a neural network with ConvLSTM cells. Prediction 3-days forward
* [ConvTimeLSTM](https://github.com/jdiaz4302/ConvTimeLSTM) -> Extension of ConvLSTM and Time-LSTM for irregularly spaced images, appropriate for Remote Sensing
* [dl-time-series](https://github.com/NexGenMap/dl-time-series) -> Deep Learning algorithms applied to characterization of Remote Sensing time-series
* [tpe](https://github.com/jnyborg/tpe) -> code for 2022 [paper](https://openaccess.thecvf.com/content/CVPR2022W/EarthVision/html/Nyborg_Generalized_Classification_of_Satellite_Image_Time_Series_With_Thermal_Positional_CVPRW_2022_paper.html): Generalized Classification of Satellite Image Time Series With Thermal Positional Encoding
* [wildfire_forecasting](https://github.com/Orion-AI-Lab/wildfire_forecasting) -> code for 2021 [paper](https://arxiv.org/abs/2111.02736): Deep Learning Methods for Daily Wildfire Danger Forecasting. Uses ConvLSTM
* [satellite_image_forecasting](https://github.com/rudolfwilliam/satellite_image_forecasting) -> predict future satellite images from past ones using features such as precipitation and elevation maps. Entry for the [EarthNet2021](https://www.earthnet.tech/) challenge

## Crop yield
* [Crop yield Prediction with Deep Learning](https://github.com/JiaxuanYou/crop_yield_prediction) -> code for the paper Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data
* [Deep-Transfer-Learning-Crop-Yield-Prediction](https://github.com/sustainlab-group/Deep-Transfer-Learning-Crop-Yield-Prediction)
* [Crop-Yield-Prediction-using-ML](https://github.com/VaibhavSaini19/Crop-Yield-Prediction-using-ML) -> A simple Web application developed in order to provide the farmers/users an approximation on how much amount of crop yield will be produced depending upon the given input
* [Building a Crop Yield Prediction App in Senegal Using Satellite Imagery and Jupyter Voila](https://omdena.com/blog/yield-prediction/)
* [Crop Yield Prediction Using Deep Neural Networks and LSTM](https://omdena.com/blog/deep-learning-yield-prediction/)
* [Deep transfer learning techniques for crop yield prediction, published in COMPASS 2018](https://github.com/AnnaXWang/deep-transfer-learning-crop-prediction)
* [Understanding crop yield predictions from CNNs](https://github.com/brad-ross/crop-yield-prediction-project)
* [Advanced Deep Learning Techniques for Predicting Maize Crop Yield using Sentinel-2 Satellite Imagery](https://zionayomide.medium.com/advanced-deep-learning-techniques-for-predicting-maize-crop-yield-using-sentinel-2-satellite-1b63ac8b0789)
* [pycrop-yield-prediction](https://github.com/gabrieltseng/pycrop-yield-prediction) -> A PyTorch Implementation of Jiaxuan You's Deep Gaussian Process for Crop Yield Prediction
* [PredictYield](https://github.com/dberm312/PredictYield) -> using data scraped from Google Earth Engine, this predicts the yield of Corn, Soybean, and Wheat in the USA with Keras
* [Crop-Yield-Prediction-and-Estimation-using-Time-series-remote-sensing-data](https://github.com/mahimatendulkar/Crop-Yield-Prediction-and-Estimation-using-Time-series-remote-sensing-data.) -> student research
* [Yield-Prediction-Using-Sentinel-Data](https://github.com/meet-sapu/Crop-Yield-Prediction-Using-Satellite-Imagery)
* [SPACY](https://github.com/rlee360/PLaTYPI) -> Satellite Prediction of Aggregate Corn Yield
* [cropyieldArticle](https://github.com/myliheik/cropyieldArticle) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/17/4193): Scalable Crop Yield Prediction with Sentinel-2 Time Series and Temporal Convolutional Network

## Wealth and economic activity
The goal is to predict economic activity from satellite imagery rather than conducting labour intensive ground surveys
* [Using publicly available satellite imagery and deep learning to understand economic well-being in Africa, Nature Comms 22 May 2020](https://www.nature.com/articles/s41467-020-16185-w) -> Used CNN on Ladsat imagery (night & day) to predict asset wealth of African villages
* [Combining Satellite Imagery and machine learning to predict poverty](https://towardsdatascience.com/combining-satellite-imagery-and-machine-learning-to-predict-poverty-884e0e200969) -> review article
* [Measuring Human and Economic Activity from Satellite Imagery to Support City-Scale Decision-Making during COVID-19 Pandemic](https://arxiv.org/abs/2004.07438) -> arxiv article
* [Predicting Food Security Outcomes Using CNNs for Satellite Tasking](https://arxiv.org/pdf/1902.05433.pdf) -> arxiv article
* [Measuring the Impacts of Poverty Alleviation Programs with Satellite Imagery and Deep Learning](https://github.com/luna983/beyond-nightlight) -> code and paper
* [Building a Spatial Model to Classify Global Urbanity Levels](https://towardsdatascience.com/building-a-spatial-model-to-classify-global-urbanity-levels-e2fb9da7252) -> estimage global urbanity levels from population data, nightime lights and road networks
* [deeppop](https://deeppop.github.io/) -> Deep Learning Approach for Population Estimation from Satellite Imagery, also [on Github](https://github.com/deeppop)
* [Estimating telecoms demand in areas of poor data availability](https://github.com/edwardoughton/taddle) -> with papers on [arxiv](https://arxiv.org/abs/2006.07311) and [Science Direct](https://www.sciencedirect.com/science/article/abs/pii/S0736585321000617)
* [satimage](https://github.com/mani-shailesh/satimage) -> Code and models for the manuscript "Predicting Poverty and Developmental Statistics from Satellite Images using Multi-task Deep Learning". Predict the main material of a roof, source of lighting and source of drinking water for properties, from satellite imagery
* [africa_poverty](https://github.com/sustainlab-group/africa_poverty) -> Using publicly available satellite imagery and deep learning to understand economic well-being in Africa
* [Predicting-Poverty](https://github.com/jmather625/predicting-poverty-replication) -> Combining satellite imagery and machine learning to predict poverty, in PyTorch
* [income-prediction](https://github.com/tnarayanan/income-prediction) -> Predicting average yearly income based on satellite imagery using CNNs, uses pytorch
* [urban_score](https://github.com/Sungwon-Han/urban_score) -> Pytorch Implementation of paper: Learning to score economic development from satellite imagery
* [READ](https://github.com/Sungwon-Han/READ) -> Pytorch Implementation of paper: Lightweight and robust representation of economic scales from satellite imagery
* [Slum-classification](https://github.com/Jesse-DE/Slum-classification) -> Binary classification on a very high-resolution satellite image in case of mapping informal settlements using unet
* [Predicting_Poverty](https://github.com/cyuancheng/Predicting_Poverty) -> uses daytime & luminosity of nighttime satellite images
* [Cancer-Prevalence-Satellite-Images](https://github.com/theJamesChen/Cancer-Prevalence-Satellite-Images) -> Predict Health Outcomes from Features of Satellite Images
* [Mapping Poverty in Bangladesh with Satellite Images and Deep Learning](https://github.com/huydang90/Mapping-Poverty-With-Satellite-Images) -> combines health data with OpenStreetMaps Data & night and daytime satellite imagery
* [Population Estimation from Satellite Imagery](https://github.com/ManuelSerranoR/Population-Estimation-from-Satellite-Imagery-using-Deep-Learning)
* [Deep_Learning_Satellite_Imd](https://github.com/surendran-berkeley/Deep_Learning_Satellite_Imd) -> code for "Project Bhoomi" - Using Deep Learning on Satellite Imagery to predict population and economic indicators
* [satellite_led_liverpool](https://github.com/darribas/satellite_led_liverpool) -> code for 2017 paper: Remote Sensing-Based Measurement of Living Environment Deprivation - Improving Classical Approaches with Machine Learning
* [uganda-poverty-project](https://github.com/vinceranga/uganda-poverty-project) -> use through Object Detection on high-resolution satellite imagery to identify indicators of poverty and economic inequality within Uganda
* [Predicting_Energy_Consumption_With_Convolutional_Neural_Networks](https://github.com/healdz/Predicting_Energy_Consumption_With_Convolutional_Neural_Networks)

## Disaster response
Also checkout the sections on change detection and water/fire/building segmentation
* [DisaVu](https://github.com/SrzStephen/DisaVu) -> combines building & damage detection and provides an app for viewing predictions
* [Soteria](https://github.com/Soteria-ai/Soteria) -> uses machine learning with satellite imagery to map natural disaster impacts for faster emergency response
* [DisasterHack](https://github.com/MarjorieRWillner/DisasterHack) -> Wildfire Mitigation: Computer Vision Identification of Hazard Fuels Using Landsat
* [forestcasting](https://github.com/ivanzvonkov/forestcasting) -> Forest fire prediction powered by analytics
* [Machine Learning-based Damage Assessment for Disaster Relief on Google AI blog](https://ai.googleblog.com/2020/06/machine-learning-based-damage.html) -> uses object detection to locate buildings, then a classifier to determine if a building is damaged. Challenge of generalising due to small dataset
* [RaVAEn](https://github.com/spaceml-org/RaVAEn) -> RaVAEn is a lightweight, unsupervised approach for change detection in satellite data based on Variational Auto-Encoders (VAEs) with the specific purpose of on-board deployment
* [hurricane_damage](https://github.com/allankapoor/hurricane_damage) -> Post-hurricane structure damage assessment based on aerial imagery with CNN
* [rescue](https://github.com/dbdmg/rescue) -> code of the paper: Attention to fires: multi-channel deep-learning models forwildfire severity prediction
* [Disaster-Classification](https://github.com/bostankhan6/Disaster-Classification) -> A disaster classification model to predict the type of disaster given an input image, trained on [this dataset](https://github.com/engrhamzaaliimran/cvassignmentdataset)
* [Coarse-to-fine weakly supervised learning method for green plastic cover segmentation](https://github.com/lauraset/Coarse-to-fine-weakly-supervised-GPC-segmentation) -> with [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271622001095)
* [Detection of destruction in satellite imagery](https://github.com/usmanali414/Destruction-Detection-in-Satellite-Imagery)
* [BDD-Net](https://github.com/jinyuan30/Recognize-damaged-buildings) -> code for 2020 paper: A General Protocol for Mapping Buildings Damaged by a Wide Range of Disasters Based on Satellite Imagery
* [Automatic_Disaster_Detection](https://github.com/yoji-kuretake-like/Automatic_Disaster_Detection) -> detect the affected area by natural disasters by using the way of semantic segmentation and change detection method
* [Flooding Damage Detection from Post-Hurricane Satellite Imagery Based on Convolutional Neural Networks](https://github.com/weining20000/Flooding-Damage-Detection-from-Post-Hurricane-Satellite-Imagery-Based-on-CNN)
* [IBM-Disaster-Response-Hack](https://github.com/NicoDeshler/IBM-Disaster-Response-Hack) -> identifying optimal terrestrial routes through calamity-stricken areas. Satellite image data informs road condition assessment and obstruction detection
* [Automatic Damage Annotation on Post-Hurricane Satellite Imagery](https://dds-lab.github.io/disaster-damage-detection/) -> detect damaged buildings using tensorflow object detection API. With repos [here](https://github.com/DDS-Lab/disaster-image-processing) and [here](https://github.com/annieyan/PreprocessSatelliteImagery-ObjectDetection)
* [Hurricane-Damage-Detection](https://github.com/Ryan-Awad/Hurricane-Damage-Detection) -> Waterloo's Hack the North 2020++ submission. A convolutional neural network model used to detect hurricane damage in RGB satellite images
* [wildfire_forecasting](https://github.com/Orion-AI-Lab/wildfire_forecasting) -> code for 2021 [paper](https://arxiv.org/abs/2111.02736): Deep Learning Methods for Daily Wildfire Danger Forecasting. Uses ConvLSTM
* [Satellite Image Analysis with fast.ai for Disaster Recovery](https://appsilon.com/satellite-image-analysis-with-fast-ai-for-disaster-recovery/)
* [shackleton](https://github.com/avanetten/shackleton) -> leverages remote sensing imagery and machine learning techniques to provide insights into various transportation and evacuation scenarios in an interactive dashboard that conducts real-time computation

## Weather phenomena
* [EddyData](https://github.com/zmokokokok/EddyData) -> code for paper: A Deep Framework for Eddy Detection and Tracking from Satellite Sea Surface Height Data
* [python-windspeed](https://github.com/h-fuzzy-logic/python-windspeed) -> Predicting windspeed of hurricanes from satellite images, uses CNN regression in keras
* [hurricane-wind-speed-cnn](https://github.com/23ccozad/hurricane-wind-speed-cnn) -> Predicting windspeed of hurricanes from satellite images, uses CNN regression in keras

## Super-resolution
Super-resolution attempts to enhance the resolution of an imaging system, and can be applied as a pre-processing step to improve the detection of small objects or boundaries. Its use is controversial since it can introduce artefacts at the same rate as real features. These techniques are generally grouped into single image super resolution (SISR) **or** a multi image super resolution (MISR)
* [The value of super resolution — real world use case](https://medium.com/sentinel-hub/the-value-of-super-resolution-real-world-use-case-2ba811f4cd7f) -> Medium article on parcel boundary detection with super-resolved satellite imagery
* [Super-Resolution on Satellite Imagery using Deep Learning](https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-1-ec5c5cd3cd2) -> Nov 2016 blog post by CosmiQ Works with a nice introduction to the topic. Proposes and demonstrates a new architecture with perturbation layers with practical guidance on the methodology and [code](https://github.com/CosmiQ/super-resolution). [Three part series](https://medium.com/the-downlinq/super-resolution-on-satellite-imagery-using-deep-learning-part-3-2e2f61eee1d3)
* [Introduction to spatial resolution](https://medium.com/sentinel-hub/the-most-misunderstood-words-in-earth-observation-d0106adbe4b0)
* [Awesome-Super-Resolution](https://github.com/ptkin/Awesome-Super-Resolution) -> another 'awesome' repo, getting a little out of date now
* [Super-Resolution (python) Utilities for managing large satellite images](https://github.com/jshermeyer/SR_Utils)
* [pytorch-enhance](https://github.com/isaaccorley/pytorch-enhance) -> Library of Image Super-Resolution Models, Datasets, and Metrics for Benchmarking or Pretrained Use. Also [checkout this implementation in Jax](https://github.com/isaaccorley/jax-enhance)
* [Super Resolution in OpenCV](https://learnopencv.com/super-resolution-in-opencv/)
* [AI-based Super resolution and change detection to enforce Sentinel-2 systematic usage](https://medium.com/@sistema_gmbh/ai-based-super-resolution-and-change-detection-to-enforce-sentinel-2-systematic-usage-65aa37d0365) -> Worldview-2 images (2m) were used to create a reference dataset and increase the spatial resolution of the Copernicus sensor from 10m to 5m
* [SRCDNet](https://github.com/liumency/SRCDNet) -> The pytorch implementation for "Super-resolution-based Change Detection Network with Stacked Attention Module for Images with Different Resolutions ". SRCDNet is designed to learn and predict change maps from bi-temporal images with different resolutions
* [Model-Guided Deep Hyperspectral Image Super-resolution](https://github.com/chengerr/Model-Guided-Deep-Hyperspectral-Image-Super-resolution) -> code accompanying the paper [Model-Guided Deep Hyperspectral Image Super-Resolution](https://ieeexplore.ieee.org/document/9429905)
* [Super-resolving beyond satellite hardware](https://github.com/smpetrie/superres) -> [paper](https://arxiv.org/abs/2103.06270) assessing SR performance in reconstructing realistically degraded satellite images
* [satellite-pixel-synthesis-pytorch](https://github.com/KellyYutongHe/satellite-pixel-synthesis-pytorch) -> PyTorch implementation of NeurIPS 2021 paper: Spatial-Temporal Super-Resolution of Satellite Imagery via Conditional Pixel Synthesis
* [SRE-HAN](https://github.com/bostankhan6/SRE-HAN) -> Squeeze-and-Residual-Excitation Holistic Attention Network improves super-resolution (SR) on remote-sensing imagery compared to other state-of-the-art attention-based SR models
* [satsr](https://github.com/deephdc/satsr) -> A project to perform super-resolution on multispectral images from any satellite, including Sentinel 2, Landsat 8, VIIRS &MODIS
* [OLI2MSI](https://github.com/wjwjww/OLI2MSI) -> dataset for remote sensing imagery super-resolution composed of Landsat8-OLI and Sentinel2-MSI images
* [MMSR](https://github.com/palmdong/MMSR) -> Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution
* [HSRnet](https://github.com/liangjiandeng/HSRnet) -> code for the 2021 [paper](https://arxiv.org/abs/2005.14400): Hyperspectral Image Super-resolution via Deep Spatio-spectral Attention Convolutional Neural Networks
* [RRSGAN](https://github.com/dongrunmin/RRSGAN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9328132): RRSGAN: Reference-Based Super-Resolution for Remote Sensing Image

### Single image super-resolution (SISR)
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
* [RFSR_TGRS](https://github.com/wxywhu/RFSR_TGRS) -> code for the paper Hyperspectral Image Super-Resolution via Recurrent Feedback Embedding and Spatial-Spectral Consistency Regularization
* [SEN2VENµS](https://zenodo.org/record/6514159#.YoRxM5PMK3I) -> a dataset for the training of Sentinel-2 super-resolution algorithms. With [paper](https://www.mdpi.com/2306-5729/7/7/96)
* [TransENet](https://github.com/Shaosifan/TransENet) -> code for 2021 paper: Transformer-based Multi-Stage Enhancement for Remote Sensing Image Super-Resolution
* [SG-FBGAN](https://github.com/hanlinwu/SG-FBGAN) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9301233): Remote Sensing Image Super-Resolution via Saliency-Guided Feedback GANs
* [finetune_ESRGAN](https://github.com/johnjaniczek/finetune_ESRGAN) -> finetune the ESRGAN super resolution generator for remote sensing images and video
* [MIP](https://github.com/jiaming-wang/MIP) -> code for 2021 [paper](https://arxiv.org/abs/2105.03579): Unsupervised Remote Sensing Super-Resolution via Migration Image Prior
* [Optical-RemoteSensing-Image-Resolution](https://github.com/wenjiaXu/Optical-RemoteSensing-Image-Resolution) -> code for 2018 [paper](https://www.mdpi.com/2072-4292/10/12/1893): Deep Memory Connected Neural Network for Optical Remote Sensing Image Restoration. Two applications: Gaussian image denoising and single image super-resolution
* [HSENet](https://github.com/Shaosifan/HSENet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9400474): Hybrid-Scale Self-Similarity Exploitation for Remote Sensing Image Super-Resolution
* [SR_RemoteSensing](https://github.com/Jing25/SR_RemoteSensing) -> Super-Resolution deep learning models for remote sensing data based on [BasicSR](https://github.com/XPixelGroup/BasicSR)
* [RSI-Net](https://github.com/EricBrock/RSI-Net) -> code for 2022 paper: A Deep Multi-task Convolutional Neural Network for Remote Sensing Image Super-resolution and Colorization
* [EDSR-Super-Resolution](https://github.com/RakeshRaj97/EDSR-Super-Resolution) -> EDSR model using PyTorch applied to satellite imagery
* [CycleCNN](https://github.com/haopzhang/CycleCNN) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9151194): Nonpairwise-Trained Cycle Convolutional Neural Network for Single Remote Sensing Image Super-Resolution
* [SISR with with Real-World Degradation Modeling](https://github.com/zhangjizhou-bit/Single-image-Super-Resolution-of-Remote-Sensing-Images-with-Real-World-Degradation-Modeling) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/12/2895): Single-Image Super Resolution of Remote Sensing Images with Real-World Degradation Modeling
* [pixel-smasher](https://github.com/ekcomputer/pixel-smasher) -> code for 2020 [paper](https://www.tandfonline.com/doi/abs/10.1080/07038992.2021.1924646?journalCode=ujrs20): Super-Resolution Surface Water Mapping on the Canadian Shield Using Planet CubeSat Images and a Generative Adversarial Network

### Multi image super-resolution (MISR)
Note that nearly all the MISR publications resulted from the [PROBA-V Super Resolution competition](https://kelvins.esa.int/proba-v-super-resolution/)
* [deepsum](https://github.com/diegovalsesia/deepsum) -> Deep neural network for Super-resolution of Unregistered Multitemporal images (ESA PROBA-V challenge)
* [3DWDSRNet](https://github.com/frandorr/3DWDSRNet) -> code to reproduce Satellite Image Multi-Frame Super Resolution (MISR) Using 3D Wide-Activation Neural Networks
* [RAMS](https://github.com/EscVM/RAMS) -> Official TensorFlow code for paper Multi-Image Super Resolution of Remotely Sensed Images Using Residual Attention Deep Neural Networks
* [TR-MISR](https://github.com/Suanmd/TR-MISR) ->  Transformer-based MISR framework for the the PROBA-V super-resolution challenge. With [paper](https://ieeexplore.ieee.org/abstract/document/9684717)
* [HighRes-net](https://github.com/ElementAI/HighRes-net) -> Pytorch implementation of HighRes-net, a neural network for multi-frame super-resolution, trained and tested on the European Space Agency’s Kelvin competition
* [ProbaVref](https://github.com/centreborelli/ProbaVref) -> Repurposing the Proba-V challenge for reference-aware super resolution
* [The missing ingredient in deep multi-temporal satellite image super-resolution](https://towardsdatascience.com/the-missing-ingredient-in-deep-multi-temporal-satellite-image-super-resolution-78cac0f063d9) -> Permutation invariance harnesses the power of ensembles in a single model, with repo [piunet](https://github.com/diegovalsesia/piunet)
* [MSTT-STVSR](https://github.com/XY-boy/MSTT-STVSR) -> Space-time Super-resolution for Satellite Video: A Joint Framework Based on Multi-Scale Spatial-Temporal Transformer, JAG, 2022
* [Self-Supervised Super-Resolution for Multi-Exposure Push-Frame Satellites](https://centreborelli.github.io/HDR-DSP-SR/)
* [DDRN](https://github.com/kuijiang94/DDRN) -> Deep Distillation Recursive Network for Video Satellite Imagery Super-Resolution
* [worldstrat](https://github.com/worldstrat/worldstrat) -> SISR and MISR implementations of SRCNN

## Pansharpening
Image fusion of low res multispectral with high res pan band.
* Several algorithms described [in the ArcGIS docs](http://desktop.arcgis.com/en/arcmap/10.3/manage-data/raster-and-images/fundamentals-of-panchromatic-sharpening.htm), with the simplest being taking the mean of the pan and RGB pixel value.
* For into to classical methods [see this notebook](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/panchromatic_sharpening.ipynb) and [this kaggle kernel](https://www.kaggle.com/resolut/panchromatic-sharpening)
* [rio-pansharpen](https://github.com/mapbox/rio-pansharpen) -> pansharpening Landsat scenes
* [Simple-Pansharpening-Algorithms](https://github.com/ThomasWangWeiHong/Simple-Pansharpening-Algorithms)
* [Working-For-Pansharpening](https://github.com/yuanmaoxun/Working-For-Pansharpening) -> long list of pansharpening methods and update of [Awesome-Pansharpening](https://github.com/Lihui-Chen/Awesome-Pansharpening)
* [PSGAN](https://github.com/liuqingjie/PSGAN) -> A Generative Adversarial Network for Remote Sensing Image Pan-sharpening, [arxiv paper](https://arxiv.org/abs/1805.03371)
* [Pansharpening-by-Convolutional-Neural-Network](https://github.com/ThomasWangWeiHong/Pansharpening-by-Convolutional-Neural-Network)
* [PBR_filter](https://github.com/dbuscombe-usgs/PBR_filter) -> {P}ansharpening by {B}ackground {R}emoval algorithm for sharpening RGB images
* [py_pansharpening](https://github.com/codegaj/py_pansharpening) -> multiple algorithms implemented in python
* [Deep-Learning-PanSharpening](https://github.com/xyc19970716/Deep-Learning-PanSharpening) -> deep-learning based pan-sharpening code package, we reimplemented include PNN, MSDCNN, PanNet, TFNet, SRPPNN, and our purposed network DIPNet
* [HyperTransformer](https://github.com/wgcban/HyperTransformer) -> A Textural and Spectral Feature Fusion Transformer for Pansharpening
* [DIP-HyperKite](https://github.com/wgcban/DIP-HyperKite) -> Hyperspectral Pansharpening Based on Improved Deep Image Prior and Residual Reconstruction
* [D2TNet](https://github.com/Meiqi-Gong/D2TNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9761261): A ConvLSTM Network with Dual-direction Transfer for Pan-sharpening
* [PanColorGAN-VHR-Satellite-Images](https://github.com/esertel/PanColorGAN-VHR-Satellite-Images) -> code for 2020 [paper](https://arxiv.org/abs/2006.16644): Rethinking CNN-Based Pansharpening: Guided Colorization of Panchromatic Images via GANs
* [MTL_PAN_SEG](https://github.com/andrewekhalel/MTL_PAN_SEG) -> code for 2019 paper: Multi-task deep learning for satellite image pansharpening and segmentation
* [Z-PNN](https://github.com/matciotola/Z-PNN) -> code for 2022 paper: Pansharpening by convolutional neural networks in the full resolution framework
* [GTP-PNet](https://github.com/HaoZhang1018/GTP-PNet) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S092427162030352X): GTP-PNet: A residual learning network based on gradient transformation prior for pansharpening
* [UDL](https://github.com/XiaoXiao-Woo/UDL) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9710135): Dynamic Cross Feature Fusion for Remote Sensing Pansharpening
* [PSData](https://github.com/yisun98/PSData) -> A Large-Scale General Pan-sharpening DataSet, which contains PSData3 (QB, GF-2, WV-3) and PSData4 (QB, GF-1, GF-2, WV-2).
* [AFPN](https://github.com/yisun98/AFPN) -> Adaptive Detail Injection-Based Feature Pyramid Network For Pan-sharpening
* [pan-sharpening](https://github.com/yisun98/pan-sharpening) -> multiple methods demonstrated for multispectral and panchromatic images
* [PSGan-Family](https://github.com/zhysora/PSGan-Family) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9306912): PSGAN: A Generative Adversarial Network for Remote Sensing Image Pan-Sharpening
* [PanNet-Landsat](https://github.com/oyam/PanNet-Landsat) -> code for 2017 paper: A Deep Network Architecture for Pan-Sharpening
* [DLPan-Toolbox](https://github.com/liangjiandeng/DLPan-Toolbox) -> code for 2022 paper: Machine Learning in Pansharpening: A Benchmark, from Shallow to Deep Networks
* [LPPN](https://github.com/ChengJin-git/LPPN) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S1566253521001809): Laplacian pyramid networks: A new approach for multispectral pansharpening
* [S2_SSC_CNN](https://github.com/hvn2/S2_SSC_CNN) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9323614): Zero-shot Sentinel-2 Sharpening Using A Symmetric Skipped Connection Convolutional Neural Network
* [S2S_UCNN](https://github.com/hvn2/S2S_UCNN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9464640): Sentinel 2 sharpening using a single unsupervised convolutional neural network with MTF-Based degradation model
* [SSE-Net](https://github.com/RSMagneto/SSE-Net) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9810290): Spatial and Spectral Extraction Network With Adaptive Feature Fusion for Pansharpening
* [UCGAN](https://github.com/zhysora/UCGAN) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9755137): Unsupervised Cycle-consistent Generative Adversarial Networks for Pan-sharpening
* [GCPNet](https://github.com/Keyu-Yan/GCPNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9758796): When Pansharpening Meets Graph Convolution Network and Knowledge Distillation
* [PanFormer](https://github.com/zhysora/PanFormer) -> code for 2022 [paper](https://arxiv.org/abs/2203.02916): PanFormer: a Transformer Based Model for Pan-sharpening
* [Pansharpening](https://github.com/nithin-gr/Pansharpening) -> code for 2021 [paper](https://www.researchgate.net/publication/356974466_Pansformers_Transformer-Based_Self-Attention_Network_for_Pansharpening): Pansformers: Transformer-Based Self-Attention Network for Pansharpening

## Image-to-image translation
Translate images e.g. from SAR to RGB.
* [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/) -> how to develop a Pix2Pix model for translating satellite photographs to Google map images. A good intro to GANS
* [A growing problem of ‘deepfake geography’: How AI falsifies satellite images](https://www.washington.edu/news/2021/04/21/a-growing-problem-of-deepfake-geography-how-ai-falsifies-satellite-images/)
* [Kaggle Pix2Pix Maps](https://www.kaggle.com/alincijov/pix2pix-maps) -> dataset for pix2pix to take a google map satellite photo and build a street map
* [guided-deep-decoder](https://github.com/tuezato/guided-deep-decoder) -> With guided deep decoder, you can solve different image pair fusion problems, allowing super-resolution, pansharpening or denoising
* [hackathon-ci-2020](https://github.com/paulaharder/hackathon-ci-2020) -> generate nighttime imagery from infrared observations
* [satellite-to-satellite-translation](https://github.com/anonymous-ai-for-earth/satellite-to-satellite-translation) -> VAE-GAN architecture for unsupervised image-to-image translation with shared spectral reconstruction loss. Model is trained on GOES-16/17 and Himawari-8 L1B data
* [Pytorch implementation of UNet for converting aerial satellite images into google maps kinda images](https://github.com/greed2411/unet_pytorch)
* [Seamless-Satellite-image-Synthesis](https://github.com/Misaliet/Seamless-Satellite-image-Synthesis) -> generate abitrarily large RGB images from a map
* [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/) -> article on machinelearningmastery.com
* [Satellite-Imagery-to-Map-Translation-using-Pix2Pix-GAN-framework](https://github.com/anh-nn01/Satellite-Imagery-to-Map-Translation-using-Pix2Pix-GAN-framework)
* [RSIT_SRM_ISD](https://github.com/summitgao/RSIT_SRM_ISD) -> PyTorch implementation of Remote sensing image translation via style-based recalibration module and improved style discriminator
* [pix2pix_google_maps](https://github.com/manishemirani/pix2pix_google_maps) -> Converts satellite images to map images using pix2pix models
* [sar2color-igarss2018-chainer](https://github.com/enomotokenji/sar2color-igarss2018-chainer) -> code for 2018 paper: Image Translation Between Sar and Optical Imagery with Generative Adversarial Nets
* [HSI2RGB](https://github.com/JakobSig/HSI2RGB) -> Create realistic looking RGB images using remote sensing hyperspectral images
* [sat_to_map](https://github.com/shagunuppal/sat_to_map) -> Learning mappings to generate city maps images from corresponding satellite images
* [pix2pix-GANs](https://github.com/shashi7679/pix2pix-GANs) -> Generate Map using Satellite Image & PyTorch

## GANS
GANS are famously used for generating synthetic data, see the section [Synthetic data](https://github.com/robmarkcole/satellite-image-deep-learning#synthetic-data)
* [Anomaly Detection on Mars using a GAN](https://omdena.com/projects/anomaly-detection-mars/)
* [Using Generative Adversarial Networks to Address Scarcity of Geospatial Training Data](https://medium.com/radiant-earth-insights/using-generative-adversarial-networks-to-address-scarcity-of-geospatial-training-data-e61cacec986e) -> GAN perform better than CNN in segmenting land cover classes outside of the training dataset (article, no code)
* [Building-A-Nets](https://github.com/lixiang-ucas/Building-A-Nets) -> robust building extraction from high-resolution remote sensing images with adversarial networks
* [GANmapper](https://github.com/ualsg/GANmapper) -> a building footprint generator using Generative Adversarial Networks
* [CSA-CDGAN](https://github.com/wangle53/CSA-CDGAN) -> Channel Self-Attention Based Generative Adversarial Network for Change Detection of Remote Sensing Images
* [DSGAN](https://github.com/lzhengchun/DSGAN) -> a conditinal GAN for dynamic precipitation downscaling
* [MarsGAN](https://github.com/kheyer/MarsGAN) -> GAN trained on satellite photos of Mars
* [HC_ADGAN](https://github.com/summitgao/HC_ADGAN) -> codes for the paper Adaptive Dropblock Enhanced GenerativeAdversarial Networks for Hyperspectral Image Classification
* [SCALAE](https://github.com/LendelTheGreat/SCALAE) -> code for our [paper](https://arxiv.org/abs/2101.05069) Formatting the Landscape: Spatial conditional GAN for varying population in satellite imagery. Method to generate satellite imagery from custom 2D population maps
* [Satellite-Image-Forgery-Detection-and-Localization](https://github.com/tailongnguyen/Satellite-Image-Forgery-Detection-and-Localization)
* [STGAN](https://github.com/ermongroup/STGAN) -> PyTorch Implementation of STGAN for Cloud Removal in Satellite Images, with [paper](https://arxiv.org/abs/1912.06838)
* [ds-gan-spatiotemporal-evaluation](https://github.com/Cervest/ds-gan-spatiotemporal-evaluation) -> evaluating use of deep generative models in remote sensing applications
* [pub-ffi-gan](https://github.com/awweide/pub-ffi-gan) -> code for 2018 paper: Applying generative adversarial networks for anomaly detection in hyperspectral remote sensing imagery
* [GAN-based method to generate high-resolution remote sensing for data augmentation and image classification](https://github.com/weihancug/GAN-based-HRRS-Sample-Generation-for-Image-Classification)
* [Remote-Sensing-Image-Generation](https://github.com/aashishrai3799/Remote-Sensing-Image-Generation) -> Generate RS Images using Generative Adversarial Networks (GAN)
* [RoadDA](https://github.com/LANMNG/RoadDA) -> code for 2021 [paper](https://arxiv.org/abs/2108.12611): Stagewise Unsupervised Domain Adaptation with Adversarial Self-Training for Road Segmentation of Remote Sensing Images
* [PSGan-Family](https://github.com/zhysora/PSGan-Family) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9306912): PSGAN: A Generative Adversarial Network for Remote Sensing Image Pan-Sharpening
* [Satellite Image Augmetation with GANs](https://github.com/Oarowolo11/11785-Project) -> code for 2022 [paper](https://arxiv.org/abs/2207.14580): Image Augmentation for Satellite Images

## Transformers
* [Transformer-in-Remote-Sensing](https://github.com/VIROBO-15/Transformer-in-Remote-Sensing) -> code for 2022 [paper](https://arxiv.org/abs/2209.01206): Transformers in Remote Sensing: A Survey
* [Transformers in remote sensing](https://robmarkcole.com/markdown/2022/08/15/transformers.html) -> blog post by robmarkcole

## Adversarial ML
Efforts to detect falsified images & deepfakes. Also checkout [Synthetic data](https://github.com/robmarkcole/satellite-image-deep-learning#synthetic-data)
* [UAE-RS](https://github.com/YonghaoXu/UAE-RS) -> dataset that provides black-box adversarial samples in the remote sensing field
* [PSGAN](https://github.com/xuxiangsun/PSGAN) -> code for paper: Perturbation Seeking Generative Adversarial Networks: A Defense Framework for Remote Sensing Image Scene Classification
* [SACNet](https://github.com/YonghaoXu/SACNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9573256): Self-Attention Context Network: Addressing the Threat of Adversarial Attacks for Hyperspectral Image Classification

## Autoencoders, dimensionality reduction, image embeddings & similarity search
* [Autoencoders & their Application in Remote Sensing](https://towardsdatascience.com/autoencoders-their-application-in-remote-sensing-95f6e2bc88f) -> intro article and example use case applied to SAR data for land classification
* [LEt-SNE](https://github.com/meghshukla/LEt-SNE) -> Dimensionality Reduction and visualization technique that compensates for the curse of dimensionality
* [AutoEncoders for Land Cover Classification of Hyperspectral Images](https://towardsdatascience.com/autoencoders-for-land-cover-classification-of-hyperspectral-images-part-1-c3c847ebc69b) -> An autoencoder nerual net is used to reduce 103 band data to 60 features (dimensionality reduction), keras. Also read [part 2](https://syamkakarla.medium.com/auto-encoders-for-land-cover-classification-in-hyperspectral-images-part-2-f8978d443d6d) which implements K-NNC, SVM and Gradient Boosting
* [Image-Similarity-Search](https://github.com/spaceml-org/Image-Similarity-Search) -> an app that helps perform super fast image retrieval on PyTorch models for better embedding space interpretability
* [Interactive-TSNE](https://github.com/spaceml-org/Interactive-TSNE) -> a tool that provides a way to visually view a PyTorch model's feature representation for better embedding space interpretability
* [How Airbus Detects Anomalies in ISS Telemetry Data Using TFX](https://blog.tensorflow.org/2020/04/how-airbus-detects-anomalies-iss-telemetry-data-tfx.html) -> uses an autoencoder
* [RoofNet](https://github.com/ultysim/RoofNet) -> identify roof age using historical satellite images to lower the customer acquisition cost for new solar installations. Uses a VAE: Variational Autoencoder
* [Visual search over billions of aerial and satellite images](https://arxiv.org/abs/2002.02624) -> implemented [at Descartes labs](https://blog.descarteslabs.com/geovisual-search-for-rapid-generation-of-annotated-datasets)
* [parallax](https://github.com/uber-research/parallax) -> Tool for interactive embeddings visualization
* [Deep-Gapfill](https://github.com/remicres/Deep-Gapfill) -> Official implementation of Optical image gap filling using deep convolutional autoencoder from optical and radar images
* [Mxnet repository for generating embeddings on satellite images](https://github.com/fisch92/Metric-embeddings-for-satellite-image-classification) -> Includes sampling of images, mining algorithms, different architectures, error functions, measures for evaluation.
* [Fine tuning CLIP with Remote Sensing (Satellite) images and captions](https://huggingface.co/blog/fine-tune-clip-rsicd) -> fine tuning CLIP on the [RSICD](https://github.com/201528014227051/RSICD_optimal) image captioning dataset, to enable querying large catalogues in natural language. With [repo](https://github.com/arampacha/CLIP-rsicd), uses 🤗
* [Image search with 🤗 datasets](https://huggingface.co/blog/image-search-datasets) -> tutorial on fine tuning an image search model
* [SynImageAnalysis](https://github.com/FlorenceJiang/SynImageAnalysis) -> comparing syn and real sattlelite images in the latent feature space (embeddings)
* [GRN-SNDL](https://github.com/jiankang1991/GRN-SNDL) -> model the relations between samples (or scenes) by making use of a graph structure which is fed into network learning
* [SauMoCo](https://github.com/jiankang1991/SauMoCo) -> codes for TGRS paper: Deep Unsupervised Embedding for Remotely Sensed Images Based on Spatially Augmented Momentum Contrast
* [TGRS_RiDe](https://github.com/jiankang1991/TGRS_RiDe) -> Rotation Invariant Deep Embedding for RemoteSensing Images
* [RaVAEn](https://github.com/spaceml-org/RaVAEn) -> RaVAEn is a lightweight, unsupervised approach for change detection in satellite data based on Variational Auto-Encoders (VAEs) with the specific purpose of on-board deployment
* [Reverse image search using deep discrete feature extraction and locality-sensitive hashing](https://github.com/martenjostmann/deep-discrete-image-retrieval) 
* [SNCA_CE](https://github.com/jiankang1991/SNCA_CE) -> code for the paper Deep Metric Learning based on Scalable Neighborhood Components for Remote Sensing Scene Characterization
* [LandslideDetection-from-satellite-imagery](https://github.com/shulavkarki/LandslideDetection-from-satellite-imagery) -> Using Attention and Autoencoder boosted CNN
* [split-brain-remote-sensing](https://github.com/vladan-stojnic/split-brain-remote-sensing) -> code for 2018 paper: Analysis of Color Space Quantization in Split-Brain Autoencoder for Remote Sensing Image Classification
* [image-similarity-measures](https://github.com/up42/image-similarity-measures) -> Implementation of eight evaluation metrics to access the similarity between two images. [Blog post here](https://up42.com/blog/tech/image-similarity-measures)
* [Large_Scale_GeoVisual_Search](https://github.com/sdhayalk/Large_Scale_GeoVisual_Search) -> ResNet architecture on UC Merced Land Use Dataset with hamming distance for similarity based search
* [geobacter](https://github.com/JakeForsey/geobacter) -> Generates useful feature embeddings for geospatial locations
* [Satellite-Image-Segmentation](https://github.com/kunnalparihar/Satellite-Image-Segmentation) -> the KV-Net model uses this feature of autoencoders to reconnect the disconnected roads
* [Satellite-Image-Enhancement](https://github.com/VNDhanush/Satellite-Image-Enhancement) -> Image enhancement using GAN's and autoencoders
* [Variational-Autoencoder-For-Satellite-Imagery](https://github.com/RayanAAY-ops/Variational-Autoencoder-For-Satellite-Imagery) -> a special VAE to squeeze N images into one single representation with colors segmentating the different objects
* [DINCAE](https://github.com/gher-ulg/DINCAE) -> Data-Interpolating Convolutional Auto-Encoder is a neural network to reconstruct missing data in satellite observations
* [3D_SITS_Clustering](https://github.com/ekalinicheva/3D_SITS_Clustering) -> code for 2020 [paper](https://www.researchgate.net/publication/341902683_Unsupervised_Satellite_Image_Time_Series_Clustering_Using_Object-Based_Approaches_and_3D_Convolutional_Autoencoder): Unsupervised Satellite Image Time Series Clustering Using Object-Based Approaches and 3D Convolutional Autoencoder
* [sat_cnn](https://github.com/GDSL-UL/sat_cnn) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S0198971522000461?via%3Dihub): Estimating Generalized Measures of Local Neighbourhood Context from Multispectral Satellite Images Using a Convolutional Neural Network. Uses a convolutional autoencoder (CAE)
* [you-are-here](https://github.com/ZhouMengjie/you-are-here) -> Matlab code for 2020 paper: You Are Here: Geolocation by Embedding Maps and Images
* [Tensorflow similarity](https://github.com/tensorflow/similarity) -> offers state-of-the-art algorithms for metric learning and all the necessary components to research, train, evaluate, and serve similarity-based models
* [Train SimSiam on Satellite Images](https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html) using [Lightly](https://docs.lightly.ai/index.html) to generate embeddings that can be used for data exploration and understanding
* [Airbus_SDC_dup](https://github.com/WillieMaddox/Airbus_SDC_dup) -> Project focused on detecting duplicate regions of overlapping satellite imagery. Applied to Airbus ship detection dataset

## Image retreival
* [Demo_AHCL_for_TGRS2022](https://github.com/weiweisong415/Demo_AHCL_for_TGRS2022) -> code for 2022 paper: Asymmetric Hash Code Learning (AHCL) for remote sensing image retreival
* [GaLR](https://github.com/xiaoyuan1996/GaLR) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9745546): Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information
* [retrievalSystem](https://github.com/xiaoyuan1996/retrievalSystem) -> cross-modal image retrieval system
* [AMFMN](https://github.com/xiaoyuan1996/AMFMN) -> code for the 2021 paper: Exploring a Fine-grained Multiscale Method for Cross-modal Remote Sensing Image Retrieval
* [Active-Learning-for-Remote-Sensing-Image-Retrieval](https://github.com/flateon/Active-Learning-for-Remote-Sensing-Image-Retrieval) -> unofficial implementation of paper: A Novel Active Learning Method in Relevance Feedback for Content-Based Remote Sensing Image Retrieval
* [CMIR-NET](https://github.com/ushasi/CMIR-NET-A-deep-learning-based-model-for-cross-modal-retrieval-in-remote-sensing) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0167865520300453?via%3Dihub): A deep learning based model for cross-modal retrieval in remote sensing
* [Deep-Hash-learning-for-Remote-Sensing-Image-Retrieval](https://github.com/smallsmallflypigtang/Deep-Hash-learning-for-Remote-Sensing-Image-Retrieval) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9143474): Deep Hash Learning for Remote Sensing Image Retrieval
* [MHCLN](https://github.com/MLEnthusiast/MHCLN) -> code for 2018 [paper](https://ieeexplore.ieee.org/abstract/document/8518381): Deep Metric and Hash-Code Learning for Content-Based Retrieval of Remote Sensing Images
* [HydroViet_VOR](https://github.com/lannguyen0910/HydroViet_VOR) -> Object Retrieval in satellite images with Triplet Network
* [AMFMN](https://github.com/AICyberTeam/AMFMN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9437331): Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval

## Image Captioning & Visual Question Answering
* See the section [Image captioning datasets](https://github.com/robmarkcole/satellite-image-deep-learning#image-captioning-datasets)
* [remote-sensing-image-caption](https://github.com/TalentBoy2333/remote-sensing-image-caption) -> image classification and image caption by PyTorch
* [Fine tuning CLIP with Remote Sensing (Satellite) images and captions](https://huggingface.co/blog/fine-tune-clip-rsicd) -> fine tuning CLIP on the [RSICD](https://github.com/201528014227051/RSICD_optimal) image captioning dataset, to enable querying large catalogues in natural language. With [repo](https://github.com/arampacha/CLIP-rsicd), uses 🤗
* [VQA-easy2hard](https://gitlab.lrz.de/ai4eo/reasoning/VQA-easy2hard) -> code for 2022 [paper](https://arxiv.org/abs/2205.03147): From Easy to Hard: Learning Language-guided Curriculum for Visual Question Answering on Remote Sensing Data
* [CapFormer](https://github.com/Junjue-Wang/CapFormer) -> Pure transformer for remote sensing image caption
* [remote_sensing_image_captioning](https://github.com/chan64/remote_sensing_image_captioning) -> code for 2019 [paper](https://www.sciencedirect.com/science/article/pii/S1877050920300752): Region Driven Remote Sensing Image Captioning
* [Remote Sensing Image Captioning with Transformer and Multilabel Classification](https://github.com/hiteshK03/Remote-sensing-image-captioning-with-transformer-and-multilabel-classification)
* [Siamese-spatial-Graph-Convolution-Network](https://github.com/ushasi/Siamese-spatial-Graph-Convolution-Network) -> code for 2019 [paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314219300578): Siamese graph convolutional network for content based remote sensing image retrieval
* [MLAT](https://github.com/Chen-Yang-Liu/MLAT) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9709791): Remote-Sensing Image Captioning Based on Multilayer Aggregated Transformer
* [WordSent](https://github.com/hw2hwei/WordSent) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9308980): Word–Sentence Framework for Remote Sensing Image Captioning
* [a-mask-guided-transformer-with-topic-token](https://github.com/Meditation0119/a-mask-guided-transformer-with-topic-token-for-remote-sensing-image-captioning) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/12/2939): A Mask-Guided Transformer Network with Topic Token for Remote Sensing Image Captioning
* [MetaCaptioning](https://github.com/QiaoqiaoYang/MetaCaptioning) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271622000351): Meta captioning: A meta learning based remote sensing image captioning framework
* [Transformer-for-image-captioning](https://github.com/RicRicci22/Transformer-for-image-captioning) -> a transformer for image captioning, trained on the UCM dataset

## Mixed data learning
These techniques combine multiple data types, e.g. imagery and text data.
* [Predicting the locations of traffic accidents with satellite imagery and convolutional neural networks](https://towardsdatascience.com/teaching-a-neural-network-to-see-roads-74bff240c3e5) -> Combining satellite imagery and structured data to predict the location of traffic accidents with a neural network of neural networks, with [repo](https://github.com/L-Lewis/Predicting-traffic-accidents-CNN)
* [Multi-Input Deep Neural Networks with PyTorch-Lightning - Combine Image and Tabular Data](https://rosenfelder.ai/multi-input-neural-network-pytorch/) -> excellent intro article using pytorch, not actually applied to satellite data but to real estate data, with [repo](https://github.com/MarkusRosen/pytorch_multi_input_example)
* [Joint Learning from Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps](https://arxiv.org/abs/1705.06057) -> fusion based architectures and coarse-to-fine segmentation to include the OpenStreetMap layer into multispectral-based deep fully convolutional networks, arxiv paper
* [Composing Decision Forest and Neural Network models](https://www.tensorflow.org/decision_forests/tutorials/model_composition_colab) tensorflow documentation
* [pyimagesearch article on mixed-data](https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/)
* [pytorch-widedeep](https://github.com/jrzaurin/pytorch-widedeep) -> A flexible package for multimodal-deep-learning to combine tabular data with text and images using Wide and Deep models in Pytorch
* [accidentRiskMap](https://github.com/songtaohe/accidentRiskMap) -> Inferring high-resolution traffic accident risk maps based on satellite imagery and GPS trajectories

## Few-shot learning
This is a class of techniques which attempt to make predictions for classes with few, one or even zero examples provided during training. In zero shot learning (ZSL) the model is assisted by the provision of auxiliary information which typically consists of descriptions/semantic attributes/word embeddings for both the seen and unseen classes at train time ([ref](https://learnopencv.com/zero-shot-learning-an-introduction/)). These approaches are particularly relevant to remote sensing, where there may be many examples of common classes, but few or even zero examples for other classes of interest.
* [Unseen Land Cover Classification from High-Resolution Orthophotos Using Integration of Zero-Shot Learning and Convolutional Neural Networks](https://www.mdpi.com/2072-4292/12/10/1676)
* [FSODM](https://github.com/lixiang-ucas/FSODM) -> Official Code for paper "Few-shot Object Detection on Remote Sensing Images" on [arxiv](https://arxiv.org/abs/2006.07826)
* [Few-Shot Classification of Aerial Scene Images via Meta-Learning](https://www.mdpi.com/2072-4292/13/1/108/htm) -> 2020 publication, a classification model that can quickly adapt to unseen categories using only a few labeled samples
* [Papers about Few-shot Learning / Meta-Learning on Remote Sensing](https://github.com/lx709/Few-shot-Learning-Meta-Learning-on-Remote-Sensing-Papers)
* [SPNet](https://github.com/zoraup/SPNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9501951): Siamese-Prototype Network for Few-Shot Remote Sensing Image Scene Classification
* [MDL4OW](https://github.com/sjliu68/MDL4OW) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9186822): Few-Shot Hyperspectral Image Classification With Unknown Classes Using Multitask Deep Learning
* [P-CNN](https://github.com/Ybowei/P-CNN) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9435769): Prototype-CNN for Few-Shot Object Detection in Remote Sensing Images
* [CIR-FSD-2022](https://github.com/Li-ZK/CIR-FSD-2022) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/14/3255): Context Information Refinement for Few-Shot Object Detection in Remote Sensing Images
* [IEEE_TNNLS_Gia-CFSL](https://github.com/YuxiangZhang-BIT/IEEE_TNNLS_Gia-CFSL) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9812472): Graph Information Aggregation Cross-Domain Few-Shot Learning for Hyperspectral Image Classification
* [TIP_2022_CMFSL](https://github.com/B-Xi/TIP_2022_CMFSL) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9841445): Few-shot Learning with Class-Covariance Metric for Hyperspectral Image Classification
* [sen12ms-human-few-shot-classifier](https://github.com/MarcCoru/sen12ms-human-few-shot-classifier) -> code for paper: Humans are poor few-shot classifiers for Sentinel-2 land cover
* [S3Net](https://github.com/ZhaohuiXue/S3Net) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9791365): S3Net: Spectral–Spatial Siamese Network for Few-Shot Hyperspectral Image Classification
* [SiameseNet-for-few-shot-Hyperspectral-Classification](https://github.com/jjwwczy/jjwwczy-SiameseNet-for-few-shot-Hyperspectral-Classification) -> code for 2020 paper: 3DCSN:SiameseNet-for-few-shot-Hyperspectral-Classification

## Self-supervised, unsupervised & contrastive learning
These techniques use unlabelled datasets. [Yann LeCun](https://braindump.jethro.dev/posts/lecun_cake_analogy/) has described self/unsupervised learning as the 'base of the cake': *If we think of our brain as a cake, then the cake base is unsupervised learning. The machine predicts any part of its input for any observed part, all without the use of labelled data. Supervised learning forms the icing on the cake, and reinforcement learning is the cherry on top.*
* [Seasonal Contrast: Unsupervised Pre-Training from Uncurated Remote Sensing Data](https://devblog.pytorchlightning.ai/seasonal-contrast-transferable-visual-representations-for-remote-sensing-73a17863ed07) -> Seasonal Contrast (SeCo) is an effective pipeline to leverage unlabeled data for in-domain pre-training of remote sensing representations. Models trained with SeCo achieve better performance than their ImageNet pre-trained counterparts and state-of-the-art self-supervised learning methods on multiple downstream tasks. [paper](https://arxiv.org/abs/2103.16607) and [repo](https://github.com/ElementAI/seasonal-contrast)
* [Unsupervised Learning for Land Cover Classification in Satellite Imagery](https://omdena.com/blog/land-cover-classification/)
* [Tile2Vec: Unsupervised representation learning for spatially distributed data](https://ermongroup.github.io/blog/tile2vec/)
* [Contrastive Sensor Fusion](https://github.com/descarteslabs/contrastive_sensor_fusion) -> Code implementing Contrastive Sensor Fusion, an approach for unsupervised learning of multi-sensor representations targeted at remote sensing imagery
* [hyperspectral-autoencoders](https://github.com/lloydwindrim/hyperspectral-autoencoders) -> Tools for training and using unsupervised autoencoders and supervised deep learning classifiers for hyperspectral data, built on tensorflow. Autoencoders are unsupervised neural networks that are useful for a range of applications such as unsupervised feature learning and dimensionality reduction.
* [Sentinel-2 image clustering in python](https://towardsdatascience.com/sentinel-2-image-clustering-in-python-58f7f2c8a7f6)
* [MARTA GANs: Unsupervised Representation Learning for Remote Sensing Image Classification](https://arxiv.org/abs/1612.08879) and [code](https://github.com/BUPTLdy/MARTA-GAN)
* [A generalizable and accessible approach to machine learning with global satellite imagery](https://www.nature.com/articles/s41467-021-24638-z) nature publication -> MOSAIKS is designed to solve an unlimited number of tasks at planet-scale quickly using feature vectors, [with repo](https://github.com/Global-Policy-Lab/mosaiks-paper). Also see [mosaiks-api](https://github.com/calebrob6/mosaiks-api)
* [contrastive-satellite](https://github.com/hakeemtfrank/contrastive-satellite) -> Using contrastive learning to create embeddings from optical EuroSAT Satellite-2 imagery
* [Self-Supervised Learning of Remote Sensing Scene Representations Using Contrastive Multiview Coding](https://arxiv.org/abs/2104.07070) -> arxiv paper and [code](https://github.com/vladan-stojnic/CMC-RSSR)
* [Self-Supervised-Learner by spaceml-org](https://github.com/spaceml-org/Self-Supervised-Learner) -> train a classifier with fewer labeled examples needed using self-supervised learning, example applied to UC Merced land use dataset
* [deepsentinel](https://github.com/Lkruitwagen/deepsentinel) -> a sentinel-1 and -2 self-supervised sensor fusion model for general purpose semantic embedding
* [contrastive_SSL_ship_detection](https://github.com/alina2204/contrastive_SSL_ship_detection) -> Contrastive self supervised learning for ship detection in Sentinel 2 images
* [geography-aware-ssl](https://github.com/sustainlab-group/geography-aware-ssl) -> uses spatially aligned images over time to construct temporal positive pairs in contrastive learning and geo-location to design pre-text tasks
* [CNN-Supervised Classification](https://github.com/geojames/CNN-Supervised-Classification) -> Python code for self-supervised classification of remotely sensed imagery - part of the Deep Riverscapes project
* [clustimage](https://github.com/erdogant/clustimage) -> a python package for unsupervised clustering of images
* [LandSurfaceClustering](https://github.com/lhalloran/LandSurfaceClustering) -> Land surface classification using remote sensing data with unsupervised machine learning (k-means)
* [K-Means Clustering for Surface Segmentation of Satellite Images](https://medium.com/@maxfieldeland/k-means-clustering-for-surface-segmentation-of-satellite-images-ad1902791ebf)
* [Sentinel-2 satellite imagery for crop classification using unsupervised clustering](https://medium.com/devseed/sentinel-2-satellite-imagery-for-crop-classification-part-2-47db3745eb49) -> label groups of pixels based on temporal trends of their NDVI values
* [TheColorOutOfSpace](https://github.com/stevinc/TheColorOutOfSpace) -> Pytorch code for the paper "The color out of space: learning self-supervised representations for Earth Observation imagery" using the BigEarthNet dataset
* [Semantic segmentation of SAR images using a self supervised technique](https://github.com/cattale93/pytorch_self_supervised_learning)
* [STEGO](https://github.com/mhamilton723/STEGO) -> Unsupervised Semantic Segmentation by Distilling Feature Correspondences, with [paper](https://arxiv.org/abs/2203.08414)
* [Unsupervised Segmentation of Hyperspectral Remote Sensing Images with Superpixels](https://github.com/mpBarbato/Unsupervised-Segmentation-of-Hyperspectral-Remote-Sensing-Images-with-Superpixels)
* [SoundingEarth](https://github.com/khdlr/SoundingEarth) -> Self-supervised Audiovisual Representation Learning for Remote Sensing Data, uses the SoundingEarth [Dataset](https://zenodo.org/record/5600379#.Yom4W5PMK3I)
* [singleSceneSemSegTgrs2022](https://github.com/sudipansaha/singleSceneSemSegTgrs2022) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9773162): Unsupervised Single-Scene Semantic Segmentation for Earth Observation
* [SSLRemoteSensing](https://github.com/flyakon/SSLRemoteSensing) -> code for 2021 [paper](https://ieeexplore.ieee.org/abstract/document/9460820): Semantic Segmentation of Remote Sensing Images With Self-Supervised Multitask Representation Learning
* [CBT](https://github.com/VMarsocci/CBT) code for 2022 [paper](https://arxiv.org/abs/2205.11319): Continual Barlow Twins: continual self-supervised learning for remote sensing semantic segmentation
* [Unsupervised Satellite Image Classification based on Partial Adversarial Domain Adaptation](https://github.com/lwpyh/Unsupervised-Satellite-Image-Classfication-based-on-Partial-Domain-Adaptation) -> Code for course project
* [T2FTS](https://github.com/wdzhao123/T2FTS) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9781379): Teaching Teachers First and Then Student: Hierarchical Distillation to Improve Long-Tailed Object Recognition in Aerial Images
* [SSLTransformerRS](https://github.com/HSG-AIML/SSLTransformerRS) -> code for 2022 paper: Self-supervised Vision Transformers for Land-cover Segmentation and
Classification
* [DINO-MM](https://github.com/zhu-xlab/DINO-MM) -> code for 2022 [paper](https://arxiv.org/abs/2204.05381): Self-supervised Vision Transformers for Joint SAR-optical Representation Learning
* [SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12) -> a large-scale dataset for self-supervised learning in Earth observation
* [SSL4EO-Review](https://github.com/zhu-xlab/SSL4EO-Review) -> code for 2022 [paper](https://arxiv.org/abs/2206.13188): Self-supervised Learning in Remote Sensing: A Review
* [transfer_learning_cspt](https://github.com/ZhAnGToNG1/transfer_learning_cspt) -> code for 2022 [paper](https://arxiv.org/abs/2207.03860): Consecutive Pretraining: A Knowledge Transfer Learning Strategy with Relevant Unlabeled Data for Remote Sensing Domain
* [OTL](https://github.com/qlilx/OTL) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/14/3361): Clustering-Based Representation Learning through Output Translation and Its Application to Remote-Sensing Images
* [Push-and-Pull-Network](https://github.com/WindVChen/Push-and-Pull-Network) -> code for 2022 paper: Contrastive Learning for Fine-grained Ship Classification in Remote Sensing Images
* [vissl_experiments](https://github.com/lewfish/ssl/tree/main/vissl_experiments) -> Self-supervised Learning using Facebook [VISSL](https://github.com/facebookresearch/vissl) on the RESISC-45 satellite imagery classification dataset
* [MS2A-Net](https://github.com/Kasra2020/MS2A-Net) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9855229): MS 2 A-Net: Multi-scale spectral-spatial association network for hyperspectral image clustering

## Weakly & semi-supervised learning
These techniques use a partially annotated dataset
* [MARE](https://github.com/VMarsocci/MARE) -> self-supervised Multi-Attention REsu-net for semantic segmentation in remote sensing
* [SSGF-for-HRRS-scene-classification](https://github.com/weihancug/SSGF-for-HRRS-scene-classification) -> code for 2018 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271617303428): A semi-supervised generative framework with deep learning features for high-resolution remote sensing image scene classification
* [SFGAN](https://github.com/MLEnthusiast/SFGAN) -> code for 2018 [paper](https://ieeexplore.ieee.org/abstract/document/8451836): Semantic-Fusion Gans for Semi-Supervised Satellite Image Classification
* [SSDAN](https://github.com/alhichri/SSDAN) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/19/3861): Multi-Source Semi-Supervised Domain Adaptation Network for Remote Sensing Scene Classification
* [HR-S2DML](https://github.com/jiankang1991/HR-S2DML) -> code for 2020 [paper](https://www.mdpi.com/2072-4292/12/16/2603): High-Rankness Regularized Semi-Supervised Deep Metric Learning for Remote Sensing Imagery
* [Semantic Segmentation of Satellite Images Using Point Supervision](https://github.com/KambachJannis/MasterThesis)
* [fcd](https://github.com/jnyborg/fcd) -> code for 2021 paper: Fixed-Point GAN for Cloud Detection. A weakly-supervised approach, training with only image-level labels
* [weak-segmentation](https://github.com/LendelTheGreat/weak-segmentation) -> Weakly supervised semantic segmentation for aerial images in pytorch
* [TNNLS_2022_X-GPN](https://github.com/B-Xi/TNNLS_2022_X-GPN) -> Code for paper: Semisupervised Cross-scale Graph Prototypical Network for Hyperspectral Image Classification
* [weakly_supervised](https://github.com/LobellLab/weakly_supervised) -> code for the paper Weakly Supervised Deep Learning for Segmentation of Remote Sensing Imagery. Demonstrates that segmentation can be performed using small datasets comprised of pixel or image labels
* [wan](https://github.com/engrjavediqbal/wan) -> Weakly-Supervised Domain Adaptation for Built-up Region Segmentation in Aerial and Satellite Imagery, with [arxiv paper](https://arxiv.org/abs/2007.02277)
* [sourcerer](https://github.com/benjaminmlucas/sourcerer) -> A Bayesian-inspired deep learning method for semi-supervised domain adaptation designed for land cover mapping from satellite image time series (SITS). [Paper](https://link.springer.com/article/10.1007/s10994-020-05942-z)
* [MSMatch](https://github.com/gomezzz/MSMatch) -> Semi-Supervised Multispectral Scene Classification with Few Labels. Includes code to work with both the RGB and the multispectral (MS) versions of EuroSAT dataset and the UC Merced Land Use (UCM) dataset. [Paper](https://arxiv.org/abs/2103.10368)
* [Flood Segmentation on Sentinel-1 SAR Imagery with Semi-Supervised Learning](https://github.com/sidgan/ETCI-2021-Competition-on-Flood-Detection) with [arxiv paper](https://arxiv.org/abs/2107.08369)
* [Semi-supervised learning in satellite image classification](https://medium.com/sentinel-hub/semi-supervised-learning-in-satellite-image-classification-e0874a76fc61) -> experimenting with MixMatch and the EuroSAT data set
* [ScRoadExtractor](https://github.com/weiyao1996/ScRoadExtractor) -> code for 2020 [paper](https://arxiv.org/abs/2010.13106): Scribble-based Weakly Supervised Deep Learning for Road Surface Extraction from Remote Sensing Images
* [ICSS](https://github.com/alteia-ai/ICSS) -> code for 2022 [paper](https://arxiv.org/abs/2201.01029): Weakly-supervised continual learning for class-incremental segmentation
* [es-CP](https://github.com/majidseydgar/Res-CP) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9849704): Semi-Supervised Hyperspectral Image Classification Using a Probabilistic Pseudo-Label Generation Framework

## Active learning
Supervised deep learning techniques typically require a huge number of annotated/labelled examples to provide a training dataset. However labelling at scale take significant time, expertise and resources. Active learning techniques aim to reduce the total amount of annotation that needs to be performed by selecting the most useful images to label from a large pool of unlabelled images, thus reducing the time to generate useful training datasets. These processes may be referred to as [Human-in-the-Loop Machine Learning](https://medium.com/pytorch/https-medium-com-robert-munro-active-learning-with-pytorch-2f3ee8ebec)
* [Active learning for object detection in high-resolution satellite images](https://arxiv.org/abs/2101.02480) -> arxiv paper
* [AIDE V2 - Tools for detecting wildlife in aerial images using active learning](https://github.com/microsoft/aerial_wildlife_detection)
* [AstronomicAL](https://github.com/grant-m-s/AstronomicAL) -> An interactive dashboard for visualisation, integration and classification of data using Active Learning
* Read about [active learning on the lightly platform](https://docs.lightly.ai/getting_started/active_learning.html) and [in label-studio](https://labelstud.io/guide/ml.html#Active-Learning)
* [Active-Labeler by spaceml-org](https://github.com/spaceml-org/Active-Labeler) -> a CLI Tool that facilitates labeling datasets with just a SINGLE line of code
* [Labelling platform for Mapping Africa active learning project](https://github.com/agroimpacts/labeller)
* [ChangeDetectionProject](https://github.com/previtus/ChangeDetectionProject) -> Trying out Active Learning in with deep CNNs for Change detection on remote sensing data
* [ALS4GAN](https://github.com/immuno121/ALS4GAN) -> Active Learning for Improved Semi Supervised Semantic Segmentation in Satellite Images, with [paper](https://arxiv.org/abs/2110.07782)
* [Active-Learning-for-Remote-Sensing-Image-Retrieval](https://github.com/flateon/Active-Learning-for-Remote-Sensing-Image-Retrieval) -> unofficial implementation of paper: A Novel Active Learning Method in Relevance Feedback for Content-Based Remote Sensing Image Retrieval
* [DIAL](https://github.com/alteia-ai/DIAL) -> code for 2022 [paper](https://arxiv.org/abs/2201.01047): DIAL: Deep Interactive and Active Learning for Semantic Segmentation in Remote Sensing

## Federated learning
Federated learning is a process for training models in a distributed fashion without sharing of data
* [Federated-Learning-for-Remote-Sensing](https://github.com/anandcu3/Federated-Learning-for-Remote-Sensing) ->  implementation of three Federated Learning models

## Image registration
Image registration is the process of registering one or more images onto another (typically well georeferenced) image. Traditionally this is performed manually by identifying control points (tie-points) in the images, for example using QGIS. This section lists approaches which mostly aim to automate this manual process. There is some overlap with the data fusion section but the distinction I make is that image registration is performed as a prerequisite to downstream processes which will use the registered data as an input.
* [Wikipedia article on registration](https://en.wikipedia.org/wiki/Image_registration) -> register for change detection or [image stitching](https://mono.software/2018/03/14/Image-stitching/)
* [Phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) is used to estimate the XY translation between two images with sub-pixel accuracy. Can be used for accurate registration of low resolution imagery onto high resolution imagery, or to register a [sub-image on a full image](https://www.mathworks.com/help/images/registering-an-image-using-normalized-cross-correlation.html) -> Unlike many spatial-domain algorithms, the phase correlation method is resilient to noise, occlusions, and other defects. With [additional pre-processing](https://scikit-image.org/docs/dev/auto_examples/registration/plot_register_rotation.html) image rotation and scale changes can also be calculated.
* [ImageRegistration](https://github.com/jandremarais/ImageRegistration) -> Interview assignment for multimodal image registration using SIFT
* [imreg_dft](https://github.com/matejak/imreg_dft) -> Image registration using discrete Fourier transform. Given two images it can calculate the difference between scale, rotation and position of imaged features. Used by the [up42 co-registration service](https://up42.com/marketplace/blocks/processing/up42-coregistration)
* [arosics](https://danschef.git-pages.gfz-potsdam.de/arosics/doc/about.html) -> Perform automatic subpixel co-registration of two satellite image datasets using phase-correlation, XY translations only.
* [SubpixelAlignment](https://github.com/vldkhramtsov/SubpixelAlignment) -> Implementation of tiff image alignment through phase correlation for pixel- and subpixel-bias
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
* [image-to-db-registration](https://gitlab.orfeo-toolbox.org/remote_modules/image-to-db-registration) -> This remote module implements an algorithm for automated vector Database registration onto an Image. Implemented in the orfeo-toolbox
* [MS_HLMO_registration](https://github.com/MrPingQi/MS_HLMO_registration) -> Multi-scale Histogram of Local Main Orientation for Remote Sensing Image Registration, with [paper](https://arxiv.org/abs/2204.00260)
* [cnn-matching](https://github.com/lan-cz/cnn-matching) -> code and datadset for paper: Deep learning algorithm for feature matching of cross modality remote sensing images
* [Imatch-P](https://github.com/geoyee/Imatch-P) -> A demo using SuperGlue and SuperPoint to do the image matching task based PaddlePaddle
* [NBR-Net](https://github.com/xuyingxiao/NBR-Net) -> A Non-rigid Bi-directional Registration Network for Multi-temporal Remote Sensing Images
* [MU-Net](https://github.com/yeyuanxin110/MU-Net) -> code for paper: A Multi-Scale Framework with Unsupervised Learning for Remote Sensing Image Registration. Also checkout [this implementation](https://github.com/woshiybc/Multi-Scale-Unsupervised-Framework-MSUF)
* [unsupervisedDeepHomographyRAL2018](https://github.com/tynguyen/unsupervisedDeepHomographyRAL2018) -> Unsupervised Deep Homography applied to aerial data
* [registration_cnn_ntg](https://github.com/zhangliukun/registration_cnn_ntg) -> code for paper: A Multispectral Image Registration Method Based on Unsupervised Learning
* [remote-sensing-images-registration-dataset](https://github.com/liliangzhi110/remote-sensing-images-registration-dataset) -> at 0.23m, 3.75m & 30m resolution
* [semantic-template-matching](https://github.com/liliangzhi110/semantictemplatematching) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271621002446): A deep learning semantic template matching framework for remote sensing image registration
* [GMN-Generative-Matching-Network](https://github.com/ei1994/GMN-Generative-Matching-Network) -> code for 2018 paper: Deep Generative Matching Network for Optical and SAR Image Registration
* [SOMatch](https://github.com/system123/SOMatch) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/pii/S0924271620302598): A deep learning framework for matching of SAR and optical imagery
* [Interspectral image registration dataset](https://medium.com/dronehub/datasets-96fc4f9a92e5) -> including satellite and drone imagery
* [RISG-image-matching](https://github.com/lan-cz/RISG-image-matching) -> A rotation invariant SuperGlue image matching algorithm
* [DeepAerialMatching_pytorch](https://github.com/jaehyunnn/DeepAerialMatching_pytorch) -> code for 2020 [paper](https://arxiv.org/abs/2002.01325): A Two-Stream Symmetric Network with Bidirectional Ensemble for Aerial Image Matching
* [DPCN](https://github.com/ZJU-Robotics-Lab/DPCN) -> code for 2020 [paper](https://arxiv.org/abs/2008.09474): Deep Phase Correlation for End-to-End Heterogeneous Sensor Measurements Matching
* [FSRA](https://github.com/Dmmm1997/FSRA) -> code for 2022 [paper](https://arxiv.org/abs/2201.09206): A Transformer-Based Feature Segmentation and Region Alignment Method For UAV-View Geo-Localization
* [IHN](https://github.com/imdumpl78/IHN) -> code for 2022 [paper](https://arxiv.org/abs/2203.15982): Iterative Deep Homography Estimation
* [OSMNet](https://github.com/zhanghan9718/OSMNet) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9609993): Explore Better Network Framework for High-Resolution Optical and SAR Image Matching
* [L2_Siamese](https://github.com/TheKiteFlier/L2_Siamese) -> code for the 2020 [paper](https://ieeexplore.ieee.org/document/9264687): Registration of Multiresolution Remote Sensing Images Based on L2-Siamese Model

## Data fusion
Data fusion covers techniques which integrate multiple datasources, for example fusing SAR & optical to make predictions about crop type. It can also cover fusion with non imagery data such as IOT sensor data
* [Awesome-Data-Fusion-for-Remote-Sensing](https://github.com/px39n/Awesome-Data-Fusion-for-Remote-Sensing)
* [UDALN_GRSL](https://github.com/JiaxinLiCAS/UDALN_GRSL) -> Deep Unsupervised Blind Hyperspectral and Multispectral Data Fusion
* [CropTypeMapping](https://github.com/ellaampy/CropTypeMapping) -> Crop type mapping from optical and radar (Sentinel-1&2) time series using attention-based deep learning
* [Multimodal-Remote-Sensing-Toolkit](https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit) -> uses Hyperspectral and LiDAR Data
* [Aerial-Template-Matching](https://github.com/m-hamza-mughal/Aerial-Template-Matching) -> development of an algorithm for template Matching on aerial imagery applied to UAV dataset
* [DS_UNet](https://github.com/SebastianHafner/DS_UNet) -> code for 2021 paper: Sentinel-1 and Sentinel-2 Data Fusion for Urban Change Detection using a Dual Stream U-Net, uses Onera Satellite Change Detection dataset
* [DDA_UrbanExtraction](https://github.com/SebastianHafner/DDA_UrbanExtraction) -> Unsupervised Domain Adaptation for Global Urban Extraction using Sentinel-1 and Sentinel-2 Data
* [swinstfm](https://github.com/LouisChen0104/swinstfm) -> code for paper: Remote Sensing Spatiotemporal Fusion using Swin Transformer
* [LoveCS](https://github.com/Junjue-Wang/LoveCS) -> code for 2022 [paper](https://www.researchgate.net/publication/360484883_Cross-sensor_domain_adaptation_for_high_spatial_resolution_urban_land-cover_mapping_From_airborne_to_spaceborne_imagery): Cross-sensor domain adaptation for high-spatial resolution urban land-cover mapping: from airborne to spaceborne imagery
* [comingdowntoearth](https://github.com/aysim/comingdowntoearth) -> code for 2021 paper: Implementation of 'Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization'
* [Matching between acoustic and satellite images](https://github.com/giovgiac/neptune)
* [MapRepair](https://github.com/zorzi-s/MapRepair) -> Deep Cadastre Maps Alignment and Temporal Inconsistencies Fix in Satellite Images
* [Compressive-Sensing-and-Deep-Learning-Framework](https://github.com/rahulgite94/Compressive-Sensing-and-Deep-Learning-Framework) ->  Compressive Sensing is used as an initial guess to combine data from multiple sources, with LSTM used to refine the result
* [DeepSim](https://github.com/wangxiaodiu/DeepSim) -> code for paper: DeepSIM: GPS Spoofing Detection on UAVs using Satellite Imagery Matching
* [MHF-net](https://github.com/XieQi2015/MHF-net) -> code for 2019 [paper](https://ieeexplore.ieee.org/document/8953470): Multispectral and Hyperspectral Image Fusion by MS/HS Fusion Net
* [Remote_Sensing_Image_Fusion](https://github.com/huangshanshan33/Remote_Sensing_Image_Fusion) -> code for 2021 [paper](https://www.researchgate.net/publication/352580177_Semi-Supervised_Remote_Sensing_Image_Fusion_Using_Multi-Scale_Conditional_Generative_Adversarial_network_with_Siamese_Structure): Semi-Supervised Remote Sensing Image Fusion Using Multi-Scale Conditional Generative Adversarial network with Siamese Structure
* [CNNs for Multi-Source Remote Sensing Data Fusion](https://github.com/yyyyangyi/CNNs-for-Multi-Source-Remote-Sensing-Data-Fusion) -> code for 2021 [paper](https://arxiv.org/abs/2109.06094): Single-stream CNN with Learnable Architecture for Multi-source Remote Sensing Data
* [Deep Generative Reflectance Fusion](https://github.com/Cervest/ds-generative-reflectance-fusion) -> Achieving Landsat-like reflectance at any date by fusing Landsat and MODIS surface reflectance with deep generative models
* [IEEE_TGRS_MDL-RS](https://github.com/danfenghong/IEEE_TGRS_MDL-RS) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9174822): More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classification
* [SSRNET](https://github.com/hw2hwei/SSRNET) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9186332): SSR-NET: Spatial-Spectral Reconstruction Network for Hyperspectral and Multispectral Image Fusion
* [cross-view-image-matching](https://github.com/kregmi/cross-view-image-matching) -> code for 2019 paper: Bridging the Domain Gap for Ground-to-Aerial Image Matching
* [CoF-MSMG-PCNN](https://github.com/WeiTan1992/CoF-MSMG-PCNN) -> code for 2020 paper: Remote Sensing Image Fusion via Boundary Measured Dual-Channel PCNN in Multi-Scale Morphological Gradient Domain
* [robust_matching_network_on_remote_sensing_imagery_pytorch](https://github.com/mrk1992/robust_matching_network_on_remote_sensing_imagery_pytorch) -> code for 2019 paper: A Robust Matching Network for Gradually Estimating Geometric Transformation on Remote Sensing Imagery
* [edcstfn](https://github.com/theonegis/edcstfn) -> code for 2019 [paper](https://www.mdpi.com/2072-4292/11/24/2898): An Enhanced Deep Convolutional Model for Spatiotemporal Image Fusion
* [ganstfm](https://github.com/theonegis/ganstfm) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9336033): A Flexible Reference-Insensitive Spatiotemporal Fusion Model for Remote Sensing Images Using Conditional Generative Adversarial Network
* [CMAFF](https://github.com/DocF/CMAFF) -> code for 2021 [paper](https://arxiv.org/abs/2112.02991): Cross-Modality Attentive Feature Fusion for Object Detection in Multispectral Remote Sensing Imagery
* [SOLC](https://github.com/yisun98/SOLC) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/pii/S0303243421003457): MCANet: A joint semantic segmentation framework of optical and SAR images for land use classification. Uses [WHU-OPT-SAR-dataset](https://github.com/AmberHen/WHU-OPT-SAR-dataset)
* [MFT](https://github.com/AnkurDeria/MFT) -> code for 2022 [paper](https://arxiv.org/abs/2203.16952): Multimodal Fusion Transformer for Remote Sensing Image Classification
* [ISPRS_S2FL](https://github.com/danfenghong/ISPRS_S2FL) -> code for 2021 [paper](https://www.sciencedirect.com/science/article/pii/S0924271621001362): Multimodal Remote Sensing Benchmark Datasets for Land Cover Classification with A Shared and Specific Feature Learning Model
* [HSHT-Satellite-Imagery-Synthesis](https://github.com/yuvalofek/HSHT-Satellite-Imagery-Synthesis) -> code for thesis - Improving Flood Maps by Increasing the Temporal Resolution of Satellites Using Hybrid Sensor Fusion
* [MDC](https://github.com/Kasra2020/MDC) -> code for 2021 [paper](https://ieeexplore.ieee.org/document/9638348): Unsupervised Data Fusion With Deeper Perspective: A Novel Multisensor Deep Clustering Algorithm
* [FusAtNet](https://github.com/ShivamP1993/FusAtNet) -> code for 2020 [paper](https://ieeexplore.ieee.org/document/9150738): FusAtNet: Dual Attention based SpectroSpatial Multimodal Fusion Network for Hyperspectral and LiDAR Classification

## Terrain mapping, Disparity Estimation, Lidar, DEMs & NeRF
Measure surface contours & locate 3D points in space from 2D images. NeRF stands for Neural Radiance Fields and is the term used in deep learning communities to describe a model that generates views of complex 3D scenes based on a partial set of 2D images
* [Wikipedia DEM article](https://en.wikipedia.org/wiki/Digital_elevation_model) and [phase correlation](https://en.wikipedia.org/wiki/Phase_correlation) article
* [Intro to depth from stereo](https://github.com/IntelRealSense/librealsense/blob/master/doc/depth-from-stereo.md)
* Map terrain from stereo images to produce a digital elevation model (DEM) -> high resolution & paired images required, typically 0.3 m, e.g. [Worldview](https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/37/DG-WV2ELEVACCRCY-WP.pdf)
* Process of creating a DEM [here](https://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XLI-B1/327/2016/isprs-archives-XLI-B1-327-2016.pdf) and [here](https://www.geoimage.com.au/media/brochure_pdfs/Geoimage_DEM_brochure_Oct10_LR.pdf).
* [ArcGIS can generate DEMs from stereo images](http://pro.arcgis.com/en/pro-app/help/data/imagery/generate-elevation-data-using-the-dems-wizard.htm)
* https://github.com/MISS3D/s2p -> produces elevation models from images taken by high resolution optical satellites -> demo code on https://gfacciol.github.io/IS18/
* [Automatic 3D Reconstruction from Multi-Date Satellite Images](http://dev.ipol.im/~facciolo/pub/CVPRW2017.pdf)
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
* [SatelliteNeRF](https://github.com/Kai-46/SatelliteNeRF) -> PyTorch-based Neural Radiance Fields adapted to satellite domain
* [SatelliteSfM](https://github.com/Kai-46/SatelliteSfM) -> A library for solving the satellite structure from motion problem
* [SatelliteSurfaceReconstruction](https://github.com/SBCV/SatelliteSurfaceReconstruction) -> 3D Surface Reconstruction From Multi-Date Satellite Images, ISPRS, 2021
* [son2sat](https://github.com/giovgiac/son2sat) -> A neural network coded in TensorFlow 1 that produces satellite images from acoustic images
* [aerial_mtl](https://github.com/marcelampc/aerial_mtl) -> PyTorch implementation for multi-task learning with aerial images to learn both semantics and height from aerial image datasets; fuses RGB & lidar
* [ReKlaSat-3D](https://github.com/MacOS/ReKlaSat-3D) -> 3D Reconstruction and Classification from Very High Resolution Satellite Imagery
* [M3Net](https://github.com/lauraset/BuildingHeightModel) -> A deep learning method for building height estimation using high-resolution multi-view imagery over urban areas
* [HMSM-Net](https://github.com/Sheng029/HMSM-Net) -> code for 2022 [paper](https://www.sciencedirect.com/science/article/abs/pii/S092427162200123X): Hierarchical multi-scale matching network for disparity estimation of high-resolution satellite stereo images
* [StereoMatchingRemoteSensing](https://github.com/Sheng029/StereoMatchingRemoteSensing) -> code for 2021 [paper](https://www.mdpi.com/2072-4292/13/24/5050): Dual-Scale Matching Network for Disparity Estimation of High-Resolution Remote Sensing Images
* [satnerf](https://centreborelli.github.io/satnerf/) -> Learning Multi-View Satellite Photogrammetry With Transient Objects and Shadow Modeling Using RPC Cameras
* [SatMVS](https://github.com/WHU-GPCV/SatMVS) -> code for 2021 paper: Rational Polynomial Camera Model Warping for Deep Learning Based Satellite Multi-View Stereo Matching
* [ImpliCity](https://github.com/prs-eth/ImpliCity) -> reconstructs digital surface models (DSMs) from raw photogrammetric 3D point clouds and ortho-images with the help of an implicit neural 3D scene representation
* [WHU-Stereo](https://github.com/Sheng029/WHU-Stereo) -> a large-scale dataset for stereo matching of high-resolution satellite imagery & several deep learning methods for stereo matching. Methods include StereoNet, Pyramid Stereo Matching Network & HMSM-Net
* [Photogrammetry-Guide](https://github.com/mikeroyal/Photogrammetry-Guide) -> A guide covering Photogrammetry including the applications, libraries and tools that will make you a better and more efficient Photogrammetry development
* [DSM-to-DTM](https://github.com/mdmeadows/DSM-to-DTM) -> Exploring the use of machine learning to convert a Digital Surface Model (e.g. SRTM) to a Digital Terrain Model

## Thermal Infrared
* [The World Needs (a lot) More Thermal Infrared Data from Space](https://towardsdatascience.com/the-world-needs-a-lot-more-thermal-infrared-data-from-space-dbbba389be8a)
* [IR2VI thermal-to-visible image translation framework based on GANs](https://arxiv.org/abs/1806.09565) with [code](https://github.com/linty5/IR2VI_CycleGAN)
* [The finest resolution urban outdoor heat exposure maps in major US cities](https://xiaojianggis.github.io/heatexpo/) -> urban microclimate modeling based on high resolution 3D urban models and meteorological data makes it possible to examine how people are exposed to heat stress at a fine spatio-temporal level.
* [Object_Classification_in_Thermal_Images](https://www.researchgate.net/publication/328400392_Object_Classification_in_Thermal_Images_using_Convolutional_Neural_Networks_for_Search_and_Rescue_Missions_with_Unmanned_Aerial_Systems) -> classification accuracy was improved by adding the object size as a feature directly within the CNN
* [Thermal imaging with satellites](https://chrieke.medium.com/thermal-imaging-with-satellites-34f381856dd1) blog post by Christoph Rieke
* [Object Detection on Thermal Images](https://medium.com/@joehoeller/object-detection-on-thermal-images-f9526237686a) -> using YOLO-v3 and applied to a terrestrial dataset from FLIR, the article offers some usful insights into the model training
* [Fire alerts service by Descartes Labs](https://www.businessinsider.com/how-descartes-labs-leveraging-artificial-intelligence-fight-wildfires-2019-12)
* [Background Invariant Classification on Infrared Imagery by Data Efficient Training and Reducing Bias in CNNs](https://arxiv.org/abs/2201.09144)
* [CMAFF](https://github.com/DocF/CMAFF) -> code for 2021 [paper](https://arxiv.org/abs/2112.02991): Cross-Modality Attentive Feature Fusion for Object Detection in Multispectral Remote Sensing Imagery
* [LEO-GEO-landsurfacetemp](https://github.com/KateDuffy/LEO-GEO-landsurfacetemp) -> Multi-sensor machine learning to retrieve high spatiotemporal resolution land surface temperature

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
* [xView3](https://iuu.xview.us/) is a competition to detect dark vessels using computer vision and global SAR satellite imagery. [First place solution](https://github.com/DIUx-xView/xView3_first_place) and [second place solution](https://github.com/DIUx-xView/xView3_second_place). Additional places up to fifth place are available at the [xView GitHub Organization page](https://github.com/DIUx-xView/)
* [Winners of the STAC Overflow: Map Floodwater from Radar Imagery competition](https://github.com/drivendataorg/stac-overflow)
* [deSpeckNet-TF-GEE](https://github.com/adugnag/deSpeckNet-TF-GEE) -> implementation of the paper 'deSpeckNet: Generalizing Deep Learning Based SAR Image Despeckling'
* [cnn_sar_image_classification](https://github.com/diogosens/cnn_sar_image_classification) -> CNN for classifying SAR images of the Amazon Rainforest
* [s1_icetype_cnn](https://github.com/nansencenter/s1_icetype_cnn) -> Retrieve sea ice type from Sentinel-1 SAR with CNN
* [SARSeg](https://github.com/ggsDing/SARSeg) -> pytorch code for the paper 'MP-ResNet: Multi-path Residual Network for the Semantic segmentation of PolSAR Images'
* [TGRS_DisOptNet](https://github.com/jiankang1991/TGRS_DisOptNet) -> Distilling Semantic Knowledge from Optical Images for Weather-independent Building Segmentation
* [SAR_CD_DDNet](https://github.com/summitgao/SAR_CD_DDNet) -> PyTorch implementation of Change Detection in Synthetic Aperture Radar Images Using a Dual Domain Network
* [SAR_CD_MS_CapsNet](https://github.com/summitgao/SAR_CD_MS_CapsNet) -> Code for the paper "Change Detection in SAR Images Based on Multiscale Capsule Network" IEEE GRSL 2021
* [anomaly-detection-in-SAR-imagery](https://github.com/iamyadavabhishek/anomaly-detection-in-SAR-imagery) -> identify an unknown ship in docks using keras &  retinanet
* [sar_transformer](https://github.com/malshaV/sar_transformer) -> Transformer based SAR image despeckling, trained with synthetic imagery, with [paper](https://arxiv.org/abs/2201.09355)
* [SSDD ship detection dataset](https://github.com/TianwenZhang0825/Official-SSDD)
* [Semantic segmentation of SAR images using a self supervised technique](https://github.com/cattale93/pytorch_self_supervised_learning)
* [Ship Detection on Remote Sensing Synthetic Aperture Radar Data](https://github.com/JasonManesis/Ship-Detection-on-Remote-Sensing-Synthetic-Aperture-Radar-Data) -> based on the architectures of the Faster-RCNN and YOLOv5 networks
* [Target Recognition in SAR](https://github.com/NateDiR/sar_target_recognition_deep_learning) -> Identify Military Vehicles in Satellite Imagery with TensorFlow, with [article](https://python.plainenglish.io/identifying-military-vehicles-in-satellite-imagery-with-tensorflow-96015634129d)
* [DSN](https://github.com/Alien9427/DSN) -> code for 2020 paper: Deep SAR-Net: Learning objects from signals
* [SAR_denoising](https://github.com/MathieuRita/SAR_denoising) -> project on application of FFDNet to SAR images
* [sarCdUsingDeepTranscoding](https://github.com/sudipansaha/sarCdUsingDeepTranscoding) -> Details of a SAR to optical transcoder training. The generator of the transcoder is subsequently used for transfer learning in a change detection framework
* [cnninsar](https://github.com/subhayanmukherjee/cnninsar) -> code for 2018 [paper](https://ieeexplore.ieee.org/document/8589920): CNN-Based InSAR Denoising and Coherence Metric
* [sar](https://github.com/GeomaticsAndRS/sar) -> Despeckling Synthetic Aperture Radar Images using a Deep Residual CNN
* [GCBANet](https://github.com/TianwenZhang0825/GCBANet) -> code for 2022 [paper](https://www.mdpi.com/2072-4292/14/9/2165): A Global Context Boundary-Aware Network for SAR Ship Instance Segmentation
* [SAR_CD_GKSNet](https://github.com/summitgao/SAR_CD_GKSNet) -> code for 2022 [paper](https://arxiv.org/abs/2201.08954): Change Detection from Synthetic Aperture Radar Images via Graph-Based Knowledge Supplement Network
* [pixel-wise-segmentation-of-sar](https://github.com/flyingshan/pixel-wise-segmentation-of-sar-imagery-using-encoder-decoder-network-and-fully-connected-crf) -> code for 2020 [paper](https://link.springer.com/chapter/10.1007/978-3-030-39431-8_15): Pixel-Wise Segmentation of SAR Imagery Using Encoder-Decoder Network and Fully-Connected CRF
* [CenterSAR](https://github.com/EdisonLeeeee/CenterSAR) -> CenterNet for SAR ship detection based on Detectron2
* [SAR_Ship_detection_CFAR](https://github.com/Rc-W024/SAR_Ship_detection_CFAR) -> An improved two-parameter CFAR algorithm based on Rayleigh distribution and Mathematical Morphology for SAR ship detection
* [sar_snow_melt_timing](https://github.com/egagli/sar_snow_melt_timing) -> notebooks and tools to identify snowmelt timing using timeseries analysis of backscatter of Sentinel-1 C-band SAR
* [Denoising radar satellite images using deep learning in Python](https://medium.com/@petebch/denoising-radar-satellite-images-using-deep-learning-in-python-946daad31022) -> Medium article on [deepdespeckling](https://github.com/hi-paris/deepdespeckling)
* [random-wetlands](https://github.com/ekcomputer/random-wetlands) -> Random forest classification for wetland vegetation from synthetic aperture radar dataset
* [AGSDNet](https://github.com/RTSIR/AGSDNet) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9755131): AGSDNet: Attention and Gradient-Based SAR Denoising Network
* [LFG-Net](https://github.com/Evarray/LFG-Net) -> code for 2022 [paper](https://ieeexplore.ieee.org/abstract/document/9815311): LFG-Net: Low-Level Feature Guided Network for Precise Ship Instance Segmentation in SAR Images
* [sar_sift](https://github.com/yishiliuhuasheng/sar_sift) -> Image registration algorithm
* [SAR-Despeckling](https://github.com/ImageRestorationToolbox/SAR-Despeckling) -> toolbox
* [cogsima2022](https://github.com/galatolofederico/cogsima2022) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9830661): Enhancing land subsidence awareness via InSAR data and Deep Transformers
* [XAI4SAR-PGIL](https://github.com/Alien9427/XAI4SAR-PGIL) -> code for 2021 [paper](https://arxiv.org/abs/2110.14144): Physically Explainable CNN for SAR Image Classification

## NVDI - vegetation index
* Calculated via band math `ndvi = np.true_divide((ir - r), (ir + r))` but challenging due to the size of the imagery
* [Example notebook local](http://nbviewer.jupyter.org/github/HyperionAnalytics/PyDataNYC2014/blob/master/ndvi_calculation.ipynb)
* [Landsat data in cloud optimised (COG) format analysed for NVDI](https://github.com/pangeo-data/pangeo-example-notebooks/blob/master/landsat8-cog-ndvi.ipynb) with [medium article here](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2).
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”
* [Seeing Through the Clouds - Predicting Vegetation Indices Using SAR](https://medium.com/descarteslabs-team/seeing-through-the-clouds-34a24f84b599)
* [A walkthrough on calculating NDWI water index for flooded areas](https://towardsdatascience.com/how-to-compute-satellite-image-statistics-and-use-it-in-pandas-81864a489144) -> Derive zonal statistics from Sentinel 2 images using Rasterio and Geopandas
* [NDVI-Net](https://github.com/HaoZhang1018/NDVI-Net) -> code for 2020 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0924271620302185): NDVI-Net: A fusion network for generating high-resolution normalized difference vegetation index in remote sensing
* [Awesome-Vegetation-Index](https://github.com/px39n/Awesome-Vegetation-Index)
* [Remote-Sensing-Indices-Derivation-Tool](https://github.com/rander38/Remote-Sensing-Indices-Derivation-Tool) -> Calculate spectral remote sensing indices from satellite imagery

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
* [Optical-RemoteSensing-Image-Resolution](https://github.com/wenjiaXu/Optical-RemoteSensing-Image-Resolution) -> code for 2018 [paper](https://www.mdpi.com/2072-4292/10/12/1893): Deep Memory Connected Neural Network for Optical Remote Sensing Image Restoration. Two applications: Gaussian image denoising and single image super-resolution
* [Hyperspectral-Deblurring-and-Destriping](https://github.com/ImageRestorationToolbox/Hyperspectral-Deblurring-and-Destriping)
* [HyDe](https://github.com/Helmholtz-AI-Energy/HyDe) -> Hyperspectral Denoising algorithm toolbox in Python, with [paper](https://arxiv.org/abs/2204.06979)
* [HLF-DIP](https://github.com/Keiv4n/HLF-DIP) -> code for 2022 [paper](https://ieeexplore.ieee.org/document/9813381): Unsupervised Hyperspectral Denoising Based on Deep Image Prior and Least Favorable Distribution

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
* [Palantir Edge AI in Space](https://blog.palantir.com/edge-ai-in-space-93d793433a1e) -> using NVIDIA Jetson for ship/aircraft/cloud detection & land cover segmentation
* [Spiral Blue](https://spiralblue.space/) -> startup building edge computers to run AI analytics on-board satellites

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
* [Transfer Learning on Greyscale Images: How to Fine-Tune Pretrained Models on Black-and-White Datasets](https://towardsdatascience.com/transfer-learning-on-greyscale-images-how-to-fine-tune-pretrained-models-on-black-and-white-9a5150755c7a)
* [How to create a DataBlock for Multispectral Satellite Image Segmentation with the Fastai](https://towardsdatascience.com/how-to-create-a-datablock-for-multispectral-satellite-image-segmentation-with-the-fastai-v2-bc5e82f4eb5)
* [A comprehensive list of ML and AI acronyms and abbreviations](https://github.com/AgaMiko/machine-learning-acronyms)
* [Finding an optimal number of “K” classes for unsupervised classification on Remote Sensing Data](https://medium.com/@tobyzawthuhtet/finding-an-optimal-number-of-k-classes-for-unsupervised-classification-on-remote-sensing-data-35a5faa0a608) -> i.e 'elbow' method
* Supplement your training data with 'negative' examples which are created through random selection of regions of the image that contain no objects of interest, read [Setting a Foundation for Machine Learning](https://medium.com/the-downlinq/setting-a-foundation-for-machine-learning-datasets-and-labeling-9733ec48a592)
* The law of diminishing returns often applies to dataset size, read [Quantifying the Effects of Resolution on Image Classification Accuracy](https://medium.com/the-downlinq/quantifying-the-effects-of-resolution-on-image-classification-accuracy-7d657aca7701)
* [Implementing Transfer Learning from RGB to Multi-channel Imagery](https://towardsdatascience.com/implementing-transfer-learning-from-rgb-to-multi-channel-imagery-f87924679166) -> Medium article which discusses how to convert a model trained on 3 channels to more channels, adding an additional 12 channels to the original 3 channel RGB image, uses Keras
* [satellite-segmentation-pytorch](https://github.com/obravo7/satellite-segmentation-pytorch) -> explores a wide variety of image augmentations to increase training dataset size
* [Quantifying uncertainty in deep learning systems](https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-quantifying-uncertainty/welcome.html)
* [How to create a custom Dataset / Loader in PyTorch, from Scratch, for multi-band Satellite Images Dataset from Kaggle](https://medium.com/analytics-vidhya/how-to-create-a-custom-dataset-loader-in-pytorch-from-scratch-for-multi-band-satellite-images-c5924e908edf) -> uses the 38-Cloud dataset

# Metrics
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
* [Metrics to Evaluate your Semantic Segmentation Model](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)

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

## iSAID
* https://captain-whu.github.io/iSAID/dataset.html
* A Large-scale Dataset for Instance Segmentation in Aerial Images

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
* [Quick tour of the WorldStrat Dataset](https://robmarkcole.com/markdown/2022/08/01/worldstrat.html) -> blog post by robmarkcole

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
* [Microsoft Planetary Computer](https://innovation.microsoft.com/en-us/planetary-computer) is a Dask-Gateway enabled JupyterHub deployment focused on supporting scalable geospatial analysis, [source repo](https://github.com/microsoft/planetary-computer-hub)
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

# Synthetic data
Training data can be hard to acquire, particularly for rare events such as change detection after disasters, or imagery of rare classes of objects. In these situations, generating synthetic training data might be the only option. This has become quite sophisticated, with 3D models being use with open source games engines such as [Unreal](https://www.unrealengine.com/en-US/). 
* [The Synthinel-1 dataset: a collection of high resolution synthetic overhead imagery for building segmentation](https://arxiv.org/ftp/arxiv/papers/2001/2001.05130.pdf) with [repo](https://github.com/timqqt/Synthinel)
* [RarePlanes](https://www.cosmiqworks.org/RarePlanes/) ->  incorporates both real and synthetically generated satellite imagery including aircraft. Read the [arxiv paper](https://arxiv.org/abs/2006.02963) and checkout [this repo](https://github.com/jdc08161063/RarePlanes). Note the dataset is available through the AWS Open-Data Program for free download
* Read [this article from NVIDIA](https://developer.nvidia.com/blog/preparing-models-for-object-detection-with-real-and-synthetic-data-and-tao-toolkit/) which discusses fine tuning a model pre-trained on synthetic data (Rareplanes) with 10% real data, then pruning the model to reduce its size, before quantizing the model to improve inference speed
* [Combining Synthetic Data with Real Data to Improve Detection Results in Satellite Imagery](https://one-view.ai/combining-synthetic-data-with-real-data-to-improve-detection-results-in-satellite-imagery-case-study/)
* [BlenderGIS](https://github.com/domlysz/BlenderGIS) could be used for synthetic data generation
* [bifrost.ai](https://www.bifrost.ai/) -> simulated data service with geospatial output data formats
* [oktal-se](https://www.oktal-se.fr/deep-learning/) -> software for generating simulated data across a wide range of bands including optical and SAR
* [The Nuances of Extracting Utility from Synthetic Data](https://www.iqt.org/synthesizing-robustness-yoltv4-results-part-1/) -> We find that strategically augmenting the real dataset is nearly as effective as adding synthetic data in the quest to improve the detection or rare object classes, and that fully extracting the utility of synthetic data is a nuanced process
* [Synthesizing Robustness](https://www.iqt.org/synthesizing-robustness/) -> explores how to best leverage and enhance synthetic data
* [rendered.ai](https://rendered.ai/) -> The Platform as a Service for Creating Synthetic Data
* [synthetic_xview_airplanes](https://github.com/yangxu351/synthetic_xview_airplanes) -> creation of airplanes synthetic dataset using ArcGIS CityEngine
* [Combining Synthetic Data with Real Data to Improve Detection Results in Satellite Imagery: Case Study](https://one-view.ai/combining-synthetic-data-with-real-data-to-improve-detection-results-in-satellite-imagery-case-study/)
* [SynImageAnalysis](https://github.com/FlorenceJiang/SynImageAnalysis) -> comparing syn and real sattlelite images in the latent feature space (embeddings)
* [Import OpenStreetMap data into Unreal Engine 4](https://github.com/ue4plugins/StreetMap)
* [deepfake-satellite-images](https://github.com/RijulGupta-DM/deepfake-satellite-images) -> dataset that includes over 1M images of synthetic aerial images
* [synthetic-disaster](https://github.com/JakeForsey/synthetic-disaster) -> Generate synthetic satellite images of natural disasters using deep neural networks
* [STPLS3D](https://github.com/meidachen/STPLS3D) -> A Large-Scale Synthetic and Real Aerial Photogrammetry 3D Point Cloud Dataset
* [LESS](https://github.com/jianboqi/lessrt) -> LargE-Scale remote sensing data and image Simulation framework over heterogeneous 3D scenes
* [Synthesizing Robustness: Dataset Size Requirements and Geographic Insights](https://avanetten.medium.com/synthesizing-robustness-dataset-size-requirements-and-geographic-insights-a687192e8004) -> Medium article, concludes that synthetic data is most beneficial to the rarest object classes and that extracting utility from synthetic data often takes significant effort and creativity
* [rs_img_synth](https://github.com/gbaier/rs_img_synth) -> code for 2020 [paper](https://arxiv.org/abs/2011.11314): Synthesizing Optical and SAR Imagery From Land Cover Maps and Auxiliary Raster Data

# Online platforms for analytics
* [This article discusses some of the available platforms](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2)
* [Pangeo](http://pangeo.io/index.html) -> There is no single software package called “pangeo”; rather, the Pangeo project serves as a coordination point between scientists, software, and computing infrastructure. Includes open source resources for parallel processing using Dask and Xarray. Pangeo recently announced their 2.0 goals: pivoting away from directly operating cloud-based JupyterHubs, and towards eductaion and research
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
* Compute and data storage are on the cloud. Read how [Planet](https://cloud.google.com/customers/planet) and [Airbus](https://cloud.google.com/customers/airbus) use the cloud
* Traditional data formats aren't designed for processing on the cloud, so new standards are evolving such as [COG](https://github.com/robmarkcole/satellite-image-deep-learning#cloud-optimised-geotiff-cog) and [STAC](https://github.com/robmarkcole/satellite-image-deep-learning#spatiotemporal-asset-catalog-specification-stac)
* Google Earth Engine and Microsoft Planetary Computer are democratising access to 'planetary scale' compute
* Google Colab and others are providing free acces to GPU compute to enable training deep learning models
* No-code platforms and auto-ml are making ML techniques more accessible than ever
* Serverless compute (e.g. AWS Lambda) mean that managing servers may become a thing of the past
* Custom hardware is being developed for rapid training and inferencing with deep learning models, both in the datacenter and at the edge
* Supervised ML methods typically require large annotated datasets, but approaches such as self-supervised and active learning require less or even no annotation
* Computer vision traditionally delivered high performance image processing on a CPU by using compiled languages like C++, as used by OpenCV for example. The advent of GPUs are changing the paradigm, with alternatives optimised for GPU being created, such as [Kornia](https://github.com/kornia/kornia)
* Whilst the combo of python and keras/tensorflow/pytorch are currently preeminent, new python libraries such as [Jax](https://github.com/google/jax) and alternative languages such as [Julia](https://julialang.org/) are showing serious promise

# Cloud providers
An overview of the most relevant services provided by AWS, Google and Microsoft. Also consider one of the many smaller but more specialised platorms such as [paperspace](https://www.paperspace.com/)

## AWS
* Host your data on [S3](https://aws.amazon.com/s3/) and metadata in a db such as [postgres](https://aws.amazon.com/rds/postgresql/)
* For batch processing use [Batch](https://aws.amazon.com/batch/). GPU instances are available for [batch deep learning](https://aws.amazon.com/blogs/compute/deep-learning-on-aws-batch/) inferencing. See how Rastervision implement this [here](https://docs.rastervision.io/en/0.13/cloudformation.html)
* If processing can be performed in 15 minutes or less, serverless [Lambda](https://aws.amazon.com/lambda/) functions are an attractive option owing to their ability to scale. Note that lambda may not be a particularly quick solution for deep learning applications, since you do not have the option to batch inference on a GPU. Creating a docker container with all the required dependencies can be a challenge. To get started read [Using container images to run PyTorch models in AWS Lambda](https://aws.amazon.com/blogs/machine-learning/using-container-images-to-run-pytorch-models-in-aws-lambda/) and for an image classification example [checkout this repo](https://github.com/aws-samples/aws-lambda-docker-serverless-inference). Also read [Processing satellite imagery with serverless architecture](https://aws.amazon.com/blogs/compute/processing-satellite-imagery-with-serverless-architecture/) which discusses queuing & lambda. Sagemaker also supports server less inference, see  [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html). For managing a serverless infrastructure composed of multiple lambda functions use [AWS SAM](https://docs.aws.amazon.com/serverless-application-model/index.html) and read [How to continuously deploy a FastAPI to AWS Lambda with AWS SAM](https://iwpnd.pw/articles/2020-01/deploy-fastapi-to-aws-lambda)
* [Sagemaker](https://aws.amazon.com/sagemaker/) is an ecosystem of ML tools accessed via a hosted Jupyter environment & API. Read [Build GAN with PyTorch and Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/build-gan-with-pytorch-and-amazon-sagemaker/), [Run computer vision inference on large videos with Amazon SageMaker asynchronous endpoints](https://aws.amazon.com/blogs/machine-learning/run-computer-vision-inference-on-large-videos-with-amazon-sagemaker-asynchronous-endpoints/)
* [SageMaker Studio Lab](https://studiolab.sagemaker.aws/) competes with Google colab being free to use with no credit card or AWS account required
* [Deep learning AMIs](https://aws.amazon.com/machine-learning/amis/) are EC2 instances with deep learning frameworks preinstalled. They do require more setup from the user than Sagemaker but in return allow access to the underlying hardware, which makes debugging issues more straightforward. There is a [good guide to setting up your AMI instance on the Keras blog](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html). Read [Deploying the SpaceNet 6 Baseline on AWS](https://medium.com/the-downlinq/deploying-the-spacenet-6-baseline-on-aws-c811ad82da1)
* Specifically created for deep learning inferencing is [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/)
* [Rekognition](https://aws.amazon.com/rekognition/custom-labels-features/) custom labels is a 'no code' annotation, training and inferencing service. Read [Training models using Satellite (Sentinel-2) imagery on Amazon Rekognition Custom Labels](https://ryfeus.medium.com/training-models-using-satellite-imagery-on-amazon-rekognition-custom-labels-dd44ac6a3812). For a comparison with Azure and Google alternatives [read this article](https://blog.roboflow.com/automl-vs-rekognition-vs-custom-vision/)
* Use [Glue](https://aws.amazon.com/glue) for data preprocessing - or use Sagemaker
* To orchestrate basic data pipelines use [Step functions](https://aws.amazon.com/step-functions/). Use the [AWS Step Functions Workflow Studio](https://aws.amazon.com/blogs/aws/new-aws-step-functions-workflow-studio-a-low-code-visual-tool-for-building-state-machines/) to get started. Read [Orchestrating and Monitoring Complex, Long-running Workflows Using AWS Step Functions](https://aws.amazon.com/blogs/architecture/field-notes-orchestrating-and-monitoring-complex-long-running-workflows-using-aws-step-functions/) and checkout the [aws-step-functions-data-science-sdk-python](https://github.com/aws/aws-step-functions-data-science-sdk-python)
* If step functions are too limited or you want to write pipelines in python and use Directed Acyclic Graphs (DAGs) for workflow management, checkout hosted [AWS managed Airflow](https://aws.amazon.com/managed-workflows-for-apache-airflow/). Read [Orchestrate XGBoost ML Pipelines with Amazon Managed Workflows for Apache Airflow](https://aws.amazon.com/blogs/machine-learning/orchestrate-xgboost-ml-pipelines-with-amazon-managed-workflows-for-apache-airflow/) and checkout [amazon-mwaa-examples](https://github.com/aws-samples/amazon-mwaa-examples)
* When developing you will definitely want to use [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) and probably [aws-data-wrangler](https://github.com/awslabs/aws-data-wrangler)
* For managing infrastructure use [Terraform](https://www.terraform.io/). Alternatively if you wish to use TypeScript, JavaScript, Python, Java, or C# checkout [AWS CDK](https://aws.amazon.com/cdk/), although I found relatively few examples to get going using python
* [AWS Ground Station now supports data delivery to Amazon S3](https://aws.amazon.com/about-aws/whats-new/2021/04/aws-ground-station-now-supports-data-delivery-to-amazon-s3/)
* [Redshift](https://aws.amazon.com/redshift/) is a fast, scalable data warehouse that can extend queries to S3. Redshift is based on PostgreSQL but [has some differences](https://docs.aws.amazon.com/redshift/latest/dg/c_redshift-and-postgres-sql.html). Redshift supports geospatial data.
* [AWS App Runner](https://aws.amazon.com/blogs/containers/introducing-aws-app-runner/) enables quick deployment of containers as apps
* [AWS Athena](https://aws.amazon.com/athena/) allows running SQL queries against CSV files stored on S3. Serverless so pay only for the queries you run
* If you are using pytorch checkout [the S3 plugin for pytorch](https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/) which provides streaming data access
* [Amazon AppStream 2.0](https://aws.amazon.com/appstream2/) is a service to securely share desktop apps over the internet
* [aws-gdal-robot](https://github.com/mblackgeo/aws-gdal-robot) -> A proof of concept implementation of running GDAL based jobs using AWS S3/Lambda/Batch
* [Building a robust data pipeline for processing Satellite Imagery at scale](https://medium.com/fasal-engineering/building-a-robust-data-pipeline-for-processing-satellite-imagery-at-scale-808700b008cd) using AWS services & Airflow
* [Using artificial intelligence to detect product defects with AWS Step Functions](https://aws.amazon.com/blogs/compute/using-artificial-intelligence-to-detect-product-defects-with-aws-step-functions/) -> demonstrates image classification workflow
* [sagemaker-defect-detection](https://github.com/awslabs/sagemaker-defect-detection) -> demonstrates object detection training and deployment
* [How do you process space data and imagery in low earth orbit?](https://www.aboutamazon.com/news/aws/how-do-you-process-space-data-and-imagery-in-low-earth-orbit) -> Snowcone is a standalone computer that can run AWS services at the edge, and has been demonstraed on the ISS (International space station)
* [Amazon OpenSearch](https://aws.amazon.com/opensearch-service/) -> can be used to create a visual search service
* [Automated Earth observation using AWS Ground Station Amazon S3 data delivery](https://aws.amazon.com/blogs/publicsector/automated-earth-observation-aws-ground-station-amazon-s3-data-delivery/)
* [Satellogic makes Earth observation data more accessible and affordable with AWS](https://aws.amazon.com/blogs/publicsector/satellogic-makes-earth-observation-data-more-accessible-affordable-aws/)
* [Analyze terabyte-scale geospatial datasets with Dask and Jupyter on AWS](https://aws.amazon.com/blogs/publicsector/analyze-terabyte-scale-geospatial-datasets-with-dask-and-jupyter-on-aws/)
* [How SkyWatch built its satellite imagery solution using AWS Lambda and Amazon EFS](https://aws.amazon.com/blogs/storage/how-skywatch-built-its-imagery-solution-using-aws-lambda-and-amazon-efs/)
* [Identify mangrove forests using satellite image features using Amazon SageMaker Studio and Amazon SageMaker Autopilot](https://aws.amazon.com/blogs/machine-learning/part-2-identify-mangrove-forests-using-satellite-image-features-using-amazon-sagemaker-studio-and-amazon-sagemaker-autopilot/)
* [Detecting invasive Australian tree ferns in Hawaiian forests](https://aws.amazon.com/blogs/machine-learning/automated-scalable-and-cost-effective-ml-on-aws-detecting-invasive-australian-tree-ferns-in-hawaiian-forests/)
* [Improve ML developer productivity with Weights & Biases: A computer vision example on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/improve-ml-developer-productivity-with-weights-biases-a-computer-vision-example-on-amazon-sagemaker/)
* [terraform-aws-tile-service](https://github.com/addresscloud/terraform-aws-tile-service) -> Terraform module to create a vector tile service using Amazon API Gateway and S3

## Google Cloud
* For storage use [Cloud Storage](https://cloud.google.com/storage) (AWS S3 equivalent)
* For data warehousing use [BigQuery](https://cloud.google.com/bigquery) (AWS Redshift equivalent). Visualize massive spatial datasets directly in BigQuery using [CARTO](https://carto.com/bigquery-tiler/)
* For model training use [Vertex](https://cloud.google.com/vertex-ai) (AWS Sagemaker equivalent)
* For containerised apps use [Cloud Run](https://cloud.google.com/run) (AWS App Runner equivalent but can scale to zero)

## Microsoft Azure
* [Azure Orbital](https://azure.microsoft.com/en-us/services/orbital/) -> Satellite ground station and scheduling services for fast downlinking of data
* [ShipDetection](https://github.com/microsoft/ShipDetection) -> use the Azure Custom Vision service to train an object detection model that can detect and locate ships in a satellite image
* [SwimmingPoolDetection](https://github.com/retkowsky/SwimmingPoolDetection) -> Swimming pool detection with Azure Custom Vision
* [Geospatial analysis with Azure Synapse Analytics](https://docs.microsoft.com/en-us/azure/architecture/industries/aerospace/geospatial-processing-analytics) and [repo](https://github.com/Azure/Azure-Orbital-Analytics-Samples)
* [AIforEarthDataSets](https://github.com/microsoft/AIforEarthDataSets) -> Notebooks and documentation for [AI-for-Earth](https://www.microsoft.com/en-us/ai/ai-for-earth) managed datasets on Azure

# Deploying models
This section discusses how to get a trained machine learning & specifically deep learning model into production. For an overview on serving deep learning models checkout [Practical-Deep-Learning-on-the-Cloud](https://github.com/PacktPublishing/-Practical-Deep-Learning-on-the-Cloud). There are many options if you are happy to dedicate a server, although you may want a GPU for batch processing. For serverless use AWS lambda.

## Rest API on dedicated server
A common approach to serving up deep learning model inference code is to wrap it in a rest API. The API can be implemented in python (flask or FastAPI), and hosted on a dedicated server e.g. EC2 instance. Note that making this a scalable solution will require significant experience.
* Basic API: https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html with code [here](https://github.com/jrosebr1/simple-keras-rest-api)
* Advanced API with request queuing: https://www.pyimagesearch.com/2018/01/29/scalable-keras-deep-learning-rest-api/
* [How to make a geospatial Rest Api web service with Python, Flask and Shapely - Tutorial](https://hatarilabs.com/ih-en/how-to-make-a-geospatial-rest-api-web-service-with-python-flask-and-shapely-tutorial)
* [BMW-YOLOv4-Training-Automation](https://github.com/BMW-InnovationLab/BMW-YOLOv4-Training-Automation) -> project that demos training ML model via rest API
* [Basic REST API for a keras model using FastAPI](https://github.com/SoySauceNZ/backend)
* [NI4OS-RSSC](https://github.com/risojevicv/NI4OS-RSSC) -> Web Service for Remote Sensing Scene Classification (RS2C) using TensorFlow Serving and Flask
* [Sat2Graph Inference Server](https://github.com/songtaohe/Sat2Graph/tree/master/docker) -> API in Go for road segmentation model inferencing
* [API algorithm to apply object detection model to terabyte size satellite images with 800% better performance and 8 times less resources usage](https://github.com/orhannurkan/API-algorithm-for-terabyte-size-images-)
* [clearcut_detection](https://github.com/QuantuMobileSoftware/clearcut_detection) -> django backend
* [airbus-ship-detection](https://github.com/jancervenka/airbus-ship-detection) -> CNN with REST API

## Framework specific model serving
If you are happy to live with some lock-in, these are good options:
* [Tensorflow serving](https://www.tensorflow.org/tfx/guide/serving) is limited to Tensorflow models
* [Pytorch serve](https://github.com/pytorch/serve) is easy to use, limited to Pytorch models, can be deployed via AWS Sagemaker
* [sagemaker-inference-toolkit](https://github.com/aws/sagemaker-inference-toolkit) -> Serve machine learning models within a Docker container using AWS SageMaker

## Framework agnostic model serving
* The [Triton Inference Server](https://github.com/triton-inference-server/server) provides an optimized cloud and edge inferencing solution. Read [CAPE Analytics Uses Computer Vision to Put Geospatial Data and Risk Information in Hands of Property Insurance Companies](https://blogs.nvidia.com/blog/2021/05/21/cape-analytics-computer-vision/)
* [RedisAI](https://oss.redis.com/redisai/) is a Redis module for executing Deep Learning/Machine Learning models and managing their data

## Using lambda functions - i.e. serverless
Using lambda functions allows inference without having to configure or manage the underlying infrastructure
* On AWS either use regular lambdas from AWS or [SageMaker Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
* [Object detection inference with AWS Lambda and IceVision (PyTorch)](https://laurenzstrothmann.com/object-detection-inference-aws-lambda-icevision) with [repo](https://github.com/2649/laurenzstrothmann)
* [Deploying PyTorch on AWS Lambda](https://segments.ai/blog/pytorch-on-lambda)
* [Example deployment behind an API Gateway Proxy](https://github.com/philschmid/cdk-samples/tree/master/sagemaker-serverless-huggingface-endpoint)

## Models in the browser
The model is run in the browser itself on live images, ensuring processing is always with the latest model available and removing the requirement for dedicated server side inferencing
* [Classifying satellite imagery - Made with TensorFlow.js YoutTube video](https://www.youtube.com/watch?v=9zqjgeqc-ew)

## Model optimisation for deployment
The general approaches are outlined in [this article from NVIDIA](https://developer.nvidia.com/blog/preparing-models-for-object-detection-with-real-and-synthetic-data-and-tao-toolkit/) which discusses fine tuning a model pre-trained on synthetic data (Rareplanes) with 10% real data, then pruning the model to reduce its size, before quantizing the model to improve inference speed. There are also toolkits for optimisation, in particular [ONNX](https://github.com/microsoft/onnxruntime) which is framework agnostic.

## MLOps
[MLOps](https://en.wikipedia.org/wiki/MLOps) is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently.
* [How to Build MLOps Pipelines with GitHub Actions](https://neptune.ai/blog/build-mlops-pipelines-with-github-actions-guide/)

## Model monitoring
Once your model is deployed you will want to monitor for data errors, broken pipelines, and model performance degradation/drift [ref](https://towardsdatascience.com/deploy-and-monitor-your-ml-application-with-flask-and-whylabs-4cd1e757c94b)
* [Blog post by Neptune: Doing ML Model Performance Monitoring The Right Way](https://neptune.ai/blog/ml-model-performance-monitoring)
* [whylogs](https://github.com/whylabs/whylogs) -> Profile and monitor your ML data pipeline end-to-end

# Image annotation
For supervised machine learning, you will require annotated images. For example if you are performing object detection you will need to annotate images with bounding boxes. Check that your annotation tool of choice supports large image (likely geotiff) files, as not all will. Note that GeoJSON is widely used by remote sensing researchers but this annotation format is not commonly supported in general computer vision frameworks, and in practice you may have to convert the annotation format to use the data with your chosen framework. There are both closed and open source tools for creating and converting annotation formats. Some of these tools are simply for performing annotation, whilst others add features such as dataset management and versioning. Note that self-supervised and active learning approaches might circumvent the need to perform a large scale annotation exercise. Note that tiffs/geotiffs cannot be displayed by most browsers (Chrome), but CAN render in Safari.

## Annotation tools with GEO features
Also check the section **Image handling, manipulation & dataset creation**
* [GroundWork](https://groundwork.azavea.com/) is designed for annotating and labeling geospatial data like satellite imagery, from Azavea
* [labelbox.com](https://labelbox.com/) -> free tier is quite generous, supports annotating Geotiffs & returning annotations with geospatial coordinates. Watch [this webcast](https://www.arturo.ai/webcastbuilding-ai-products-from-the-ground-up/)
* [diffgram](https://github.com/diffgram/diffgram) describes itself as a complete training data platform for machine learning delivered as a single application, supports [streaming data to pytorch & tensorflow](https://medium.com/diffgram/stream-training-data-to-your-models-with-diffgram-f0f25f6688c5). [COGS can be annotated](https://diffgram.readme.io/docs/geospatial-annotation-guide)
* [iris](https://github.com/ESA-PhiLab/iris) -> Tool for manual image segmentation and classification of satellite imagery
* If you are considering building an in house annotation platform [read this article](https://medium.com/earthcube-stories/ai-products-and-remote-sensing-yes-it-is-hard-and-yes-you-need-a-good-infra-4b5d6cf822f1). Used PostGis database, GeoJson format and GIS standard in a stateless architecture
* [satellite-imagery-labeling-tool](https://github.com/microsoft/satellite-imagery-labeling-tool) -> from Microsoft, this is a lightweight web-interface for creating and sharing vector annotations over satellite/aerial imagery scenes
* [RSLabel](https://github.com/yonglinZ/RSLabel) -> remote sensing (RS) image annotation tool for deep learning
* [encord](https://encord.com/) -> supports annotatin SAR
* [Flood-Annotation-Tool](https://github.com/saugatadhikari/Flood-Annotation-Tool) -> Annotation tool to annotate flooded and non-flooded regions on satellite image, uses jupyter notebook

## Open source annotation tools
* [awesome-data-labeling](https://github.com/heartexlabs/awesome-data-labeling) -> long list of annotation tools
* [awesome-open-data-annotation](https://github.com/zenml-io/awesome-open-data-annotation) -> another long list of annotation tools
* [labelImg](https://github.com/tzutalin/labelImg) is the classic desktop tool, limited to bounding boxes for object detection. Also checkout [roLabelImg](https://github.com/cgvict/roLabelImg) which supports ROTATED rectangle regions, as often occurs in aerial imagery. [labelImg_OBB](https://github.com/heshameraqi/labelImg_OBB) is another fork supporting orinted bounding boxes (OBB)
* [Labelme](https://github.com/wkentaro/labelme) is a very popular & simple dektop app for polygonal annotation suitable for object detection and semantic segmentation. Note it outputs annotations in a custom LabelMe JSON format which you will need to convert, e.g. using [labelme2coco](https://github.com/fcakyon/labelme2coco). Read [Labelme Image Annotation for Geotiffs](https://medium.com/@wvsharber/labelme-image-annotation-for-geotiffs-b460ba83804f)
* [Label Studio](https://labelstud.io/) is a multi-type data labeling and annotation tool with standardized output format, syncing to buckets, and supports importing pre-annotations (create with a model). Checkout [label-studio-converter](https://github.com/heartexlabs/label-studio-converter) for converting Label Studio annotations into common dataset formats
* [CVAT](https://github.com/cvat-ai/cvat) suports object detection, segmentation and classification via a local web app. [This article on Roboflow](https://blog.roboflow.com/cvat/) gives a good intro to CVAT. Checkout [CVAT images validator](https://github.com/developmentseed/cvat-images-validator)
* [VoTT](https://github.com/Microsoft/VoTT) -> an electron app for building end to end Object Detection Models from Images and Videos, by Microsoft
* Create your own annotation tool using [Bokeh Holoviews](https://examples.pyviz.org/ml_annotators/ml_annotators.html#ml-annotators-gallery-ml-annotators), [tkinter](https://github.com/matpalm/bnn#labelling), or see these dash examples for [object detection](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-image-annotation) and [segmentation](https://github.com/plotly/dash-sample-apps/tree/main/apps/dash-image-segmentation)
* [Deeplabel](https://github.com/jveitchmichaelis/deeplabel) is a cross-platform tool for annotating images with labelled bounding boxes. Deeplabel also supports running inference using state-of-the-art object detection models like Faster-RCNN and YOLOv4. With support out-of-the-box for CUDA, you can quickly label an entire dataset using an existing model.
* [Alturos.ImageAnnotation](https://github.com/AlturosDestinations/Alturos.ImageAnnotation) is a collaborative tool for labeling image data on S3 for yolo
* [pigeonXT](https://github.com/dennisbakhuis/pigeonXT) -> create custom image classification annotators within Jupyter notebooks
* [ipyannotations](https://github.com/janfreyberg/ipyannotations) -> Image annotations in python using Jupyter notebooks
* [Label-Detect](https://github.com/Jakaria08/Label-Detect) -> is a graphical image annotation tool and using this tool a user can also train and test large satellite images, fork of the popular labelImg tool
* [Swipe-Labeler](https://github.com/spaceml-org/Swipe-Labeler) -> Swipe Labeler is a Graphical User Interface based tool that allows rapid labeling of image data
* SuperAnnotate can be run [locally](https://github.com/opencv-ai/superannotate) or used via a [cloud service](https://superannotate.com/)
* [dash_doodler](https://github.com/dbuscombe-usgs/dash_doodler) -> A web application built with plotly/dash for image segmentation with minimal supervision
* [remo](https://remo.ai) -> A webapp and Python library that lets you explore and control your image datasets
* TensorFlow Object Detection API provides a [handy utility](https://github.com/tensorflow/models/blob/6a55ecdea7afda51f9dc42dc17104bd6444395d9/research/object_detection/utils/colab_utils.py#L384) for object annotation within Google Colab notebooks. See usage [here](https://github.com/yasserius/tf2-object-detection-api#label-images-in-colab)
* [coco-annotator](https://github.com/jsbroks/coco-annotator) -> Web-based image segmentation tool for object detection, localization, and keypoints
* [pylabel](https://github.com/pylabel-project/pylabel) -> Python library for computer vision labeling tasks. The core functionality is to translate bounding box annotations between different formats-for example, from coco to yolo. PyLabel also includes an image labeling tool that runs in a Jupyter notebook that can annotate images manually or perform automatic labeling using a pre-trained model
* [BMW-Labeltool-Lite](https://github.com/BMW-InnovationLab/BMW-Labeltool-Lite) -> bounding box annotator
* [django-labeller](https://github.com/Britefury/django-labeller) -> An image labelling tool for creating segmentation data sets, for Django and Flask
* [scalabel](https://github.com/scalabel/scalabel) -> supports 2D images and 3D point clouds
* [Detection-Label-Tool](https://github.com/px39n/Detection-Label-Tool) -> Change detection and object annotation, uses PyQt
* [image_sorter](https://github.com/clcr/image_sorter) -> A quick interface for sorting a folder of images into two other folders

## Cloud hosted & paid annotation tools & services
Several open source tools are also available on the cloud, including CVAT, label-studio & Diffgram. In general cloud solutions will provide a lot of infrastructure and storage for you, as well as integration with outsourced annotators.
* [GroundWork](https://groundwork.azavea.com/) is designed for annotating and labeling geospatial data like satellite imagery, from Azavea
* [labelbox.com](https://labelbox.com/) -> free tier is quite generous, supports annotating Geotiffs & returning annotations with geospatial coordinates. Watch [this webcast](https://www.arturo.ai/webcastbuilding-ai-products-from-the-ground-up/)
* [Roboflow](https://roboflow.com/robincole) -> in addition to annotation this platform makes it easy to convert between annotation formats & manage datasets, as well as train and deploy custom models to private API endpoints. Read [How to Train Computer Vision Models on Aerial Imagery](https://blog.roboflow.com/how-to-use-roboflow-with-aerial-imagery/)
* [supervise.ly](https://supervise.ly) is one of the more fully featured platforms, decent free tier
* AWS supports image annotation via the Rekognition Custom Labels console
* [rectlabel](https://rectlabel.com/) is a desktop app for MacOS to annotate images for bounding box object detection and segmentation, paid and free (rectlabel-lite) versions
* [hasty.ai](https://hasty.ai/) -> supports model assisted annotation & inferencing

## Annotation formats
Note there are many annotation formats, although PASCAL VOC and coco-json are the most commonly used. I recommend using geojson for storing polygons, then converting these to the required format when needed.
* PASCAL VOC format: XML files in the format used by ImageNet
* coco-json format: JSON in the format used by the 2015 COCO dataset
* YOLO Darknet TXT format: contains one text file per image, used by YOLO
* Tensorflow TFRecord: a proprietary binary file format used by the Tensorflow Object Detection API
* Many more formats listed [here](https://roboflow.com/formats)
* OBB: orinted bounding boxes are polygons representing rotated rectangles

## Annotation visualisation & conversion tools
Tools to visualise annotations & convert between formats. Note that most annotation software will allow you to visualise existing annotations
* [Dataset-Converters](https://github.com/ISSResearch/Dataset-Converters) -> a conversion toolset between different object detection and instance segmentation annotation formats
* [FiftyOne](https://github.com/voxel51/fiftyone) -> open-source tool for building high quality datasets and computer vision models. Visualise labels, evaluate model predictions, explore scenarios of interest, identify failure modes, find annotation mistakes, and much more! Read [Nearest Neighbor Embeddings Search with Qdrant and FiftyOne](https://medium.com/voxel51/nearest-neighbor-embeddings-search-with-qdrant-and-fiftyone-adc9aa01b6db)
* [rebox](https://github.com/tensorturtle/rebox) -> Easily convert between bounding box annotation formats
* [Pascal VOC BBox Viewer](https://github.com/zchrissirhcz/imageset-viewer)
* [COCO-Assistant](https://github.com/ashnair1/COCO-Assistant) -> Helper for dealing with MS-COCO annotations; Merge datasets, Remove specfic category from dataset, Generate annotations statistics - distribution of object areas and category distribution
* [pybboxes](https://github.com/devrimcavusoglu/pybboxes) -> Light weight toolkit for bounding boxes providing conversion between bounding box types and simple computations
* [voc2coco](https://github.com/yukkyo/voc2coco) -> Convert VOC format XMLs to COCO format json
* [ObjectDetectionEval](https://github.com/laclouis5/ObjectDetectionEval) -> Parse all kinds of object detection databases (ImageNet, COCO, YOLO, PascalVOC, OpenImage, CVAT, LabelMe, etc.) & save to other formats
* [LabelMeYoloConverter](https://github.com/ivder/LabelMeYoloConverter) -> Convert LabelMe Annotation Tool JSON format to YOLO text file format
* [mask-to-polygons](https://github.com/azavea/mask-to-polygons) -> Routines for extracting and working with polygons from semantic segmentation masks
* [labelme2coco](https://github.com/fcakyon/labelme2coco) -> Converts LabelMe JSON format into COCO object detection and instance segmentation format.

# Open source software
By software, I here mean desktop type apps. [A note on licensing](https://www.gislounge.com/businesses-using-open-source-gis/): The two general types of licenses for open source are copyleft and permissive. Copyleft requires that subsequent derived software products also carry the license forward, e.g. the GNU Public License (GNU GPLv3). For permissive, options to modify and use the code as one please are more open, e.g. MIT & Apache 2. Checkout [choosealicense.com/](https://choosealicense.com/)
* [awesome-earthobservation-code](https://github.com/acgeospatial/awesome-earthobservation-code) -> lists many useful tools and resources
* [Orfeo toolbox](https://www.orfeo-toolbox.org/) - remote sensing toolbox with python API (just a wrapper to the C code). Do activites such as [pansharpening](https://www.orfeo-toolbox.org/CookBook/Applications/app_Pansharpening.html), ortho-rectification, image registration, image segmentation & classification. Not much documentation.
* [QUICK TERRAIN READER - view DEMS, Windows](http://appliedimagery.com/download/)
* [dl-satellite-docker](https://github.com/sshuair/dl-satellite-docker) -> docker files for geospatial analysis, including tensorflow, pytorch, gdal, xgboost...
* [AIDE V2 - Tools for detecting wildlife in aerial images using active learning](https://github.com/microsoft/aerial_wildlife_detection)
* [Land Cover Mapping web app from Microsoft](https://github.com/microsoft/landcover)
* [Solaris](https://github.com/CosmiQ/solaris) -> An open source ML pipeline for overhead imagery by [CosmiQ Works](https://www.cosmiqworks.org/), similar to Rastervision but with some unique very vool features
* [openSAR](https://github.com/EarthBigData/openSAR) -> Synthetic Aperture Radar (SAR) Tools and Documents from Earth Big Data LLC
* [YMIR](https://github.com/industryessentials/ymir) -> YMIR provides a Rapid Data-centric Development Platform
for Vision Applications. Read the paper [here](https://arxiv.org/abs/2111.10046).
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
* [SatDump](https://github.com/altillimity/SatDump) -> A generic satellite data processing software.

## General utilities
Scripts and command line applications
* [geospatial-cli](https://github.com/JakobMiksch/geospatial-cli) -> a collection of geospatial programs with commandline interface
* [PyShp](https://github.com/GeospatialPython/pyshp) -> The Python Shapefile Library (PyShp) reads and writes Shapefiles in pure Python
* [s2p](https://github.com/cmla/s2p) -> a Python library and command line tool that implements a stereo pipeline which produces elevation models from images taken by high resolution optical satellites such as Pléiades, WorldView, QuickBird, Spot or Ikonos
* [EarthPy](https://github.com/earthlab/earthpy) -> A set of helper functions to make working with spatial data in open source tools easier. read[Exploratory Data Analysis (EDA) on Satellite Imagery Using EarthPy](https://towardsdatascience.com/exploratory-data-analysis-eda-on-satellite-imagery-using-earthpy-c0e186fe4293)
* [pygeometa](https://geopython.github.io/pygeometa/) -> provides a lightweight and Pythonic approach for users to easily create geospatial metadata in standards-based formats using simple configuration files
* [pesto](https://airbusdefenceandspace.github.io/pesto/) -> PESTO is designed to ease the process of packaging a Python algorithm as a processing web service into a docker image. It contains shell tools to generate all the boiler plate to build an OpenAPI processing web service compliant with the Geoprocessing-API. By [Airbus Defence And Space](https://github.com/AirbusDefenceAndSpace)
* [GEOS](https://geos.readthedocs.io/en/latest/index.html) -> Google Earth Overlay Server (GEOS) is a python-based server for creating Google Earth overlays of tiled maps. Your can also display maps in the web browser, measure distances and print maps as high-quality PDF’s.
* [GeoDjango](https://docs.djangoproject.com/en/3.1/ref/contrib/gis/) intends to be a world-class geographic Web framework. Its goal is to make it as easy as possible to build GIS Web applications and harness the power of spatially enabled data. [Some features of GDAL are supported.](https://docs.djangoproject.com/en/3.1/ref/contrib/gis/gdal/)
* [rasterstats](https://pythonhosted.org/rasterstats/) -> summarize geospatial raster datasets based on vector geometries
* [turfpy](https://turfpy.readthedocs.io/en/latest/index.html) -> a Python library for performing geospatial data analysis which reimplements turf.js
* [rsgislib](https://github.com/remotesensinginfo/rsgislib) -> Remote Sensing and GIS Software Library; python module tools for processing spatial and image data
* [eo-learn](https://eo-learn.readthedocs.io/en/latest/index.html) -> seamlessly access and process spatio-temporal image sequences acquired by any satellite fleet in a timely and automatic manner. See [eo-learn-examples](https://github.com/sentinel-hub/eo-learn-examples)
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
* [buzzard](https://github.com/earthcube-lab/buzzard) -> Advanced raster and geometry manipulations
* [sentinel1denoised](https://github.com/nansencenter/sentinel1denoised) -> Thermal noise subtraction, scalloping correction, angular correction
* [RStoolbox](https://github.com/bleutner/RStoolbox) -> Remote Sensing Data Analysis in R
* [kart](https://github.com/koordinates/kart) -> Distributed version-control for geospatial and tabular data
* [picogeojson](https://github.com/fortyninemaps/picogeojson) -> a Python library for reading, writing, and working with GeoJSON
* [shareloc](https://github.com/CNES/shareloc) -> a simple remote sensing geometric library, to perform image coordinates projections between sensor and ground and vice versa
* [geoblaze](https://github.com/GeoTIFF/geoblaze) -> Blazing Fast JavaScript Raster Processing Engine
* [nasa-wildfires](https://github.com/datadesk/nasa-wildfires) -> Download wildfire hotspots detected by NASA satellites and the Fire Information for Resource Management System (FIRMS)
* [SSGP-toolbox](https://github.com/Dreamlone/SSGP-toolbox) -> Simple Spatial Gapfilling Processor. Toolbox for filling gaps in spatial datasets
* [imgreg2D](https://github.com/BrancoLab/imgreg2D) -> 2D image registration in python, using napari
* [georust](https://github.com/georust) -> A collection of geospatial tools and libraries written in Rust
* [DataPillager](https://github.com/gdherbert/DataPillager) -> Download data from Esri REST service
* [litexplore](https://github.com/litements/litexplore) -> a Python web app that lets you explore remote SQLite databases over SSH connections
* [tifeatures](https://github.com/developmentseed/tifeatures) -> Simple and Fast Geospatial Features API for PostGIS
* [pyroSAR](https://github.com/johntruckenbrodt/pyroSAR) -> framework for large-scale SAR satellite data processing
* [S1_NRB](https://github.com/SAR-ARD/S1_NRB) -> A prototype processor for the Sentinel-1 Normalised Radar Backscatter product
* [AGBench](https://github.com/gyrrei/AGBench) -> a Python library that benchmarks satellite-based aboveground biomass or carbon estimate maps
* [mbtiles-s3-server](https://github.com/uktrade/mbtiles-s3-server) -> Python server to on-the-fly extract and serve vector tiles from an mbtiles file on S3
* [matico](https://github.com/Matico-Platform/matico) -> a set of tools and services that allow users to manage geospatial datasets, build APIs that use those datasets and full geospatial applications with little to no code
* [gmtsar](https://github.com/mobigroup/gmtsar) -> easy and fast satellite interferometry (InSAR) processing

## Low level numerical & data formats
* [xarray](http://xarray.pydata.org/en/stable/) -> N-D labeled arrays and datasets. Read [Handling multi-temporal satellite images with Xarray](https://medium.com/@bonnefond.virginie/handling-multi-temporal-satellite-images-with-xarray-30d142d3391). Checkout [xarray_leaflet](https://github.com/davidbrochart/xarray_leaflet) for tiled map plotting and [sklearn-xarray](https://github.com/phausamann/sklearn-xarray) for metadata-aware machine learning. Publish Xarray Datasets via a REST API uisng [xpublish](https://github.com/xarray-contrib/xpublish)
* [wxee](https://github.com/aazuspan/wxee) -> Export data from GEE to xarray using wxee then train with pytorch or tensorflow models. Useful since GEE only suports tfrecord export natively
* [xarray-spatial](https://github.com/makepath/xarray-spatial) -> Fast, Accurate Python library for Raster Operations. Implements algorithms using Numba and Dask, free of GDAL
* [xarray-beam](https://github.com/google/xarray-beam) -> Distributed Xarray with Apache Beam by Google
* [Geowombat](https://geowombat.readthedocs.io/) -> geo-utilities applied to air and space-borne imagery, uses Rasterio, Xarray and Dask for I/O and distributed computing with named coordinates. [Create Land Use Classification using Geowombat & Sklearn](https://pygis.io/docs/f_rs_ml_predict.html)
* [NumpyTiles](https://github.com/planetlabs/numpytiles-spec) -> a specification for providing multiband full-bit depth raster data in the browser
* [Zarr](https://zarr.readthedocs.io/en/stable/) -> Zarr is a format for the storage of chunked, compressed, N-dimensional arrays. Zarr depends on NumPy
* [geoparquet](https://github.com/opengeospatial/geoparquet) -> Specification for storing geospatial vector data (point, line, polygon) in Parquet
* [TFRecord reader for PyTorch](https://github.com/vahidk/tfrecord)

## Image processing, handling, manipulation
* [Pillow is the Python Imaging Library](https://pillow.readthedocs.io/en/stable/) -> this will be your go-to package for image manipulation in python
* [opencv-python](https://github.com/opencv/opencv-python) is pre-built CPU-only OpenCV packages for Python
* [kornia](https://github.com/kornia/kornia) is a differentiable computer vision library for PyTorch, like openCV but on the GPU. Perform image transformations, epipolar geometry, depth estimation, and low-level image processing such as filtering and edge detection that operate directly on tensors.
* [tifffile](https://github.com/cgohlke/tifffile) -> Read and write TIFF files
* [xtiff](https://github.com/BodenmillerGroup/xtiff) -> A small Python 3 library for writing multi-channel TIFF stacks
* [geotiff](https://github.com/Open-Source-Agriculture/geotiff) -> A noGDAL tool for reading and writing geotiff files
* [geolabel-maker](https://github.com/makinacorpus/geolabel-maker) -> combine satellite or aerial imagery with vector spatial data to create your own ground-truth dataset in the COCO format for deep-learning models
* [imagehash](https://github.com/JohannesBuchner/imagehash) -> Image hashes tell whether two images look nearly identical
* [fake-geo-images](https://github.com/up42/fake-geo-images) -> A module to programmatically create geotiff images which can be used for unit tests
* [imagededup](https://github.com/idealo/imagededup) -> Finding duplicate images made easy! Uses perceptual hashing
* [duplicate-img-detection](https://github.com/mattpodolak/duplicate-img-detection) -> A basic duplicate image detection service using perceptual image hash functions and nearest neighbor search, implemented using faiss, fastapi, and imagehash
* [rmstripes](https://github.com/DHI-GRAS/rmstripes) -> Remove stripes from images with a combined wavelet/FFT approach
* [activeloopai Hub](https://github.com/activeloopai/hub) -> The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow. Works locally or on any cloud. Scalable data pipelines.
* [sewar](https://github.com/andrewekhalel/sewar) -> All image quality metrics you need in one package
* [Satellite imagery label tool](https://github.com/calebrob6/labeling-tool) -> provides an easy way to collect a random sample of labels over a given scene of satellite imagery
* [Missing-Pixel-Filler](https://github.com/spaceml-org/Missing-Pixel-Filler) -> given images that may contain missing data regions (like satellite imagery with swath gaps), returns these images with the regions filled
* [color_range_filter](https://github.com/developmentseed/color_range_filter) -> a script that allows us to find range of colors in images using openCV, and then convert them into geo vectors
* [eo4ai](https://github.com/ESA-PhiLab/eo4ai) -> easy-to-use tools for preprocessing datasets for image segmentation tasks in Earth Observation
* [rasterix](https://github.com/mogasw/rasterix) -> a cross-platform utility built around the GDAL library and the Qt framework designed to process geospatial raster data
* [datumaro](https://github.com/openvinotoolkit/datumaro) -> Dataset Management Framework, a Python library and a CLI tool to build, analyze and manage Computer Vision datasets
* [sentinelPot](https://github.com/LLeiSong/sentinelPot) -> a python package to preprocess Sentinel 1&2 imagery
* [ImageAnalysis](https://github.com/UASLab/ImageAnalysis) -> Aerial imagery analysis, processing, and presentation scripts.
* [rastertodataframe](https://github.com/mblackgeo/rastertodataframe) -> Convert any GDAL compatible raster to a Pandas DataFrame
* [yeoda](https://github.com/TUW-GEO/yeoda) -> provides lower and higher-level data cube classes to work with well-defined and structured earth observation data
* [tiles-to-tiff](https://github.com/jimutt/tiles-to-tiff) -> Python script for converting XYZ raster tiles for slippy maps to a georeferenced TIFF image
* [telluric](https://github.com/satellogic/telluric) -> a Python library to manage vector and raster geospatial data in an interactive and easy way
* [Sniffer](https://github.com/2320sharon/Sniffer) -> A python application for sorting through geospatial imagery
* [pyjeo](https://github.com/ec-jrc/jeolib-pyjeo) -> a library for image processing for geospatial data implemented in JRC Ispra, with [paper](https://www.mdpi.com/2220-9964/8/10/461)
* [vpv](https://github.com/kidanger/vpv) -> Image viewer designed for image processing experts
* [arop](https://github.com/george-silva/arop) -> Automated Registration and Orthorectification Package
* [satellite_image](https://github.com/dgketchum/satellite_image) -> Python package to process images from Landsat satellites and return geographic information, cloud mask, numpy array, geotiff
* [large_image](https://github.com/girder/large_image) -> Python modules to work with large multiresolution images
* [ResizeRight](https://github.com/assafshocher/ResizeRight) -> The correct way to resize images or tensors. For Numpy or Pytorch (differentiable)
* [pysat](https://github.com/pysat/pysat) -> a package providing a simple and flexible interface for downloading, loading, cleaning, managing, processing, and analyzing scientific measurements
* [plcompositor](https://github.com/planetlabs/plcompositor) -> c++ tool from Planet to create seamless and cloudless image mosaics from deep stacks of satellite imagery

## Image chipping/tiling & merging
Since raw images can be very large, it is usually necessary to chip/tile them into smaller images before annotation & training
* [image_slicer](https://github.com/samdobson/image_slicer) -> Split images into tiles. Join the tiles back together
* [tiler by nuno-faria](https://github.com/nuno-faria/tiler) -> split images into tiles and merge tiles into a large image
* [tiler by the-lay](https://github.com/the-lay/tiler) -> N-dimensional NumPy array tiling and merging with overlapping, padding and tapering
* [xbatcher](https://github.com/pangeo-data/xbatcher) -> Xbatcher is a small library for iterating xarray DataArrays in batches. The goal is to make it easy to feed xarray datasets to machine learning libraries such as Keras
* [GeoTagged_ImageChip](https://github.com/Hejarshahabi/GeoTagged_ImageChip) -> A simple script to create geo tagged image chips from high resolution RS iamges for training deep learning models such as Unet
* [geotiff-crop-dataset](https://github.com/tayden/geotiff-crop-dataset) -> A Pytorch Dataloader for tif image files that dynamically crops the image
* [Train-Test-Validation-Dataset-Generation](https://github.com/salarghaffarian/Train-Test-Validation-Dataset-Generation) ->  app to crop images and create small patches of a large image e.g. Satellite/Aerial Images, which will then be used for training and testing Deep Learning models specifically semantic segmentation models
* [satproc](https://github.com/dymaxionlabs/satproc) -> Python library and CLI tools for processing geospatial imagery for ML
* [Sliding Window](https://github.com/adamrehn/slidingwindow) ->  break large images into a series of smaller chunks
* [patchify](https://github.com/dovahcrow/patchify.py) -> A library that helps you split image into small, overlappable patches, and merge patches into original image
* [split-rs-data](https://github.com/Youssef-Harby/split-rs-data) -> Divide remote sensing images and their labels into data sets of specified size
* [image-reconstructor-patches](https://github.com/marijavella/image-reconstructor-patches) -> Reconstruct Image from Patches with a Variable Stride
* [rpc_cropper](https://github.com/carlodef/rpc_cropper) -> A small standalone tool to crop satellite images and their RPC
* [geotile](https://github.com/iamtekson/geotile) -> python library for tiling the geographic raster data
* [GeoPatch](https://github.com/Hejarshahabi/GeoPatch) -> generating patches from remote sensing data
* [ImageTilingUtils](https://github.com/vfdev-5/ImageTilingUtils) -> Minimalistic set of image reader agnostic tools to easily iterate over large images
* [split_raster](https://github.com/cuicaihao/split_raster) -> Creates a tiled output from an input raster dataset. pip installable
* [SAHI](https://github.com/obss/sahi) -> Utilties for slicing COCO formatted annotations and image files, performing sliced inference using MMDetection, Detectron2, YOLOv5, HuggingFace detectors and calculating AP over image slices.

## Image dataset creation
Many datasets on kaggle & elsewhere have been created by screen-clipping Google Maps or browsing web portals. The tools below are to create datasets programatically
* [MapTilesDownloader](https://github.com/AliFlux/MapTilesDownloader) -> A super easy to use map tiles downloader built using Python
* [jimutmap](https://github.com/Jimut123/jimutmap) -> get enormous amount of high resolution satellite images from apple / google maps quickly through multi-threading
* [google-maps-downloader](https://github.com/yildirimcagatay34/google-maps-downloader) -> A short python script that downloads satellite imagery from Google Maps
* [ExtractSatelliteImagesFromCSV](https://github.com/thewati/ExtractSatelliteImagesFromCSV) -> extract satellite images using a CSV file that contains latitude and longitude, uses mapbox
* [sentinelsat](https://github.com/sentinelsat/sentinelsat) -> Search and download Copernicus Sentinel satellite images
* [SentinelDownloader](https://github.com/cordmaur/SentinelDownloader) -> a high level wrapper to the SentinelSat that provides an object oriented interface, asynchronous downloading, quickview & simpler searching methods
* [GEES2Downloader](https://github.com/cordmaur/GEES2Downloader) -> Downloader for GEE S2 bands
* [Sentinel-2 satellite tiles images downloader from Copernicus](https://github.com/flaviostutz/sentinelloader) -> Minimizes data download and combines multiple tiles to return a single area of interest
* [felicette](https://github.com/plant99/felicette) -> Satellite imagery for dummies. Generate JPEG earth imagery from coordinates/location name with publicly available satellite data
* [Easy Landsat Download](https://github.com/dgketchum/Landsat578)
* [A simple python scrapper to get satellite images of Africa, Europe and Oceania's weather using the Sat24 website](https://github.com/luistripa/sat24-image-scrapper)
* [RGISTools](https://github.com/spatialstatisticsupna/RGISTools) -> Tools for Downloading, Customizing, and Processing Time Series of Satellite Images from Landsat, MODIS, and Sentinel
* [DeepSatData](https://github.com/michaeltrs/DeepSatData) -> Automatically create machine learning datasets from satellite images
* [landsat_ingestor](https://github.com/landsat-pds/landsat_ingestor) -> Scripts and other artifacts for landsat data ingestion into Amazon public hosting
* [satpy](https://github.com/pytroll/satpy) -> a python library for reading and manipulating meteorological remote sensing data and writing it to various image and data file formats
* [GIBS-Downloader](https://github.com/spaceml-org/GIBS-Downloader) -> a command-line tool which facilitates the downloading of NASA satellite imagery and offers different functionalities in order to prepare the images for training in a machine learning pipeline
* [eodag](https://github.com/CS-SI/eodag) -> Earth Observation Data Access Gateway
* [pylandsat](https://github.com/yannforget/pylandsat) -> Search, download, and preprocess Landsat imagery
* [landsatxplore](https://github.com/yannforget/landsatxplore) -> Search and download Landsat scenes from EarthExplorer
* [OpenSarToolkit](https://github.com/ESA-PhiLab/OpenSarToolkit) -> High-level functionality for the inventory, download and pre-processing of Sentinel-1 data in the python language
* [lsru](https://github.com/loicdtx/lsru) -> Query and Order Landsat Surface Reflectance data via ESPA
* [eoreader](https://github.com/sertit/eoreader) -> Remote-sensing opensource python library reading optical and SAR sensors, loading and stacking bands, clouds, DEM and index in a sensor-agnostic way
* [Export thumbnails from Earth Engine](https://gorelick.medium.com/fast-er-downloads-a2abd512aa26)
* [deepsentinel-osm](https://github.com/Lkruitwagen/deepsentinel-osm) -> A repository to generate land cover labels from OpenStreetMap
* [img2dataset](https://github.com/rom1504/img2dataset) -> Easily turn large sets of image urls to an image dataset. Can download, resize and package 100M urls in 20h on one machine
* [ohsome2label](https://github.com/GIScience/ohsome2label) -> Historical OpenStreetMap (OSM) Objects to Machine Learning Training Samples
* [Label Maker](https://github.com/developmentseed/label-maker) -> a library for creating machine-learning ready data by pairing satellite images with OpenStreetMap (OSM) vector data. [Example usage with dask using Planetary Computer](https://github.com/microsoft/PlanetaryComputerExamples/blob/main/tutorials/label-maker-dask.ipynb)
* [sentinel2tools](https://github.com/QuantuMobileSoftware/sentinel2tools) -> downloading & basic processing of Sentinel 2 imagesry. Read [Sentinel2tools: simple lib for downloading Sentinel-2 satellite images](https://medium.com/geekculture/sentinel2tools-simple-lib-for-downloading-sentinel-2-satellite-images-f8a6be3ee894)
* [Aerial-Satellite-Imagery-Retrieval](https://github.com/chiragkhandhar/Aerial-Satellite-Imagery-Retrieval) -> A program using Bing maps tile system to automatically download Aerial / Satellite Imagery given a lat/lon bounding box and level of detail
* [google-maps-at-88-mph](https://github.com/doersino/google-maps-at-88-mph) -> Google Maps keeps old satellite imagery around for a while – this tool collects what's available for a user-specified region in the form of a GIF
* [srtmDownloader](https://github.com/Abdi-Ghasem/srtmDownloader) -> Python library (multi-threaded) for retrieving SRTM elevation map of CGIAR-CSI
* [ImageDatasetViz](https://github.com/vfdev-5/ImageDatasetViz) -> create a mosaic of images in a dataset for previewing purposes
* [landsatlinks](https://github.com/ernstste/landsatlinks) -> A simple CLI interface to generate download urls for Landsat Collection 2 Level 1 product bundles
* [pyeo](https://github.com/clcr/pyeo) -> a set of portable, extensible and modular Python scripts for machine learning in earth observation and GIS, including downloading, preprocessing, creation of base layers, classification and validation.
* [metaearth](https://github.com/bair-climate-initiative/metaearth) -> Download and access remote sensing data from any platform

## Image augmentation packages
Image augmentation is a technique used to expand a training dataset in order to improve ability of the model to generalise
* [AugLy](https://github.com/facebookresearch/AugLy) -> A data augmentations library for audio, image, text, and video. By Facebook
* [albumentations](https://github.com/albumentations-team/albumentations) -> Fast image augmentation library and an easy-to-use wrapper around other libraries
* [FoHIS](https://github.com/noahzn/FoHIS) -> Towards Simulating Foggy and Hazy Images and Evaluating their Authenticity
* [Kornia](https://kornia.readthedocs.io/en/latest/augmentation.html) provides augmentation on the GPU
* [toolbox by ming71](https://github.com/ming71/toolbox) -> various cv tools, such as label tools, data augmentation, label conversion, etc.
* [AstroAugmentations](https://github.com/mb010/AstroAugmentations) -> augmentations designed around astronomical instruments
* [Chessmix](https://github.com/matheusbarrosp/chessmix) -> data augmentation method for remote sensing semantic segmentation
* [satellite_object_augmentation](https://github.com/LanaLana/satellite_object_augmentation) -> Object-based augmentation for remote sensing images segmentation via CNN
* [hypernet](https://github.com/ESA-PhiLab/hypernet) -> hyperspectral data augmentation

## Image formats, data management and catalogues
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
* [fastdup](https://github.com/visualdatabase/fastdup) -> a tool for gaining insights from a large image collection. It can find anomalies, duplicate and near duplicate images
* [Nucleus](https://dashboard.scale.com/nucleus/) is a platform for image dataset management with advanced features including [autotagging](https://nucleus.scale.com/docs/introduction-to-autotag) and finding [instances with mismatched predictions & annotations](https://nucleus.scale.com/docs/find-inaccurate-predictions)

# Deep learning packages, frameworks & projects
* [TorchGeo](https://github.com/microsoft/torchgeo) -> a PyTorch domain library providing datasets, samplers, transforms, and pre-trained models specific to geospatial data, supported by Microsoft. Read [Geospatial deep learning with TorchGeo](https://pytorch.org/blog/geospatial-deep-learning-with-torchgeo/)
* [rastervision](https://docs.rastervision.io/) -> An open source Python framework for building computer vision models on aerial, satellite, and other large imagery sets
* [torchrs](https://github.com/isaaccorley/torchrs) -> PyTorch implementation of popular datasets and models in remote sensing tasksenhance) -> Enhance PyTorch vision for semantic segmentation, multi-channel images and TIF file
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
* [SAHI](https://github.com/obss/sahi) -> Python library for slicing image datasets, performing sliced inference with MMDetection, Detectron2, YOLOv5, Torchvision detectors and generating error analysis plots. Read the [arxiv paper](https://arxiv.org/abs/2202.06934) and article [SAHI: A vision library for large-scale object detection & instance segmentation](https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80)
* [terragpu](https://github.com/nasa-cisto-ai/terragpu) -> Python library to process and classify remote sensing imagery by means of GPUs and AI/ML
* [EOTorchLoader](https://github.com/ndavid/EOTorchLoader) -> Pytorch dataloader and pytorch lightning datamodule for Earth Observation imagery
* [satellighte](https://github.com/canturan10/satellighte) -> an image classification library that consist state-of-the-art deep learning methods, using PyTorch Lightning
* [aeronetlib](https://github.com/Geoalert/aeronetlib) -> Python library to work with geospatial raster and vector data for deep learning
* [rsi-semantic-segmentation](https://github.com/xdu-jjgs/rsi-semantic-segmentation) -> A unified PyTorch framework for semantic segmentation from remote sensing imagery
* [AiTLAS](https://github.com/biasvariancelabs/aitlas) -> implements state-of-the-art AI methods for exploratory and predictive analysis of satellite images
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) -> Semantic Segmentation Toolbox with support for many remote sensing datasets including LoveDA
, Potsdam, Vaihingen & iSAID
* [ODEON landcover](https://github.com/IGNF/odeon-landcover) -> a set of command-line tools performing semantic segmentation on remote sensing images (aerial and/or satellite) with as many layers as you wish
* [aitlas-arena](https://github.com/biasvariancelabs/aitlas-arena) -> An open-source benchmark framework for evaluating state-of-the-art deep learning approaches for image classification in Earth Observation (EO)
* [PaddleRS](https://github.com/PaddlePaddle/PaddleRS) -> remote sensing image processing development kit

## Model tracking, versioning, specification & compilation
* [dvc](https://dvc.org/) -> a git extension to keep track of changes in data, source code, and ML models together
* [Weights and Biases](https://wandb.ai/) -> keep track of your ML projects. Log hyperparameters and output metrics from your runs, then visualize and compare results and quickly share findings with your colleagues
* [geo-ml-model-catalog](https://github.com/radiantearth/geo-ml-model-catalog) -> provides a common metadata definition for ML models that operate on geospatial data
* [hummingbird](https://github.com/microsoft/hummingbird) ->  a library for compiling trained traditional ML models into tensor computations, e.g. scikit learn model to pytorch for fast inference on a GPU
* [deepchecks](https://github.com/deepchecks/deepchecks) -> Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort
* [pachyderm](https://www.pachyderm.com/) -> Data Versioning and Pipelines for MLOps. Read [Pachyderm + Label Studio](https://medium.com/pachyderm-data/pachyderm-label-studio-ecc09f1f9329) which discusses versioning and lineage of data annotations

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
* [Embedding Projector in Wandb](https://docs.wandb.ai/ref/app/features/panels/weave/embedding-projector) -> allows users to plot multi-dimensional embeddings on a 2D plane using common dimension reduction algorithms like PCA, UMAP, and t-SNE
* [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) -> Latex code for making neural networks diagrams
* [Damage Assessment Visualizer](https://github.com/microsoft/Nonprofits/tree/master/Damage%20Assessment%20Visualizer) -> leverages satellite imagery from a disaster region to visualize conditions of building and structures before and after a disaster
* [NN-SVG](https://github.com/alexlenail/NN-SVG) -> is a tool for creating Neural Network (NN) architecture drawings parametrically rather than manually
* [bbox-visualizer](https://github.com/shoumikchow/bbox-visualizer) -> Make drawing and labeling bounding boxes easy as cake
* [jupyter-bbox-widget](https://github.com/gereleth/jupyter-bbox-widget) -> A Jupyter widget for annotating images with bounding boxes
* [EOmaps](https://github.com/raphaelquast/EOmaps) -> A library to create interactive maps of geographical datasets
* [H3-Pandas](https://github.com/DahnJ/H3-Pandas) -> Integrates H3 with GeoPandas and Pandas
* [gmplot](https://github.com/gmplot/gmplot) -> a matplotlib-like interface to render all the data you'd like on top of Google Maps
* [NPYViewer](https://github.com/csmailis/NPYViewer) ->  a simple GUI tool that provides multiple ways to view `.npy` files containing 2D NumPy Arrays
* [pyGEOVis](https://github.com/geoyee/pyGEOVis) -> Visualize geo-tiff/json based on folium
* [bokeh-tiler](https://github.com/avanetten/bokeh-tiler) -> Tile large geospatial images for use in Bokeh. Read [Serving up SpaceNet Imagery for Bokeh](https://medium.com/geodesic/serving-up-spacenet-imagery-for-bokeh-e85b8fffe05)
* [torchshow](https://github.com/xwying/torchshow) -> Visualize PyTorch tensor in one-line of code
* [pixels](https://github.com/jwasilgeo/pixels) -> Mapping and charting pixels from remote sensing Earth observation data with JavaScript
* [MulimgViewer](https://github.com/nachifur/MulimgViewer) -> a multi-image viewer that can open multiple images in one interface
* [cnn-explainer](https://github.com/poloclub/cnn-explainer) -> Learning Convolutional Neural Networks with Interactive Visualization
* [Overlay-GeoTiff-Raster-with-nodata-On-Interactive-Map](https://github.com/royalosyin/Overlay-GeoTiff-Raster-with-nodata-On-Interactive-Map)
* [shapefile2gif](https://github.com/johannesuhl/shapefile2gif) -> Given a shapefile with time-annotated vector objects (e.g., building footprints + construction year), this script will automatically create an animated GIF illustrating the dynamics for a user-specified period of time
* [insat3d_imagen](https://github.com/rupeshs/insat3d_imagen) -> Processes INSAT HDF file and generates satellite images
* [pygieons](https://github.com/pygieons/pygieons) -> A simple package to visualize and keep track of GIS and Earth Observation libraries in Python
* [regionmask](https://github.com/regionmask/regionmask) -> Create masks of geographical regions for arbitrary longitude and latitude grids
* [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/)
* [SAHI](https://github.com/obss/sahi) -> Creates [error analysis plots](https://github.com/obss/sahi/issues/356) for your object detector and [interactively visualizes your predictions](https://github.com/obss/sahi/issues/357)

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
* [sarsen](https://github.com/bopen/sarsen) -> Algorithms and utilities for Synthetic Aperture Radar (SAR) sensors
* [sun-position](https://github.com/s-bear/sun-position) -> code for computing sun position
* [simple_ortho](https://github.com/dugalh/simple_ortho) -> Fast and simple orthorectification of images with known DEM and camera model
* [imageResolution](https://github.com/geojames/imageResolution) -> Simple spatial resolution calculator for nadir & oblique aerial imagery
* [Spectral-Clustering](https://github.com/zhangyk8/Spectral-Clustering) -> normalized and unnormalized spectral clustering algorithms
* [Fogpy](https://github.com/pytroll/fogpy) -> nowcasting of fog and low stratus clouds
* [orthorectification](https://github.com/mpfaffenberger/orthorectification) -> Orthorectification in Python. Note that all of this functionality already exists in libraries like GDAL and others. The goal of this codebase was to present and deep dive into these subroutines
* [Flood-Severity-Estimation](https://github.com/jorgemspereira/Flood-Severity-Estimation) -> estimate the height of the water in geo-referenced photos that depict floods using DEMs from JAXA
* [coastline-extraction](https://github.com/Ricardo-C-Oliveira/coastline-extraction) -> Methods to identify and extract coastline from remote sensed data
* [Near real-time shadow detection and removal in remote sensing imagery application](https://github.com/BIT-zhwang/remote-sensing-image-shadow-detection-and-removal)
* [image-registration](https://github.com/satish1901/image-registration) -> using Point Feature Detection, Normalized DLT, RANSAC & Image Warping
* [pyTSEB](https://github.com/hectornieto/pyTSEB) -> A python Two Source Energy Balance model for estimation of evapotranspiration with remote sensing data
* [libpredict](https://github.com/la1k/libpredict) -> satellite orbit prediction library
* [GOTCHA](https://github.com/jveitchmichaelis/gotcha) -> Command line implementation of the GOTCHA stereo matching algorithm
* [SREM](https://github.com/oyam/srem) -> A Simplified and Robust Surface Reflectance Estimation Method for Satellite Imagery
* [kaizen](https://github.com/fuzailpalnak/kaizen) -> A library to map match and help tackle the problem of overlapping/intersecting road and building footprint that arises in the process of map making
* [CoastSat.PlanetScope](https://github.com/ydoherty/CoastSat.PlanetScope) -> Batch shoreline extraction toolkit for PlanetScope Dove satellite imagery

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
* [gdal2tiles](https://github.com/tehamalab/gdal2tiles) -> A python library for generating map tiles based on gdal2tiles.py from GDAL project
* [gdal3.js](https://github.com/bugra9/gdal3.js) -> Convert raster and vector geospatial data to various formats and coordinate systems entirely in the browser

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
* [stac-fastapi-elasticsearch](https://github.com/stac-utils/stac-fastapi-elasticsearch) -> Elasticsearch backend for stac-fastapi
* [ml-aoi](https://github.com/stac-extensions/ml-aoi) -> An Item and Collection extension to provide labeled training data for machine learning models
* Discoverable and Reusable ML Workflows for Earth Observation -> [part 1](https://medium.com/radiant-earth-insights/discoverable-and-reusable-ml-workflows-for-earth-observation-part-1-e198507b5eaa) and [part 2](https://medium.com/radiant-earth-insights/discoverable-and-reusable-ml-workflows-for-earth-observation-part-2-ebe2b4812d5a) with the Geospatial Machine Learning Model Catalog (GMLMC)
* [eoAPI](https://github.com/developmentseed/eoAPI) -> Earth Observation API with STAC + dynamic Raster/Vector Tiler
* [stac-nb](https://github.com/darrenwiens/stac-nb) -> STAC in Jupyter Notebooks
* [xstac](https://github.com/TomAugspurger/xstac) -> Generate STAC Collections from xarray datasets
* [qgis-stac-plugin](https://github.com/stac-utils/qgis-stac-plugin) -> QGIS plugin for reading STAC APIs
* [cirrus-geo](https://github.com/cirrus-geo/cirrus-geo) -> a STAC-based processing pipeline
* [stac-interactive-search](https://github.com/calebrob6/stac-interactive-search) -> A simple (browser based) UI for searching STAC APIs
* [easystac](https://github.com/cloudsen12/easystac) -> A Python package for simple STAC queries
* [stacmap](https://github.com/aazuspan/stacmap) -> Explore STAC items with an interactive map
* [odc-stac](https://github.com/opendatacube/odc-stac) -> Load STAC items into xarray Datasets. Process locally or distribute data loading and computation with Dask.
* [AWS Lambda SenCloud Monitoring](https://github.com/ahuarte47/aws-sencloud-monitoring) -> keep up-to-date your own derived data from the Sentinel-2 COG imagery archive using AWS lambda
* [stac-geoparquet](https://github.com/TomAugspurger/stac-geoparquet) -> Convert STAC items to geoparquet

## OpenStreetMap
[OpenStreetMap](https://www.openstreetmap.org/) (OSM) is a map of the world, created by people like you and free to use under an open license. Quite a few publications use OSM data for annotations & ground truth. Note that the data is created by volunteers and the quality can be variable
* [osmnx](https://github.com/gboeing/osmnx) -> Retrieve, model, analyze, and visualize data from OpenStreetMap
* [ohsome2label](https://github.com/GIScience/ohsome2label) -> Historical OpenStreetMap Objects to Machine Learning Training Samples
* [Label Maker](https://github.com/developmentseed/label-maker) -> downloads OpenStreetMap QA Tile information and satellite imagery tiles and saves them as an `.npz` file for use in machine learning training. This should be used instead of the deprecated [skynet-data](https://github.com/developmentseed/skynet-data)
* [prettymaps](https://github.com/marceloprates/prettymaps) -> A small set of Python functions to draw pretty maps from OpenStreetMap data
* [Joint Learning from Earth Observation and OpenStreetMap Data to Get Faster Better Semantic Maps](https://arxiv.org/abs/1705.06057) -> fusion based architectures and coarse-to-fine segmentation to include the OpenStreetMap layer into multispectral-based deep fully convolutional networks, arxiv paper
* [Identifying Buildings in Satellite Images with Machine Learning and Quilt](https://github.com/jyamaoka/LandUse) -> NDVI & edge detection via gaussian blur as features, fed to TPOT for training with labels from OpenStreetMap, modelled as a two class problem, “Buildings” and “Nature”
* [Import OpenStreetMap data into Unreal Engine 4](https://github.com/ue4plugins/StreetMap)
* [OSMDeepOD](https://github.com/geometalab/OSMDeepOD) ->  perform object detection with retinanet
* [Match Bing Map Aerial Imagery with OpenStreetMap roads](https://github.com/whywww/Aerial-Imagery-and-OpenStreetMap-Retrieval)
* [Computer Vision With OpenStreetMap and SpaceNet — A Comparison](https://medium.com/the-downlinq/computer-vision-with-openstreetmap-and-spacenet-a-comparison-cc70353d0ace)
* [url-map](https://simonwillison.net/2022/Jun/12/url-map/) -> A tiny web app to create images from OpenStreetMap maps
* [Label Maker](https://github.com/developmentseed/label-maker) -> a library for creating machine-learning ready data by pairing satellite images with OpenStreetMap (OSM) vector data. [Example usage with dask using Planetary Computer](https://github.com/microsoft/PlanetaryComputerExamples/blob/main/tutorials/label-maker-dask.ipynb)
* [baremaps](https://github.com/baremaps/baremaps) -> Create custom vector tiles from OpenStreetMap and other data sources with Postgis and Java.

## QGIS
A popular open source alternative to ArcGIS, QGIS is a desktop appication written in python and extended with plugins which are essentially python scripts
* [QGIS](https://qgis.org/en/site/)
* Create, edit, visualise, analyse and publish geospatial information. Open source alternative to ArcGIS.
* [Python scripting](https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/intro.html#scripting-in-the-python-console)
* Create your own plugins using the [QGIS Plugin Builder](http://g-sherman.github.io/Qgis-Plugin-Builder/)
* [DeepLearningTools plugin](https://plugins.qgis.org/plugins/DeepLearningTools/) -> aid training Deep Learning Models
* [Mapflow.ai plugin](https://www.gislounge.com/run-ai-mapping-in-qgis-over-high-resolution-satellite-imagery/) -> various models to extract building footprints etc from Maxar imagery
* [dzetsaka plugin](https://github.com/nkarasiak/dzetsaka) -> classify different kind of vegetation
* [Coregistration-Qgis-processing](https://github.com/SMByC/Coregistration-Qgis-processing) -> Qgis processing plugin for image co-registration; projection and pixel alignment based on a target image, uses Arosics
* [qgis-stac-plugin](https://github.com/stac-utils/qgis-stac-plugin) -> QGIS plugin for reading STAC APIs
* [buildseg](https://github.com/deepbands/buildseg) -> a building extraction plugin of QGIS based on ONNX
* [deep-learning-datasets-maker](https://github.com/deepbands/deep-learning-datasets-maker) -> a QGIS plugin to make datasets creation easier for raster and vector data
* [Modzy-QGIS-Plugin](https://github.com/modzy/Modzy-QGIS-Plugin) -> demos Vehicle Detection model
* [kart](https://plugins.qgis.org/plugins/kart/) -> provides modern, open source, distributed version-control for geospatial and tabular datasets
* [Plugin for Landcover Classification](https://github.com/atishayjn/QGIS-Plugin) -> capable of implementing machine learning algorithms such as Random forest, SVM and CNN algorithms such as UNET through a simple GUI framework.
* [pg_tileserv])(https://github.com/CrunchyData/pg_tileserv) -> A very thin PostGIS-only tile server in Go. Takes in HTTP tile requests, executes SQL, returns MVT tiles.
* [pg_featureserv](https://github.com/CrunchyData/pg_featureserv) -> Lightweight RESTful Geospatial Feature Server for PostGIS in Go
* [osm-instance-segmentation](https://github.com/mnboos/osm-instance-segmentation) -> QGIS plugin for finding changes in vector data from orthophotos (i.e. aerial imagery) using tensorflow
* [Semi-Automatic Classification Plugin](https://github.com/semiautomaticgit/SemiAutomaticClassificationPlugin) -> supervised classification of remote sensing images, providing tools for the download, the preprocessing and postprocessing of images

## Parallel procesing with Dask
Dask provides advanced parallelism and distributed out-of-core computation with a `dask.dataframe` module designed to scale pandas.
* [Dask](https://docs.dask.org/en/latest/) works with your favorite PyData libraries to provide performance at scale for the tools you love
* [Coiled](https://coiled.io) is a managed Dask service. Get started by reading [Democratizing Satellite Imagery Analysis with Dask](https://coiled.io/blog/democratizing-satellite-imagery-analysis-with-dask/)
* [Dask with PyTorch for large scale image analysis](https://blog.dask.org/2021/03/29/apply-pretrained-pytorch-model)
* [dask-geopandas](https://github.com/geopandas/dask-geopandas) -> offers geospatial capabilities of GeoPandas backed by Dask
* [stackstac](https://github.com/gjoseph92/stackstac) -> Turn a STAC catalog into a dask-based xarray
* [dask-geomodeling](https://github.com/nens/dask-geomodeling) -> On-the-fly operations on geographical maps
* [dask-image](https://github.com/dask/dask-image) -> many SciPy ndimage functions implemented
* [Detecting Green Roofs in Toronto](https://toarches.medium.com/geospatial-big-data-processing-with-python-detecting-green-roofs-in-toronto-bd7bf08900f2) -> compares deep learning (Mask R-CNN & fast.ai) and classical approach using NDVI scaled on Dask
* [Analyze terabyte-scale geospatial datasets with Dask and Jupyter on AWS](https://aws.amazon.com/blogs/publicsector/analyze-terabyte-scale-geospatial-datasets-with-dask-and-jupyter-on-aws/)
* [austin-ml-change-detection-demo](https://github.com/makepath/austin-ml-change-detection-demo) -> A change detection demo for the Austin area using a pre-trained PyTorch model scaled with Dask on Planet imagery

## Web apps
Flask is often used to serve up a simple web app based on templated HTML files
* [FastMap](https://github.com/butlerbt/FastMap) -> Flask deployment of deep learning model performing segmentation task on aerial imagery building footprints
* [Querying Postgres with Python Fastapi Backend and Leaflet-Geoman Frontend](https://geo.rocks/post/leaflet-geoman-fastapi-postgis/)
* [cropcircles](https://github.com/doersino/cropcircles) -> a purely-client-side web app originally designed for accurately cropping circular center pivot irrigation fields from aerial imagery
* [django-large-image](https://github.com/ResonantGeoData/django-large-image) -> Django endpoints for working with large images for tile serving
* [Earth Classification API](https://github.com/conlamon/satellite-classification-flask-api) -> Flask based app that serves a CNN model and interfaces with a React and Leaflet front-end
* [Demo flask map app](https://github.com/kdmayer/flask_tutorial) -> Building Python-based, database-driven web applications (with maps!) using Flask, SQLite, SQLAlchemy and MapBox
* [Building a Web App for Instance Segmentation using Docker, Flask and Detectron2](https://towardsdatascience.com/instance-segmentation-web-app-63016b8ed4ae)
* [greppo](https://github.com/greppo-io/greppo) -> Build & deploy geospatial applications quick and easy. Read [Build a geospatial dashboard in Python using Greppo](https://towardsdatascience.com/build-a-geospatial-dashboard-in-python-using-greppo-60aff44ba6c9)
* [localtileserver](https://github.com/banesullivan/localtileserver) -> image tile server for viewing geospatial rasters with ipyleaflet, folium, or CesiumJS locally in Jupyter or remotely in Flask applications. Checkout [bokeh-tiler](https://github.com/avanetten/bokeh-tiler) 
* [flask-geocoding-webapp](https://github.com/mblackgeo/flask-geocoding-webapp) -> A quick example Flask application for geocoding and rendering a webmap using Folium/Leaflet
* [flask-vector-tiles](https://github.com/mblackgeo/flask-vector-tiles) -> A simple Flask/leaflet based webapp for rendering vector tiles from PostGIS
* [Crash Severity Prediction](https://github.com/SoySauceNZ/web-app) -> using CAS Open Data and Maxar Satellite Imagery, React app
* [wildfire-detection-from-satellite-images-ml](https://github.com/shrey24/wildfire-detection-from-satellite-images-ml) -> simple flask app for classification
* [SlumMappingViaRemoteSensingImagery](https://github.com/hamna-moieez/SlumMappingViaRemoteSensingImagery) -> learning slum segmentation and localization using satellite imagery and visualising on a flask app
* [cloud-removal-deploy](https://github.com/XavierJiezou/cloud-removal-deploy) -> flask app for cloud removal
* [clearcut_detection](https://github.com/QuantuMobileSoftware/clearcut_detection) -> research & web-service for clearcut detection

## Jupyter
The [Jupyter](https://jupyter.org/) Notebook is a web-based interactive computing platform. There are many extensions which make it a powerful environment for analysing satellite imagery
* [jupyterlite](https://jupyterlite.readthedocs.io/en/latest/) -> JupyterLite is a JupyterLab distribution that runs entirely in the browser
* [jupyter_compare_view](https://github.com/Octoframes/jupyter_compare_view) -> Blend Between Multiple Images
* [folium](https://python-visualization.github.io/folium/quickstart.html) -> display interactive maps in Jupyter notebooks
* [ipyannotations](https://github.com/janfreyberg/ipyannotations) -> Image annotations in python using jupyter notebooks
* [pigeonXT](https://github.com/dennisbakhuis/pigeonXT) -> create custom image classification annotators within Jupyter notebooks
* [jupyter-innotater](https://github.com/ideonate/jupyter-innotater) -> Inline data annotator for Jupyter notebooks
* [jupyter-bbox-widget](https://github.com/gereleth/jupyter-bbox-widget) -> A Jupyter widget for annotating images with bounding boxes
* [mapboxgl-jupyter](https://github.com/mapbox/mapboxgl-jupyter) -> Use Mapbox GL JS to visualize data in a Python Jupyter notebook
* [pylabel](https://github.com/pylabel-project/pylabel) -> includes an image labeling tool that runs in a Jupyter notebook that can annotate images manually or perform automatic labeling using a pre-trained model
* [jupyterlab-s3-browser](https://github.com/IBM/jupyterlab-s3-browser) -> extension for browsing S3-compatible object storage
* [papermill](https://github.com/nteract/papermill) -> Parameterize, execute, and analyze notebooks
* [pretty-jupyter](https://github.com/JanPalasek/pretty-jupyter) -> Creates dynamic html report from jupyter notebook

## Streamlit
[Streamlit](https://streamlit.io/) is an awesome python framework for creating apps with python. Additionally they will host the apps free of charge. Here I list resources which are EO related. Note that a component is an addon which extends Streamlits basic functionality
* [cogviewer](https://github.com/mykolakozyr/cogviewer) -> Simple Cloud Optimized GeoTIFF viewer
* [cogcreator](https://github.com/mykolakozyr/cogcreator) -> Simple Cloud Optimized GeoTIFF Creator. Generates COG from GeoTIFF files.
* [cogvalidator](https://github.com/mykolakozyr/cogvalidator) -> Simple Cloud Optimized GeoTIFF validator
* [streamlit-image-comparison](https://github.com/fcakyon/streamlit-image-comparison) -> compare images with a slider. Used in [example-app-image-comparison](https://github.com/streamlit/example-app-image-comparison)
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
* [Streamlit-Authenticator](https://github.com/mkhorasani/Streamlit-Authenticator) -> A secure authentication module to validate user credentials in a Streamlit application
* [prettymapp](https://github.com/chrieke/prettymapp) -> Create beautiful maps from OpenStreetMap data in a webapp
* [mapa-streamlit](https://github.com/fgebhart/mapa-streamlit) -> creating 3D-printable models of the earth surface based on mapa
* [BoulderAreaDetector](https://github.com/pszemraj/BoulderAreaDetector) -> CNN to classify whether a satellite image shows an area would be a good rock climbing spot or not, deployed to streamlit app
* [streamlit-remotetileserver](https://github.com/banesullivan/streamlit-remotetileserver) -> Easily visualize a remote raster given a URL and check if it is a valid Cloud Optimized GeoTiff (COG)
* [Streamlit_Image_Sorter](https://github.com/2320sharon/Streamlit_Image_Sorter) -> Generic Image Sorter Interface for Streamlit
* [Streamlit-Folium + Snowflake + OpenStreetMap](https://github.com/randyzwitch/streamlit-folium-snowflake-openstreetmap) -> demonstrates the power of Snowflake Geospatial data types and queries combined with Streamlit
* [observing-earth-from-space-with-streamlit](https://blog.streamlit.io/observing-earth-from-space-with-streamlit/) -> blog post on the [SatSchool](https://github.com/Spiruel/SatSchool) app
* [vector-validator](https://github.com/chrieke/vector-validator) -> Webapp that validates and automatically fixes your geospatial vector data

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
* [Rasters.jl](https://github.com/rafaqz/Rasters.jl) -> types and methods for reading, writing and manipulating rasterized spatial data including GeoTIFF and NetCDF
* [RemoteS.jl](https://github.com/GenericMappingTools/RemoteS.jl) -> Remote sensing data processing
* [SatelliteToolbox.jl](https://github.com/JuliaSpace/SatelliteToolbox.jl) -> This package contains several functions to build simulations related with satellites
* [SatelliteDynamics.jl](https://github.com/sisl/SatelliteDynamics.jl) -> a satellite dynamics modeling package
* [Sentinel.jl](https://github.com/mhudecheck/Sentinel.jl) -> library for processing ESA Sentinel 2 satellite data
* [DeepSat-Kaggle](https://github.com/athulsudheesh/DeepSat-Kaggle) -> uses Julia

# Movers and shakers on Github
* [Adam Van Etten](https://github.com/avanetten) is doing interesting things in object detection and segmentation
* [Adeel Hassan](https://github.com/adeelh) is ML engineer at Azavea
* [Alastair Graham](https://github.com/ajggeoger) of Scene From Above podcast fame
* [Andrew Cutts](https://github.com/acgeospatial) of Scene From Above podcast fame
* [Ankit Kariryaa](https://github.com/ankitkariryaa) published a recent nature paper on tree detection
* [Chris Holmes](https://github.com/cholmes) is doing great things at Planet
* [Christoph Rieke](https://github.com/chrieke) maintains a very popular imagery repo and has published his thesis on segmentation
* [Daniel J Dufour](https://github.com/DanielJDufour) builds [geotiff.io](https://geotiff.io/) and more
* [Daniel Moraite](https://daniel-moraite.medium.com/) is publishing some excellent articles
* [Even Rouault](https://github.com/rouault) maintains several of the most critical tools in this domain such as GDAL
* [Fatih Cagatay Akyon](https://github.com/fcakyon) aka fcakyon is maintaining SAHI and many other useful projects
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
* [Rohit Singh](https://github.com/rohitgeo) has some great [Medium articles](https://medium.com/@rohitgeo)

# Companies & organisations on Github
For a full list of companies, on and off Github, checkout [awesome-geospatial-companies](https://github.com/chrieke/awesome-geospatial-companies). The following lists companies with interesting Github profiles
* [Airbus Defence And Space](https://github.com/AirbusDefenceAndSpace)
* [Agricultural Impacts Research Group](https://github.com/agroimpacts)
* [Astraea](https://github.com/s22s)
* [Applied-GeoSolutions](https://github.com/Applied-GeoSolutions)
* [Azavea](https://github.com/azavea) -> lots of interesting repos around STAC
* [CARTO](https://github.com/CartoDB) -> "The leading platform for Location Intelligence and Spatial Data Science"
* [Cervest](https://github.com/Cervest) -> Climate Intelligence
* [Citymapper](https://github.com/citymapper)
* [Defense Innovation Unit (DIU)](https://diu.mil/) -> run the xView challenges
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
* [HSG-AIML](https://github.com/HSG-AIML) -> Artificial Intelligence & Machine Learning (AI:ML Lab) at HSG
* [Hummingbird Technologies Ltd](https://github.com/HummingbirdTechGroup) -> sustainability and optimised food production
* [ICEYE](https://github.com/iceye-ltd)
* [indigo-ag](https://github.com/indigo-ag)
* [L3Harris Geospatial Solutions (HGSI)](https://github.com/hgsolutions)
* [Mapbox](https://github.com/mapbox) -> thanks for Rasterio!
* [Maxar-Analytics](https://github.com/maxar-analytics)
* [ml6team](https://github.com/ml6team)
* [NASA](https://github.com/nasa)
* [NASA-JPL](https://github.com/nasa-jpl) -> THE Jet Propulsion Laboratory
* [NASA Harvest Mission](https://github.com/nasaharvest) -> NASA’s Food Security and Agriculture Program
* [National Geospatial-Intelligence Agency USA](https://github.com/ngageoint)
* [Near Space Labs](https://github.com/nearspacelabs)
* [Open Business Software Solutions](https://github.com/obss)
* [OpenGeoScales](https://github.com/OpenGeoScales)
* [Open Geospatial Consortium](https://github.com/opengeospatial) -> the [OGC](https://www.ogc.org/)
* [Ordnance Survey](https://github.com/OrdnanceSurvey)
* [OroraTech](https://github.com/OroraTech)
* [Planet Labs](https://github.com/planetlabs) -> thanks for COGS!
* [Preligens](https://github.com/earthcube-lab) -> formerly Earthcube Lab
* [pyronear](https://github.com/pyronear) -> Preserving forests from wildfires one commit at a time
* [SatelliteVu](https://github.com/SatelliteVu) -> thermal imagery from space!
* [Sinergise](https://github.com/sentinel-hub) -> maintaining Sentinel-hub
* [SIAnalytics](https://github.com/SIAnalytics) -> Korean AI firm
* [SkyTruth](https://github.com/SkyTruth)
* [SpaceKnow](https://github.com/SpaceKnow)
* [Sparkgeo](https://github.com/sparkgeo)
* [Tracasa](https://github.com/tracasa) -> created SEN2ROAD
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
* [Machine Learning on Earth Observation: ML4EO Bootcamp](https://online.atingi.org/course/view.php?id=1107%27)
* [Artificial Intelligence (AI) for Earth Monitoring](https://www.futurelearn.com/courses/artificial-intelligence-for-earth-monitoring)
* [Deep Learning DIY](https://dataflowr.github.io/website/)
* [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html)
* [Practical Data Science Specialization](https://www.coursera.org/specializations/practical-data-science) -> AWS specific, develop and scale your data science projects into the cloud using Amazon SageMaker
* [Disaster Risk Monitoring Using Satellite Imagery by NVIDIA](https://courses.nvidia.com/courses/course-v1:DLI+S-ES-01+V1/)
* [Course materials for: Geospatial Data Science](https://github.com/mszell/geospatialdatascience)
* [Data-Science-For-Beginners](https://github.com/microsoft/Data-Science-For-Beginners) -> by Microsoft
* [AI-For-Beginners](https://github.com/microsoft/AI-For-Beginners) -> by Microsoft
* [Remote Sensing Tutorials](https://www.nrcan.gc.ca/maps-tools-and-publications/satellite-imagery-and-air-photos/tutorial-fundamentals-remote-sensing/9309) -> by the Canada Centre for Mapping and Earth Observation
* [RUS Copernicus Training](https://www.youtube.com/channel/UCB01WjameYMvL7-XfI8vRIA)
* [Practical Deep Learning for Coders](https://course.fast.ai/) -> the popular course by Jeremy Howard using fast.ai
* [engn3903](https://github.com/nicolasyounes/engn3903) -> Environmental Sensing, Mapping and Modelling
* [PyTorch Fundamentals](https://docs.microsoft.com/en-us/learn/paths/pytorch-fundamentals/) -> 4 hour course by Microsoft

# Books
* I highly recommend [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) by François Chollet
* [Image Analysis, Classification and Change Detection in Remote Sensing With Algorithms for Python, Fourth Edition, By Morton John Canty](https://www.routledge.com/Image-Analysis-Classification-and-Change-Detection-in-Remote-Sensing-With/Canty/p/book/9781138613225) -> code [here](https://github.com/mortcanty/CRC4Docker)
* [Practical Deep Learning for Cloud, Mobile & Edge](https://github.com/PracticalDL/Practical-Deep-Learning-Book)
* [d2l.ai](https://www.d2l.ai/) -> Interactive deep learning book with code, math, and discussions
* [interpretable-ml-book](https://github.com/christophM/interpretable-ml-book)
* [pyGIS](https://github.com/mmann1123/pyGIS) -> an online textbook covering all the core geospatial functionality available in Python
* [Deep Learning Interviews book](https://github.com/BoltzmannEntropy/interviews.ai)
* [Geographic Data Science with Python](https://geographicdata.science/book/intro.html) -> data science applied to geographic problems and data
* [eBook: Introduction to Datascience with Julia](https://datascience-book.gitlab.io/)
* [Deep Learning In Production Book](https://github.com/The-AI-Summer/Deep-Learning-In-Production)
* [Land Use Cover Datasets and Validation Tools](https://link.springer.com/book/10.1007/978-3-030-90998-7)
* [Intermediate Python](https://book.pythontips.com/en/latest/)

# Bedtime reading
* [Meditations: A Requiem for Descartes Labs](https://philosophygeek.medium.com/meditations-a-requiem-for-descartes-labs-8b913b5e898)
* [Why We Haven’t Taken on Investment by Robert Cheetham at Azavea](https://www.azavea.com/blog/2022/05/03/why-we-havent-taken-on-investment/)
* [State of AI for Earth Observation](https://sa.catapult.org.uk/digital-library/white-paper-state-of-ai-for-earth-observation/) -> White paper by catapult.org.uk

# Podcasts
* [Scene From Above Podcast](https://scenefromabove.podbean.com/)
* [Mapscaping podcast](https://mapscaping.com/blogs/the-mapscaping-podcast)
* [Minds Behind Maps](https://minds-behind-maps.simplecast.com/)
* [Terrawatch Space](https://anchor.fm/terrawatch-space)
* [Geomob](https://thegeomob.com/)
* [Project Geospatial](https://podcasts.apple.com/us/podcast/project-geospatial/id1486384184)
* [Eyes on Earth by USGS](https://www.usgs.gov/centers/eros/eyes-earth)
* [Zenml episode: Satellite Vision with Robin Cole](https://podcast.zenml.io/satellite-vision-robin-cole)

# Newsletters
* [Radiant Earth ML4EO market news](https://www.radiant.earth/category/ml4eo-market-news/)
* [A Closer Look with Joe Morrison](https://joemorrison.substack.com/) -> examines the business and technology of mapping
* [TerraWatch Space by Aravind](https://terrawatch.substack.com/)
* [Payload Space](https://payloadspace.com/)
* [Geoscience and Remote Sensing eNewsletter from grss-ieee](https://www.grss-ieee.org/publications/geoscience-and-remote-sensing-enewsletter/)
* [Weekly Remote Sensing and Geosciences news by Rafaela Tiengo](https://www.getrevue.co/profile/rafaelatiengo)

# Online communities
* [fast AI geospatial study group](https://forums.fast.ai/t/geospatial-deep-learning-resources-study-group/31044)
* [Kaggle Intro to Satellite imagery Analysis group](https://www.kaggle.com/getting-started/131455)
* [Omdena](https://omdena.com/) brings together small teams of engineers to work on AI projects

# Jobs
Signup for the [geospatial-jobs-newsletter](https://geospatial.substack.com/p/welcome-to-geospatial-jobs-newsletter) and [Pangeo discourse](https://discourse.pangeo.io/c/news/jobs) lists multiple jobs, global. List of job portals below:
* [Capella Space](https://apply.workable.com/capellaspace/)
* [Development SEED](http://devseed.com/careers)
* [Earthdaily](https://earthdaily.applytojob.com/apply)
* [EO-jobs](https://github.com/DahnJ/EO-jobs) -> List of earth observation companies and job sites
* [Planet](https://boards.greenhouse.io/planetlabs)
* [Satellite Vu](https://www.satellitevu.com/careers)
* [Sparkgeo](https://sparkgeo.com/jobs/)

----
- *Logo created with* [*Brandmark*](https://app.brandmark.io/v3/)
