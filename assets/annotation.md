# Annotation
For supervised machine learning, you will require annotated images. For example if you are performing object detection you will need to annotate images with bounding boxes. Check that your annotation tool of choice supports large image (likely geotiff) files, as not all will. Note that GeoJSON is widely used by remote sensing researchers but this annotation format is not commonly supported in general computer vision frameworks, and in practice you may have to convert the annotation format to use the data with your chosen framework. There are both closed and open source tools for creating and converting annotation formats. Some of these tools are simply for performing annotation, whilst others add features such as dataset management and versioning. Note that self-supervised and active learning approaches might circumvent the need to perform a large scale annotation exercise. Note that tiffs/geotiffs cannot be displayed by most browsers (Chrome), but CAN render in Safari.

## Annotation tools with GEO features
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
