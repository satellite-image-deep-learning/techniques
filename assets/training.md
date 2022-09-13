# Training models
This section discusses training machine learning models.

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

## Others
* [Paperspace gradient](https://gradient.run/notebooks) -> free tier includes GPU usage
* [Deepnote](https://deepnote.com/) -> many features for collaboration, GPU use is paid