## Pangeo
* Pangeo == A community platform for Big Data geoscience
* https://github.com/pangeo-data/pangeo
* [Read this post](https://medium.com/pangeo/cloud-native-geoprocessing-of-earth-observation-satellite-data-with-pangeo-997692d91ca2)
* Pangeo is a environment made up of many different open-source software packages. Pangeo is run on cloud infrastructure (typically GCP) in order to enable parallel processing of geospatial and climate data via [Xarray and Dask](http://xarray.pydata.org/en/latest/dask.html). Kubernetes is used to distribute analysis across many machines ('workers'). In the default configuration for Pangeo Binder, each worker has 2 vCPUs and 7Gb of RAM. As we are free of RAM constraints, there is no need to do analysis on downsampled data.
* Modern python stack of rasterio, xarray etc, unlike many of the other cloud platforms. [See the Dockerfile for a complete list](https://github.com/pangeo-data/helm-chart/blob/master/docker-images/notebook/Dockerfile).
* [Use case gallery](http://pangeo.io/use_cases/index.html#use-cases)
* [Try on Binder](http://binder.pangeo.io/v2/gh/pangeo-data/pangeo-example-notebooks/master) -> screenshot below. Note this is launching the [pangeo-example-notebooks](https://github.com/pangeo-data/pangeo-example-notebooks) repo but you could launch your own repo.
* There are several places where you can access a Pangeo deployment -> most require you to be a member of the Pangeo organisation http://pangeo.io/deployments.html#deployments


<p align="center">
<img src="https://github.com/robmarkcole/satellite-image-deep-learning/blob/master/data/images/pangeo_binder.png" width="1000">
</p>
