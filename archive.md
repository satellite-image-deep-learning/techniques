# Neural nets in space
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
