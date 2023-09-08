# Research
General repo for the code developed as part of my personal research work.

Some notes:

* For now, `pyins` seems to sufficiently meet my needs. It has a standard legacy error state Kalman Filter and does the typical navigation corrections. It still feels a little under documented and clunky, so I might modify it on my fork.

* I kind of have this crazy idea of replicating Matlab's [Navigation Toolbox](https://www.mathworks.com/products/navigation.html) though that would be a big project. 

    * There's already several libraries out there with some functionalities: 
        * [`pyins`](https://github.com/nmayorov/pyins), a basic INS system and simulator
        * [`gnss-ins-sim`](https://github.com/Aceinna/gnss-ins-sim)), a more complex GNSS/INS system and simulator; [Article](https://medium.com/@mikehorton/open-source-python-based-gnss-ins-simulation-dd38d7dc729a)
        * [`filterpy`](https://github.com/rlabbe/filterpy), a general Kalman Filtering library
        * [`haversine`](https://github.com/mapado/haversine), used primarily for the haversine great circle distance calculation, but has some useful unit conversions too
        * [`ins-nav`](https://github.com/the-guild-of-calamitous-intent/ins_nav), under development, good references
        * [`navigation`](https://github.com/ngfgrant/navigation), has the name I want, appears dead, mostly just units
        * [`lat-lon-parser`](https://github.com/NOAA-ORR-ERD/lat_lon_parser), tool from NOAA that parses and convert various latitude and longitude conventions
    * These cover most of the Sensor Modeling and Multisensor Pose Estimation packages, but not in a complete comprehensive system
    * Still need: 
        * Map Representation
        * SLAM
        * Path Planning
        * Navigation in Dynamic Environments

* For now, let's get my version of the geophysical particle filter up and running on the NOAA trackline datasets. Use `pyins` position and/or velocity observations to test closed-loop functionality.