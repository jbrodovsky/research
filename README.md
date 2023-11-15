# Research

General repo for the code developed as part of my personal research work as part of my Ph.D. This repo and my dissertation work will largely be focusing on autonomous global scale marine-quality inertial navigation. Primary sub-focus will be on geophysical navigation (bathymetry, gravity, and magnetics).

Some notes:

* Couple of general research topics:
  * Using a particle filter to accurately track position using pseudo-INS velocities and geophysical measurements (bathymetry, gravity anomaly, magnetic anomaly)
  * Using tracked particle filter position to provide INS position fixes
  * Analysis of general maps to create a more robust fixibility criteria (position error correction as a function of trajectory)
    * Trajectory is already pretty complex: course, speed, duration
    * Various geophysical phenomena criteria (profile range, deviation of that range, how those relate to position)
    * Position error correction as a valid criteria? Also potentially a function of INS drift profile?
    * *Really* non-linear problem, suitable for AI/ML analysis
  * With such a map, we can do a guidance alogirthm that approaches it from the opposite of the robotic SLAM problem: given a reasonably well-known map, how do we derive a guidance and control scheme that minimizes uncertainty in the pose estimate (rather than the map uncertainty)
* Development environment:
  * Completely shifted over to WSL Ubuntu 22.04. Seperating out my development environment on my personal laptop to all be on the Linux side.
  * Office applications and other things will sit on the Windows side and manuscripts will sit where appropriate, depending on how/where I end up writing them (TexStudio, or VSCode).
  * Remote server is still running Ubuntu 20.04. Upgrading to 22.04 broke networking drivers (Ethernet and WiFi). Should still be ok. Enables a good transition to proper cloud infrastructure if needed.
  * For now, Python is sufficient. Going to incorporate GitHub Copilot and squeeze as much performance out of Python and possibly set up some sort of "web"-app for transfering data to my local remote server.
  * WSL environment will enable easy transition to C/C++ if (when) needed or desired for speed (or portfolio)
* For now, `pyins` seems to sufficiently meet my needs. It has a standard legacy error state (both feed-forward and feed-back) Kalman Filter and does the typical navigation corrections. It still feels a little under documented and clunky, so I might modify it on my fork.
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
    * Would also only benefit if there would be bindings in some interpreted language and/or other compiled languages
      * Could possibly do all the base coding in C/C++ and easily bind with PyBind11
      * Alternatively, could use SWIG to write multiple bindings, but I don't know SWIG yet
      * If we do go down this route, can leave this as something open-source with source code in C/C++ and contributors adding in bindings
