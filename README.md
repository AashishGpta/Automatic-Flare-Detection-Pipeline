# Automatic-Flare-Detection-Pipeline
The aims of the project were to make a search pipeline for searching flares in Kepler's Data and analyse the flares found in accordance with existing (exponential) models. 

Download all the given code files and a database made available at https://drive.google.com/open?id=17Q0IYO2vsiFL3UUP3VMJyVQ3-Gtz_g0k. Store there files in a same folder.

The user need to run main.py code. It will ask user to enter kepler IDs of objects user is interested in. User can enter multiple IDs. 

This pipeline will automatically download the short-cadence lightcurves corresponding to those objects, detrend those lightcurves, detect flares, analyse all the detected flaring events individually, and store all the relevant results in a new folder. The name of the new folder will be the Kepler ID. The most important results in the new folder are saved in 'flare_details.txt', refer to it's header for it's contents.

For a breif technical overview of the pipeline 
