This code repository includes code and data to run an event based spatio-temporal network epidemiology model. To use it, one needs only python installed (all together should take only a few minutes).

To run the model, first download a mobility and population data from:
https://drive.google.com/drive/folders/1DxHZpE7t16Vkqc337dufjMfs5TssJ7aH?usp=sharing
and put the files in Root/Seed_v3_0/
These files are are pre-run mobility model execution, they contain information about the locations of agents throughout the week.


Secondly, create and activate a python virtual environment using "py -m venv venv" > "venv\Scripts\activate"
Download dependencies using "pip install -r "requirements.txt"
(The model was tested using python 3.8.5 and 3.11.0, so it should work using those versions)

Thirdly, run the transmission model: Go into the TransmissionModel folder, and run __MainT__gamma.py by using "py __MainT__gamma.py a b c d e"
Where a, b, c, d, e:
    - a: seed location 
    - b: demographic group of infected agents at start
    - c: Ndays (number of days you want to run for)
    - d: run id start (ie, 0 for the first run)
    - e: run id end (ie 5, if you want to do 5 runs from 0 to 5)

The seed locations are the municipalities of the Netherlands (in 2019). They are indexed from 0 to 355, and can be found in the Gemeenten.pkl and GemeentenID.pkl files in Data/Model_V1/Data. To simulate one run with the same setup as the runs in Fig. 2 for Heemstede as the seed location, use a = 126, b = 5, c = 21, d = 0, e = 1. The results are formulated in a .npz file, which needs to be analyzed after to obtain the transmission figures. To do this, run Transmission_data_analysis.ipynb. This notebook analyses  data from specified runs, the current execution holds the analysis of a single run in municipality 0, The Hague ('s Gravenhage).

To build your own mobility seed, run the MobilityModel:
In the MobilityModel folder, run __MainM__.py, with only a seed number as a command line argument. Running this model will take quite some time. The resulting mobility data can be found in the Seed_v3_X folder. To use this data for the transmission model, it needs to be copied, folder and all, to the root folder. This is done to keep these files separate so as to not overwrite accidentally. 

The maps in the paper can be made by doing runs for each municipality, and analysing them in a similar way to the the Transmission_data_analysis.ipynb notebook. One needs to plot the data using the mapdf package. Further instructions are not included since doing these runs is computationally expensive.