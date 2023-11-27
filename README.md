# EEIService
Energy Efficiency Investments Derisking Service (Pilot 8)

The Database.sql creates the necessary tables in a PostgreSQL\PostGIS database based in NTUA's premises.

DATA-MANIPULATION

Each script splits the unstructured excel files into csv files in programming readable form. The only exception is read_gpkg.py which reads the State Land Dataset from the .gpkg file given by REA, including Geometrical information of the housing in Riga. Accordingly, read_riga_dhs.py makes operation in Riga DHS Database and creates the table validation in the corresponding database, read_envelope.py summarizes the operations done in the second sheet of the Audits excel files containing information about the envelope components of the already audited buildings.Same goes for read_energy_efficiency.py holding information from the Energy_Efficiency_Measures dataset. Finally, read_audits.py creates most of the tables in the database, reading, splitting and comprehending unstructured information from the two Audit Datasets.

APISRC

Contains a Dockerfile for the api and all the necessary files in order for the back end side of the application to be functional.

DATASETS

Here are all the datasets used for the service. Some were provided by the partner REA and some other were created using the data-manipulation scripts provided in another folder.

MACHINE LEARNING

There are 3 files in this folder. The pytorch-kfold creates the physics-informed neural network, while the ml_algorithm creates the final datasets in form for the machine learning algorithm. Finally the ml-for-backend file creates some shallow regression models to predict the segments of the envelope as well as the energy consumption of a building with less accuracy than the physics informed algorithms.

Paper

Here lies a preprint of our paper, coming from the machine learning algorithm developed for this service.