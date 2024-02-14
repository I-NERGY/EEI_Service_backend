# EEIService
Energy Efficiency Investments Derisking Service backend(Pilot 8)

## Database
The `Database.sql` script creates the necessary tables in a PostgreSQL/PostGIS database located at NTUA's premises.

## DATA-MANIPULATION
This section includes scripts that split unstructured Excel files into CSV files, except for `read_gpkg.py`, which reads the State Land Dataset from a .gpkg file provided by REA, including geometrical information of housing in Riga. Other scripts like `read_riga_dhs.py`, `read_envelope.py`, `read_energy_efficiency.py`, and `read_audits.py` handle operations on various datasets and create tables in the corresponding database.

## APISRC
This section contains a Dockerfile for the API and all necessary files for the backend of the application to be functional.

## DATASETS
This section contains all datasets used for the service. Some were provided by the partner REA, while others were created using data-manipulation scripts.

## MACHINE LEARNING
This section contains files related to machine learning. `pytorch-kfold` creates a physics-informed neural network, `ml_algorithm` prepares final datasets for machine learning algorithms, and `ml-for-backend` creates shallow regression models to predict envelope segments and energy consumption of buildings with less accuracy than the physics-informed algorithms.

## Paper
This section contains a preprint of the paper derived from the machine learning algorithm developed for this service.
