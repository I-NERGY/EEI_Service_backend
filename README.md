# EEI_Service_backend
Energy Efficiency Investments Derisking Service backend(Pilot 8)

## Database
The `Database.sql` script creates the necessary tables in a PostgreSQL/PostGIS database located at NTUA's premises.

## DATA-MANIPULATION
This folder includes scripts that split unstructured Excel files into CSV files, except for `read_gpkg.py`, which reads the State Land Dataset from a .gpkg file provided by REA, including geometrical information of housing in Riga. Other scripts like `read_riga_dhs.py`, `read_envelope.py`, `read_energy_efficiency.py`, and `read_audits.py` handle operations on various datasets and create tables in the corresponding database.

## APISRC
This folder contains a Dockerfile for the API and all necessary files for the backend of the application to be functional.

## DATASETS
This folder contains all datasets used for the service. Some were provided by the partner REA, while others were created using data-manipulation scripts.

## MACHINE LEARNING
This folder contains files related to machine learning. `pytorch-kfold` creates a physics-informed neural network, `ml_algorithm` prepares final datasets for machine learning algorithms, and `ml-for-backend` creates shallow regression models to predict envelope segments and energy consumption of buildings with less accuracy than the physics-informed algorithms.

## Paper
This folder contains a preprint of the paper derived from the machine learning algorithm developed for this service.


### Instructions
To run the API:
1. Build the Docker image using the provided Dockerfile. 
```bash
   docker build -t rea-fastapi:2.0 .
   ```
2. Run the image to create the container of the api 
```bash
   docker run -d -p 8000:8000 -v ~/rea-uc/apisrc:/app/src --name rea-api-final-2 rea-fastapi:2.0
   ```
To run the PostGIS:
```bash
   docker compose up -d --build
   ```
