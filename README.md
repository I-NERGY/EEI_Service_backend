# EEIService
Energy Efficiency Investments Derisking Service (Pilot 8)

The Database.sql creates the necessary tables in a PostgreSQL\PostGIS database based in NTUA's premises.

SCRIPTS

Each script splits the unstructured excel files into csv files in programming readable form. The only exception is read_gpkg.py which reads the State Land Dataset from the .gpkg file given by REA, including Geometrical information of the housing in Riga. Accordingly, read_riga_dhs.py makes operation in Riga DHS Database and creates the table validation in the corresponding database, read_envelope.py summarizes the operations done in the second sheet of the Audits excel files containing information about the envelope components of the already audited buildings.Same goes for read_energy_efficiency.py holding information from the Energy_Efficiency_Measures dataset. Finally, read_audits.py creates most of the tables in the database, reading, splitting and comprehending unstructured information from the two Audit Datasets.
