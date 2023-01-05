CREATE TABLE "buildings" (
  "cadastre_number" varchar PRIMARY KEY,
  "address" varchar,
  "geometry" geometry,
  "serie" int,
  "floors" float,
  "usefull_area" float,
  "total_area" float
);

CREATE TABLE "audits" (
  "cadastre_number" varchar PRIMARY KEY,
  "serie" varchar,
  "width" float,
  "length" float,
  "floors" float,
  "volume" float,
  "Avg_indoor_height" float,
  "apartments" int,
  "type_of_heating" varchar
);

CREATE TABLE "calculations" (
  "cadastre_number" varchar PRIMARY KEY,
  "total_area" float,
  "volume" float,
  "temperature" float,
  "heating_period" int,
  "air_exchange" float
);

CREATE TABLE "energy_consumption" (
  "cadastre_number" varchar PRIMARY KEY,
  "heating_system" varchar,
  "hot_water_system" varchar,
  "heating_system_energy_consumption" float,
  "hot_water_energy_consumption" float,
  "total_energy_consumption" float,
  "benefit_utilization_factor" float
);

CREATE TABLE "envelope" (
  "cadastre_number" varchar PRIMARY KEY,
  "structure_heat_loss_coefficient" float,
  "energy_consumption" float
);

CREATE TABLE "investments" (
  "cadastre_number" varchar PRIMARY KEY,
  "total_renovation_cost" float,
  "total_renovation_cost_per_m2" float
);

CREATE TABLE "validation" (
  "cadastre_number" varchar PRIMARY KEY,
  "1.2017" float,
  "2.2017" float,
  "3.2017" float,
  "4.2017" float,
  "5.2017" float,
  "6.2017" float,
  "7.2017" float,
  "8.2017" float,
  "9.2017" float,
  "10.2017" float,
  "11.2017" float,
  "12.2017" float,
  "2017" float,
  "1.2018" float,
  "2.2018" float,
  "3.2018" float,
  "4.2018" float,
  "5.2018" float,
  "6.2018" float,
  "7.2018" float,
  "8.2018" float,
  "9.2018" float,
  "10.2018" float,
  "11.2018" float,
  "12.2018" float,
  "2018" float,
  "1.2019" float,
  "2.2019" float,
  "3.2019" float,
  "4.2019" float,
  "5.2019" float,
  "6.2019" float,
  "7.2019" float,
  "8.2019" float,
  "9.2019" float,
  "10.2019" float,
  "11.2019" float,
  "12.2019" float,
  "2019" float,
  "1.2020" float,
  "2.2020" float,
  "3.2020" float,
  "4.2020" float,
  "5.2020" float,
  "6.2020" float,
  "7.2020" float,
  "8.2020" float,
  "9.2020" float,
  "10.2020" float,
  "11.2020" float,
  "12.2020" float,
  "2020" float,
  "1.2021" float,
  "2.2021" float,
  "3.2021" float,
  "4.2021" float,
  "5.2021" float
);

CREATE TABLE "envelope_components" (
  "cadastre_number" varchar ,
  "enclosing_structure" varchar,
  "material" varchar,
  "energy_consumption" float,
  "area" float,
  "structure_heat_loss_coefficient" float,
  "total_structure_heat_loss_coefficient" float,
  "total_energy_consumption" float,
  "total_area" float
);

CREATE TABLE "energy_efficiency_measures" (
  "code" varchar PRIMARY KEY,
  "place" varchar,
  "measure" varchar,
  "thickness" int,
  "λ" float,
  "time_norm" float,
  "rate" float,
  "salary" float,
  "materials" float,
  "transport_mechanisms" float,
  "total_per_unit" float,
  "total_per_unit_with_profit" float
);

COMMENT ON TABLE "buildings" IS 'table "buildings" contains information from the State Land Building Dataset';

COMMENT ON TABLE "audits" IS 'table "audits" contains audit building information';

COMMENT ON TABLE "calculations" IS 'table contains information about calculations in audit buildings';

COMMENT ON TABLE "energy_consumption" IS 'table contains information about energy consumption in audit buildings';

COMMENT ON COLUMN "energy_consumption"."heating_system" IS 'single pipe or double pipe';

COMMENT ON COLUMN "envelope"."energy_consumption" IS '10 x 9 x number of heating days x hours';

COMMENT ON TABLE "validation" IS ' Contains information from Riga DHS Database (energy_consumptions)';

COMMENT ON TABLE "envelope_components" IS 'Information about each component of the envelope';

COMMENT ON COLUMN "envelope_components"."energy_consumption" IS '10 x 9 x number of heating days x hours';

COMMENT ON TABLE "energy_efficiency_measures" IS 'Infromation from Energyefficiency_measures Dataset';

ALTER TABLE "audits" ADD FOREIGN KEY ("cadastre_number") REFERENCES "buildings" ("cadastre_number");

ALTER TABLE "calculations" ADD FOREIGN KEY ("cadastre_number") REFERENCES "audits" ("cadastre_number");

ALTER TABLE "energy_consumption" ADD FOREIGN KEY ("cadastre_number") REFERENCES "audits" ("cadastre_number");

ALTER TABLE "envelope" ADD FOREIGN KEY ("cadastre_number") REFERENCES "audits" ("cadastre_number");

ALTER TABLE "investments" ADD FOREIGN KEY ("cadastre_number") REFERENCES "audits" ("cadastre_number");

ALTER TABLE "validation" ADD FOREIGN KEY ("cadastre_number") REFERENCES "buildings" ("cadastre_number");

ALTER TABLE "envelope_components" ADD FOREIGN KEY ("cadastre_number") REFERENCES "audits" ("cadastre_number");
