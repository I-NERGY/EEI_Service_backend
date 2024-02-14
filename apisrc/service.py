from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, Mapped
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Depends, FastAPI
from sqlalchemy import Column, Integer, String, Float, Index
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from typing import List
from src.pytorch_model_backend import DNN

#from sklearn.preprocessing import  MinMaxScaler
import torch
import numpy as np
import joblib

tags_metadata = [
    {"name": "Building Information", "description": "REST API for retrieving elements from the buildings table"},
    {"name": "Building Series", "description": "REST API for posting the building series of the examined buildings"},
    {"name": "Predicted Audit", "description": "REST APIs for exposing the output of the ML Model"},
    {"name": "Energy Measures", "description": "REST APIs for addressing the admin capabilities of the application"},
    {"name": "Visualizations", "description": "REST APIs for providing some visualizations to the predicted audit"},
    {"name": "Recalculations", "description": "REST APIs for recalculating the audit after the renovation"},
]




my_database_connection = "postgresql://<user>:<password>@<ip>:<port>""

#engine1 = create_engine(my_database_connection, pool_pre_ping = True)
#Base1 = automap_base()
#base = Base1.prepare(engine1, reflect = True)
#basesession = sessionmaker(bind=engine1)
#session = basesession()

#xt = Base1.classes.energy_efficiency_measures


engine = create_engine(my_database_connection, pool_pre_ping = True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=True, bind=engine)
Base = declarative_base()

series_list = []


class Serie(BaseModel):
    serie: int
    cadastre_number: str
    heavy: bool

#vale ta epipleon (bold)
class Energy_efficiency_measure(BaseModel):
    unit: str
    code: str
    measure: str
    energy_measure_id: int
    labda: float
    thickness: int
    time_norm: float
    rate: float
    salary: float
    materials: float
    transport_mechanisms: float
    total_per_unit: float
    total_per_unit_with_profit: float

class Recalculation(BaseModel):
    id: int
    thickness: int
    checked: bool
    U: float
    measureName: str
    cost: float
    category : int
   

class Building(Base):
    __tablename__ = "buildings"
    cadastre_number = Column(String, primary_key=True)
    address = Column(String)
    longitude_centroid = Column(Float)
    #total_area = Column(Float)
    latitude_centroid = Column(Float)
    perimeter = Column(Float)
    building_id = Index(Integer)

class Riga_dhs(Base):
    __tablename__ = "riga_dhs_only_years"
    cadastre_number = Column(String, primary_key=True)
    energy_consumption = Column(Float)

class Building_ml(Base):
    __tablename__ = "buildings_ml"
    cadastre_number = Column(String, primary_key=True)
    total_area = Column(Float)
    useful_area = Column(Float)
    apartments = Column(Integer)
    floors = Column(Integer)
    

class Energy_measures(Base):
    __tablename__ = "energy_efficiency_measures"
    code = Column(String, primary_key=True)
    unit = Column(String)
    measure = Column(String)
    labda = Column(Float)
    time_norm = Column(Float)
    rate = Column(Float)
    salary = Column(Float)
    materials = Column(Float)
    transport_mechanisms = Column(Float)
    total_per_unit = Column(Float)
    thickness = Column(Integer)
    total_per_unit_with_profit = Column(Float)
    u_value = Column(Float)
    energy_measure_id = Index(Integer)

Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="EEI service API",
    description="Collection of REST APIs for Serving Execution of I-NERGY REA UC",
    version="0.0.1",
    openapi_tags=tags_metadata,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.all_u_columns = None
app.all_area_columns = None
app.all_remaining_columns = None
app.all_inputs = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def scale(data):
    scaler = joblib.load('./src/models/my_scaler_input_pytorch.save')
    scaled_data = scaler.transform(data.reshape(1,-1))
    return scaled_data

def unscale_output(predicted_data):
    scaler = joblib.load('./src/models/my_scaler_output_pytorch.save')
    unscaled_output_data = scaler.inverse_transform(predicted_data)
    return unscaled_output_data

def unscale_input(input_data):
    scaler = joblib.load('./src/models/my_scaler_input.save')
    unscaled_input_data = scaler.inverse_transform(input_data.reshape(1,-1))
    return unscaled_input_data

def predict(data_scaled):
    #decision_tree = joblib.load('./src/models/decision_tree_for_backend.sav')
    model = DNN.load_from_checkpoint('src/checkpoint_200/epoch=12-step=2548.ckpt')
    tensor_input = torch.from_numpy(data_scaled.copy())
    tensor_input = tensor_input.to(torch.float32)
    y_pred = model(tensor_input)
    y_pred = y_pred.to(torch.float64)
    return y_pred.detach().numpy()

def encode(serie):
    try:
        label_encoder = joblib.load('./src/models/my_encoder.pkl')

        str_serie_value = str(serie)

        for i in label_encoder.classes_:
            if i == str_serie_value:
                encoded_serie = label_encoder.transform([str(serie)])




    except Exception as e:
        return {'Something went wrong with the encoder': str(e)}

    #encoded_serie = label_encoder.transform([str_serie_value])[0]
    return(encoded_serie)



def calculate_energy_consumption(predictions_unscaled, inputs_unscaled):
    areas_columns = predictions_unscaled[:, :5]
    u_columns = predictions_unscaled[:, 5:10]
    #FOR VISUALIZATION
    app.all_u_columns = u_columns
   
    app.all_area_columns = areas_columns[-1]
    
    remaining_columns = predictions_unscaled[:, 10:]
    app.all_remaining_columns = remaining_columns
    
    # Extracting specific values from the remaining columns
    h = remaining_columns[:, 0]
    specific_heat_gains = remaining_columns[:, 1]

    # Calculate envelope heat losses
    heat_losses = areas_columns * u_columns * (18.9) * 192 * 24 / 1000
    thermal_bridges = np.sum(heat_losses, axis=1, keepdims=True) * 0.03
    envelope_heat_losses = np.sum(np.concatenate((heat_losses, thermal_bridges), axis= 1), axis = 1)
    # thermal
    # Calculate ventilation heat losses
    useful_area = inputs_unscaled[:, 0]
    V = useful_area * 2.5
    ventilation_heat_loss_coefficient = V * h * 0.34
    ventilation_heat_losses = ventilation_heat_loss_coefficient * 18.9 * 192 * 24 / 1000

    # Calculate total heat losses and gains
    total_heat_losses = envelope_heat_losses + ventilation_heat_losses
    total_heat_gains = specific_heat_gains * V

    # Calculate energy consumption
    ratio = np.abs(total_heat_gains / total_heat_losses)
    if inputs_unscaled[-1][-1] == 0:
        # Heavy buildings
        building_type = 54.2
    else:
        # Light buildings
        building_type = 23.1
    building_time_constant = building_type * useful_area / (total_heat_losses / (192 * 24) / 18.9 * 10 ** 6)
    divide = building_time_constant / 30
    numerical_parameter = divide + 0.8
    heat_gain_usage_factor = (1 - ratio ** numerical_parameter) / (1 - ratio ** (numerical_parameter + 1))
    energy_consumption = total_heat_losses - total_heat_gains * heat_gain_usage_factor
    
    return energy_consumption / 1000

def calculate_energy_class(consumption):
    if consumption <= 30:
        return "APlus"
    elif consumption <= 40:
        return "A"
    elif consumption <= 60:
        return "B"
    elif consumption <= 80:
        return "C"
    elif consumption <= 100:
        return "D"
    elif consumption <= 125:
        return "E"
    else:
        return "E"

#na doume ti tha kanoume me to inference
def ml_output(cadastre_number, buildings):
    total_area = buildings.total_area
    useful_area = buildings.useful_area
    apartments = buildings.apartments
    floors = buildings.floors
    try:
        if series_list:
            #serie_item =  series_list.pop()
            serie_item = series_list[-1]
            serie = serie_item.serie
            heavy = serie_item.heavy
        else:
            raise Exception("BACKEND ERROR")
    except Exception as e:
        return {'Something went wrong with the database': str(e)} 
    
    encoded_serie = encode(serie)
    encoded_serie = encoded_serie[0]

    encoded_serie_columns = [int(i == encoded_serie) for i in range(12)]
    if heavy:
        extra_columns = [1, 0]  
    else:
        extra_columns = [0, 1]

    building_data = np.array([useful_area, floors, apartments, total_area] + encoded_serie_columns + extra_columns, dtype=float)
    app.all_inputs = building_data
    scaled_data = scale(building_data)

    predicted_data = predict(scaled_data)
    unscaled_predicted_data = unscale_output(predicted_data)
    unscaled_input_data = unscale_input(scaled_data)
    #send for calculating the cost
    areas_columns = unscaled_predicted_data[:, :5].copy()
    
    basement_slab = areas_columns[:, 0][0]
    roof_atic = areas_columns[:, 1][0]
    walls = areas_columns[:, 2][0]
    doors = areas_columns[:, 3][0]
    windows = areas_columns[:, 4][0]
    energy_consumption = calculate_energy_consumption(unscaled_predicted_data, unscaled_input_data)
    energy_class = calculate_energy_class(energy_consumption[0])
    return { "energy_class": energy_class, "total_energy_consumption": round(energy_consumption[0],2), "basement": basement_slab, "roof": roof_atic, "walls": walls, "doors": doors, "windows":windows } 

    #return { "energy_class": energy_class, "total_energy_consumption": round(energy_consumption[0],2) } 


def ml_output_envelope(cadastre_number,energy_measures):
    categories = {'english': {}, 'latvian': {}}
    #
    for index, measure in enumerate(energy_measures):
        unit = measure.unit
        measure.energy_measure_id = index

        category_name_english, category_number = get_category_name_english(unit)
        if category_name_english not in categories['english']:
            categories['english'][category_name_english] = {
                'categoryName': category_name_english,
                'categoryItems': [],
            }
        #['english']
        #'categoryLanguage': 'english',
        
        category_name_latvian, category_number = get_category_name_latvian(unit)
        if category_name_latvian not in categories['latvian']:
            categories['latvian'][category_name_latvian] = {
                'categoryName': category_name_latvian,
                'categoryItems': [],
            }

        item_english = {
            'id': measure.energy_measure_id,
            'measureName': measure.measure,
            'cost': abs(measure.total_per_unit_with_profit),
            'thickness': measure.thickness,
            'checked': False,
            'U': measure.u_value,
            'category': category_number
        }
        # Translate the measureName for the Latvian category
        item_latvian = {
            'id': measure.energy_measure_id,
            'measureName': translate_measure_name_english_to_latvian(measure.measure),
            'cost': abs(measure.total_per_unit_with_profit),
            'thickness': measure.thickness,
            'checked': False,
            'U': measure.u_value,
            'category': category_number
        }
        categories['english'][category_name_english]['categoryItems'].append(item_english)
        #['english']
        categories['latvian'][category_name_latvian]['categoryItems'].append(item_latvian)
    english = list(categories['english'].values())
    latvian = list(categories['latvian'].values())

    combined = {
    'english': english,
    'latvian': latvian
    }
    return combined


def translate_measure_name_english_to_latvian(english_measure_name):
    translation_dict = {
    'Attic insulation with bulk stone wool': 'Bēniņu siltināšana ar akmens vati',
    'Attic insulation with ECO wool': 'Bēniņu siltināšana ar EKO vati',
    'Roof insulation with stone wool': 'Jumta siltināšana ar akmens vati',
    'Roof insulation with polystyrene foam': 'Jumta siltināšana ar putupolistirolu',
    'Roof insulation with ECO wool': 'Jumta siltināšana ar EKO vati',
    'Roof insulation with polyurethane': 'Jumta siltināšana ar poliuretānu',
    'Roof insulation with phenolic foam': 'Jumta siltināšana ar fenola putām',
    'Facade insulation with stone wool': 'Fasādes siltināšana ar akmens vati',
    'Facade insulation with polystyrene foam': 'Fasādes siltināšana ar putupolistirolu',
    'Ventilated facade with stone wool insulation': 'Ventilējamā fasāde ar akmens vates izolāciju',
    'Ventilated facade with ECO wool insulation': 'Ventilējamā fasāde ar EKO vates izolāciju',
    'Assembly of internal wall vacuum insulation panels': 'Iekšējo sienu vakuumizolācijas paneļu montāža',
    'Insulation of the basement cover with stone wool': 'Pagraba pārseguma siltināšana ar akmens vati',
    '2-Pane window (U=1.0)': 'Divkameru logs (U=1.0)',
    'Entrance doors (U=0.9)': 'Ieejas durvis (U=0.9)',
    '2-Pane window (U=0.8)': 'Divkameru logs (U=0.8)',
    'Entrance doors (U=1.8)': 'Ieejas durvis (U=1.8)',
    '2-Pane window (U=1.4)': 'Divkameru logs (U=1.4)',
    'Insulation of the plinth with polystyrene foam': 'Cokola siltināšana ar putupolistirolu',
    'Insulation of the plinth with polyurethane': 'Cokola siltināšana ar poliuretānu',
    
    }
    return translation_dict.get(english_measure_name, english_measure_name)


def get_category_name_english(unit):
    if "Windows/doors" in unit:
        return "Replacement of windows", 5
    elif "Front door" in unit:
        return "Replacement of doors", 4
    elif "Attic cover" in unit:
        return "Insulation of the roof/attic slab", 2
    elif "roof" in unit:
        return "Insulation of the roof/attic slab", 2
    elif "Exterior wall"  in unit:
        return "Facade insulation" , 3
    elif "External wall" in unit:
        return "Facade insulation", 3
    elif "Thermal insulation of internal walls" in unit:
        return "Floor slab insulation", 1
    elif "Basement" in unit:
        return "Floor slab insulation", 1
    else:
        return "Other", 6  # A default category for units that don't match any specific category

def get_category_name_latvian(unit):
    if "Windows/doors" in unit:
        return "Logu nomaiņa", 5
    elif "Front door" in unit:
        return "Durvju nomaiņa", 4
    elif "Attic cover" in unit:
        return "Jumta/bēniņu plātnes siltināšana", 2
    elif "roof" in unit:
        return "Jumta/bēniņu plātnes siltināšana", 2
    elif "Exterior wall"  in unit:
        return "Fasādes siltināšana" , 3
    elif "External wall" in unit:
        return "Fasādes siltināšana", 3
    elif "Thermal insulation of internal walls" in unit:
        return "Grīdas siltināšana", 1
    elif "Basement" in unit:
        return "Grīdas siltināšana", 1
    else:
        return "Cits", 6
"""
def ml_output_envelope(cadastre_number):
    return [ { "id": 0, "title": "Windows/Doors", "cost": 1000, "checked": False},  
             { "id": 1, "title": "Entrance Doors", "cost": 200, "checked": False},
             { "id": 2, "title": "Exterior Walls", "cost": 300, "checked": False},
             { "id": 3, "title": "Attic Insulation with bulk stone wool", "cost": 900, "checked": False},
             { "id": 4, "title": "Basement Cover", "cost": 1500, "checked": False},
            ]
"""

def recalculate_energy_consumption(u_columns, areas_columns, remaining_columns):
    h = remaining_columns[:, 0]
    specific_heat_gains = remaining_columns[:, 1]
    
    # Calculate envelope heat losses
    heat_losses = areas_columns * u_columns * (18.9) * 192 * 24 / 1000
    thermal_bridges = np.sum(heat_losses, axis=1, keepdims=True) * 0.03
    envelope_heat_losses = np.sum(np.concatenate((heat_losses, thermal_bridges), axis=1), axis=1)
    inputs_unscaled = app.all_inputs
    # Calculate ventilation heat losses
    useful_area = inputs_unscaled[0]
    V = useful_area * 2.5
    ventilation_heat_loss_coefficient = V * h * 0.34
    ventilation_heat_losses = ventilation_heat_loss_coefficient * 18.9 * 192 * 24 / 1000

    # Calculate total heat losses and gains
    total_heat_losses = envelope_heat_losses + ventilation_heat_losses
    total_heat_gains = specific_heat_gains * V

    # Calculate energy consumption
    ratio = np.abs(total_heat_gains / total_heat_losses)
    if inputs_unscaled[-1] == 0:
        # Heavy buildings
        building_type = 54.2
    else:
        # Light buildings
        building_type = 23.1
    building_time_constant = building_type * useful_area / (total_heat_losses / (192 * 24) / 18.9 * 10 ** 6)
    divide = building_time_constant / 30
    numerical_parameter = divide + 0.8
    heat_gain_usage_factor = (1 - ratio ** numerical_parameter) / (1 - ratio ** (numerical_parameter + 1))
    energy_consumption = total_heat_losses - total_heat_gains * heat_gain_usage_factor
    
    return energy_consumption / 1000
  

@app.get("/buildings/", tags=['Building Information'])
async def read_buildings(db:Session = Depends(get_db)):
    buildings = db.query(Building).all()
    #series_list.clear()
    #all_u_columns.clear()
    #all_area_columns.clear()
    #all_remaining_columns.clear()
    #all_inputs.clear()
    for index, i in enumerate(buildings):
        i.building_id = index
    return buildings


@app.post("/series/", tags=['Building Series'])
async def send_serie(serie: Serie):
    series_list.append(serie)
    return serie

@app.get("/building/info/{cadastre_number}", tags=['Predicted Audit'])
async def audit(cadastre_number: str,db:Session = Depends(get_db)):
    cadastre_number = cadastre_number.lstrip('0')
    try:
        buildings = db.query(Building_ml).filter(Building_ml.cadastre_number == cadastre_number).first()
        if buildings is None:
            raise Exception("wrong cadastre")
    except Exception as e:
        return {'Something went wrong with the database': str(e)}
    
    return ml_output(cadastre_number, buildings)

@app.get("/energy_measures", tags=['Energy Measures'])
async def energy_measures(db:Session = Depends(get_db)):
    
    energy_measures = db.query(Energy_measures).order_by('code').all()
    for index, i in enumerate(energy_measures):
        i.energy_measure_id = index
    
    return energy_measures
   
# gia na vazoume kainourgio measure
@app.put("/energy_measures_add/", tags = ['Energy Measures'])
async def add_measure(Energy_efficiency_measure: Energy_efficiency_measure, db:Session = Depends(get_db)):
   
    try:
        new_measure = Energy_measures(
            code=Energy_efficiency_measure.code,
            unit=Energy_efficiency_measure.unit,
            measure=Energy_efficiency_measure.measure,
            labda=Energy_efficiency_measure.labda,
            time_norm=Energy_efficiency_measure.time_norm,
            rate=Energy_efficiency_measure.rate,
            salary=Energy_efficiency_measure.salary,
            materials=Energy_efficiency_measure.materials,
            transport_mechanisms=Energy_efficiency_measure.transport_mechanisms,
            total_per_unit=Energy_efficiency_measure.total_per_unit,
            thickness=Energy_efficiency_measure.thickness,
            total_per_unit_with_profit=Energy_efficiency_measure.total_per_unit_with_profit,
            energy_measure_id=Energy_efficiency_measure.energy_measure_id
        )
        db.add(new_measure)
        db.commit()
        return 201
    except Exception as e:
        return {'Something went wrong with the database': str(e)}

# gia na afairoume kainourgio measure
@app.delete("/energy_measures_delete/{id}", tags = ['Energy Measures'])
async def delete_measure(id: str, db:Session = Depends(get_db)):
    try:
        energy_measure = db.query(Energy_measures).filter(Energy_measures.code == id).first()
        if energy_measure:
            db.delete(energy_measure)
            db.commit()
            
            
            #return {"message": f"Row with ID {id} has been deleted"}
        else:
            return {"error": f"ID {id} not found"}
    except Exception as e:
        return {"error": str(e)}
    

#stelnei mono th grammh kai gurnaw th grammh kai allazw sthn vash
@app.put("/energy_measures/{id}", tags=['Energy Measures'])
async def admin(Energy_efficiency_measure: Energy_efficiency_measure, id: str, db:Session = Depends(get_db)):
    energy_measures = db.query(Energy_measures).all()
    for index, i in enumerate(energy_measures):
        i.energy_measure_id = index
        if i.code == id:
            i.code = Energy_efficiency_measure.code
            i.unit = Energy_efficiency_measure.unit
            i.measure = Energy_efficiency_measure.measure
            i.labda = Energy_efficiency_measure.labda
            i.time_norm = Energy_efficiency_measure.time_norm
            i.rate = Energy_efficiency_measure.rate
            i.salary = Energy_efficiency_measure.salary
            i.materials = Energy_efficiency_measure.materials
            i.transport_mechanisms = Energy_efficiency_measure.transport_mechanisms
            i.total_per_unit = Energy_efficiency_measure.total_per_unit
            i.thickness = Energy_efficiency_measure.thickness
            i.total_per_unit_with_profit = Energy_efficiency_measure.total_per_unit_with_profit
            db.commit()
            db.refresh(i)
            
            return i
    return {"error": f"ID {id} not found"}
"""
@app.put("/energy_measures/{id}", tags=['Energy Measures'])
async def admin(Energy_efficiency_measure: Energy_efficiency_measure, id: int, db:Session = Depends(get_db)):
    energy_measure = db.query(Energy_measures).filter_by(energy_measure_id=id).first()
    print(energy_measure)
    if energy_measure:
        energy_measure.code = Energy_efficiency_measure.code
        energy_measure.unit = Energy_efficiency_measure.unit
        energy_measure.labda = Energy_efficiency_measure.labda
        #db.commit()
        #db.refresh(energy_measure)
        return energy_measure
    else:
        return {"error": f"ID {id} not found"}
"""

@app.get("/energy_measures/{cadastre_number}", tags=['Predicted Audit'])
async def audit_envelope(cadastre_number: str, db:Session = Depends(get_db)):
    energy_measures = db.query(Energy_measures).all()
    return ml_output_envelope(cadastre_number, energy_measures)

@app.get("/visualizations/district_heating_data/{cadastre_number}", tags=['Visualizations'])
async def bar_chart(cadastre_number: str, db:Session = Depends(get_db)):
    riga_dhs = db.query(Riga_dhs).all()
    riga_dhs_sorted = sorted(riga_dhs, key=lambda x: x.energy_consumption)
    data = [record.energy_consumption for record in riga_dhs_sorted]
    labels = ["YOU ARE HERE" if ('0' + str(record.cadastre_number)) == cadastre_number else "" for record in riga_dhs_sorted]
    value = None
    for record, label in zip(riga_dhs_sorted, labels):
        if label == "YOU ARE HERE":
            value = record.energy_consumption
            break
    response_data = {
        "data": data,
        "labels": labels,
        "value": value
    }
    return response_data
    

@app.get("/visualizations/heat_loses/{cadastre_number}", tags=['Visualizations'])
async def pie_chart(cadastre_number: str):
    #area_Basement/slab	area_Roof/attic	area_Walls	area_doors	area_windows	U_Basement/slab	U_Roof/attic	U_Walls	U_doors	U_windows
    # SUPPOSING PREDICTIONS 805.064392	873.623474	3629.231934	41.275234	1070.053711	0.707313	0.796826	0.915740	3.225770	2.105200
    #[0.707313, 0.796826, 0.915740, 3.225770, 2.105200] u
    #[805.064392, 873.623474, 3629.231934, 41.275234, 1070.053711]
    u_values = app.all_u_columns
   
    areas = app.all_area_columns
    #print(u_values)
    #print(areas)
    names = ['Basement/slab', 'Roof/attic', 'Walls', 'doors', 'windows']
    areas_column = np.array(areas)
    u_column = np.array(u_values)

    calculation = areas_column * u_column * (18.9) * 192 * 24 / 1000
    total_calculation = np.sum(calculation)
    percentage_calculation = (calculation / total_calculation) * 100
    
    response_data = {
        "data": percentage_calculation.tolist(),
        "labels": names
    }
    return response_data


@app.post("/recalculations/{cadastre_number}", tags=['Recalculations'])
async def send_recalculations(cadastre_number: str, measures: List[Recalculation]):
    u_values = app.all_u_columns
    areas = app.all_area_columns
    remaining_columns = app.all_remaining_columns
    for measure in measures:
        U = measure.U
        category = measure.category
        if category != 6:
            u_values[-1][category-1] = U
    new_energy_consumption = recalculate_energy_consumption(u_values,areas,remaining_columns)
    new_energy_class = calculate_energy_class(new_energy_consumption)
    #return new_energy_consumption[0]
    return { "energy_class": new_energy_class, "total_energy_consumption": round(new_energy_consumption[0],2) } 
    

