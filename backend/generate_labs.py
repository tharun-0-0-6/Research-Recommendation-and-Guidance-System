import pandas as pd
import random
import uuid
import json
from pathlib import Path

# Domains and Equipment (from generate_data.py)
DOMAINS_EQUIPMENT = {
    "Environmental Science": [
        "pH Sensor", "Turbidity Sensor", "Dissolved Oxygen Sensor", "Temperature Sensor",
        "Conductivity Meter", "Spectrophotometer", "GPS Device", "Water Sampler",
        "Drone (DJI Matrice)", "Raspberry Pi 4", "Arduino Uno", "Datalogger",
        "Turbidity Meter", "Portable Water Quality Analyzer", "Anemometer",
        "Particulate Matter Sensor (PM2.5)", "CO2 Sensor", "Soil pH Meter",
        "Rainfall Gauge", "Breadboard & Jumper Wires", "Waterproof Enclosure"
    ],
    "Biomedical": [
        "Centrifuge", "PCR Machine", "Microscope (Fluorescent)", "Electrophoresis Apparatus",
        "ELISA Reader", "Flow Cytometer", "Biosensor Array", "ECG Sensor", "EEG Headset",
        "Blood Glucose Sensor", "High-Speed Camera", "Autoclave", "Incubator",
        "Spectrophotometer", "GPU Workstation (NVIDIA A100)", "Microcontroller (ESP32)",
        "MRI Scanner (access)"
    ],
    "CS-AI": [
        "GPU Workstation (NVIDIA RTX 4090)", "TPU Cloud Instance", "Raspberry Pi 5",
        "Jetson Nano", "High-Resolution Camera", "Microphone Array", "LiDAR Sensor",
        "Tactile Sensor", "Network Switch (10GbE)", "NAS Storage (100TB)"
    ],
    "Electronics": [
        "Oscilloscope (4-channel)", "Function Generator", "Digital Multimeter",
        "Spectrum Analyzer", "Logic Analyzer", "FPGA Board (Xilinx Artix-7)",
        "Soldering Station", "PCB Prototyping Machine", "RF Signal Generator",
        "Power Supply (Variable)", "Arduino Mega", "STM32 Nucleo Board",
        "3D Printer (FDM)", "Motor Driver Module", "CRO (Cathode Ray Oscilloscope)"
    ],
    "Materials Science": [
        "X-Ray Diffractometer (XRD)", "Scanning Electron Microscope (SEM)",
        "Transmission Electron Microscope (TEM)", "AFM (Atomic Force Microscope)",
        "Muffle Furnace", "Ball Mill", "Universal Testing Machine (UTM)",
        "Electrochemical Workstation", "Spin Coater", "Sputtering System",
        "UV-Vis Spectrophotometer", "Thermogravimetric Analyzer (TGA)",
        "Raman Spectrometer", "Hardness Tester"
    ],
    "Civil": [
        "Total Station (Survey)", "GPS RTK System", "Compression Testing Machine (CTM)",
        "Core Cutter", "Penetrometer", "Theodolite", "Strain Gauge", "Accelerometer",
        "Piezometer", "Schmidt Hammer", "Triaxial Test Apparatus",
        "Proctor Compaction Test", "Traffic Counter Sensor", "Drone (Survey Grade)"
    ]
}

LOCATIONS = [
    "Chennai, Tamil Nadu", "Bengaluru, Karnataka", "Mumbai, Maharashtra", "Delhi, NCR",
    "Hyderabad, Telangana", "Pune, Maharashtra", "Coimbatore, Tamil Nadu", "Kolkata, West Bengal",
    "Ahmedabad, Gujarat", "Jaipur, Rajasthan", "Kochi, Kerala", "Chandigarh, Punjab",
    "Lucknow, Uttar Pradesh", "Bhubaneswar, Odisha", "Guwahati, Assam",
]

LAB_PREFIXES = ["Advanced", "Regional", "Central", "Digital", "Precision", "Collaborative", "Apex"]
LAB_SUFFIXES = ["Research Lab", "Innovation Center", "Testing Facility", "Institute of Technology", "Center for Excellence"]

def generate_labs(num_labs=100):
    labs = []
    all_equip = [item for sublist in DOMAINS_EQUIPMENT.values() for item in sublist]
    
    for _ in range(num_labs):
        domain = random.choice(list(DOMAINS_EQUIPMENT.keys()))
        location = random.choice(LOCATIONS)
        
        name = f"{random.choice(LAB_PREFIXES)} {domain} {random.choice(LAB_SUFFIXES)}"
        
        # Each lab has 5-15 random equipment from its domain, plus some general ones
        domain_equip = DOMAINS_EQUIPMENT[domain]
        selected_equip = random.sample(domain_equip, k=random.randint(5, min(15, len(domain_equip))))
        
        # Add some random overlap from other domains
        other_equip = random.sample(all_equip, k=random.randint(1, 3))
        selected_equip = list(set(selected_equip + other_equip))
        
        availability = {
            "Monday-Friday": f"{random.randint(8, 10)}:00-{random.randint(17, 20)}:00",
            "Saturday": random.choice(["10:00-14:00", "Closed", "09:00-13:00"]),
            "Sunday": "Closed"
        }
        
        lab = {
            "lab_id": f"LAB_{uuid.uuid4().hex[:6].upper()}",
            "lab_name": name,
            "location": location,
            "equipment_list": ";".join(selected_equip),
            "availability": json.dumps(availability),
            "contact_person": f"{random.choice(['Dr.', 'Prof.', 'Mr.', 'Ms.'])} {random.choice(['Sharma', 'Iyer', 'Reddy', 'Patel', 'Singh', 'Das', 'Nair'])}",
            "contact_email": f"contact@{name.lower().replace(' ', '')}.edu.in"
        }
        labs.append(lab)
        
    df = pd.DataFrame(labs)
    out_path = Path(__file__).resolve().parent.parent / "datasets" / "labs_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {num_labs} labs in {out_path}")

if __name__ == "__main__":
    generate_labs(120)
