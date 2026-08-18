import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

class AIEngine:
    ALLOWED_EQUIPMENT = [
        '3D Printer (FDM)', 'AFM (Atomic Force Microscope)', 'Accelerometer', 'Anemometer', 'Arduino Mega', 
        'Arduino Uno', 'Autoclave', 'Ball Mill', 'Biosensor Array', 'Blood Glucose Sensor', 
        'Breadboard & Jumper Wires', 'CO2 Sensor', 'CRO (Cathode Ray Oscilloscope)', 'Centrifuge', 
        'Chemical Vapor Deposition', 'Compression Testing Machine (CTM)', 'Conductivity Meter', 'Core Cutter', 
        'Datalogger', 'Digital Multimeter', 'Dissolved Oxygen Sensor', 'Drone (DJI Matrice)', 
        'Drone (Survey Grade)', 'ECG Sensor', 'EEG Headset', 'ELISA Reader', 'Electrochemical Workstation', 
        'Electrophoresis Apparatus', 'FPGA Board (Xilinx Artix-7)', 'Flow Cytometer', 'Function Generator', 
        'GPS Device', 'GPS RTK System', 'GPU Workstation (NVIDIA A100)', 'GPU Workstation (NVIDIA RTX 4090)', 
        'Hardness Tester', 'High-Resolution Camera', 'High-Speed Camera', 'Incubator', 'Jetson Nano', 
        'LiDAR Sensor', 'Logic Analyzer', 'MRI Scanner (access)', 'Microcontroller (ESP32)', 
        'Microphone Array', 'Microscope (Fluorescent)', 'Motor Driver Module', 'Muffle Furnace', 
        'NAS Storage (100TB)', 'Network Switch (10GbE)', 'Oscilloscope (4-channel)', 'PCB Prototyping Machine', 
        'PCR Machine', 'Particulate Matter Sensor (PM2.5)', 'Penetrometer', 'Piezometer', 
        'Power Supply (Variable)', 'Proctor Compaction Test', 'RF Signal Generator', 'Rainfall Gauge', 
        'Raman Spectrometer', 'Raspberry Pi 4', 'Raspberry Pi 5', 'Scanning Electron Microscope (SEM)', 
        'Schmidt Hammer', 'Soil pH Meter', 'Soldering Station', 'Spectrum Analyzer', 'Spectrophotometer', 
        'Spin Coater', 'Sputtering System', 'Strain Gauge', 'TPU Cloud Instance', 'Tactile Sensor', 
        'Temperature Sensor', 'Theodolite', 'Thermogravimetric Analyzer (TGA)', 'Total Station (Survey)', 
        'Traffic Counter Sensor', 'Transmission Electron Microscope (TEM)', 'Triaxial Test Apparatus', 
        'Turbidity Meter', 'Turbidity Sensor', 'UV-Vis Spectrophotometer', 'Universal Testing Machine (UTM)', 
        'Water Sampler', 'Waterproof Enclosure', 'X-Ray Diffractometer (XRD)', 'pH Sensor'
    ]

    def __init__(self):
        if API_KEY and API_KEY != "your_api_key_here":
            try:
                genai.configure(api_key=API_KEY)
                # Dynamic Model Discovery
                available_models = [m.name for m in genai.list_models() 
                                   if 'generateContent' in m.supported_generation_methods]
                
                # Preference order (Stable first, then experimental)
                preferred = [
                    'models/gemini-1.5-flash', 
                    'models/gemini-flash-latest',
                    'models/gemini-pro',
                    'models/gemini-pro-latest',
                    'models/gemini-2.0-flash'
                ]
                
                selected = None
                for p in preferred:
                    if p in available_models:
                        selected = p
                        break
                
                if not selected and available_models:
                    selected = available_models[0]
                
                if selected:
                    self.model = genai.GenerativeModel(selected)
                    self.is_active = True
                    print(f"[+] AI Engine active using model: {selected}")
                else:
                    self.is_active = False
                    print("[!] No compatible Gemini models found.")
            except Exception as e:
                self.is_active = False
                print(f"[!] AI Engine failed to initialize: {e}")
        else:
            self.is_active = False
            print("[!] Gemini API Key not found. Falling back to Local ML.")

    def get_recommendations(self, inp: dict):
        if not self.is_active:
            return None

        prompt = f"""
        You are a specialized Research Project Consultant. 
        Based on the researcher's inputs below, generate highly specific technical recommendations.

        PROJECT DETAILS:
        - Title: {inp.get('project_title')}
        - Description: {inp.get('project_description')}
        - Domain: {inp.get('research_domain')}
        - Type: {inp.get('project_type')}

        CRITICAL CONSTRAINTS:
        1. HARDWARE EVALUATION: Before suggesting any equipment, determine if the project is a PURE SOFTWARE project (e.g., Full-stack Web Apps, Assignment Portals, Dashboard Systems, Mobile Apps, or purely Computational AI).
           * If it is a PURE SOFTWARE project, YOU MUST return an empty array `[]` for `recommended_equipment`.
           * NEVER suggest 'Tactile Sensor', 'LiDAR Sensor', or any physical lab instruments for web dashboards or management portals.
           * ONLY suggest hardware if the project involves physical prototyping, robotics, or embedded systems.
        
        2. DATASET MATCHING: For 'recommended_equipment', you MUST ONLY pick names from this EXACT list (or return `[]` if no hardware is needed):
           {", ".join(self.ALLOWED_EQUIPMENT)}

        REQUIREMENTS:
        - Generate 5-6 Methodologies (e.g., Agile, SDLC, MVC Architecture for web projects).
        - Generate 6-8 Software/Tools (e.g., React, Node.js, MongoDB for full-stack).
        - Generate a detailed 6-phase Execution Roadmap.
        - Generate 3 Similar Project Ideas.

        You MUST return the response strictly as a JSON object with this structure:
        {{
            "recommended_equipment": [
                {{"name": "Item Name from Allowed List", "category": "Sensor/Computing/Lab Instrument/Field Equipment", "score": 0.95}}
            ],
            "recommended_methodologies": [
                {{"name": "Method Name", "type": "Experimental/Computational/Statistical/Analytical", "score": 0.92}}
            ],
            "recommended_tools": [
                {{"name": "Tool Name", "category": "Programming/Data Analysis/Simulation/Visualization/Design", "score": 0.94}}
            ],
            "execution_roadmap": [
                {{"phase_number": 1, "phase_name": "Phase Title", "steps": ["Step 1", "Step 2"]}}
            ],
            "similar_project_ids": [
                {{"project_title": "Title", "domain": "Domain", "similarity": 0.85}}
            ]
        }}
        
        Return ONLY the raw JSON string. Ensure the software/tools list is modern and accurate for the project description.
        """

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"): text = text[4:]
            
            data = json.loads(text)
            
            # Post-prediction filter: Ensure hardware exists in dataset
            data["recommended_equipment"] = [
                e for e in data.get("recommended_equipment", [])
                if e["name"] in self.ALLOWED_EQUIPMENT
            ]
            
            return data
        except Exception as e:
            print(f"[!] AI Engine Error: {e}")
            return None

# Singleton-ish pattern
_ai_instance = None
def get_ai_engine():
    global _ai_instance
    if _ai_instance is None:
        _ai_instance = AIEngine()
    return _ai_instance
