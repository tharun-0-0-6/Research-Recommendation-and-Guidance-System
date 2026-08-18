"""
Research Recommendation Platform — Fixed Inference Engine (test.py)
====================================================================
Key fixes:
  1. Domain-aware filtering: predictions are filtered/boosted by the
     user-provided research_domain so Biomedical input never gets
     Environmental Science equipment.
  2. Keyword-based domain detection fallback when domain not supplied.
  3. Confidence score no longer inflated by high-scoring wrong-domain items.
  4. Roadmap now always matched to project_type correctly.
"""

import os
import json
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings("ignore")

# Import API automation engine
try:
    from api_automation import get_api_engine
    API_ENGINE_AVAILABLE = True
except ImportError:
    API_ENGINE_AVAILABLE = False
    print("Warning: API automation not available, using fallback mode")


# ─────────────────────────────────────────────────────────────
# DOMAIN → EQUIPMENT / METHODOLOGY / TOOL ALLOWLISTS
# These ensure predictions stay within the correct domain.
# ─────────────────────────────────────────────────────────────

DOMAIN_EQUIPMENT = {
    "Environmental Science": [
        "pH Sensor", "Turbidity Sensor", "Dissolved Oxygen Sensor", "Temperature Sensor",
        "Conductivity Meter", "Spectrophotometer", "GPS Device", "Water Sampler",
        "Drone (DJI Matrice)", "Raspberry Pi 4", "Arduino Uno", "Datalogger",
        "Turbidity Meter", "Portable Water Quality Analyzer", "Anemometer",
        "Particulate Matter Sensor (PM2.5)", "CO2 Sensor", "Soil pH Meter",
        "Rainfall Gauge", "Breadboard & Jumper Wires", "Waterproof Enclosure",
    ],
    "Biomedical": [
        "Centrifuge", "PCR Machine", "Microscope (Fluorescent)", "Electrophoresis Apparatus",
        "ELISA Reader", "Flow Cytometer", "Biosensor Array", "ECG Sensor", "EEG Headset",
        "Blood Glucose Sensor", "GPU Workstation (NVIDIA A100)", "Microcontroller (ESP32)",
        "MRI Scanner (access)", "High-Speed Camera", "Autoclave", "Incubator",
        "Spectrophotometer", "UV-Vis Spectrophotometer",
    ],
    "CS-AI": [
        "GPU Workstation (NVIDIA RTX 4090)", "TPU Cloud Instance", "Raspberry Pi 5",
        "Jetson Nano", "High-Resolution Camera", "Microphone Array", "LiDAR Sensor",
        "Tactile Sensor", "Network Switch (10GbE)", "NAS Storage (100TB)",
    ],
    "Electronics": [
        "Oscilloscope (4-channel)", "Function Generator", "Digital Multimeter",
        "Spectrum Analyzer", "Logic Analyzer", "FPGA Board (Xilinx Artix-7)",
        "Soldering Station", "PCB Prototyping Machine", "RF Signal Generator",
        "Power Supply (Variable)", "Arduino Mega", "STM32 Nucleo Board",
        "3D Printer (FDM)", "Motor Driver Module", "CRO (Cathode Ray Oscilloscope)",
    ],
    "Materials Science": [
        "X-Ray Diffractometer (XRD)", "Scanning Electron Microscope (SEM)",
        "Transmission Electron Microscope (TEM)", "AFM (Atomic Force Microscope)",
        "Muffle Furnace", "Ball Mill", "Universal Testing Machine (UTM)",
        "Electrochemical Workstation", "Spin Coater", "Sputtering System",
        "UV-Vis Spectrophotometer", "Thermogravimetric Analyzer (TGA)",
        "Raman Spectrometer", "Hardness Tester",
    ],
    "Civil": [
        "Total Station (Survey)", "GPS RTK System", "Compression Testing Machine (CTM)",
        "Core Cutter", "Penetrometer", "Theodolite", "Strain Gauge", "Accelerometer",
        "Piezometer", "Schmidt Hammer", "Triaxial Test Apparatus",
        "Proctor Compaction Test", "Traffic Counter Sensor", "Drone (Survey Grade)",
    ],
}

DOMAIN_METHODOLOGIES = {
    "Environmental Science": [
        "IoT Sensor Integration", "Time-Series Data Collection", "Anomaly Detection (ML)",
        "Statistical Threshold Analysis", "Geospatial Mapping", "Principal Component Analysis",
        "Regression Analysis", "Remote Sensing Analysis", "Water Quality Index Calculation",
        "Random Forest Classification",
    ],
    "Biomedical": [
        "PCR Amplification", "ELISA Assay", "Convolutional Neural Network", "Transfer Learning",
        "Signal Processing (FFT)", "Statistical Hypothesis Testing", "Logistic Regression",
        "Support Vector Machine", "k-Fold Cross Validation", "Federated Learning",
    ],
    "CS-AI": [
        "Transformer Fine-Tuning", "Reinforcement Learning from Human Feedback", "A/B Testing",
        "Ablation Study", "RAG Pipeline", "Graph Neural Network", "Active Learning",
        "Hyperparameter Optimization (Optuna)", "Benchmark Evaluation", "Monte Carlo Simulation",
        "Convolutional Neural Network", "Transfer Learning", "Logistic Regression",
        "Support Vector Machine", "k-Fold Cross Validation",
    ],
    "Electronics": [
        "VLSI RTL Design", "Hardware-in-Loop Simulation", "FFT Signal Analysis",
        "PID Control Design", "Thermal Analysis", "Finite Element Analysis",
        "Monte Carlo Fault Simulation", "Regression Testing (Hardware)",
        "Eye Diagram Analysis", "LTSpice Circuit Simulation",
    ],
    "Materials Science": [
        "XRD Phase Analysis", "SEM-EDS Characterization", "Cyclic Voltammetry",
        "Tensile Testing", "Sol-Gel Synthesis", "Chemical Vapor Deposition",
        "Density Functional Theory (DFT)", "Molecular Dynamics Simulation",
        "Taguchi Design of Experiments", "Nanoindentation",
    ],
    "Civil": [
        "Finite Element Analysis (FEA)", "HEC-RAS Flood Modeling", "Marshall Mix Design",
        "Regression Analysis", "Non-Destructive Testing (NDT)", "SCADA Monitoring",
        "Taguchi Optimization", "Geotechnical Boring", "BIM-based Scheduling",
        "Monte Carlo Risk Analysis",
    ],
}

DOMAIN_TOOLS = {
    "Environmental Science": [
        "Python", "scikit-learn", "MQTT", "Node-RED", "Grafana", "SQLite",
        "QGIS", "TensorFlow", "Pandas", "Matplotlib", "Google Earth Engine",
    ],
    "Biomedical": [
        "Python", "TensorFlow", "PyTorch", "OpenCV", "MATLAB", "R",
        "SPSS", "Keras", "scikit-learn", "Jupyter Notebook", "ITK-SNAP",
    ],
    "CS-AI": [
        "Python", "PyTorch", "Hugging Face Transformers", "LangChain", "FastAPI",
        "Docker", "Kubernetes", "MLflow", "Weights & Biases", "PostgreSQL", "Redis",
    ],
    "Electronics": [
        "MATLAB/Simulink", "Cadence Virtuoso", "Altium Designer", "Xilinx Vivado",
        "KiCad", "LTSpice", "Python", "Proteus", "Eagle PCB", "LabVIEW", "C/C++",
    ],
    "Materials Science": [
        "MATLAB", "VASP (DFT)", "LAMMPS", "Origin Pro", "ImageJ", "Python",
        "AutoCAD", "ANSYS", "HighScore Plus (XRD)", "R", "Crystal Maker",
    ],
    "Civil": [
        "STAAD Pro", "AutoCAD Civil 3D", "SAP2000", "ETABS", "MATLAB",
        "Python", "ArcGIS", "HEC-RAS", "Revit (BIM)", "MS Project", "SPSS",
    ],
}

# ─── Category / type lookup tables ───────────────────────────
EQUIPMENT_CATEGORIES = {
    "pH Sensor": "Sensor", "Turbidity Sensor": "Sensor", "Dissolved Oxygen Sensor": "Sensor",
    "Temperature Sensor": "Sensor", "Conductivity Meter": "Lab Instrument",
    "Spectrophotometer": "Lab Instrument", "GPS Device": "Field Equipment",
    "Water Sampler": "Field Equipment", "Drone (DJI Matrice)": "Field Equipment",
    "Raspberry Pi 4": "Computing", "Arduino Uno": "Computing", "Datalogger": "Field Equipment",
    "Turbidity Meter": "Lab Instrument", "Portable Water Quality Analyzer": "Field Equipment",
    "Anemometer": "Field Equipment", "Particulate Matter Sensor (PM2.5)": "Sensor",
    "CO2 Sensor": "Sensor", "Soil pH Meter": "Field Equipment", "Rainfall Gauge": "Field Equipment",
    "Breadboard & Jumper Wires": "Computing", "Waterproof Enclosure": "Computing",
    "Centrifuge": "Lab Instrument", "PCR Machine": "Lab Instrument",
    "Microscope (Fluorescent)": "Lab Instrument", "Electrophoresis Apparatus": "Lab Instrument",
    "ELISA Reader": "Lab Instrument", "Flow Cytometer": "Lab Instrument",
    "Biosensor Array": "Sensor", "ECG Sensor": "Sensor", "EEG Headset": "Sensor",
    "Blood Glucose Sensor": "Sensor", "GPU Workstation (NVIDIA A100)": "Computing",
    "Microcontroller (ESP32)": "Computing", "MRI Scanner (access)": "Lab Instrument",
    "GPU Workstation (NVIDIA RTX 4090)": "Computing", "TPU Cloud Instance": "Computing",
    "Raspberry Pi 5": "Computing", "Jetson Nano": "Computing",
    "High-Resolution Camera": "Sensor", "Microphone Array": "Sensor",
    "LiDAR Sensor": "Sensor", "NAS Storage (100TB)": "Computing",
    "Oscilloscope (4-channel)": "Lab Instrument", "Function Generator": "Lab Instrument",
    "Digital Multimeter": "Lab Instrument", "Spectrum Analyzer": "Lab Instrument",
    "Logic Analyzer": "Lab Instrument", "FPGA Board (Xilinx Artix-7)": "Computing",
    "Soldering Station": "Fabrication", "PCB Prototyping Machine": "Fabrication",
    "RF Signal Generator": "Lab Instrument", "Power Supply (Variable)": "Lab Instrument",
    "Arduino Mega": "Computing", "STM32 Nucleo Board": "Computing",
    "3D Printer (FDM)": "Fabrication", "Motor Driver Module": "Computing",
    "CRO (Cathode Ray Oscilloscope)": "Lab Instrument",
    "X-Ray Diffractometer (XRD)": "Lab Instrument", "Scanning Electron Microscope (SEM)": "Lab Instrument",
    "Transmission Electron Microscope (TEM)": "Lab Instrument", "AFM (Atomic Force Microscope)": "Lab Instrument",
    "Muffle Furnace": "Lab Instrument", "Ball Mill": "Lab Instrument",
    "Universal Testing Machine (UTM)": "Lab Instrument", "Electrochemical Workstation": "Lab Instrument",
    "Spin Coater": "Fabrication", "Sputtering System": "Fabrication",
    "UV-Vis Spectrophotometer": "Lab Instrument", "Thermogravimetric Analyzer (TGA)": "Lab Instrument",
    "Raman Spectrometer": "Lab Instrument", "Hardness Tester": "Lab Instrument",
    "Total Station (Survey)": "Field Equipment", "GPS RTK System": "Field Equipment",
    "Compression Testing Machine (CTM)": "Lab Instrument", "Core Cutter": "Lab Instrument",
    "Penetrometer": "Field Equipment", "Strain Gauge": "Sensor", "Accelerometer": "Sensor",
    "Piezometer": "Field Equipment", "Schmidt Hammer": "Field Equipment",
    "Triaxial Test Apparatus": "Lab Instrument", "Traffic Counter Sensor": "Field Equipment",
    "High-Speed Camera": "Lab Instrument", "Autoclave": "Lab Instrument",
    "Incubator": "Lab Instrument", "Network Switch (10GbE)": "Computing",
    "Tactile Sensor": "Sensor", "Proctor Compaction Test": "Lab Instrument",
    "Theodolite": "Field Equipment", "Drone (Survey Grade)": "Field Equipment",
    "NAS Storage (100TB)": "Computing",
}

TOOL_CATEGORIES = {
    "Python": "Programming", "scikit-learn": "Data Analysis", "MQTT": "Data Analysis",
    "Node-RED": "Visualization", "Grafana": "Visualization", "SQLite": "Data Analysis",
    "QGIS": "Visualization", "TensorFlow": "Data Analysis", "Pandas": "Data Analysis",
    "Matplotlib": "Visualization", "Google Earth Engine": "Simulation",
    "PyTorch": "Data Analysis", "Hugging Face Transformers": "Data Analysis",
    "LangChain": "Programming", "FastAPI": "Programming", "Docker": "Simulation",
    "Kubernetes": "Simulation", "MLflow": "Visualization", "Weights & Biases": "Visualization",
    "PostgreSQL": "Data Analysis", "Redis": "Programming", "OpenCV": "Data Analysis",
    "MATLAB": "Simulation", "MATLAB/Simulink": "Simulation", "R": "Data Analysis",
    "SPSS": "Data Analysis", "Keras": "Data Analysis", "Jupyter Notebook": "Programming",
    "ITK-SNAP": "Visualization", "Cadence Virtuoso": "Design", "Altium Designer": "Design",
    "Xilinx Vivado": "Design", "KiCad": "Design", "LTSpice": "Simulation",
    "Proteus": "Simulation", "Eagle PCB": "Design", "LabVIEW": "Visualization",
    "C/C++": "Programming", "VASP (DFT)": "Simulation", "LAMMPS": "Simulation",
    "Origin Pro": "Data Analysis", "ImageJ": "Data Analysis", "AutoCAD": "Design",
    "ANSYS": "Simulation", "HighScore Plus (XRD)": "Data Analysis", "Crystal Maker": "Visualization",
    "STAAD Pro": "Simulation", "AutoCAD Civil 3D": "Design", "SAP2000": "Simulation",
    "ETABS": "Simulation", "ArcGIS": "Visualization", "HEC-RAS": "Simulation",
    "Revit (BIM)": "Design", "MS Project": "Data Analysis",
}

METHODOLOGY_TYPES = {
    "IoT Sensor Integration": "Experimental", "Time-Series Data Collection": "Experimental",
    "Anomaly Detection (ML)": "Computational", "Statistical Threshold Analysis": "Statistical",
    "Geospatial Mapping": "Analytical", "Principal Component Analysis": "Analytical",
    "Regression Analysis": "Statistical", "Remote Sensing Analysis": "Analytical",
    "Water Quality Index Calculation": "Analytical", "Random Forest Classification": "Computational",
    "PCR Amplification": "Experimental", "ELISA Assay": "Experimental",
    "Convolutional Neural Network": "Computational", "Transfer Learning": "Computational",
    "Signal Processing (FFT)": "Analytical", "Statistical Hypothesis Testing": "Statistical",
    "Logistic Regression": "Computational", "Support Vector Machine": "Computational",
    "k-Fold Cross Validation": "Statistical", "Federated Learning": "Computational",
    "Transformer Fine-Tuning": "Computational", "A/B Testing": "Statistical",
    "Ablation Study": "Analytical", "RAG Pipeline": "Computational",
    "Graph Neural Network": "Computational", "Active Learning": "Computational",
    "Hyperparameter Optimization (Optuna)": "Computational", "Benchmark Evaluation": "Analytical",
    "Monte Carlo Simulation": "Computational", "VLSI RTL Design": "Experimental",
    "Hardware-in-Loop Simulation": "Computational", "FFT Signal Analysis": "Analytical",
    "PID Control Design": "Analytical", "Thermal Analysis": "Experimental",
    "Finite Element Analysis": "Computational", "LTSpice Circuit Simulation": "Computational",
    "XRD Phase Analysis": "Analytical", "SEM-EDS Characterization": "Analytical",
    "Cyclic Voltammetry": "Experimental", "Tensile Testing": "Experimental",
    "Sol-Gel Synthesis": "Experimental", "Chemical Vapor De-position": "Experimental",
    "Density Functional Theory (DFT)": "Computational", "Molecular Dynamics Simulation": "Computational",
    "Taguchi Design of Experiments": "Statistical", "HEC-RAS Flood Modeling": "Computational",
    "Marshall Mix Design": "Experimental", "Non-Destructive Testing (NDT)": "Experimental",
    "SCADA Monitoring": "Experimental", "BIM-based Scheduling": "Analytical",
    "Monte Carlo Risk Analysis": "Computational", "Geotechnical Boring": "Experimental",
    "Reinforcement Learning from Human Feedback": "Computational",
    "Eye Diagram Analysis": "Analytical", "Monte Carlo Fault Simulation": "Computational",
    "Regression Testing (Hardware)": "Experimental", "Nanoindentation": "Experimental",
    "Finite Element Analysis (FEA)": "Computational", "Taguchi Optimization": "Statistical",
}

DOMAIN_LOCATIONS = {
    "Environmental Science": ["Chennai, Tamil Nadu", "Bengaluru, Karnataka", "Kochi, Kerala", "Kolkata, West Bengal"],
    "Biomedical":            ["Chennai, Tamil Nadu", "Mumbai, Maharashtra", "Hyderabad, Telangana", "Delhi, NCR"],
    "CS-AI":                 ["Bengaluru, Karnataka", "Hyderabad, Telangana", "Pune, Maharashtra", "Chennai, Tamil Nadu"],
    "Electronics":           ["Bengaluru, Karnataka", "Coimbatore, Tamil Nadu", "Ahmedabad, Gujarat", "Pune, Maharashtra"],
    "Materials Science":     ["Chennai, Tamil Nadu", "Delhi, NCR", "Jaipur, Rajasthan", "Bhubaneswar, Odisha"],
    "Civil":                 ["Chennai, Tamil Nadu", "Mumbai, Maharashtra", "Delhi, NCR", "Chandigarh, Punjab"],
}

# ─── Domain keyword detector (used when domain not provided) ─
DOMAIN_KEYWORDS = {
    "Biomedical": [
        "cancer", "tumor", "histopathology", "pathology", "medical", "clinical",
        "patient", "diagnosis", "disease", "genomics", "dna", "pcr", "ecg", "eeg",
        "mri", "ct scan", "drug", "therapy", "biosensor", "glucose", "blood",
        "neural network image", "mammogram", "retina", "tissue", "biopsy",
    ],
    "CS-AI": [
        "natural language", "nlp", "transformer", "bert", "gpt", "llm",
        "reinforcement learning", "object detection", "yolo", "rag", "chatbot",
        "recommendation system", "knowledge graph", "speech recognition",
        "generative", "autonomous", "federated learning", "computer vision",
    ],
    "Electronics": [
        "vlsi", "fpga", "pcb", "circuit", "oscilloscope", "arduino", "microcontroller",
        "embedded", "signal", "rf", "antenna", "motor", "power electronics",
        "sensor interfacing", "uart", "i2c", "spi", "gpio",
    ],
    "Materials Science": [
        "nanomaterial", "nanoparticle", "xrd", "sem", "tem", "afm", "polymer",
        "crystal", "composite", "thin film", "corrosion", "battery", "electrode",
        "synthesis", "characterization", "tribology", "mechanical property",
    ],
    "Civil": [
        "structural", "concrete", "traffic", "flood", "pavement", "bridge",
        "geotechnical", "soil bearing", "construction", "hec-ras", "bim",
        "earthquake", "dam", "drainage", "road", "building", "foundation",
    ],
    "Environmental Science": [
        "water quality", "air pollution", "soil contamination", "climate",
        "biodiversity", "greenhouse", "remote sensing", "waste", "ecosystem",
        "river", "heavy metal", "microplastic", "carbon", "wetland",
        "ph", "turbidity", "dissolved oxygen",
    ],
}

FALLBACK_ROADMAPS = {
    "Experimental": [
        {"phase_number": 1, "phase_name": "Literature Review & Problem Formulation",
         "steps": ["1. Conduct systematic literature review on the topic",
                   "2. Identify research gaps and define problem statement",
                   "3. Prepare literature review document (min 30 papers)",
                   "4. Define hypothesis and success criteria"]},
        {"phase_number": 2, "phase_name": "Design & Procurement",
         "steps": ["1. Finalize system architecture and block diagram",
                   "2. Prepare Bill of Materials (BoM)",
                   "3. Procure equipment and consumables",
                   "4. Set up experimental workspace and safety protocols"]},
        {"phase_number": 3, "phase_name": "Prototype / Setup Development",
         "steps": ["1. Assemble hardware components",
                   "2. Develop initial firmware or software pipeline",
                   "3. Perform bench-level unit tests",
                   "4. Document assembly and calibration procedure"]},
        {"phase_number": 4, "phase_name": "Data Collection",
         "steps": ["1. Design data collection protocol",
                   "2. Run controlled experiments with replication",
                   "3. Log raw data with timestamps",
                   "4. Perform initial quality check on collected data"]},
        {"phase_number": 5, "phase_name": "Analysis & Modelling",
         "steps": ["1. Clean and preprocess collected data",
                   "2. Apply statistical or ML models",
                   "3. Visualise results and generate insights",
                   "4. Conduct sensitivity analysis"]},
        {"phase_number": 6, "phase_name": "Validation & Testing",
         "steps": ["1. Compare results against baseline or benchmarks",
                   "2. Conduct blind validation experiments",
                   "3. Document accuracy, precision, recall metrics"]},
        {"phase_number": 7, "phase_name": "Reporting & Dissemination",
         "steps": ["1. Write final project report",
                   "2. Prepare journal or conference manuscript",
                   "3. Archive code, data, and docs on GitHub/Zenodo"]},
    ],
    "Computational": [
        {"phase_number": 1, "phase_name": "Problem Scoping & Literature Review",
         "steps": ["1. Define problem formally with inputs, outputs, objectives",
                   "2. Review state-of-the-art models and benchmarks",
                   "3. Identify datasets and evaluation metrics"]},
        {"phase_number": 2, "phase_name": "Dataset Preparation",
         "steps": ["1. Identify and download relevant public datasets",
                   "2. Perform exploratory data analysis (EDA)",
                   "3. Handle missing values, outliers, class imbalance",
                   "4. Split data into train / validation / test sets"]},
        {"phase_number": 3, "phase_name": "Model Architecture Design",
         "steps": ["1. Propose candidate model architectures",
                   "2. Implement baseline model",
                   "3. Define loss functions and evaluation metrics",
                   "4. Set up experiment tracking (MLflow / W&B)"]},
        {"phase_number": 4, "phase_name": "Training & Optimisation",
         "steps": ["1. Train baseline model and log results",
                   "2. Perform hyperparameter tuning",
                   "3. Apply regularisation and augmentation strategies",
                   "4. Analyse learning curves and convergence"]},
        {"phase_number": 5, "phase_name": "Evaluation & Ablation",
         "steps": ["1. Evaluate on held-out test set",
                   "2. Conduct ablation studies for each component",
                   "3. Compare with state-of-the-art benchmarks",
                   "4. Perform error analysis on failure cases"]},
        {"phase_number": 6, "phase_name": "Deployment & Reporting",
         "steps": ["1. Package model as REST API (FastAPI/Flask)",
                   "2. Containerise with Docker",
                   "3. Write final report and create demo",
                   "4. Publish code and model weights on GitHub"]},
    ],
    "Survey-based": [
        {"phase_number": 1, "phase_name": "Research Design",
         "steps": ["1. Define survey objectives and target population",
                   "2. Perform power analysis to determine sample size",
                   "3. Design questionnaire with validated scales",
                   "4. Get IRB / ethics committee approval"]},
        {"phase_number": 2, "phase_name": "Pilot Survey & Refinement",
         "steps": ["1. Conduct pilot with 20-30 respondents",
                   "2. Assess Cronbach's alpha for reliability",
                   "3. Refine ambiguous questions",
                   "4. Finalise survey instrument"]},
        {"phase_number": 3, "phase_name": "Data Collection",
         "steps": ["1. Deploy survey via Google Forms / KoBoToolbox",
                   "2. Monitor response rate and send reminders",
                   "3. Ensure geographic and demographic diversity"]},
        {"phase_number": 4, "phase_name": "Statistical Analysis",
         "steps": ["1. Perform descriptive statistics",
                   "2. Test normality and homogeneity assumptions",
                   "3. Run correlation / regression or SEM",
                   "4. Visualise results with appropriate charts"]},
        {"phase_number": 5, "phase_name": "Reporting",
         "steps": ["1. Interpret findings in context of literature",
                   "2. Report limitations and generalisability",
                   "3. Write manuscript for journal submission",
                   "4. Share anonymised dataset on OSF / Zenodo"]},
    ],
}
FALLBACK_ROADMAPS["Mixed"] = FALLBACK_ROADMAPS["Experimental"]


def detect_domain_from_text(text: str) -> str:
    """Detect research domain from free text when user hasn't selected one."""
    lower = text.lower()
    scores = {domain: 0 for domain in DOMAIN_KEYWORDS}
    for domain, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in lower:
                scores[domain] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Environmental Science"


class RecommendationEngine:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._load()

    def _load(self):
        d = self.model_dir
        self.tfidf      = joblib.load(f"{d}/tfidf.pkl")
        self.ohe        = joblib.load(f"{d}/ohe.pkl")
        self.scaler     = joblib.load(f"{d}/scaler.pkl")
        self.mlb_equip  = joblib.load(f"{d}/mlb_equipment.pkl")
        self.mlb_meth   = joblib.load(f"{d}/mlb_methodology.pkl")
        self.mlb_tool   = joblib.load(f"{d}/mlb_tool.pkl")
        
        try:
            self.clf_equip  = joblib.load(f"{d}/clf_equipment.pkl")
            self.clf_meth   = joblib.load(f"{d}/clf_methodology.pkl")
            self.clf_tool   = joblib.load(f"{d}/clf_tool.pkl")
        except Exception as e:
            print(f"\n[!] ERROR: Failed to load classifier models: {e}")
            print("    This usually happens due to a scikit-learn version mismatch.")
            print("    ACTION: Please run 'python train_model.py' to retrain the models.")
            raise RuntimeError("Model loading failed. See console for details.") from e

        self.knn        = joblib.load(f"{d}/knn.pkl")
        self.X_full     = joblib.load(f"{d}/X_full.pkl")
        
        # Load Lab Dataset - now using API automation
        self.labs_df = pd.DataFrame()
        if API_ENGINE_AVAILABLE:
            self.api_engine = get_api_engine()
        else:
            self.api_engine = None
            print(f"[!] WARNING: API automation not available. Lab recommendations will be limited.")
        with open(f"{d}/project_index.json") as f:
            self.project_index = json.load(f)
        with open(f"{d}/roadmap_index.json") as f:
            self.roadmap_index = json.load(f)
        with open(f"{d}/project_type_map.json") as f:
            self.project_type_map = json.load(f)

    def _build_features(self, inp: dict):
        text = (
            inp.get("project_title", "") + " " +
            inp.get("project_description", "") + " " +
            inp.get("team_expertise", "") + " " +
            inp.get("research_domain", "")
        )
        X_text = self.tfidf.transform([text])

        cat_df = pd.DataFrame([{
            "research_domain": inp.get("research_domain", "Unknown"),
            "project_type":    inp.get("project_type", "Unknown"),
            "budget_range":    inp.get("budget_range", "Unknown"),
        }])
        X_cat = self.ohe.transform(cat_df)

        dur = float(inp.get("duration_months", 6))
        X_num = csr_matrix(self.scaler.transform([[dur]]))

        return hstack([X_text, X_cat, X_num])

    def _domain_filtered_topk(self, clf, mlb, X_feat, domain_allowlist, top_k):
        """
        Get top-k predictions restricted to the domain allowlist.
        Falls back to global top-k if allowlist yields fewer than min_k items.
        """
        proba = clf.predict_proba(X_feat)[0]
        all_labels = mlb.classes_

        # Score all labels
        label_scores = list(zip(all_labels, proba))

        # Separate in-domain vs out-of-domain
        in_domain  = [(lbl, sc) for lbl, sc in label_scores if lbl and lbl in domain_allowlist]
        out_domain = [(lbl, sc) for lbl, sc in label_scores if lbl and lbl not in domain_allowlist]

        # Sort each by score descending
        in_domain.sort(key=lambda x: x[1], reverse=True)
        out_domain.sort(key=lambda x: x[1], reverse=True)

        # Take up to top_k from in-domain; pad with out-of-domain only if needed
        result = in_domain[:top_k]
        if len(result) < max(3, top_k // 2):
            result = result + out_domain[: top_k - len(result)]

        return [(lbl, float(sc)) for lbl, sc in result[:top_k] if sc > 0.01]

    def _similar_projects(self, X_feat, domain: str, top_k=5):
        dists, idxs = self.knn.kneighbors(X_feat, n_neighbors=min(top_k * 4, len(self.project_index)))
        results = []
        for dist, idx in zip(dists[0][1:], idxs[0][1:]):
            proj = self.project_index[idx]
            # Prefer same domain
            if proj["research_domain"] == domain or len(results) < top_k:
                results.append({
                    "project_id":    proj["project_id"],
                    "project_title": proj["project_title"],
                    "domain":        proj["research_domain"],
                    "location":      proj["location"],
                    "similarity":    round(1 - float(dist), 3),
                })
            if len(results) >= top_k:
                break
        return results[:top_k]

    def _match_labs(self, recommended_equip_names, user_location, top_k=5):
        """Find labs matching recommended equipment, prioritized by location."""
        if self.labs_df is None or self.labs_df.empty:
            return []

        results = []
        equip_set = set(recommended_equip_names)
        
        # Simple distance-ish heuristic for location
        # (Exact match = 0, Same State = 1, Different State = 2)
        def get_loc_score(lab_loc, user_loc):
            if lab_loc == user_loc: return 0
            if lab_loc.split(",")[-1].strip() == user_loc.split(",")[-1].strip(): return 1
            return 2

        for _, lab in self.labs_df.iterrows():
            lab_equip = set([e.strip() for e in str(lab["equipment_list"]).split(";")])
            intersection = equip_set.intersection(lab_equip)
            
            if intersection:
                loc_score = get_loc_score(lab["location"], user_location)
                results.append({
                    "lab_id": lab["lab_id"],
                    "lab_name": lab["lab_name"],
                    "location": lab["location"],
                    "matched_equipment": list(intersection),
                    "availability": json.loads(lab["availability"]),
                    "contact": {
                        "person": lab["contact_person"],
                        "email": lab["contact_email"]
                    },
                    "loc_score": loc_score,
                    "match_count": len(intersection)
                })

        # Sort by: 1. Location Proximity (primary), 2. Number of matches (secondary)
        results.sort(key=lambda x: (x["loc_score"], -x["match_count"]))
        return results[:top_k]

    def _get_roadmap(self, similar_projects, project_type):
        for sp in similar_projects:
            pid = sp["project_id"]
            if pid in self.roadmap_index:
                phased = self.roadmap_index[pid]
                if phased:
                    return phased
        return FALLBACK_ROADMAPS.get(project_type, FALLBACK_ROADMAPS["Experimental"])

    def _match_teachers(self, domain: str, top_k=5, user_loc: str = ""):
        """Match teachers based on research domain/department with aggressive fuzzy searching."""
        teachers = []
        # Use absolute path to ensure server finds it
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "..", "datasets", "teacher_dataset.csv")
        
        if not os.path.exists(path):
            print(f"[!] Teacher dataset not found at {path}")
            return []
            
        try:
            df = pd.read_csv(path)
            # Standardize columns (strip whitespace/newline chars)
            df.columns = [c.strip() for c in df.columns]
            
            # Clean domain input
            clean_domain = str(domain).strip()
            
            # 1. Map higher-level domain to set of searchable keywords
            DOMAIN_ALIASES = {
                "Environmental Science": ["Environmental", "Ecology", "Climate", "Sustainability"],
                "Biomedical": ["Biomedical", "Bio", "Medicine", "Medical"],
                "CS-AI": ["Computer", "AI", "Software", "Data", "Vision", "Intelligence"],
                "Electronics": ["ECE", "Electronics", "Embedded", "Communication", "Signal"],
                "Materials Science": ["Materials", "Nanotech", "Metallurgy", "Ceramics"],
                "Civil": ["Civil", "Structural", "Geotechnical", "Hydrology"]
            }
            
            # Get keywords for search
            keywords = DOMAIN_ALIASES.get(clean_domain, [clean_domain])
            
            # 2. Strict Department Filter (Restricted to 'department' column only)
            # Construct a regex pattern like r'\b(Computer|AI|Software)\b'
            pattern = r"\b(" + "|".join(keywords) + r")\b"
            matches = df[
                df['department'].str.contains(pattern, case=False, na=True, regex=True)
            ]
            
            # 3. Sort by location proximity (same state first), then limit to top_k
            if user_loc and 'location' in matches.columns:
                user_loc_lower = user_loc.lower()
                matches = matches.copy()
                matches['_loc_score'] = matches['location'].apply(
                    lambda loc: 0 if str(loc).lower() in user_loc_lower or user_loc_lower in str(loc).lower() else 1
                )
                matches = matches.sort_values(['_loc_score', 'years_of_experience'], ascending=[True, False]).drop(columns=['_loc_score'])
            matches = matches.head(top_k)

            for _, row in matches.iterrows():
                pub_count = 0
                try: pub_count = int(row.get('publications_count', 0))
                except: pass
                
                # Fetch live rating from CSV
                tid = str(row["teacher_id"])
                avg_rating = "-"
                rating_count = 0
                try:
                    ratings_path = os.path.join(base_dir, "..", "datasets", "teacher_ratings.csv")
                    if os.path.exists(ratings_path):
                        rdf = pd.read_csv(ratings_path)
                        # Filter for this teacher
                        t_ratings = rdf[rdf["teacher_id"] == tid]["rating_value"]
                        if not t_ratings.empty:
                            avg_rating = round(t_ratings.mean(), 1)
                            rating_count = int(len(t_ratings))
                except: pass

                # Parse keywords
                raw_kw = str(row.get("specialization_keywords", ""))
                keywords = [kw.strip() for kw in raw_kw.split(";") if kw.strip()]

                teachers.append({
                    "teacher_id": tid,
                    "full_name": str(row["full_name"]),
                    "designation": str(row["designation"]),
                    "institution": str(row["institution"]),
                    "research_domain": str(row["department"]),
                    "guidance_mode": "Hybrid", 
                    "ratings": avg_rating,
                    "rating_count": rating_count,
                    "publications_count": pub_count,
                    "years_of_experience": int(row.get("years_of_experience", 0) or 0),
                    "specialization_keywords": keywords,
                    "location": str(row.get("location", ""))
                })
        except Exception as e:
            print(f"ERROR in _match_teachers: {e}")
            traceback.print_exc()
            
        return teachers

    def predict(self, inp: dict, top_k_equip=8, top_k_meth=5, top_k_tools=6, top_k_similar=5):
        # ── Step 1: resolve domain (auto-detect if blank) ──────────
        domain = inp.get("research_domain", "").strip()
        if not domain or domain.lower() in ("", "select domain", "unknown"):
            text_for_detect = inp.get("project_title", "") + " " + inp.get("project_description", "")
            domain = detect_domain_from_text(text_for_detect)
            inp = dict(inp)
            inp["research_domain"] = domain

        # ── Step 2: resolve project_type (default Computational for ML projects) ──
        project_type = inp.get("project_type", "").strip()
        if not project_type or project_type.lower() in ("", "select type", "unknown"):
            desc_lower = (inp.get("project_description", "") + inp.get("project_title", "")).lower()
            if any(kw in desc_lower for kw in ["deep learning", "machine learning", "neural network", "model", "train", "dataset"]):
                project_type = "Computational"
            elif any(kw in desc_lower for kw in ["survey", "questionnaire"]):
                project_type = "Survey-based"
            elif any(kw in desc_lower for kw in ["sensor", "experiment", "fabricat", "prototype"]):
                project_type = "Experimental"
            else:
                project_type = "Mixed"
            inp = dict(inp)
            inp["project_type"] = project_type

        # ── Step 3: build features ──────────────────────────────────
        X_feat = self._build_features(inp)

        # ── Step 4: domain-filtered predictions ────────────────────
        equip_allowlist = set(DOMAIN_EQUIPMENT.get(domain, []))
        meth_allowlist  = set(DOMAIN_METHODOLOGIES.get(domain, []))
        tool_allowlist  = set(DOMAIN_TOOLS.get(domain, []))

        equipment_raw   = self._domain_filtered_topk(self.clf_equip, self.mlb_equip, X_feat, equip_allowlist, top_k_equip)
        methodology_raw = self._domain_filtered_topk(self.clf_meth,  self.mlb_meth,  X_feat, meth_allowlist,  top_k_meth)
        tool_raw        = self._domain_filtered_topk(self.clf_tool,  self.mlb_tool,  X_feat, tool_allowlist,  top_k_tools)

        # ── Step 5: similar projects (domain-preferred) ─────────────
        similar = self._similar_projects(X_feat, domain, top_k_similar)

        # ── Step 6: roadmap ─────────────────────────────────────────
        # Use API automation for roadmap generation if available
        if self.api_engine:
            try:
                project_id = f"PRJ_API_{hash(str(inp)) % 1000000:06d}"
                api_roadmap = self.api_engine.generate_project_roadmap(project_id, project_type)
                roadmap = [
                    {
                        "phase_number": step.phase_number,
                        "phase_name": step.phase_name,
                        "steps": step.steps
                    }
                    for step in api_roadmap
                ]
            except Exception as e:
                print(f"API roadmap generation failed: {e}")
                roadmap = self._get_roadmap(similar, project_type)
        else:
            roadmap = self._get_roadmap(similar, project_type)

        # ── Step 7: matched labs (location-first) ────────────────────
        equip_names = [n for n, s in equipment_raw]
        user_loc = inp.get("location", "Chennai, Tamil Nadu")
        
        # Use API automation for lab recommendations if available
        if self.api_engine:
            try:
                api_labs = self.api_engine.generate_lab_recommendations(", ".join(equip_names), user_loc)
                recommended_labs = [
                    {
                        "lab_id": lab.lab_id,
                        "lab_name": lab.lab_name,
                        "location": lab.location,
                        "equipment_list": lab.equipment_list,
                        "availability": lab.availability,
                        "contact_person": lab.contact_person,
                        "contact_email": lab.contact_email
                    }
                    for lab in api_labs
                ]
            except Exception as e:
                print(f"API lab recommendation failed: {e}")
                recommended_labs = self._match_labs(equip_names, user_loc)
        else:
            recommended_labs = self._match_labs(equip_names, user_loc)

        # ── Step 8: confidence ──────────────────────────────────────
        all_scores = [s for _, s in equipment_raw] + [s for _, s in methodology_raw] + [s for _, s in tool_raw]
        confidence = round(float(np.mean(all_scores)), 3) if all_scores else 0.5

        # ── Step 9: matched teachers ───────────────────────────────
        teachers = self._match_teachers(domain, user_loc=inp.get('location',''))

        # ── Step 9: enrich and return ───────────────────────────────
        return {
            "recommended_equipment": [
                {"name": n, "category": EQUIPMENT_CATEGORIES.get(n, "Equipment"), "score": round(s, 3)}
                for n, s in equipment_raw
            ],
            "recommended_methodologies": [
                {"name": n, "type": METHODOLOGY_TYPES.get(n, "Analytical"), "score": round(s, 3)}
                for n, s in methodology_raw
            ],
            "recommended_tools": [
                {"name": n, "category": TOOL_CATEGORIES.get(n, "Data Analysis"), "score": round(s, 3)}
                for n, s in tool_raw
            ],
            "similar_project_ids":       similar,
            "location_recommendations":  DOMAIN_LOCATIONS.get(domain, ["Chennai, Tamil Nadu"]),
            "execution_roadmap":         roadmap,
            "recommended_labs":          recommended_labs,
            "teachers":                  teachers,
            "confidence_score":          confidence,
            "detected_domain":           domain,
            "detected_project_type":     project_type,
        }


# ── Self-test ────────────────────────────────────────────────
if __name__ == "__main__":
    engine = RecommendationEngine(model_dir="models")

    tests = [
        {
            "label": "Breast Cancer Detection (Biomedical)",
            "input": {
                "project_title": "AI-Powered Breast Cancer Detection from Histopathology Images",
                "project_description": (
                    "This project aims to develop a deep learning-based system for early "
                    "detection of breast cancer using histopathology slide images. The system "
                    "will use convolutional neural networks, specifically a ResNet-50 architecture "
                    "fine-tuned on the BreakHis dataset, to classify tissue samples as benign or "
                    "malignant. The pipeline includes image preprocessing, data augmentation, "
                    "model training with transfer learning, and GradCAM-based explainability."
                ),
                "research_domain": "Biomedical",
                "project_type": "Computational",
                "budget_range": "5L-20L",
                "duration_months": 12,
                "team_expertise": "ML, Data Analysis, Biology",
                "location": "Chennai, Tamil Nadu",
            }
        },
        {
            "label": "Water Quality (Environmental Science)",
            "input": {
                "project_title": "Smart Water Quality Monitoring System",
                "project_description": "IoT-based system to monitor pH, turbidity, dissolved oxygen with ML anomaly detection.",
                "research_domain": "Environmental Science",
                "project_type": "Experimental",
                "budget_range": "1L-5L",
                "duration_months": 6,
                "team_expertise": "Embedded Systems, ML, Data Analysis",
                "location": "Chennai, Tamil Nadu",
            }
        },
    ]

    for t in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {t['label']}")
        print('='*60)
        result = engine.predict(t["input"])
        print(f"Detected domain   : {result['detected_domain']}")
        print(f"Detected proj type: {result['detected_project_type']}")
        print(f"Confidence        : {result['confidence_score']}")
        print("\nEQUIPMENT:")
        for eq in result["recommended_equipment"]:
            print(f"  [{eq['category']}] {eq['name']}")
        print("\nMETHODOLOGIES:")
        for m in result["recommended_methodologies"]:
            print(f"  [{m['type']}] {m['name']}")
        print("\nTOOLS:")
        for t2 in result["recommended_tools"]:
            print(f"  [{t2['category']}] {t2['name']}")