"""
Research Recommendation Platform — Synthetic Dataset Generator
Generates realistic training data across 6 research domains with proper equipment,
methodologies, tools, and project roadmaps.
Run: python generate_dataset.py
Outputs: basic_training_dataset.csv, roadmap_training_dataset.csv
"""

import uuid
import random
import csv
import textwrap
from pathlib import Path
from itertools import product as iterproduct

# ─────────────────────────────────────────────
# MASTER KNOWLEDGE BASE
# ─────────────────────────────────────────────

DOMAINS = {
    "Environmental Science": {
        "keywords": ["water quality", "air pollution", "soil contamination", "climate", "biodiversity",
                     "greenhouse gas", "remote sensing", "waste management", "ecosystem", "river monitoring",
                     "heavy metal detection", "microplastics", "carbon sequestration", "wetland mapping"],
        "equipment": [
            ("pH Sensor", "Sensor"), ("Turbidity Sensor", "Sensor"), ("Dissolved Oxygen Sensor", "Sensor"),
            ("Temperature Sensor", "Sensor"), ("Conductivity Meter", "Lab Instrument"),
            ("Spectrophotometer", "Lab Instrument"), ("GPS Device", "Field Equipment"),
            ("Water Sampler", "Field Equipment"), ("Drone (DJI Matrice)", "Field Equipment"),
            ("Raspberry Pi 4", "Computing"), ("Arduino Uno", "Computing"),
            ("Datalogger", "Field Equipment"), ("Turbidity Meter", "Lab Instrument"),
            ("Portable Water Quality Analyzer", "Field Equipment"), ("Anemometer", "Field Equipment"),
            ("Particulate Matter Sensor (PM2.5)", "Sensor"), ("CO2 Sensor", "Sensor"),
            ("Soil pH Meter", "Field Equipment"), ("Rainfall Gauge", "Field Equipment"),
            ("Breadboard & Jumper Wires", "Computing"), ("Waterproof Enclosure", "Computing"),
        ],
        "methodologies": [
            ("IoT Sensor Integration", "Experimental"), ("Time-Series Data Collection", "Experimental"),
            ("Anomaly Detection (ML)", "Computational"), ("Statistical Threshold Analysis", "Statistical"),
            ("Geospatial Mapping", "Analytical"), ("Principal Component Analysis", "Analytical"),
            ("Regression Analysis", "Statistical"), ("Remote Sensing Analysis", "Analytical"),
            ("Water Quality Index Calculation", "Analytical"), ("Random Forest Classification", "Computational"),
        ],
        "tools": [
            ("Python", "Programming"), ("scikit-learn", "Data Analysis"), ("MQTT", "Data Analysis"),
            ("Node-RED", "Visualization"), ("Grafana", "Visualization"), ("SQLite", "Data Analysis"),
            ("QGIS", "Visualization"), ("TensorFlow", "Data Analysis"), ("Pandas", "Data Analysis"),
            ("Matplotlib", "Visualization"), ("Google Earth Engine", "Simulation"),
        ],
        "project_types": ["Experimental", "Mixed"],
        "budgets": ["1L-5L", "5L-20L", "<1L"],
        "descriptions": [
            "A real-time IoT-based system to monitor water quality parameters such as pH, turbidity, and dissolved oxygen using low-cost sensors with anomaly detection using machine learning.",
            "Development of a drone-based remote sensing framework to map wetland biodiversity and detect vegetation stress using multispectral imagery and deep learning segmentation.",
            "Design of a low-cost air quality monitoring network using distributed PM2.5 and CO2 sensors across urban Chennai to detect pollution hotspots with spatial interpolation.",
            "An end-to-end soil contamination assessment pipeline combining field sampling, spectroscopic analysis, and ML-based heavy metal concentration prediction.",
            "IoT-enabled real-time river monitoring system with automated alert generation for flood prediction using LSTM-based time series forecasting.",
        ],
        "expertise": ["Embedded Systems", "ML", "Data Analysis"],
    },
    "Biomedical": {
        "keywords": ["disease detection", "genomics", "drug discovery", "medical imaging", "biosensor",
                     "protein folding", "clinical data", "wearable health", "cancer diagnosis", "EEG signal",
                     "brain-computer interface", "MRI segmentation", "pathology", "blood glucose"],
        "equipment": [
            ("Centrifuge", "Lab Instrument"), ("PCR Machine", "Lab Instrument"),
            ("Microscope (Fluorescent)", "Lab Instrument"), ("Electrophoresis Apparatus", "Lab Instrument"),
            ("ELISA Reader", "Lab Instrument"), ("Flow Cytometer", "Lab Instrument"),
            ("Biosensor Array", "Sensor"), ("ECG Sensor", "Sensor"), ("EEG Headset", "Sensor"),
            ("Blood Glucose Sensor", "Sensor"), ("High-Speed Camera", "Lab Instrument"),
            ("Autoclave", "Lab Instrument"), ("Incubator", "Lab Instrument"),
            ("Spectrophotometer", "Lab Instrument"), ("GPU Workstation (NVIDIA A100)", "Computing"),
            ("Microcontroller (ESP32)", "Computing"), ("MRI Scanner (access)", "Lab Instrument"),
        ],
        "methodologies": [
            ("PCR Amplification", "Experimental"), ("ELISA Assay", "Experimental"),
            ("Convolutional Neural Network", "Computational"), ("Transfer Learning", "Computational"),
            ("Signal Processing (FFT)", "Analytical"), ("Statistical Hypothesis Testing", "Statistical"),
            ("Logistic Regression", "Computational"), ("Support Vector Machine", "Computational"),
            ("k-Fold Cross Validation", "Statistical"), ("Federated Learning", "Computational"),
        ],
        "tools": [
            ("Python", "Programming"), ("TensorFlow", "Data Analysis"), ("PyTorch", "Data Analysis"),
            ("OpenCV", "Data Analysis"), ("MATLAB", "Simulation"), ("R", "Data Analysis"),
            ("SPSS", "Data Analysis"), ("Keras", "Data Analysis"), ("scikit-learn", "Data Analysis"),
            ("Jupyter Notebook", "Programming"), ("ITK-SNAP", "Visualization"),
        ],
        "project_types": ["Experimental", "Computational", "Mixed"],
        "budgets": ["5L-20L", "20L-1Cr", "1L-5L"],
        "descriptions": [
            "A deep learning pipeline for early-stage breast cancer detection from mammogram images using ResNet-50 transfer learning with GradCAM-based explainability.",
            "Development of a wearable ECG patch with real-time arrhythmia classification using a lightweight CNN deployed on an edge microcontroller.",
            "Federated learning framework for multi-hospital diabetic retinopathy grading ensuring privacy-preserving collaborative model training.",
            "ELISA-based multiplex biosensor for simultaneous detection of three cancer biomarkers with ML-assisted result interpretation.",
            "EEG-based brain-computer interface for motor-impaired patients using LSTM classification of imagined movement signals.",
        ],
        "expertise": ["Biology", "ML", "Data Analysis"],
    },
    "CS-AI": {
        "keywords": ["natural language processing", "computer vision", "reinforcement learning", "knowledge graph",
                     "recommendation system", "autonomous agent", "speech recognition", "transformer model",
                     "federated learning", "anomaly detection", "edge AI", "generative AI", "chatbot", "OCR"],
        "equipment": [
            ("GPU Workstation (NVIDIA RTX 4090)", "Computing"), ("TPU Cloud Instance", "Computing"),
            ("Raspberry Pi 5", "Computing"), ("Jetson Nano", "Computing"),
            ("High-Resolution Camera", "Sensor"), ("Microphone Array", "Sensor"),
            ("LiDAR Sensor", "Sensor"), ("Tactile Sensor", "Sensor"),
            ("Network Switch (10GbE)", "Computing"), ("NAS Storage (100TB)", "Computing"),
        ],
        "methodologies": [
            ("Transformer Fine-Tuning", "Computational"), ("Reinforcement Learning from Human Feedback", "Computational"),
            ("A/B Testing", "Statistical"), ("Ablation Study", "Analytical"),
            ("RAG Pipeline", "Computational"), ("Graph Neural Network", "Computational"),
            ("Active Learning", "Computational"), ("Hyperparameter Optimization (Optuna)", "Computational"),
            ("Benchmark Evaluation", "Analytical"), ("Monte Carlo Simulation", "Computational"),
        ],
        "tools": [
            ("Python", "Programming"), ("PyTorch", "Data Analysis"), ("Hugging Face Transformers", "Data Analysis"),
            ("LangChain", "Programming"), ("FastAPI", "Programming"), ("Docker", "Simulation"),
            ("Kubernetes", "Simulation"), ("MLflow", "Visualization"), ("Weights & Biases", "Visualization"),
            ("PostgreSQL", "Data Analysis"), ("Redis", "Programming"),
        ],
        "project_types": ["Computational", "Mixed"],
        "budgets": ["1L-5L", "5L-20L", "20L-1Cr"],
        "descriptions": [
            "A retrieval-augmented generation (RAG) based question answering system for Indian legal documents using LLaMA-3 fine-tuned on IPC and CrPC text corpora.",
            "Real-time object detection and tracking system for traffic management using YOLOv9 with edge deployment on NVIDIA Jetson Nano.",
            "Multilingual hate speech detection model for Tamil and Telugu social media using cross-lingual transformer embeddings and adversarial training.",
            "Knowledge graph construction from unstructured biomedical literature using NLP entity extraction and graph neural network link prediction.",
            "Autonomous drone navigation using deep reinforcement learning with PPO algorithm for obstacle avoidance in GPS-denied indoor environments.",
        ],
        "expertise": ["ML", "Data Analysis", "Programming"],
    },
    "Electronics": {
        "keywords": ["VLSI design", "FPGA", "embedded systems", "power electronics", "RF circuit",
                     "PCB design", "motor control", "signal processing", "IoT hardware", "MEMS sensor",
                     "wireless communication", "energy harvesting", "analog circuit", "smart grid"],
        "equipment": [
            ("Oscilloscope (4-channel)", "Lab Instrument"), ("Function Generator", "Lab Instrument"),
            ("Digital Multimeter", "Lab Instrument"), ("Spectrum Analyzer", "Lab Instrument"),
            ("Logic Analyzer", "Lab Instrument"), ("FPGA Board (Xilinx Artix-7)", "Computing"),
            ("Soldering Station", "Fabrication"), ("PCB Prototyping Machine", "Fabrication"),
            ("RF Signal Generator", "Lab Instrument"), ("Power Supply (Variable)", "Lab Instrument"),
            ("Arduino Mega", "Computing"), ("STM32 Nucleo Board", "Computing"),
            ("3D Printer (FDM)", "Fabrication"), ("Motor Driver Module", "Computing"),
            ("CRO (Cathode Ray Oscilloscope)", "Lab Instrument"),
        ],
        "methodologies": [
            ("VLSI RTL Design", "Experimental"), ("Hardware-in-Loop Simulation", "Computational"),
            ("FFT Signal Analysis", "Analytical"), ("PID Control Design", "Analytical"),
            ("Thermal Analysis", "Experimental"), ("Finite Element Analysis", "Computational"),
            ("Monte Carlo Fault Simulation", "Computational"), ("Regression Testing (Hardware)", "Experimental"),
            ("Eye Diagram Analysis", "Analytical"), ("LTSpice Circuit Simulation", "Computational"),
        ],
        "tools": [
            ("MATLAB/Simulink", "Simulation"), ("Cadence Virtuoso", "Design"), ("Altium Designer", "Design"),
            ("Xilinx Vivado", "Design"), ("KiCad", "Design"), ("LTSpice", "Simulation"),
            ("Python", "Programming"), ("Proteus", "Simulation"), ("Eagle PCB", "Design"),
            ("LabVIEW", "Visualization"), ("C/C++", "Programming"),
        ],
        "project_types": ["Experimental", "Mixed"],
        "budgets": ["1L-5L", "5L-20L", "<1L"],
        "descriptions": [
            "Design and fabrication of a low-power FPGA-based real-time signal processing core for radar pulse compression with 200 MHz bandwidth.",
            "Development of an energy harvesting circuit for piezoelectric sensors in structural health monitoring with autonomous power management.",
            "Smart motor controller for BLDC motors using STM32 with FOC algorithm and CAN bus communication for EV applications.",
            "RF front-end design for a 2.4 GHz IoT transceiver with noise figure optimization and on-PCB antenna integration.",
            "VLSI implementation of a fast Fourier transform accelerator using pipelined Cooley-Tukey algorithm in 28nm CMOS technology.",
        ],
        "expertise": ["Electronics", "Embedded Systems", "Programming"],
    },
    "Materials Science": {
        "keywords": ["nanomaterials", "thin film", "corrosion", "composite material", "crystal structure",
                     "polymer synthesis", "battery electrode", "semiconductor fabrication", "tribology",
                     "thermal conductivity", "XRD characterization", "SEM imaging", "tensile strength"],
        "equipment": [
            ("X-Ray Diffractometer (XRD)", "Lab Instrument"), ("Scanning Electron Microscope (SEM)", "Lab Instrument"),
            ("Transmission Electron Microscope (TEM)", "Lab Instrument"), ("AFM (Atomic Force Microscope)", "Lab Instrument"),
            ("Muffle Furnace", "Lab Instrument"), ("Ball Mill", "Lab Instrument"),
            ("Universal Testing Machine (UTM)", "Lab Instrument"), ("Electrochemical Workstation", "Lab Instrument"),
            ("Spin Coater", "Fabrication"), ("Sputtering System", "Fabrication"),
            ("UV-Vis Spectrophotometer", "Lab Instrument"), ("Thermogravimetric Analyzer (TGA)", "Lab Instrument"),
            ("Raman Spectrometer", "Lab Instrument"), ("Hardness Tester", "Lab Instrument"),
        ],
        "methodologies": [
            ("XRD Phase Analysis", "Analytical"), ("SEM-EDS Characterization", "Analytical"),
            ("Cyclic Voltammetry", "Experimental"), ("Tensile Testing", "Experimental"),
            ("Sol-Gel Synthesis", "Experimental"), ("Chemical Vapor Deposition", "Experimental"),
            ("Density Functional Theory (DFT)", "Computational"), ("Molecular Dynamics Simulation", "Computational"),
            ("Taguchi Design of Experiments", "Statistical"), ("Nanoindentation", "Experimental"),
        ],
        "tools": [
            ("MATLAB", "Simulation"), ("VASP (DFT)", "Simulation"), ("LAMMPS", "Simulation"),
            ("Origin Pro", "Data Analysis"), ("ImageJ", "Data Analysis"), ("Python", "Programming"),
            ("AutoCAD", "Design"), ("ANSYS", "Simulation"), ("HighScore Plus (XRD)", "Data Analysis"),
            ("R", "Data Analysis"), ("Crystal Maker", "Visualization"),
        ],
        "project_types": ["Experimental", "Computational", "Mixed"],
        "budgets": ["5L-20L", "20L-1Cr", "1L-5L"],
        "descriptions": [
            "Synthesis and characterization of TiO2-graphene nanocomposites for photocatalytic degradation of textile dye effluents under visible light irradiation.",
            "Development of high-entropy alloy coatings via magnetron sputtering for tribological performance in extreme temperature aerospace applications.",
            "Electrochemical fabrication and characterization of MnO2-based supercapacitor electrodes with optimized nanostructure for high energy density.",
            "DFT and molecular dynamics study of lithium-ion diffusion kinetics in solid-state electrolyte interfaces for next-generation battery design.",
            "Fabrication of carbon fiber reinforced polymer composites with nano-clay filler and evaluation of interlaminar shear strength under fatigue loading.",
        ],
        "expertise": ["Chemistry", "Data Analysis", "Programming"],
    },
    "Civil": {
        "keywords": ["structural analysis", "geotechnical", "concrete strength", "traffic flow", "smart city",
                     "earthquake resistance", "flood modeling", "pavement design", "bridge health monitoring",
                     "groundwater", "urban planning", "construction management", "BIM", "water supply"],
        "equipment": [
            ("Total Station (Survey)", "Field Equipment"), ("GPS RTK System", "Field Equipment"),
            ("Compression Testing Machine (CTM)", "Lab Instrument"), ("Core Cutter", "Lab Instrument"),
            ("Penetrometer", "Field Equipment"), ("Theodolite", "Field Equipment"),
            ("Strain Gauge", "Sensor"), ("Accelerometer", "Sensor"),
            ("Piezometer", "Field Equipment"), ("Schmidt Hammer", "Field Equipment"),
            ("Triaxial Test Apparatus", "Lab Instrument"), ("Proctor Compaction Test", "Lab Instrument"),
            ("Traffic Counter Sensor", "Field Equipment"), ("Drone (Survey Grade)", "Field Equipment"),
        ],
        "methodologies": [
            ("Finite Element Analysis (FEA)", "Computational"), ("HEC-RAS Flood Modeling", "Computational"),
            ("Marshall Mix Design", "Experimental"), ("Regression Analysis", "Statistical"),
            ("Non-Destructive Testing (NDT)", "Experimental"), ("SCADA Monitoring", "Experimental"),
            ("Taguchi Optimization", "Statistical"), ("Geotechnical Boring", "Experimental"),
            ("BIM-based Scheduling", "Analytical"), ("Monte Carlo Risk Analysis", "Computational"),
        ],
        "tools": [
            ("STAAD Pro", "Simulation"), ("AutoCAD Civil 3D", "Design"), ("SAP2000", "Simulation"),
            ("ETABS", "Simulation"), ("MATLAB", "Simulation"), ("Python", "Programming"),
            ("ArcGIS", "Visualization"), ("HEC-RAS", "Simulation"), ("Revit (BIM)", "Design"),
            ("MS Project", "Data Analysis"), ("SPSS", "Data Analysis"),
        ],
        "project_types": ["Experimental", "Computational", "Survey-based", "Mixed"],
        "budgets": ["5L-20L", "20L-1Cr", "1L-5L"],
        "descriptions": [
            "Structural health monitoring of a pre-stressed concrete bridge using IoT-based vibration sensors and ML-based damage classification.",
            "Flood inundation mapping for the Adyar river basin using HEC-RAS 2D model calibrated with satellite imagery and rainfall data.",
            "Development of high-performance recycled aggregate concrete with industrial byproducts and ML prediction of compressive strength.",
            "Smart pavement monitoring system using embedded strain gauges and piezoelectric sensors for real-time traffic load characterization.",
            "BIM-integrated construction project scheduling optimization using genetic algorithm for resource leveling in high-rise building projects.",
        ],
        "expertise": ["Civil Engineering", "Data Analysis", "Programming"],
    },
}

LOCATIONS = [
    "Chennai, Tamil Nadu", "Bengaluru, Karnataka", "Mumbai, Maharashtra", "Delhi, NCR",
    "Hyderabad, Telangana", "Pune, Maharashtra", "Coimbatore, Tamil Nadu", "Kolkata, West Bengal",
    "Ahmedabad, Gujarat", "Jaipur, Rajasthan", "Kochi, Kerala", "Chandigarh, Punjab",
    "Lucknow, Uttar Pradesh", "Bhubaneswar, Odisha", "Guwahati, Assam",
]

ROADMAP_PHASES = {
    "Experimental": [
        ("Literature Review & Problem Formulation", [
            "Conduct systematic literature review on topic",
            "Identify research gaps and define problem statement",
            "Prepare literature review document (min 30 papers)",
            "Define hypothesis and success criteria",
        ]),
        ("Design & Procurement", [
            "Finalize system architecture and block diagram",
            "Prepare Bill of Materials (BoM)",
            "Procure equipment and consumables",
            "Set up experimental workspace and safety protocols",
        ]),
        ("Prototype Development", [
            "Assemble hardware components",
            "Develop initial firmware/software",
            "Perform bench-level unit tests",
            "Document assembly and calibration procedure",
        ]),
        ("Data Collection", [
            "Design data collection protocol",
            "Run controlled experiments with replication",
            "Log raw data with timestamps",
            "Perform initial quality check on collected data",
        ]),
        ("Analysis & Modeling", [
            "Clean and preprocess collected data",
            "Apply statistical/ML models",
            "Visualize results and generate insights",
            "Conduct sensitivity analysis",
        ]),
        ("Validation & Testing", [
            "Compare results against baseline or benchmarks",
            "Conduct blind validation experiments",
            "Document accuracy, precision, recall metrics",
            "Identify failure modes and limitations",
        ]),
        ("Reporting & Dissemination", [
            "Write final project report",
            "Prepare journal/conference manuscript",
            "Create poster or demo presentation",
            "Archive code, data, and documentation on GitHub/Zenodo",
        ]),
    ],
    "Computational": [
        ("Problem Scoping & Literature Review", [
            "Define problem formally with inputs, outputs, and objectives",
            "Review state-of-the-art models and benchmarks",
            "Identify datasets and evaluation metrics",
            "Prepare related work summary",
        ]),
        ("Dataset Preparation", [
            "Identify and download relevant public datasets",
            "Perform exploratory data analysis (EDA)",
            "Handle missing values, outliers, and class imbalance",
            "Split data into train/validation/test sets",
        ]),
        ("Model Architecture Design", [
            "Propose candidate model architectures",
            "Implement baseline model",
            "Define loss functions and evaluation metrics",
            "Set up experiment tracking (MLflow/W&B)",
        ]),
        ("Training & Optimization", [
            "Train baseline model and log results",
            "Perform hyperparameter tuning (grid/random/Bayesian search)",
            "Apply regularization and augmentation strategies",
            "Analyze learning curves and convergence",
        ]),
        ("Evaluation & Ablation", [
            "Evaluate on held-out test set",
            "Conduct ablation studies for each component",
            "Compare with state-of-the-art benchmarks",
            "Perform error analysis on failure cases",
        ]),
        ("Deployment & Reporting", [
            "Package model as REST API (FastAPI/Flask)",
            "Containerize with Docker",
            "Write final report and create demo",
            "Publish code and model weights on GitHub",
        ]),
    ],
    "Survey-based": [
        ("Research Design", [
            "Define survey objectives and target population",
            "Perform power analysis to determine sample size",
            "Design questionnaire with validated scales",
            "Get IRB/ethics committee approval",
        ]),
        ("Pilot Survey & Refinement", [
            "Conduct pilot with 20-30 respondents",
            "Assess Cronbach's alpha for reliability",
            "Refine ambiguous questions",
            "Finalize survey instrument",
        ]),
        ("Data Collection", [
            "Deploy survey via Google Forms / KoBoToolbox",
            "Monitor response rate and send reminders",
            "Ensure geographic and demographic diversity",
            "Close data collection at target sample size",
        ]),
        ("Statistical Analysis", [
            "Perform descriptive statistics",
            "Test normality and homogeneity assumptions",
            "Run correlation/regression or structural equation modeling",
            "Visualize results with appropriate charts",
        ]),
        ("Reporting", [
            "Interpret findings in context of literature",
            "Report limitations and generalizability",
            "Write manuscript for journal submission",
            "Share anonymized dataset on OSF/Zenodo",
        ]),
    ],
    "Mixed": [
        ("Literature Review", [
            "Perform systematic literature review",
            "Identify experimental and computational components",
            "Define hybrid methodology rationale",
        ]),
        ("Experimental Setup", [
            "Procure equipment and prepare lab setup",
            "Conduct preliminary experiments",
            "Collect baseline experimental data",
        ]),
        ("Computational Modeling", [
            "Develop simulation or ML model",
            "Calibrate model with experimental data",
            "Perform virtual parameter sweeps",
        ]),
        ("Integrated Analysis", [
            "Cross-validate experimental and computational results",
            "Identify discrepancies and perform root-cause analysis",
            "Optimize design based on integrated findings",
        ]),
        ("Validation", [
            "Conduct confirmatory experiments",
            "Benchmark against existing literature",
            "Document uncertainty and error bounds",
        ]),
        ("Reporting", [
            "Prepare final report combining both methodologies",
            "Submit to conference or journal",
            "Archive all artifacts",
        ]),
    ],
}

# ─────────────────────────────────────────────
# GENERATOR FUNCTIONS
# ─────────────────────────────────────────────

def new_id(prefix="PRJ"):
    return f"{prefix}_{uuid.uuid4().hex[:6].upper()}"

def random_subset(lst, min_k=2, max_k=4):
    k = random.randint(min_k, min(max_k, len(lst)))
    return random.sample(lst, k)

def generate_description(domain_name, domain_data, project_type):
    base = random.choice(domain_data["descriptions"])
    extras = [
        "The project will leverage open-source tools wherever possible to maximize reproducibility.",
        "Results will be validated against publicly available benchmark datasets.",
        "The system is designed for deployment in resource-constrained environments.",
        "A web dashboard will be developed to present results in real-time.",
        "Data privacy and security considerations are incorporated into the design.",
        "The methodology includes a rigorous statistical validation framework.",
    ]
    full = base + " " + random.choice(extras)
    return full

def generate_basic_rows(domain_name, domain_data, project_id, title):
    project_type = random.choice(domain_data["project_types"])
    budget = random.choice(domain_data["budgets"])
    duration = random.choice([3, 6, 9, 12, 18, 24])
    expertise = ", ".join(random.sample(domain_data["expertise"] + ["Electronics", "Biology", "Chemistry"][:2], k=min(3, len(domain_data["expertise"]))))
    location = random.choice(LOCATIONS)
    description = generate_description(domain_name, domain_data, project_type)

    base_row = {
        "project_id": project_id,
        "project_title": title,
        "project_description": description,
        "research_domain": domain_name,
        "project_type": project_type,
        "budget_range": budget,
        "duration_months": duration,
        "team_expertise": expertise,
        "location": location,
    }

    equipment_rows = []
    for eq_name, eq_cat in random_subset(domain_data["equipment"], 4, 7):
        row = dict(base_row)
        row["equipment_name"] = eq_name
        row["equipment_category"] = eq_cat
        equipment_rows.append(row)

    methodology_rows = []
    for meth_name, meth_type in random_subset(domain_data["methodologies"], 2, 4):
        row = dict(base_row)
        row["methodology_name"] = meth_name
        row["methodology_type"] = meth_type
        row["skill_required"] = random.choice(domain_data["expertise"])
        methodology_rows.append(row)

    tool_rows = []
    for tool_name, tool_cat in random_subset(domain_data["tools"], 3, 5):
        row = dict(base_row)
        row["tool_name"] = tool_name
        row["tool_category"] = tool_cat
        tool_rows.append(row)

    return base_row, equipment_rows, methodology_rows, tool_rows, project_type

def generate_roadmap_rows(project_id, project_type):
    phases = ROADMAP_PHASES.get(project_type, ROADMAP_PHASES["Mixed"])
    rows = []
    for i, (phase_name, steps) in enumerate(phases, start=1):
        rows.append({
            "roadmap_id": new_id("RDM"),
            "project_id": project_id,
            "phase_number": i,
            "phase_name": phase_name,
            "steps": " | ".join([f"{j+1}. {s}" for j, s in enumerate(steps)]),
        })
    return rows

def generate_title(domain_name, domain_data):
    kw1 = random.choice(domain_data["keywords"]).title()
    kw2 = random.choice(domain_data["keywords"]).title()
    prefixes = [
        f"IoT-Based {kw1} System for {kw2} Analysis",
        f"ML-Driven {kw1} Detection Framework",
        f"Real-Time {kw1} Monitoring with {kw2}",
        f"Deep Learning Approach to {kw1}",
        f"Low-Cost {kw1} Platform for {kw2}",
        f"Automated {kw1} Classification System",
        f"Smart {kw1} Prediction Using {kw2}",
        f"Multi-Modal {kw1} Assessment System",
        f"Embedded {kw1} Controller for {kw2}",
        f"Data-Driven {kw1} Optimization Framework",
    ]
    return random.choice(prefixes)

# ─────────────────────────────────────────────
# MAIN GENERATION
# ─────────────────────────────────────────────

def generate_datasets(total_projects=500):
    random.seed(42)

    # Distribute projects across domains
    projects_per_domain = total_projects // len(DOMAINS)

    basic_fieldnames = [
        "project_id", "project_title", "project_description", "research_domain",
        "project_type", "budget_range", "duration_months", "team_expertise", "location",
        "equipment_name", "equipment_category",
        "methodology_name", "methodology_type", "skill_required",
        "tool_name", "tool_category",
    ]

    roadmap_fieldnames = [
        "roadmap_id", "project_id", "phase_number", "phase_name", "steps",
    ]

    all_basic_rows = []
    all_roadmap_rows = []

    for domain_name, domain_data in DOMAINS.items():
        print(f"  Generating {projects_per_domain} projects for domain: {domain_name}")
        for _ in range(projects_per_domain):
            project_id = new_id("PRJ")
            title = generate_title(domain_name, domain_data)
            base, eq_rows, meth_rows, tool_rows, ptype = generate_basic_rows(domain_name, domain_data, project_id, title)

            # Merge: create one row per (equipment, methodology, tool) combination (flattened)
            max_len = max(len(eq_rows), len(meth_rows), len(tool_rows))
            for i in range(max_len):
                row = dict(base)
                if i < len(eq_rows):
                    row["equipment_name"] = eq_rows[i]["equipment_name"]
                    row["equipment_category"] = eq_rows[i]["equipment_category"]
                else:
                    row["equipment_name"] = ""
                    row["equipment_category"] = ""
                if i < len(meth_rows):
                    row["methodology_name"] = meth_rows[i]["methodology_name"]
                    row["methodology_type"] = meth_rows[i]["methodology_type"]
                    row["skill_required"] = meth_rows[i]["skill_required"]
                else:
                    row["methodology_name"] = ""
                    row["methodology_type"] = ""
                    row["skill_required"] = ""
                if i < len(tool_rows):
                    row["tool_name"] = tool_rows[i]["tool_name"]
                    row["tool_category"] = tool_rows[i]["tool_category"]
                else:
                    row["tool_name"] = ""
                    row["tool_category"] = ""
                all_basic_rows.append(row)

            all_roadmap_rows.extend(generate_roadmap_rows(project_id, ptype))

    # Write next to this script (portable; works on Windows / Linux / macOS)
    out_dir = Path(__file__).resolve().parent.parent / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write basic dataset
    basic_path = out_dir / "basic_training_dataset.csv"
    with open(basic_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=basic_fieldnames)
        writer.writeheader()
        writer.writerows(all_basic_rows)
    print(f"\n✅ Basic training dataset saved: {basic_path}  ({len(all_basic_rows)} rows)")

    # Write roadmap dataset
    roadmap_path = out_dir /"roadmap_training_dataset.csv"
    with open(roadmap_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=roadmap_fieldnames)
        writer.writeheader()
        writer.writerows(all_roadmap_rows)
    print(f"✅ Roadmap training dataset saved: {roadmap_path}  ({len(all_roadmap_rows)} rows)\n")

    return basic_path, roadmap_path


if __name__ == "__main__":
    print("=" * 60)
    print("  Research Recommendation Platform — Dataset Generator")
    print("=" * 60)
    generate_datasets(total_projects=600)
    print("Done! Both CSV files are ready for model training.")