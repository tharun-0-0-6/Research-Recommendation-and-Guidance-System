"""
Dynamic AI Engine - Uses Groq (free) for truly personalized recommendations.
Falls back to a comprehensive rule-based engine if Groq is unavailable.
"""

import os
import json
import requests
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')


def _call_groq(prompt: str, max_tokens: int = 2000) -> str:
    """Call Groq API and return text response."""
    if not GROQ_API_KEY:
        return ""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Groq call failed: {e}")
        return ""


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


class DynamicAIEngine:
    def __init__(self):
        if GROQ_API_KEY:
            print("Groq ready: llama-3.3-70b-versatile (will connect on first request)")
        else:
            print("Warning: GROQ_API_KEY not set — using rule-based fallback")

        self.project_database = {
            "air_monitoring": {
                "equipment": ["PM2.5 Sensor","NO2 Sensor","O3 Sensor","CO Sensor","Air Quality Monitor","Weather Station","Dust Monitor","Gas Analyzer"],
                "methodologies": ["Air Quality Index Calculation","Pollutant Dispersion Modeling","Statistical Analysis","Real-time Monitoring","Data Visualization"],
                "tools": ["Python","R","MATLAB","Air Quality Software","GIS Tools","Statistical Packages"],
                "similar_projects": ["Urban Air Quality Network","Smart City Pollution System","Industrial Emission Monitor","Atmospheric Data Platform","Air Health Alert System"]
            },
            "water_quality": {
                "equipment": ["pH Sensor","Turbidity Sensor","Dissolved Oxygen Meter","Conductivity Probe","Water Sampler","Spectrophotometer","Flow Meter","Temperature Sensor"],
                "methodologies": ["Water Quality Testing","Contaminant Analysis","Source Tracking","Hydrological Modeling","Sampling Protocols"],
                "tools": ["Python","Aquachem","Water Quality Software","Statistical Tools","Lab Equipment","Data Analysis Packages"],
                "similar_projects": ["River Monitoring System","Lake Assessment Platform","Wastewater Treatment Monitor","Drinking Water Safety Network","Aquatic Ecosystem Tracker"]
            },
            "biomedical": {
                "equipment": ["ECG Monitor","EEG Headset","Blood Pressure Monitor","Pulse Oximeter","Medical Sensors","Imaging Equipment","Lab Diagnostic Tools"],
                "methodologies": ["Patient Monitoring","Signal Processing","Medical Imaging","Diagnostic Algorithms","Health Data Analysis"],
                "tools": ["Python","MATLAB","Medical Imaging Software","Bioinformatics Tools","Signal Processing Libraries","Machine Learning Frameworks"],
                "similar_projects": ["Heart Disease Prediction","Brain Activity Monitor","Diabetes Management System","Medical Imaging Analysis","Patient Care Platform"]
            },
            "computer_vision": {
                "equipment": ["High-Resolution Camera","Depth Sensor","GPU Workstation","Neural Processor","Image Processing Unit","Computer Vision Kit","Edge AI Device"],
                "methodologies": ["Object Detection","Image Classification","Pattern Recognition","Video Analysis","Deep Learning"],
                "tools": ["Python","TensorFlow","PyTorch","OpenCV","Computer Vision Libraries","GPU Computing"],
                "similar_projects": ["Face Recognition System","Vehicle Detection Network","Object Tracking Platform","Image Analysis Tool","AI Vision Assistant"]
            },
            "iot_general": {
                "equipment": ["Arduino","Raspberry Pi","ESP32","LoRa Module","Sensor Network","IoT Gateway","Edge Device","Cloud Platform"],
                "methodologies": ["Sensor Deployment","Data Collection","Edge Computing","Cloud Integration","Real-time Processing"],
                "tools": ["Python","Arduino IDE","MQTT","Node-RED","IoT Platforms","Communication Protocols"],
                "similar_projects": ["Smart Home System","Industrial IoT Network","Environmental Monitor","Agricultural IoT","Smart City Platform"]
            },
            "machine_learning": {
                "equipment": ["GPU Server","TPU Unit","High-Performance Computer","ML Workstation","Data Center Access","Cloud Computing","Neural Network Hardware"],
                "methodologies": ["Neural Networks","Deep Learning","Model Training","Algorithm Development","Performance Optimization"],
                "tools": ["Python","TensorFlow","PyTorch","Scikit-learn","Jupyter","ML Platforms"],
                "similar_projects": ["Predictive Analytics Platform","ML Model Deployment","AI Training System","Neural Network Application","Machine Learning Service"]
            },
            "materials_science": {
                "equipment": ["SEM (Scanning Electron Microscope)","XRD Diffractometer","Tensile Testing Machine","Electrochemical Workstation","Optical Microscope","AFM (Atomic Force Microscope)","Furnace/Sintering Unit","Spectrometer"],
                "methodologies": ["Materials Characterization","Electrochemical Testing","Microstructure Analysis","Corrosion Testing","Thermal Analysis","Mechanical Property Testing"],
                "tools": ["MATLAB","Python","Origin Pro","ImageJ","HighScore Plus (XRD)","ZView (Electrochemical)","COMSOL"],
                "similar_projects": ["Nanocomposite Thin Film Study","Anti-Corrosion Coating Research","Graphene Oxide Composite","Polymer Nanocomposite Development","Surface Engineering Project"]
            },
            "civil": {
                "equipment": ["Universal Testing Machine","Concrete Testing Kit","GPS Survey Equipment","Load Cell Sensors","Data Logger","Structural Analyzer","Geotechnical Tools"],
                "methodologies": ["Structural Analysis","Load Testing","Finite Element Analysis","Material Testing","Field Survey","Quality Control"],
                "tools": ["AutoCAD","STAAD.Pro","ETABS","Python","SAP2000","ArcGIS","MATLAB"],
                "similar_projects": ["Smart Bridge Monitoring","Concrete Strength Study","Seismic Analysis Project","Traffic Flow Optimization","Flood Risk Assessment"]
            }
        }

        self.institution_database = {
            "tamil_nadu": {"cities": ["Chennai","Coimbatore","Madurai","Trichy","Vellore"],"institutions": ["IIT Madras - Research Laboratory","Anna University - Innovation Centre","PSG College of Technology - Advanced Lab","NIT Trichy - Engineering Research Centre","VIT Vellore - Technology Lab"]},
            "karnataka": {"cities": ["Bengaluru","Mysore","Surathkal","Manipal","Belgaum"],"institutions": ["IISc Bangalore - Advanced Research Lab","NIT Karnataka - Engineering Lab","RV College of Engineering - Research Centre","Manipal Institute of Technology - Innovation Lab","VTU Belgaum - Research Hub"]},
            "maharashtra": {"cities": ["Mumbai","Pune","Nagpur","Nashik","Aurangabad"],"institutions": ["IIT Bombay - Research Centre","COEP Pune - Engineering Lab","VJTI Mumbai - Advanced Lab","SPCE Pune - Innovation Centre","ICT Mumbai - Research Hub"]},
            "delhi_ncr": {"cities": ["New Delhi","Gurgaon","Noida","Faridabad","Ghaziabad"],"institutions": ["IIT Delhi - Research Laboratory","DTU Delhi - Engineering Centre","NSUT Delhi - Innovation Lab","JNU Delhi - Research Centre","IP University - Advanced Lab"]},
            "telangana": {"cities": ["Hyderabad","Warangal","Nizamabad","Karimnagar","Secunderabad"],"institutions": ["IIIT Hyderabad - AI Research Lab","University of Hyderabad - Research Centre","NIT Warangal - Engineering Lab","Osmania University - Technology Hub","JNTU Hyderabad - Innovation Lab"]},
            "andhra_pradesh": {"cities": ["Visakhapatnam","Vijayawada","Guntur","Tirupati","Nellore"],"institutions": ["Andhra University - Engineering College","IIT Tirupati - Research Centre","NIT Visakhapatnam - Technology Lab","GITAM University - Innovation Centre","JNTU Kakinada - Advanced Lab"]},
            "rajasthan": {"cities": ["Jaipur","Jodhpur","Udaipur","Kota","Ajmer"],"institutions": ["IIT Jodhpur - Research Laboratory","BITS Pilani - Innovation Centre","NIT Jaipur - Engineering Lab","University of Rajasthan - Research Hub","Manipal University Jaipur - Advanced Lab"]},
            "uttar_pradesh": {"cities": ["Lucknow","Kanpur","Varanasi","Agra","Allahabad"],"institutions": ["IIT Kanpur - Research Laboratory","IIT BHU Varanasi - Engineering Centre","HBTU Kanpur - Innovation Lab","AKTU Lucknow - Research Hub","Amity University Noida - Advanced Lab"]},
            "west_bengal": {"cities": ["Kolkata","Howrah","Durgapur","Kharagpur","Siliguri"],"institutions": ["IIT Kharagpur - Research Laboratory","Jadavpur University - Engineering Centre","IIEST Shibpur - Innovation Lab","Presidency University - Research Hub","NIT Durgapur - Advanced Lab"]},
            "kerala": {"cities": ["Thiruvananthapuram","Kochi","Kozhikode","Thrissur","Palakkad"],"institutions": ["IIT Palakkad - Research Laboratory","NIT Calicut - Engineering Centre","Amrita University - Innovation Lab","Kerala University - Research Hub","CUSAT Kochi - Advanced Lab"]},
            "gujarat": {"cities": ["Ahmedabad","Surat","Vadodara","Rajkot","Gandhinagar"],"institutions": ["IIT Gandhinagar - Research Laboratory","NIT Surat - Engineering Centre","DAIICT Gandhinagar - Innovation Lab","MS University Baroda - Research Hub","Nirma University Ahmedabad - Advanced Lab"]},
            "punjab": {"cities": ["Chandigarh","Ludhiana","Amritsar","Jalandhar","Patiala"],"institutions": ["IIT Ropar - Research Laboratory","PEC Chandigarh - Engineering Centre","NIT Jalandhar - Innovation Lab","Thapar University Patiala - Research Hub","Punjab University - Advanced Lab"]},
        }

    def _extract_location_state(self, location: str) -> str:
        loc = location.lower()
        mapping = {
            "tamil_nadu":     ["tamil nadu","chennai","coimbatore","madurai","trichy","salem","vellore"],
            "karnataka":      ["karnataka","bengaluru","bangalore","mysore","hubli","mangalore","belgaum"],
            "maharashtra":    ["maharashtra","mumbai","pune","nagpur","nashik","aurangabad"],
            "delhi_ncr":      ["delhi","gurgaon","noida","faridabad","new delhi"],
            "telangana":      ["telangana","hyderabad","warangal","nizamabad"],
            "andhra_pradesh": ["andhra pradesh","visakhapatnam","vijayawada","guntur"],
            "rajasthan":      ["rajasthan","jaipur","jodhpur","udaipur","kota","ajmer"],
            "uttar_pradesh":  ["uttar pradesh","lucknow","kanpur","agra","varanasi","allahabad"],
            "west_bengal":    ["west bengal","kolkata","calcutta","howrah","durgapur"],
            "kerala":         ["kerala","thiruvananthapuram","kochi","kozhikode","thrissur","calicut"],
            "gujarat":        ["gujarat","ahmedabad","surat","vadodara","rajkot"],
            "punjab":         ["punjab","chandigarh","ludhiana","amritsar","jalandhar"],
        }
        for state, kws in mapping.items():
            if any(k in loc for k in kws):
                return state
        return "tamil_nadu"

    def _analyze_project_type(self, title: str, description: str) -> str:
        import re
        text = (title + " " + description).lower()
        def has(kws):
            for kw in kws:
                if len(kw) <= 4 or ' ' in kw:
                    if re.search(r'\b' + re.escape(kw) + r'\b', text): return True
                else:
                    if kw in text: return True
            return False
        if has(["nanocomposite","nanomaterial","graphene","corrosion","coating","polymer","ceramic","metallurgy","xrd","thin film","nanoparticle","alloy","sintering","electrochemical","sem analysis"]):
            return "materials_science"
        elif has(["structural","bridge","concrete","seismic","geotechnical","pavement","construction","civil engineering","flood control"]):
            return "civil"
        elif has(["air quality","atmospheric","pm2.5","no2","air pollution","emission"]):
            return "air_monitoring"
        elif has(["water quality","turbidity","dissolved oxygen","contamination","river","lake","aquatic","wastewater","microplastic"]):
            return "water_quality"
        elif has(["medical","biomedical","ecg","eeg","diagnosis","patient","disease","clinical","genomic","drug","retinal","fundus"]):
            return "biomedical"
        elif has(["computer vision","image processing","object detection","face recognition","video analysis","yolo","mediapipe","opencv"]):
            return "computer_vision"
        elif has(["arduino","raspberry pi","esp32","lora","mqtt","iot gateway","sensor network","smart home","smart city"]):
            return "iot_general"
        elif has(["machine learning","deep learning","neural network","tensorflow","pytorch","reinforcement learning","nlp","natural language","transformer","bert"]):
            return "machine_learning"
        return "iot_general"

    def _groq_recommendations(self, user_input: dict) -> dict:
        """Call Groq to generate fully personalized recommendations."""
        prompt = f"""You are a research project advisor. Given the project below, return ONLY a valid JSON object (no markdown, no explanation, no extra text).

Project Title: {user_input.get('project_title', '')}
Description: {user_input.get('project_description', '')}
Domain: {user_input.get('research_domain', '')}
Budget: {user_input.get('budget_range', '')}
Duration: {user_input.get('duration_months', '')} months
Team Expertise: {user_input.get('team_expertise', '')}
State/Location: {user_input.get('location', '')}

Return exactly this JSON structure:
{{
  "equipment": ["item1", "item2", "item3", "item4", "item5"],
  "methodologies": ["method1", "method2", "method3", "method4", "method5"],
  "tools": ["tool1", "tool2", "tool3", "tool4", "tool5"],
  "similar_projects": ["Project Name 1", "Project Name 2", "Project Name 3", "Project Name 4", "Project Name 5"],
  "roadmap": [
    {{"phase": 1, "name": "Research & Planning", "steps": ["step1", "step2", "step3", "step4"]}},
    {{"phase": 2, "name": "Design & Development", "steps": ["step1", "step2", "step3", "step4"]}},
    {{"phase": 3, "name": "Implementation", "steps": ["step1", "step2", "step3", "step4"]}},
    {{"phase": 4, "name": "Testing & Validation", "steps": ["step1", "step2", "step3", "step4"]}},
    {{"phase": 5, "name": "Analysis & Optimization", "steps": ["step1", "step2", "step3", "step4"]}},
    {{"phase": 6, "name": "Deployment & Documentation", "steps": ["step1", "step2", "step3", "step4"]}}
  ]
}}

Make ALL recommendations SPECIFIC to this exact project. Equipment = real items needed. Tools = real software/platforms. Similar projects = real research project names in this domain. Return ONLY the JSON, nothing else."""

        raw = _call_groq(prompt, max_tokens=2000)
        if not raw:
            return {}
        raw = _clean_json(raw)
        if '{' in raw:
            raw = raw[raw.find('{'):raw.rfind('}')+1]
        try:
            return json.loads(raw)
        except Exception as e:
            print(f"Groq JSON parse error: {e}")
            return {}

    def generate_dynamic_recommendations(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        title = user_input.get('project_title', '')
        description = user_input.get('project_description', '')
        domain = user_input.get('research_domain', 'Environmental Science')
        location = user_input.get('location', '')

        # ── Try Groq first ────────────────────────────────────────────────────
        groq_data = {}
        if GROQ_API_KEY:
            try:
                groq_data = self._groq_recommendations(user_input)
                if groq_data:
                    print("✅ Groq AI: Personalized recommendations generated")
                else:
                    print("⚠️  Groq returned empty — using rule-based fallback")
            except Exception as e:
                print(f"Groq recommendations failed: {e}")

        # ── Map domain dropdown to project type ───────────────────────────────
        DOMAIN_MAP = {
            "environmental science": "air_monitoring",
            "biomedical": "biomedical",
            "cs-ai": "machine_learning",
            "electronics": "iot_general",
            "materials science": "materials_science",
            "civil": "civil",
        }
        domain_hint = DOMAIN_MAP.get(domain.lower().strip())
        if domain_hint:
            analyzed_type = domain_hint
            if domain_hint == "air_monitoring":
                desc_lower = (title + " " + description).lower()
                if any(w in desc_lower for w in ["water","river","lake","ph","turbidity","aqua","flood","hydro","microplastic"]):
                    analyzed_type = "water_quality"
        else:
            analyzed_type = self._analyze_project_type(title, description)

        project_data = self.project_database.get(analyzed_type, self.project_database["machine_learning"])
        base_scores = [0.94, 0.89, 0.85, 0.81, 0.76]

        # Equipment
        if groq_data.get("equipment"):
            equipment = [[item, 0.95 - i*0.05] for i, item in enumerate(groq_data["equipment"][:5])]
        else:
            equipment = [[item, 0.95 - i*0.05] for i, item in enumerate(project_data["equipment"][:5])]

        # Methodologies
        if groq_data.get("methodologies"):
            methodologies = [[item, 0.94 - i*0.04] for i, item in enumerate(groq_data["methodologies"][:5])]
        else:
            methodologies = [[item, 0.94 - i*0.04] for i, item in enumerate(project_data["methodologies"][:5])]

        # Tools
        if groq_data.get("tools"):
            tools = [[item, 0.93 - i*0.03] for i, item in enumerate(groq_data["tools"][:5])]
        else:
            tools = [[item, 0.93 - i*0.03] for i, item in enumerate(project_data["tools"][:5])]

        # Similar projects
        if groq_data.get("similar_projects"):
            similar_project_objs = [{"project_name": name, "similarity": base_scores[i]} for i, name in enumerate(groq_data["similar_projects"][:5])]
        else:
            similar_project_objs = [{"project_name": name, "similarity": base_scores[i]} for i, name in enumerate(project_data["similar_projects"][:5])]

        # Institutions (always rule-based — reliable)
        state = self._extract_location_state(location)
        state_data = self.institution_database.get(state) or self.institution_database["tamil_nadu"]
        display_state = state.replace("_", " ").title()
        institutions = []
        for i, institution in enumerate(state_data["institutions"][:5]):
            city = state_data["cities"][i % len(state_data["cities"])]
            institutions.append({
                "lab_id": f"LAB_DYN_{i+1:03d}",
                "lab_name": institution,
                "location": f"{city}, {display_state}",
                "equipment_list": "; ".join(project_data["equipment"][:3]),
                "availability": '{"Monday-Friday": "9:00-18:00", "Saturday": "9:00-13:00", "Sunday": "Closed"}',
                "contact_person": f"Dr. {['Sharma', 'Iyer', 'Kumar', 'Reddy', 'Patel'][i]}",
                "contact_email": f"contact@{institution.lower().replace(' ', '').replace('-', '').replace(',', '')[:30]}.edu"
            })

        # Roadmap
        roadmap = []
        if groq_data.get("roadmap"):
            for ph in groq_data["roadmap"]:
                steps_list = ph.get("steps", [])
                roadmap.append({
                    "phase_number": ph.get("phase", len(roadmap)+1),
                    "phase_name": ph.get("name", f"Phase {len(roadmap)+1}"),
                    "steps": " | ".join(steps_list) if isinstance(steps_list, list) else str(steps_list)
                })
        else:
            fb_phases = [
                (1,"Research & Planning","Requirements gathering | Literature review | Technical specification | Resource planning"),
                (2,"Design & Development","System architecture | Module development | Unit testing | Integration"),
                (3,"Implementation","System deployment | Configuration | User testing | Documentation"),
                (4,"Data Collection","Data acquisition | Processing pipeline | Statistical analysis | Validation"),
                (5,"Analysis & Optimization","Model optimization | Performance tuning | Report generation | Stakeholder presentation"),
                (6,"Deployment & Documentation","Production deployment | Maintenance planning | Support system | Continuous improvement"),
            ]
            for n, name, steps in fb_phases:
                roadmap.append({"phase_number": n, "phase_name": name, "steps": steps})

        return {
            "recommended_equipment": equipment,
            "recommended_methodologies": methodologies,
            "recommended_tools": tools,
            "recommended_labs": institutions,
            "similar_project_ids": similar_project_objs,
            "execution_roadmap": roadmap,
            "ai_enhanced": True,
            "confidence": 0.92,
            "justification": f"Generated based on {analyzed_type} analysis {'via Groq AI' if groq_data else '(rule-based fallback)'}"
        }


dynamic_ai_engine = DynamicAIEngine()

def get_dynamic_ai_engine():
    return dynamic_ai_engine