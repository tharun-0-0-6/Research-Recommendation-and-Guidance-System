"""
API Automation Module - Uses Groq (free) for recommendations.
"""

import os
import json
import requests
from typing import List, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')


def _call_groq(prompt: str, max_tokens: int = 1500) -> str:
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
        print(f"Groq API call failed: {e}")
        return ""


@dataclass
class Project:
    project_id: str
    project_title: str
    project_description: str
    research_domain: str
    project_type: str
    budget_range: str
    duration_months: int
    team_expertise: str
    location: str
    equipment_name: str
    equipment_category: str
    methodology_name: str
    methodology_type: str
    skill_required: str
    tool_name: str
    tool_category: str

@dataclass
class Lab:
    lab_id: str
    lab_name: str
    location: str
    equipment_list: str
    availability: str
    contact_person: str
    contact_email: str

@dataclass
class RoadmapStep:
    roadmap_id: str
    project_id: str
    phase_number: int
    phase_name: str
    steps: str


class APIAutomationEngine:
    def __init__(self):
        if GROQ_API_KEY:
            print("Groq API ready: llama-3.3-70b-versatile")
        else:
            print("Warning: GROQ_API_KEY not set — using fallback")

    def _get_fallback_labs(self, location: str = "") -> List[Lab]:
        ALL_LABS = {
            "tamil_nadu": [("LAB_TN001","IIT Madras - Research Laboratory","Chennai, Tamil Nadu","Dr. Rajan","contact@iitmadraslab.edu"),("LAB_TN002","Anna University - Innovation Centre","Chennai, Tamil Nadu","Dr. Priya","contact@annaunivlab.edu"),("LAB_TN003","PSG Tech - Advanced Research Lab","Coimbatore, Tamil Nadu","Dr. Kumar","contact@psglab.edu"),("LAB_TN004","NIT Trichy - Engineering Lab","Trichy, Tamil Nadu","Dr. Suresh","contact@nittrichylab.edu"),("LAB_TN005","VIT Vellore - Technology Centre","Vellore, Tamil Nadu","Dr. Menon","contact@vitvellab.edu")],
            "karnataka": [("LAB_KA001","IISc Bangalore - Advanced Research Lab","Bengaluru, Karnataka","Dr. Sharma","contact@iisclab.edu"),("LAB_KA002","NIT Karnataka - Engineering Lab","Surathkal, Karnataka","Dr. Patel","contact@nitklab.edu"),("LAB_KA003","RV College - Research Centre","Bengaluru, Karnataka","Dr. Iyer","contact@rvcollegelab.edu"),("LAB_KA004","Manipal Institute - Innovation Lab","Manipal, Karnataka","Dr. Reddy","contact@manipallab.edu"),("LAB_KA005","VTU Belgaum - Research Hub","Belgaum, Karnataka","Dr. Rao","contact@vtulab.edu")],
            "maharashtra": [("LAB_MH001","IIT Bombay - Research Centre","Mumbai, Maharashtra","Dr. Joshi","contact@iitbombaylab.edu"),("LAB_MH002","COEP Pune - Engineering Lab","Pune, Maharashtra","Dr. Deshmukh","contact@coeplab.edu"),("LAB_MH003","VJTI Mumbai - Advanced Lab","Mumbai, Maharashtra","Dr. Kulkarni","contact@vjtilab.edu"),("LAB_MH004","SPCE Pune - Innovation Centre","Pune, Maharashtra","Dr. Patil","contact@spcelab.edu"),("LAB_MH005","ICT Mumbai - Research Hub","Mumbai, Maharashtra","Dr. Shah","contact@ictlab.edu")],
            "delhi_ncr": [("LAB_DL001","IIT Delhi - Research Laboratory","New Delhi, Delhi","Dr. Gupta","contact@iitdelhilab.edu"),("LAB_DL002","DTU Delhi - Engineering Centre","New Delhi, Delhi","Dr. Verma","contact@dtulab.edu"),("LAB_DL003","NSUT Delhi - Innovation Lab","New Delhi, Delhi","Dr. Singh","contact@nsutlab.edu"),("LAB_DL004","JNU Delhi - Research Centre","New Delhi, Delhi","Dr. Sharma","contact@jnulab.edu"),("LAB_DL005","IP University - Advanced Lab","New Delhi, Delhi","Dr. Kumar","contact@ipulab.edu")],
            "telangana": [("LAB_TS001","IIIT Hyderabad - AI Lab","Hyderabad, Telangana","Dr. Reddy","contact@iiitlab.edu"),("LAB_TS002","University of Hyderabad - Research Centre","Hyderabad, Telangana","Dr. Rao","contact@uohlab.edu"),("LAB_TS003","NIT Warangal - Engineering Lab","Warangal, Telangana","Dr. Naik","contact@nitwlab.edu"),("LAB_TS004","Osmania University - Tech Hub","Hyderabad, Telangana","Dr. Ali","contact@osmanialab.edu"),("LAB_TS005","JNTU Hyderabad - Innovation Lab","Hyderabad, Telangana","Dr. Iyer","contact@jntulab.edu")],
            "west_bengal": [("LAB_WB001","IIT Kharagpur - Research Laboratory","Kharagpur, West Bengal","Dr. Bose","contact@iitkharagpurlab.edu"),("LAB_WB002","Jadavpur University - Engineering Centre","Kolkata, West Bengal","Dr. Chatterjee","contact@julab.edu"),("LAB_WB003","IIEST Shibpur - Innovation Lab","Howrah, West Bengal","Dr. Das","contact@iiestlab.edu"),("LAB_WB004","Presidency University - Research Hub","Kolkata, West Bengal","Dr. Sen","contact@presidencylab.edu"),("LAB_WB005","NIT Durgapur - Advanced Lab","Durgapur, West Bengal","Dr. Ghosh","contact@nitdurgapurlab.edu")],
            "gujarat": [("LAB_GJ001","IIT Gandhinagar - Research Laboratory","Gandhinagar, Gujarat","Dr. Patel","contact@iitgnlab.edu"),("LAB_GJ002","NIT Surat - Engineering Centre","Surat, Gujarat","Dr. Shah","contact@nitsurlab.edu"),("LAB_GJ003","DAIICT Gandhinagar - Innovation Lab","Gandhinagar, Gujarat","Dr. Joshi","contact@daiictlab.edu"),("LAB_GJ004","MS University Baroda - Research Hub","Vadodara, Gujarat","Dr. Mehta","contact@msunivlab.edu"),("LAB_GJ005","Nirma University - Advanced Lab","Ahmedabad, Gujarat","Dr. Trivedi","contact@nirmalab.edu")],
        }
        state_keywords = {
            "tamil_nadu":["tamil nadu","chennai","coimbatore","madurai","trichy","vellore","salem"],
            "karnataka":["karnataka","bengaluru","bangalore","mysore","hubli","mangalore","belgaum"],
            "maharashtra":["maharashtra","mumbai","pune","nagpur","nashik"],
            "delhi_ncr":["delhi","gurgaon","noida","new delhi","faridabad"],
            "telangana":["telangana","hyderabad","warangal","nizamabad"],
            "west_bengal":["west bengal","kolkata","calcutta","kharagpur","durgapur"],
            "gujarat":["gujarat","ahmedabad","surat","vadodara","gandhinagar"],
        }
        loc_lower = location.lower()
        chosen_state = "tamil_nadu"
        for state, kws in state_keywords.items():
            if any(k in loc_lower for k in kws):
                chosen_state = state
                break
        chosen = ALL_LABS.get(chosen_state, ALL_LABS["tamil_nadu"])
        avail = '{"Monday-Friday": "9:00-18:00", "Saturday": "9:00-13:00", "Sunday": "Closed"}'
        equip = "GPU Server;Sensors;Lab Equipment;Computers;Network Equipment"
        return [Lab(lab_id=l[0], lab_name=l[1], location=l[2], equipment_list=equip, availability=avail, contact_person=l[3], contact_email=l[4]) for l in chosen]

    def _get_fallback_roadmap(self, project_id: str) -> List[RoadmapStep]:
        phases = [(1,"Literature Review","1. Review existing literature | 2. Identify gaps | 3. Define objectives"),(2,"Experimental Setup","1. Setup equipment | 2. Calibrate instruments | 3. Test procedures"),(3,"Data Collection","1. Collect baseline data | 2. Monitor parameters | 3. Record observations"),(4,"Analysis","1. Process data | 2. Apply statistical methods | 3. Interpret results"),(5,"Validation","1. Verify findings | 2. Peer review | 3. Quality assurance"),(6,"Reporting","1. Prepare report | 2. Create visualizations | 3. Submit findings")]
        return [RoadmapStep(roadmap_id=f"RDM_FB{n}",project_id=project_id,phase_number=n,phase_name=name,steps=steps) for n,name,steps in phases]

    def generate_lab_recommendations(self, equipment_needed: str, location: str = None) -> List[Lab]:
        return self._get_fallback_labs(location or "")

    def generate_project_roadmap(self, project_id: str, project_type: str) -> List[RoadmapStep]:
        return self._get_fallback_roadmap(project_id)


api_engine = APIAutomationEngine()

def get_api_engine():
    return api_engine