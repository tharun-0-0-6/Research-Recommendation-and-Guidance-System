"""
teacher_engine.py — Teacher/Guide Engine for ResearchIQ
Handles guide and customer authentication, domain inference, and persistent chat storage.
"""

import csv
import json
import time
from pathlib import Path

TEACHER_CSV  = Path(__file__).resolve().parent.parent / "datasets" / "teacher_dataset.csv"
CUSTOMER_CSV = Path(__file__).resolve().parent.parent / "datasets" / "customer_dataset.csv"
CHATS_JSON   = Path(__file__).resolve().parent / "chats.json"

_teachers = []
# (Removed global _customers cache to ensure real-time CSV reloading)

# Map department keywords → ResearchIQ domain names
DEPT_TO_DOMAIN = {
    "environmental": "Environmental Science", "ecology": "Environmental Science",
    "climate": "Environmental Science", "biomedical": "Biomedical",
    "biomaterial": "Biomedical", "biotechnology": "Biomedical",
    "computer": "CS-AI", "software": "CS-AI", "ai": "CS-AI", "data": "CS-AI",
    "ece": "Electronics", "electronics": "Electronics", "electrical": "Electronics",
    "communication": "Electronics", "materials": "Materials Science",
    "metallurgy": "Materials Science", "polymer": "Materials Science",
    "civil": "Civil", "structural": "Civil", "geotechnical": "Civil",
    "construction": "Civil", "transportation": "Civil",
}

def _infer_domain(department: str) -> str:
    dept_lower = department.strip().lower()
    for key, domain in DEPT_TO_DOMAIN.items():
        if key in dept_lower: return domain
    return "CS-AI"

def _load_teachers():
    global _teachers
    if _teachers: return _teachers
    _teachers = []
    if not TEACHER_CSV.exists(): return _teachers
    with open(TEACHER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_kw = row.get("specialization_keywords", "")
            keywords = [kw.strip() for kw in raw_kw.split(";") if kw.strip()]
            _teachers.append({
                "teacher_id": row.get("teacher_id", ""),
                "username": row.get("username", ""),
                "full_name": row.get("full_name", ""),
                "email": row.get("email", ""),
                "password": row.get("password", ""),
                "designation": row.get("designation", ""),
                "institution": row.get("institution", ""),
                "department": row.get("department", ""),
                "highest_qualification": row.get("highest_qualification", ""),
                "years_of_experience": int(row.get("years_of_experience", 0) or 0),
                "specialization_keywords": keywords,
                "publications_count": int(row.get("publications_count", 0) or 0),
                "primary_domain": _infer_domain(row.get("department", "")),
                "rating": 4.5, "available_for_guidance": True, "guidance_mode": "Both"
            })
    return _teachers

def _load_customers():
    if not CUSTOMER_CSV.exists(): return []
    customers = []
    with open(CUSTOMER_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            customers.append({
                "cust_id": row.get("cust_id", ""),
                "name": row.get("name", ""),
                "username": row.get("username", ""),
                "password": row.get("password", ""),
                "age": row.get("age", ""),
                "university": row.get("university", ""),
                "department": row.get("department", ""),
                "type": row.get("type", "free").strip().lower()
            })
    return customers

def recommend_teachers(domain: str, keywords: list = None, top_k: int = 5) -> list:
    teachers = _load_teachers()
    keywords = [kw.strip().lower() for kw in (keywords or [])]
    scored = []
    for t in teachers:
        score = 0.0
        # Robust case-insensitive match
        if t["primary_domain"].strip().lower() == domain.strip().lower():
            score += 5.0
        
        t_kws = [kw.lower() for kw in t["specialization_keywords"]]
        for kw in keywords:
            for tkw in t_kws:
                if kw in tkw or tkw in kw:
                    score += 1.0
                    break
        score += min(t["publications_count"] / 100.0, 1.0)
        score += min(t["years_of_experience"] * 0.1, 2.0)
        scored.append((score, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, t in scored[:top_k]:
        results.append({**t, "match_score": round(score, 2)})
    return results

# ── AUTH ──
def login_guide(username, password):
    teachers = _load_teachers()
    for t in teachers:
        if t["username"] == username and t["password"] == password:
            return {"success": True, "guide": {k: v for k, v in t.items() if k != 'password'}}
    return {"success": False, "error": "Invalid username or password"}

def login_customer(username, password):
    customers = _load_customers()
    for c in customers:
        if c["username"] == username and c["password"] == password:
            return {"success": True, "customer": {k: v for k, v in c.items() if k != 'password'}}
    return {"success": False, "error": "Invalid username or password"}

def get_customer(username):
    customers = _load_customers()
    for c in customers:
        if c["username"] == username:
            return {k: v for k, v in c.items() if k != 'password'}
    return None

# ── CHAT (LOCAL DB) ──
def get_chat_history(user_a, user_b):
    if not CHATS_JSON.exists(): return []
    try:
        with open(CHATS_JSON, "r") as f:
            chats = json.load(f)
    except: return []
    conv_id = "-".join(sorted([user_a, user_b]))
    return chats.get(conv_id, [])

def save_message(sender, receiver, text):
    chats = {}
    if CHATS_JSON.exists():
        try:
            with open(CHATS_JSON, "r") as f:
                chats = json.load(f)
        except: chats = {}
    
    conv_id = "-".join(sorted([sender, receiver]))
    if conv_id not in chats: chats[conv_id] = []
    
    msg = {"sender": sender, "text": text, "timestamp": time.time()}
    chats[conv_id].append(msg)
    
    with open(CHATS_JSON, "w") as f:
        json.dump(chats, f, indent=2)
    return msg

def get_chat_list(guide_id):
    """Return all students the guide has chatted with, sorted by most recent."""
    if not CHATS_JSON.exists(): return []
    try:
        with open(CHATS_JSON, "r") as f:
            chats = json.load(f)
    except: return []
    
    customers = {c["cust_id"]: c for c in _load_customers()}
    contacts = []
    for conv_id, messages in chats.items():
        if guide_id in conv_id:
            # conv_id is sorted, so guide_id could be first or second
            ids = conv_id.split("-")
            other_id = ids[1] if ids[0] == guide_id else ids[0]
            
            # Check if other_id is a customer
            if other_id.startswith("C"):
                cust = customers.get(other_id, {})
                last_msg = messages[-1]
                contacts.append({
                    "id": other_id,
                    "name": cust.get("name", other_id),
                    "last_text": last_msg["text"],
                    "timestamp": last_msg["timestamp"]
                })
    return sorted(contacts, key=lambda x: x["timestamp"], reverse=True)

def get_customer_chat_list(cust_id):
    chats = _load_chats()
    teachers = {t["teacher_id"]: t for t in _load_teachers()}
    contacts = []
    
    for conv_id, messages in chats.items():
        if cust_id in conv_id:
            ids = conv_id.split("-")
            other_id = ids[1] if ids[0] == cust_id else ids[0]
            
            if other_id.startswith("T"):
                teacher = teachers.get(other_id, {})
                last_msg = messages[-1]
                contacts.append({
                    "id": other_id,
                    "name": teacher.get("full_name", other_id),
                    "last_text": last_msg["text"],
                    "timestamp": last_msg["timestamp"]
                })
    return sorted(contacts, key=lambda x: x["timestamp"], reverse=True)

# ── MOCK STUDENTS (Enhanced) ──
def get_guide_students(guide_id):
    # In a real app, this would be a join table
    # For now, we'll return a few "assigned" students from the dataset
    all_custs = _load_customers()
    # Let's say every guide is assigned 3 students for demo
    # We'll use deterministic assignment based on guide_id hash
    start_idx = sum(ord(c) for c in guide_id) % max(1, len(all_custs) - 3)
    assigned = all_custs[start_idx : start_idx + 3]
    
    return [
        {
            "id": c["cust_id"],
            "name": c["name"],
            "topic": "Research Exploration",
            "domain": c["department"],
            "status": "In Progress"
        } for c in assigned
    ]
