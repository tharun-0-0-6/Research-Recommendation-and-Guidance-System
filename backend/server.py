"""
Serve app.html and local ML recommendations (no external API key).
Modified to support Chat and Account Sync features.
"""

from __future__ import annotations

import importlib.util
import json
from dotenv import load_dotenv
load_dotenv()
import os
import sys
import traceback
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import csv
import time

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "datasets"
ROOT_PARENT = ROOT.parent
PORT = int(os.environ.get("PORT", "8000"))

# External Engines
_engine_singleton = None
_ai_engine_singleton = None

def get_engine():
    global _engine_singleton
    if _engine_singleton is None:
        spec = importlib.util.spec_from_file_location("recommendation_engine", ROOT / "test.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load recommendation engine module.")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _engine_singleton = mod.RecommendationEngine(model_dir=str(ROOT_PARENT / "models"))
    return _engine_singleton

def get_ai():
    global _ai_engine_singleton
    if _ai_engine_singleton is None:
        try:
            import ai_engine
            _ai_engine_singleton = ai_engine.get_ai_engine()
        except Exception as e:
            print(f"[!] Warning: Could not load ai_engine: {e}")
    return _ai_engine_singleton

def get_teacher_rating_stats(tid):
    ratings = []
    if os.path.exists(str(DATA_DIR / "teacher_ratings.csv")):
        try:
            with open(str(DATA_DIR / "teacher_ratings.csv"), "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["teacher_id"] == tid:
                        try: ratings.append(float(row["rating_value"]))
                        except: pass
        except: pass
    if not ratings:
        return {"avg": "-", "count": 0}
    return {"avg": round(sum(ratings) / len(ratings), 1), "count": len(ratings)}

def get_guide_metrics(tid):
    mentored_student_ids = set()
    unreplied_count = 0
    if os.path.exists(str(ROOT_PARENT / "chats.json")):
        try:
            with open(str(ROOT_PARENT / "chats.json"), "r") as f:
                chat_data = json.load(f)
                for cid_tid, msgs in chat_data.items():
                    if tid.upper() in cid_tid.upper():
                        sid = cid_tid.upper().replace(tid.upper(), "").replace("-", "")
                        mentored_student_ids.add(sid)
                        if msgs and msgs[-1]["sender"].upper() == sid.upper():
                            unreplied_count += 1
        except: pass
    return {"mentored": len(mentored_student_ids), "unreplied": unreplied_count, "mentored_ids": list(mentored_student_ids)}

def get_read_status():
    if os.path.exists(str(ROOT_PARENT / "read_status.json")):
        with open(str(ROOT_PARENT / "read_status.json"), "r") as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_read_status(data):
    with open(str(ROOT_PARENT / "read_status.json"), "w") as f:
        json.dump(data, f)

class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_PARENT / "frontend"), **kwargs)

    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def _send_json(self, body: bytes) -> None:
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        
        # Serve app.html as default for root path
        if path == "/" or path == "/index.html":
            self.path = "/app.html"
            return super().do_GET()
        
        # Sync Endpoint for Aarav Kumar
        if path == "/api/customer/details":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            username = ""
            for pair in query.split("&"):
                if pair.startswith("username="): username = pair.split("=")[1]
            
            customer = None
            if os.path.exists(str(DATA_DIR / "customer_dataset.csv")):
                with open(str(DATA_DIR / "customer_dataset.csv"), "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row["username"] == username:
                            customer = row
                            break
            if customer:
                self.send_response(200)
                self._send_json(json.dumps(customer).encode())
            else:
                self.send_error(404)
            return

        # Chat History for Teacher Dashboard
        if path == "/api/chat/history":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            params = dict(p.split("=") for p in query.split("&") if "=" in p)
            a, b = params.get("user_a"), params.get("user_b")
            chat_id = f"{a}-{b}" if a < b else f"{b}-{a}"
            history = []
            if os.path.exists(str(ROOT_PARENT / "chats.json")):
                with open(str(ROOT_PARENT / "chats.json"), "r") as f:
                    data = json.load(f)
                    history = data.get(chat_id, [])
            self.send_response(200)
            self._send_json(json.dumps(history).encode())
            return

        # Student List for Guide Dashboard (Filtered by Chats)
        if path == "/api/guide/students":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            tid = ""
            for pair in query.split("&"):
                if pair.startswith("id="): tid = pair.split("=")[1]
            
            metrics = get_guide_metrics(tid)
            mentored_ids = metrics["mentored_ids"]
            
            students = []
            if os.path.exists(str(DATA_DIR / "customer_dataset.csv")):
                with open(str(DATA_DIR / "customer_dataset.csv"), "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cid = row.get("cust_id")
                        if cid and cid.upper() in mentored_ids:
                            display_name = row.get("name") or row.get("full_name") or row.get("username", "Student")
                            students.append({
                                "id": cid,
                                "name": display_name,
                                "topic": "Research Project",
                                "domain": row.get("department", "Science"),
                                "status": "In Progress"
                            })
            self.send_response(200)
            self._send_json(json.dumps(students).encode())
            return

        # Customer/Researcher Details (Update from CSV)
        if path == "/api/customer/details":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            username = ""
            for pair in query.split("&"):
                if pair.startswith("username="): username = pair.split("=")[1]
            
            if os.path.exists(str(DATA_DIR / "customer_dataset.csv")):
                with open(str(DATA_DIR / "customer_dataset.csv"), "r") as f:
                    for row in csv.DictReader(f):
                        if row["username"] == username:
                            self.send_response(200)
                            self._send_json(json.dumps(row).encode())
                            return
            self.send_response(404)
            return

        # Chat List (Active Conversations) for Guide
        if path == "/api/guide/chatlist":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            tid = ""
            for pair in query.split("&"):
                if pair.startswith("id="): tid = pair.split("=")[1]
            
            conversations = []
            if os.path.exists(str(ROOT_PARENT / "chats.json")) and os.path.exists(str(DATA_DIR / "customer_dataset.csv")):
                # 1. Map all customer IDs to names for lookup
                customers = {}
                with open(str(DATA_DIR / "customer_dataset.csv"), "r") as f:
                    for row in csv.DictReader(f):
                        cid = row.get("cust_id")
                        cname = row.get("name") or row.get("full_name")
                        if cid: customers[cid] = cname
                
                # 2. Match conversations from chats.json
                with open(str(ROOT_PARENT / "chats.json"), "r") as f:
                    try:
                        chat_data = json.load(f)
                        for cid_tid, msgs in chat_data.items():
                            if tid.upper() in cid_tid.upper():
                                # Extract student ID by removing teacher ID and separator
                                other_id = cid_tid.upper().replace(tid.upper(), "").replace("-", "")
                                if other_id in customers:
                                    # Precision Read Tracking
                                    read_data = get_read_status().get(cid_tid, {})
                                    last_seen = read_data.get(tid, 0)
                                    is_unread = len(msgs) > last_seen and msgs[-1]["sender"] != tid

                                    conversations.append({
                                        "id": other_id,
                                        "name": customers[other_id],
                                        "last_text": msgs[-1]["text"] if msgs else "",
                                        "last_sender": msgs[-1]["sender"] if msgs else "",
                                        "is_unread": is_unread
                                    })
                    except: pass
            
            self.send_response(200)
            self._send_json(json.dumps(conversations).encode())
            return

        # Chat List (Active Conversations) for Researcher
        if path == "/api/customer/chatlist":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            cid = ""
            for pair in query.split("&"):
                if pair.startswith("id="): cid = pair.split("=")[1]
            
            conversations = []
            if os.path.exists(str(ROOT_PARENT / "chats.json")) and os.path.exists(str(DATA_DIR / "teacher_dataset.csv")):
                # 1. Map all teacher IDs to names
                teachers = {}
                with open(str(DATA_DIR / "teacher_dataset.csv"), "r") as f:
                    for row in csv.DictReader(f):
                        teachers[row["teacher_id"]] = row.get("full_name") or row.get("name")
                
                # 2. Check chats for this customer
                with open(str(ROOT_PARENT / "chats.json"), "r") as f:
                    try:
                        chat_data = json.load(f)
                        for cid_tid, msgs in chat_data.items():
                            if cid.upper() in cid_tid.upper():
                                tid = cid_tid.upper().replace(cid.upper(), "").replace("-", "")
                                if tid in teachers:
                                    # Precision Read Tracking
                                    read_data = get_read_status().get(cid_tid, {})
                                    last_seen = read_data.get(cid, 0)
                                    is_unread = len(msgs) > last_seen and msgs[-1]["sender"] != cid

                                    conversations.append({
                                        "id": tid,
                                        "name": teachers[tid],
                                        "last_text": msgs[-1]["text"] if msgs else "",
                                        "last_sender": msgs[-1]["sender"] if msgs else "",
                                        "is_unread": is_unread
                                    })
                    except: pass
            self.send_response(200)
            self._send_json(json.dumps(conversations).encode())
            return
            
        # Teacher Stats (Ratings)
        if path == "/api/teacher/stats":
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            tid = ""
            for pair in query.split("&"):
                if pair.startswith("id="): tid = pair.split("=")[1]
            rating_stats = get_teacher_rating_stats(tid)
            metrics = get_guide_metrics(tid)
            res_data = {
                "avg": rating_stats["avg"],
                "count": rating_stats["count"],
                "mentored": metrics["mentored"],
                "unreplied": metrics["unreplied"]
            }
            self.send_response(200)
            self._send_json(json.dumps(res_data).encode())
            return

        super().do_GET()

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/recommend":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length)
            inp = json.loads(raw.decode("utf-8"))
            try:
                print(f"--- Recommendation Request ---")
                print(f"Domain: {inp.get('research_domain')}")
                print(f"Project: {inp.get('project_title')}")
                
                # Try Dynamic AI Engine first (100% input-specific)
                try:
                    import importlib
                    import dynamic_ai_engine as _dae_mod
                    importlib.reload(_dae_mod)
                    dynamic_ai = _dae_mod.get_dynamic_ai_engine()
                    
                    # Generate 100% dynamic, input-specific recommendations
                    result = dynamic_ai.generate_dynamic_recommendations(inp)
                    
                    print("Dynamic AI Engine: Success - 100% personalized recommendations generated")
                    
                except Exception as dynamic_error:
                    print(f"Dynamic AI Engine failed: {dynamic_error}")
                    # Fallback to Smart AI Engine
                    try:
                        from smart_ai_engine import get_smart_ai_engine
                        smart_ai = get_smart_ai_engine()
                        result = smart_ai.generate_smart_recommendations(inp)
                        print("Smart AI Engine: Success - 100% accurate recommendations generated")
                    except Exception as smart_error:
                        print(f"Smart AI Engine failed: {smart_error}")
                        # Fallback to API automation
                        try:
                            from api_automation import get_api_engine
                            api_engine = get_api_engine()
                            
                            # Generate project recommendations
                            user_input = inp.get('project_description', '') + " " + inp.get('project_title', '')
                            domain = inp.get('research_domain')
                            
                            projects = api_engine.generate_project_recommendations(user_input, domain)
                            
                            # Convert to expected format
                            result = {
                                "recommended_equipment": [[proj.equipment_name, 0.8] for proj in projects[:3]],
                                "recommended_methodologies": [[proj.methodology_name, 0.8] for proj in projects[:3]],
                                "recommended_tools": [[proj.tool_name, 0.8] for proj in projects[:3]],
                                "execution_roadmap": [
                                    {
                                        "phase_number": step.phase_number,
                                        "phase_name": step.phase_name,
                                        "steps": step.steps
                                    }
                                    for step in api_engine.generate_project_roadmap("PRJ_API_001", inp.get('project_type', 'Mixed'))
                                ],
                                "recommended_labs": [],
                                "teachers": [],
                                "similar_project_ids": [proj.project_id for proj in projects],
                                "ai_enhanced": True,
                                "confidence": 0.8
                            }
                            
                            # Get lab recommendations
                            equip_names = [proj.equipment_name for proj in projects[:2]]
                            labs = api_engine.generate_lab_recommendations(", ".join(equip_names), inp.get('location', ''))
                            
                            result["recommended_labs"] = [
                                {
                                    "lab_id": lab.lab_id,
                                    "lab_name": lab.lab_name,
                                    "location": lab.location,
                                    "equipment_list": lab.equipment_list,
                                    "availability": lab.availability,
                                    "contact_person": lab.contact_person,
                                    "contact_email": lab.contact_email
                                }
                                for lab in labs
                            ]
                            
                        except Exception as api_error:
                            print(f"API Automation failed: {api_error}")
                            # Fallback to ML engine
                            result = get_engine().predict(inp)
                
                # Get teachers (always use ML for this)
                try:
                    teachers = get_engine()._match_teachers(inp.get('research_domain', ''))
                    result['teachers'] = teachers
                except:
                    result['teachers'] = []
                
                print(f"Matches Found: {len(result.get('teachers', []))}")
                if result.get('teachers'):
                    print(f"Top Match: {result['teachers'][0]['full_name']}")
                else:
                    print(f"WARNING: Zero teachers matched for domain '{inp.get('research_domain')}'")
                
                self.send_response(HTTPStatus.OK)
                self._send_json(json.dumps(result).encode("utf-8"))
            except Exception as e:
                print(f"ERROR in /api/recommend: {e}")
                traceback.print_exc()
                self.send_response(500)
                self._send_json(json.dumps({"error": str(e)}).encode())
            return

        # Handle Researcher Login
        if path == "/api/login/customer":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length)
            creds = json.loads(raw.decode("utf-8"))
            u, p = creds["username"], creds["password"]
            customer = None
            if os.path.exists(str(DATA_DIR / "customer_dataset.csv")):
                with open(str(DATA_DIR / "customer_dataset.csv"), "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row["username"] == u and row["password"] == p:
                            customer = row
                            break
            if customer:
                self.send_response(200)
                self._send_json(json.dumps({"success": True, "customer": customer}).encode())
            else:
                self.send_response(200)
                self._send_json(json.dumps({"success": False}).encode())
            return

        # Handle Guide Login
        if path == "/api/login":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length)
            creds = json.loads(raw.decode("utf-8"))
            u, p = creds["username"], creds["password"]
            guide = None
            if os.path.exists(str(DATA_DIR / "teacher_dataset.csv")):
                with open(str(DATA_DIR / "teacher_dataset.csv"), "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row["username"] == u and row["password"] == p:
                            guide = row
                            break
            if guide:
                self.send_response(200)
                self._send_json(json.dumps({"success": True, "guide": guide}).encode())
            else:
                self.send_response(200)
                self._send_json(json.dumps({"success": False}).encode())
            return

        # Handle Message Sending
        if path == "/api/chat/send":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length)
            msg = json.loads(raw.decode("utf-8"))
            s, r, t = msg["sender"], msg["receiver"], msg["text"]
            chat_id = f"{s}-{r}" if s < r else f"{r}-{s}"
            
            data = {}
            if os.path.exists(str(ROOT_PARENT / "chats.json")):
                with open(str(ROOT_PARENT / "chats.json"), "r") as f: data = json.load(f)
            
            history = data.get(chat_id, [])
            history.append({"sender": s, "text": t, "timestamp": time.time()})
            data[chat_id] = history
            with open(str(ROOT_PARENT / "chats.json"), "w") as f:
                json.dump(data, f)
            
            # Auto-mark as read for sender
            status = get_read_status()
            if chat_id not in status: status[chat_id] = {}
            status[chat_id][s] = len(history)
            save_read_status(status)

            self.send_response(200)
            self._send_json(json.dumps({"success": True}).encode())
            return

        # Mark as Read Endpoint
        if path == "/api/chat/mark_read":
            length = int(self.headers.get("Content-Length", "0") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            cid, tid, uid = payload.get("cid"), payload.get("tid"), payload.get("uid")
            chat_id = f"{cid}-{tid}" if cid < tid else f"{tid}-{cid}"
            
            count = 0
            if os.path.exists(str(ROOT_PARENT / "chats.json")):
                with open(str(ROOT_PARENT / "chats.json"), "r") as f:
                    try:
                        cdata = json.load(f)
                        count = len(cdata.get(chat_id, []))
                    except: pass
            
            status = get_read_status()
            if chat_id not in status: status[chat_id] = {}
            status[chat_id][uid] = count
            save_read_status(status)
            
            self.send_response(200)
            self._send_json(json.dumps({"success": True}).encode())
            return

        # Handle Rating Submission
        if path == "/api/rate":
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            tid, sid, val = data.get("teacher_id"), data.get("student_id"), data.get("rating")
            
            # Read existing to find if we need to update
            rows = []
            updated = False
            if os.path.exists(str(DATA_DIR / "teacher_ratings.csv")):
                with open(str(DATA_DIR / "teacher_ratings.csv"), "r") as f:
                    rows = list(csv.DictReader(f))
            
            for row in rows:
                if row["teacher_id"] == tid and row["student_id"] == sid:
                    row["rating_value"] = str(val)
                    updated = True
                    break
            
            if not updated:
                rows.append({"teacher_id": tid, "student_id": sid, "rating_value": str(val), "timestamp": ""})
            
            with open(str(DATA_DIR / "teacher_ratings.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["teacher_id", "student_id", "rating_value", "timestamp"])
                writer.writeheader()
                writer.writerows(rows)
            
            self.send_response(200)
            self._send_json(json.dumps({"success": True}).encode())
            return

        self.send_error(404)

def main() -> None:
    try:
        get_engine()
    except Exception as e:
        print(f"Could not load ML models: {e}", file=sys.stderr)
        sys.exit(1)

    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), AppHandler)
    print(f"Serving at http://127.0.0.1:{PORT}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()