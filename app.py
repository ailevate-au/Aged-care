import base64
import datetime
import json
import os

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI, OpenAIError

load_dotenv()

# .env uses a SQLAlchemy-style URL; psycopg2 wants plain postgresql://
DATABASE_URL = os.environ["DATABASE"].strip().replace("postgresql+psycopg2://", "postgresql://")

SOAP_FIELDS = ("subjective", "objective", "assessment", "plan")

# New patients get the next ID in this series, e.g. SYNTHETIC_083
PATIENT_ID_PREFIX = "SYNTHETIC_"
PATIENT_FIELDS = (
    "first_name", "last_name", "gender", "date_of_birth", "phone_number", "emergency_contact_name",
    "emergency_contact_relationship", "emergency_contact_phone", "address", "primary_language", "blood_type",
    "allergies", "chronic_conditions", "primary_care_physician", "physician_contact", "care_status",
    "admission_date", "mobility_status", "insurance_provider",
)
REQUIRED_PATIENT_FIELDS = {"first_name": "first name", "last_name": "last name",
                           "gender": "gender", "date_of_birth": "date of birth"}

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # uploads up to 20 MB


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "That file is too large. Please upload a file under 20 MB."}), 413


def query(sql, params=None):
    with psycopg2.connect(DATABASE_URL) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


# ---------- Patients and SOAP notes ----------

@app.route("/api/patients")
def patients():
    return jsonify(query("""
        select patient_id, first_name, last_name, care_status
        from acare_master_patients
        order by last_name, first_name
    """))


@app.route("/api/patients", methods=["POST"])
def add_patient():
    data = request.get_json()
    patient = {f: (str(data.get(f) or "")).strip() or None for f in PATIENT_FIELDS}
    missing = [label for f, label in REQUIRED_PATIENT_FIELDS.items() if not patient[f]]
    if missing:
        return jsonify({"error": "Please fill in: " + ", ".join(missing)}), 400
    patient["primary_language"] = patient["primary_language"] or "English"
    patient["care_status"] = patient["care_status"] or "Active Home Nursing"

    columns = ", ".join(PATIENT_FIELDS)
    placeholders = ", ".join(["%s"] * len(PATIENT_FIELDS))
    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # The lock stops two saves at the same time from picking the same ID
                cur.execute("lock table acare_master_patients in exclusive mode")
                cur.execute("""
                    select coalesce(max(substring(patient_id from '[0-9]+$')::int), 0)
                    from acare_master_patients where patient_id like %s
                """, (PATIENT_ID_PREFIX + "%",))
                patient_id = f"{PATIENT_ID_PREFIX}{cur.fetchone()[0] + 1:03d}"
                # Age is worked out from the date of birth rather than typed in
                cur.execute(f"""
                    insert into acare_master_patients (patient_id, age, {columns})
                    values (%s, date_part('year', age(%s::date)), {placeholders})
                """, (patient_id, patient["date_of_birth"], *patient.values()))
    except psycopg2.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

    return jsonify({"patient_id": patient_id})


@app.route("/api/soap-notes/<patient_id>/<date>")
def get_soap_note(patient_id, date):
    # Lets the form show an existing note, since there can only be one per patient per day
    rows = query("""
        select subjective, objective, assessment, plan
        from acare_soap_notes where patient_id = %s and date = %s
    """, (patient_id, date))
    return jsonify(rows[0] if rows else None)


@app.route("/api/soap-notes", methods=["POST"])
def save_soap_note():
    data = request.get_json()
    note = {f: (data.get(f) or "").strip() or None for f in SOAP_FIELDS}
    if not data.get("patient_id") or not data.get("date"):
        return jsonify({"error": "Please choose a patient and a date."}), 400
    if not any(note.values()):
        return jsonify({"error": "Please fill in at least one of Subjective, Objective, Assessment or Plan."}), 400
    # acare_soap_notes has no foreign key to the patients table, so check here
    if not query("select 1 from acare_master_patients where patient_id = %s", (data["patient_id"],)):
        return jsonify({"error": f"Unknown patient {data['patient_id']}."}), 400

    try:
        # One note per patient per day: insert, or update the existing note for that day
        row = query("""
            insert into acare_soap_notes (patient_id, date, subjective, objective, assessment, plan)
            values (%s, %s, %s, %s, %s, %s)
            on conflict (patient_id, date) do update
              set subjective = excluded.subjective, objective = excluded.objective,
                  assessment = excluded.assessment, plan = excluded.plan
            returning (xmax = 0) as inserted
        """, (data["patient_id"], data["date"], *note.values()))[0]
    except psycopg2.Error as e:
        return jsonify({"error": f"Database error: {e}"}), 500

    return jsonify({"status": "inserted" if row["inserted"] else "updated"})


# ---------- Reading a SOAP note from a PDF or photo ----------

UPLOAD_TYPES = {"application/pdf", "image/png", "image/jpeg", "image/webp", "image/gif"}

VISION_PROMPT = """You read scanned, photographed or typed nursing SOAP notes and transcribe them.

Return the text of each section: Subjective, Objective, Assessment and Plan.
- Transcribe what is written. Do not add, summarise, interpret or correct clinical content.
- Leave out anything that identifies a person or a visit: names, dates, times, patient IDs, addresses,
  phone numbers and signatures.
- If the sections are not labelled, place each sentence in the section it clearly belongs to
  (Subjective = what the patient reports; Objective = measurements and observations;
  Assessment = the clinician's assessment; Plan = next steps).
- Write [illegible] for any word you cannot read.
- If a section is missing, return an empty string for it. If the document is not a SOAP or nursing note,
  return empty strings for all four."""

SOAP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "soap_note",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {f: {"type": "string"} for f in SOAP_FIELDS},
            "required": list(SOAP_FIELDS),
            "additionalProperties": False,
        },
    },
}


@app.route("/api/soap-notes/extract", methods=["POST"])
def extract_soap_note():
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY is not set. Add it to .env and restart the app."}), 500
    upload = request.files.get("file")
    if not upload or not upload.filename:
        return jsonify({"error": "Please choose a file."}), 400
    if upload.mimetype not in UPLOAD_TYPES:
        return jsonify({"error": "Please upload a PDF, PNG, JPEG, WEBP or GIF file."}), 400

    # The file is only held for this request and sent to OpenAI; it is never stored
    data_url = f"data:{upload.mimetype};base64,{base64.b64encode(upload.read()).decode()}"
    if upload.mimetype == "application/pdf":
        document = {"type": "file", "file": {"filename": upload.filename, "file_data": data_url}}
    else:
        document = {"type": "image_url", "image_url": {"url": data_url}}

    try:
        response = OpenAI().chat.completions.create(
            model=os.environ.get("OPENAI_VISION_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
            messages=[
                {"role": "system", "content": VISION_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": "Transcribe the SOAP note in this document."},
                                             document]},
            ],
            response_format=SOAP_SCHEMA,
        )
    except OpenAIError as e:
        return jsonify({"error": f"OpenAI error: {e}"}), 502

    message = response.choices[0].message
    if not message.content:
        return jsonify({"error": message.refusal or "The model did not return any text."}), 502
    note = json.loads(message.content)
    if not any(note[f].strip() for f in SOAP_FIELDS):
        return jsonify({"error": "No SOAP note content was found in this file."}), 422
    return jsonify(note)


# ---------- Reports ----------
# Unlike the chatbot, the code (not the model) chooses which notes go in: every note in the period.
# Facts like name, DOB and physician come straight from the database; the model only writes the narrative.

REPORT_TYPES = {
    "coordinator": {
        "title": "Coordinator follow-up summary",
        "audience": "an internal care coordinator. Use plain, direct language and short bullet points.",
        "sections": [
            ("overview", "Overview", "2-4 sentences on the patient's overall course during the period."),
            ("changes", "Changes during the period",
             "Bullets of new, worsening or improving problems, each with when it was first and last noted."),
            ("outstanding_plan_items", "Outstanding plan items",
             "Bullets of actions from the Plan sections that are ongoing or not yet shown as done."),
            ("suggested_followups", "Suggested follow-up actions",
             "Bullets of follow-up actions for the coordinator to consider (scheduling, reviews, contacting "
             "the physician or family). These are suggestions for review, not clinical orders."),
        ],
    },
    "gp": {
        "title": "GP update",
        "audience": "the patient's primary care physician. Write in a formal, concise clinical letter style, "
                    "in full sentences, referring to the person as 'the patient'.",
        "sections": [
            ("reason", "Reason for this update", "1-2 sentences on why the physician is being updated."),
            ("findings", "Relevant findings",
             "The clinically relevant findings and changes during the period, most important first."),
            ("current_plan", "Current home nursing plan", "What the home nursing team is currently doing."),
            ("requests", "Questions or requests for the GP",
             "Specific things the physician is asked to review or advise on, based on the notes. "
             "Do not recommend treatment; ask for the physician's review instead."),
        ],
    },
}

REPORT_SYSTEM_PROMPT = """You write a {title} for {audience}

You are given the patient's clinical profile and every home nursing SOAP note in the period, oldest first.
Rules:
- Use ONLY the notes and profile provided. Never invent findings, dates or results.
- Cite the note behind each statement by its date in square brackets, e.g. [2026-10-17].
- Do not diagnose. Describe what was documented and flag it for clinical review where relevant.
- Do not include the patient's name or contact details; they are added separately.
- Write plain text. Use "- " at the start of a line for bullets. No markdown headings.
- If there is nothing to report for a section, say so in one short sentence.

Sections to write:
{sections}"""

MAX_REPORT_DAYS = 730


def report_schema(report_type):
    keys = [key for key, _, _ in REPORT_TYPES[report_type]["sections"]]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{report_type}_report",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {key: {"type": "string"} for key in keys},
                "required": keys,
                "additionalProperties": False,
            },
        },
    }


@app.route("/api/reports/generate", methods=["POST"])
def generate_report():
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY is not set. Add it to .env and restart the app."}), 500
    data = request.get_json()
    report_type = data.get("report_type")
    if report_type not in REPORT_TYPES:
        return jsonify({"error": "Please choose a report type."}), 400
    try:
        start = datetime.date.fromisoformat(data.get("start_date") or "")
        end = datetime.date.fromisoformat(data.get("end_date") or "")
    except ValueError:
        return jsonify({"error": "Please choose a valid period."}), 400
    if start > end:
        return jsonify({"error": "The start date must be before the end date."}), 400
    if (end - start).days > MAX_REPORT_DAYS:
        return jsonify({"error": "Please choose a period of 2 years or less."}), 400

    patient = query("select * from acare_master_patients where patient_id = %s", (data.get("patient_id"),))
    if not patient:
        return jsonify({"error": "Please choose a patient."}), 400
    patient = patient[0]
    notes = query("""
        select date, subjective, objective, assessment, plan from acare_soap_notes
        where patient_id = %s and date between %s and %s
        order by date
    """, (patient["patient_id"], start, end))
    if not notes:
        return jsonify({"error": f"No SOAP notes for this patient between {start} and {end}."}), 422

    spec = REPORT_TYPES[report_type]
    # Only clinical details go to the model, not name, address or phone numbers
    profile = {k: patient[k] for k in ("gender", "age", "chronic_conditions", "allergies",
                                        "mobility_status", "care_status", "admission_date")}
    system = REPORT_SYSTEM_PROMPT.format(
        title=spec["title"], audience=spec["audience"],
        sections="\n".join(f"- {key} ({heading}): {guide}" for key, heading, guide in spec["sections"]),
    )
    user = json.dumps({"period": {"from": start, "to": end}, "profile": profile, "notes": notes}, default=str)

    try:
        response = OpenAI().chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=report_schema(report_type),
        )
    except OpenAIError as e:
        return jsonify({"error": f"OpenAI error: {e}"}), 502
    message = response.choices[0].message
    if not message.content:
        return jsonify({"error": message.refusal or "The model did not return a report."}), 502
    written = json.loads(message.content)

    return jsonify({
        "title": spec["title"],
        "report_type": report_type,
        "patient": {**{k: patient[k] for k in ("patient_id", "first_name", "last_name", "gender", "age",
                                               "chronic_conditions", "allergies", "mobility_status",
                                               "primary_care_physician", "physician_contact", "care_status")},
                    "date_of_birth": patient["date_of_birth"].isoformat()},
        "period": {"from": start.isoformat(), "to": end.isoformat()},
        "note_count": len(notes),
        "generated_on": datetime.date.today().isoformat(),
        "sections": [{"key": key, "heading": heading, "text": written[key]}
                     for key, heading, _ in spec["sections"]],
    })


# ---------- Chatbot ----------
# The LLM never writes SQL. It can only call the fixed, parameterised queries below,
# then it summarises what they return.

CHAT_SYSTEM_PROMPT = """You are an assistant for home nursing staff. Today's date is {today}.

You answer questions about patients using ONLY their profile and SOAP notes (Subjective, Objective,
Assessment, Plan) returned by your tools. Always call a tool before answering a question about a patient;
use find_patients first if you only have a name.

Rules:
- Cite the note behind every fact by its date in square brackets, one date per bracket, e.g. [2026-10-17].
  If the answer covers several patients, name the patient next to each fact.
- To say whether something is new or has changed, compare the notes in the period with the earlier notes
  returned as previous_notes. Say clearly when nothing new was recorded.
- Keep recorded facts separate from your interpretation.
- Do not diagnose or recommend treatment beyond what the notes say. Describe documented patterns and say
  they need clinical review.
- If the data does not answer the question, say so. Never invent observations.
- Be concise: short bullet points with bold headings, no more than about 250 words."""

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_patients",
            "description": "Find patients by full or partial first or last name. Returns patient IDs and "
                           "profile details (conditions, allergies, mobility, physician, care status).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Full or partial patient name"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_soap_notes",
            "description": "Get one patient's SOAP notes from the last N days, plus the few notes just before "
                           "that period (previous_notes) as a baseline for spotting what is new.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string", "description": "Patient ID, e.g. SYNTHETIC_074"},
                    "days": {"type": "integer", "description": "How many days back to look (1-1095)", "default": 30},
                },
                "required": ["patient_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_soap_notes",
            "description": "Search SOAP notes for a word or phrase (e.g. 'fall', 'edema', 'glucose') across all "
                           "patients, or one patient if patient_id is given, within the last N days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Word or phrase to look for"},
                    "patient_id": {"type": "string", "description": "Optional: limit to one patient"},
                    "days": {"type": "integer", "description": "How many days back to look (1-1095)", "default": 365},
                },
                "required": ["text"],
            },
        },
    },
]


def clamp_days(days, default):
    return max(1, min(int(days or default), 1095))


def tool_find_patients(name):
    pattern = f"%{name.strip()}%"
    return query("""
        select patient_id, first_name, last_name, gender, date_of_birth, age, allergies, chronic_conditions,
               mobility_status, primary_care_physician, care_status, admission_date
        from acare_master_patients
        where first_name || ' ' || last_name ilike %s or last_name ilike %s
        order by last_name, first_name
    """, (pattern, pattern))


def tool_get_soap_notes(patient_id, days=30):
    days = clamp_days(days, 30)
    patient = query("select patient_id, first_name, last_name from acare_master_patients where patient_id = %s",
                    (patient_id,))
    if not patient:
        return {"error": f"No patient with ID {patient_id}"}
    window = (patient_id, days)
    return {
        "patient": patient[0],
        "period": f"last {days} days",
        "notes": query("""
            select date, subjective, objective, assessment, plan from acare_soap_notes
            where patient_id = %s and date >= current_date - %s
            order by date
        """, window),
        "previous_notes": query("""
            select date, subjective, objective, assessment, plan from acare_soap_notes
            where patient_id = %s and date < current_date - %s
            order by date desc limit 3
        """, window),
    }


def tool_search_soap_notes(text, patient_id=None, days=365):
    pattern = f"%{text.strip()}%"
    return query("""
        select n.patient_id, p.first_name || ' ' || p.last_name as patient_name, n.date,
               n.subjective, n.objective, n.assessment, n.plan
        from acare_soap_notes n join acare_master_patients p using (patient_id)
        where (n.subjective ilike %s or n.objective ilike %s or n.assessment ilike %s or n.plan ilike %s)
          and (%s::text is null or n.patient_id = %s)
          and n.date >= current_date - %s
        order by n.date
        limit 40
    """, (pattern, pattern, pattern, pattern, patient_id, patient_id, clamp_days(days, 365)))


def run_tool(name, args):
    if name == "find_patients":
        return tool_find_patients(args.get("name", ""))
    if name == "get_soap_notes":
        return tool_get_soap_notes(args.get("patient_id", ""), args.get("days", 30))
    if name == "search_soap_notes":
        return tool_search_soap_notes(args.get("text", ""), args.get("patient_id"), args.get("days", 365))
    return {"error": f"Unknown tool: {name}"}


@app.route("/api/chat", methods=["POST"])
def chat():
    if not os.environ.get("OPENAI_API_KEY"):
        return jsonify({"error": "OPENAI_API_KEY is not set. Add it to .env and restart the app."}), 500

    history = request.get_json().get("messages", [])
    today = query("select current_date as today")[0]["today"]
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT.format(today=today)}]
    # Only plain user/assistant text comes from the browser
    messages += [{"role": m["role"], "content": m["content"]} for m in history if m.get("role") in ("user", "assistant")]

    client = OpenAI()
    tools_used = []
    try:
        for _ in range(6):  # model may call several tools before answering
            response = client.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
                messages=messages,
                tools=CHAT_TOOLS,
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return jsonify({"reply": message.content, "tools_used": tools_used})

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                               for tc in message.tool_calls],
            })
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                tools_used.append({"name": tc.function.name, "args": args})
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(run_tool(tc.function.name, args), default=str),
                })
    except OpenAIError as e:
        return jsonify({"error": f"OpenAI error: {e}"}), 502

    return jsonify({"error": "The assistant made too many lookups without answering. Try a more specific question."}), 500


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    # Port 5000 is taken by AirPlay Receiver on macOS
    app.run(debug=True, port=8000)
