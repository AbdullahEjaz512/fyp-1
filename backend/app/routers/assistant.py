from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
import os, json

from app.database import get_db, AnalysisResult, File as DBFile, User, FileAccessPermission
from app.dependencies.auth import get_current_user

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])

# ── GROQ LLM CLIENT ──────────────────────────────────────────────────────────
def _get_groq_client():
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            return Groq(api_key=api_key)
    except Exception:
        pass
    return None

SYSTEM_PROMPT = """You are a clinical AI assistant integrated into Seg-Mind, a brain tumor MRI analysis platform.
You have deep knowledge of:
- Brain tumor types: Glioblastoma (GBM), Anaplastic Astrocytoma, Astrocytoma, Oligodendroglioma, Meningioma, Medulloblastoma, Ependymoma, Pituitary adenoma
- WHO grading system (Grade I-IV), IDH mutation status, MGMT methylation
- Tumor subregions: NCR (Necrotic Core), ET (Enhancing Tumor), ED (Edema)
- MRI sequences: T1, T2, FLAIR, T1ce
- Treatments: surgical resection, radiation, temozolomide chemotherapy, bevacizumab
- Platform features: upload NIfTI scans, AI segmentation, classification, 3D visualization, report generation

When given patient case data, summarize it clearly in clinical terms.
Be concise, accurate, and helpful. If asked about a specific tumor type, explain: definition, WHO grade, prognosis, treatment.
"""

# ── MEDICAL KNOWLEDGE FALLBACK (when no Groq key) ────────────────────────────
MEDICAL_KB = {
    "anaplastic astrocytoma": """**Anaplastic Astrocytoma (WHO Grade III)**
- A malignant diffuse glioma arising from astrocytes
- IDH mutation found in ~70% of cases (IDH-mutant = better prognosis)
- Median survival: 3-5 years (IDH-mutant), 1.5-2 years (IDH-wildtype)
- Treatment: Surgical resection → Radiation (60 Gy) → Temozolomide chemotherapy
- MRI: Heterogeneous T2/FLAIR signal, may show enhancement; less necrosis than GBM
- Recurrence common; often progresses to Glioblastoma (Grade IV)""",

    "glioblastoma": """**Glioblastoma Multiforme (GBM) - WHO Grade IV**
- Most aggressive primary brain tumor; median survival 14-16 months with treatment
- IDH-wildtype in 90% of cases
- MGMT promoter methylation = better response to temozolomide
- Treatment: Stupp protocol — resection + 60 Gy radiation + temozolomide
- MRI: Ring-enhancing lesion with central necrosis, perilesional edema""",

    "astrocytoma": """**Diffuse Astrocytoma (WHO Grade II)**
- Slow-growing; IDH-mutant in most cases
- Median survival: 6-8 years
- Treatment: Observation or surgery; radiation/chemo if high-risk
- MRI: T2/FLAIR hyperintensity, typically no enhancement""",

    "oligodendroglioma": """**Oligodendroglioma (WHO Grade II-III)**
- IDH-mutant + 1p/19q co-deletion (defines this tumor)
- Best prognosis among gliomas; median survival 10-15 years
- Highly responsive to PCV chemotherapy and temozolomide
- MRI: Frontal lobe predominance, calcifications common""",

    "meningioma": """**Meningioma (WHO Grade I-III)**
- Arises from meninges; most are benign (Grade I)
- Grade I: Excellent prognosis with surgical cure
- Grade II/III: Higher recurrence; requires radiation
- MRI: Extra-axial, dural tail sign, homogeneous enhancement""",

    "medulloblastoma": """**Medulloblastoma (WHO Grade IV)**
- Most common malignant pediatric brain tumor
- Located in cerebellum/posterior fossa
- 5-year survival: 70-80% with craniospinal radiation + chemo
- Molecular subgroups: WNT (best), SHH, Group 3, Group 4""",

    "ncr": "**NCR (Necrotic Core)**: The non-enhancing necrotic center of the tumor — dead tissue with no blood supply. Large NCR volume is associated with higher grade tumors like GBM.",
    "enhancing tumor": "**ET (Enhancing Tumor)**: Active tumor cells that break the blood-brain barrier, visible on T1-contrast MRI. Volume correlates with aggressiveness.",
    "edema": "**ED (Edema/Peritumoral)**: Swelling around the tumor. Large edema can cause mass effect, headaches, and neurological deficits.",

    "who grade": """**WHO Brain Tumor Grading:**
- Grade I: Benign, slow growth, surgical cure possible (e.g., Pilocytic Astrocytoma)
- Grade II: Slow-growing, may recur (e.g., Diffuse Astrocytoma)
- Grade III: Malignant, anaplastic (e.g., Anaplastic Astrocytoma)
- Grade IV: Most aggressive, short survival (e.g., Glioblastoma)""",

    "idh": """**IDH Mutation Status:**
- IDH-mutant: Better prognosis, younger patients, secondary GBMs
- IDH-wildtype: Worse prognosis, primary GBMs, more aggressive
- Tested by immunohistochemistry or sequencing""",

    "mgmt": """**MGMT Methylation:**
- MGMT promoter methylation silences a DNA repair gene
- Result: Tumor cannot repair temozolomide-induced DNA damage
- Methylated = better response to temozolomide chemotherapy
- Present in ~40% of GBMs""",
}

PLATFORM_KB = {
    "upload": "To upload a scan: Go to Upload tab → select NIfTI (.nii/.nii.gz) file → enter Patient ID → click Upload. The system auto-runs segmentation and classification.",
    "visualization": "Two modes: 2D (slice viewer with windowing controls) and 3D Reconstruction (interactive volumetric rendering with tumor overlay).",
    "segmentation": "Uses a 3D U-Net trained on BraTS dataset. Segments 3 tumor subregions: NCR (necrotic core), ET (enhancing tumor), ED (edema). Outputs volume in mm³.",
    "classification": "ResNet50 model classifies tumor type with confidence score. Categories: Glioblastoma, Astrocytoma, Oligodendroglioma, Meningioma, Medulloblastoma, Ependymoma.",
    "report": "Auto-generate PDF clinical report: Go to Results → Download Report. Includes AI findings, volumes, classification, and doctor assessment.",
    "collaboration": "Share cases: Results page → Share button → enter colleague's email. They get access to view and add clinical opinions.",
    "growth prediction": "LSTM model forecasts tumor volume over time using sequential scans. Requires 2+ scans uploaded for the same patient.",
    "xai": "Explainable AI uses Grad-CAM heatmaps showing which brain regions drove the classification decision.",
}


def _llm_answer(message: str, context: str = "") -> str:
    client = _get_groq_client()
    if not client:
        return None
    try:
        prompt = message
        if context:
            prompt = f"Context from patient record:\n{context}\n\nQuestion: {message}"
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def _local_answer(message: str) -> Optional[str]:
    msg = message.lower()
    # Medical knowledge
    for key, val in MEDICAL_KB.items():
        if key in msg:
            return val
    # Platform knowledge
    for key, val in PLATFORM_KB.items():
        if key in msg:
            return val
    return None


# ── FAISS DOC INDEX (project docs) ───────────────────────────────────────────
_DOC_INDEX = {"ready": False, "documents": [], "paths": [], "model": None, "index": None}


def _collect_docs():
    try:
        from sentence_transformers import SentenceTransformer
        import faiss, numpy as np
    except Exception:
        return
    root = Path(__file__).resolve().parents[3]
    docs, paths = [], []
    for p in root.glob("*.md"):
        try:
            docs.append(p.read_text(encoding="utf-8", errors="ignore"))
            paths.append(str(p.relative_to(root)))
        except Exception:
            continue
    if not docs:
        return
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb = model.encode(docs, show_progress_bar=False).astype('float32')
        idx = faiss.IndexFlatL2(emb.shape[1])
        idx.add(emb)
        _DOC_INDEX.update({"ready": True, "documents": docs, "paths": paths, "model": model, "index": idx})
    except Exception as e:
        print(f"Doc index error: {e}")


def _search_docs(query: str, k: int = 2) -> List[dict]:
    if not _DOC_INDEX["ready"]:
        _collect_docs()
    if not _DOC_INDEX["ready"]:
        return []
    try:
        import numpy as np
        qv = _DOC_INDEX["model"].encode([query]).astype('float32')
        dists, idxs = _DOC_INDEX["index"].search(qv, k)
        return [{"path": _DOC_INDEX["paths"][i], "score": float(1/(1+d)),
                 "snippet": _DOC_INDEX["documents"][i][:400]}
                for i, d in zip(idxs[0], dists[0]) if i < len(_DOC_INDEX["documents"])]
    except Exception:
        return []


# ── PATIENT CASE LOOKUP ───────────────────────────────────────────────────────
def _get_case_context(message: str, db: Session, user: dict) -> str:
    msg = message.lower()
    user_id = int(user.get("user_id", 0))
    user_role = user.get("role", "")

    # Extract file_id if mentioned
    import re
    fid_match = re.search(r'file[_\s]?(?:id[:\s#]?)?\s*(\d+)', msg)
    pid_match = re.search(r'patient[_\s]?(?:id[:\s#]?)?\s*([A-Za-z0-9\-]+)', msg)

    file_record = None
    if fid_match:
        fid = int(fid_match.group(1))
        file_record = db.query(DBFile).filter(DBFile.file_id == fid).first()
    elif pid_match:
        pid = pid_match.group(1)
        file_record = db.query(DBFile).filter(DBFile.patient_id == pid).order_by(DBFile.upload_date.desc()).first()

    if not file_record:
        return ""

    # Access control
    if user_role == "patient":
        db_user = db.query(User).filter(User.user_id == user_id).first()
        if file_record.patient_id != (db_user.medical_record_number if db_user else None):
            return ""
    elif user_role in ["doctor", "radiologist", "oncologist"]:
        has_access = db.query(FileAccessPermission).filter(
            FileAccessPermission.file_id == file_record.file_id,
            FileAccessPermission.doctor_id == user_id,
            FileAccessPermission.status == "active"
        ).first()
        if not has_access and file_record.user_id != user_id:
            return ""

    analysis = db.query(AnalysisResult).filter(
        AnalysisResult.file_id == file_record.file_id
    ).order_by(AnalysisResult.analysis_date.desc()).first()

    ctx = f"File ID: {file_record.file_id} | File: {file_record.filename} | Patient ID: {file_record.patient_id} | Status: {file_record.status}\n"
    if analysis:
        ctx += (
            f"Tumor Type: {analysis.classification_type} | "
            f"Confidence: {analysis.classification_confidence:.1f}% | "
            f"WHO Grade: {analysis.who_grade} | "
            f"Volume: {analysis.tumor_volume:.2f} mm³ | "
            f"Malignancy: {analysis.malignancy_level}\n"
        )
        if analysis.doctor_interpretation:
            ctx += f"Doctor Notes: {analysis.doctor_interpretation}\n"
    return ctx


# ── MAIN CHAT ENDPOINT ────────────────────────────────────────────────────────
@router.post("/chat")
def chat_assistant(body: dict, user=Depends(get_current_user), db: Session = Depends(get_db)):
    message: str = body.get("message", "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message'")

    sources = []

    # 1. Try to get patient case context
    case_ctx = _get_case_context(message, db, user)

    # 2. Try Groq LLM (best path)
    response = _llm_answer(message, context=case_ctx)

    if not response:
        # 3. Try local knowledge base
        response = _local_answer(message)

    if not response:
        # 4. Try FAISS doc search
        docs = _search_docs(message, k=2)
        if docs and docs[0]["score"] > 0.3:
            response = f"From the documentation:\n\n{docs[0]['snippet']}..."
            sources = [{"path": d["path"], "score": d["score"]} for d in docs]
        else:
            response = (
                "I can help with:\n\n"
                "**Medical Questions:** Ask about any tumor type (e.g. 'Tell me about Anaplastic Astrocytoma'), "
                "WHO grades, IDH/MGMT status, treatments.\n\n"
                "**Platform Features:** uploading scans, segmentation, visualization, reports, collaboration.\n\n"
                "**Patient Cases:** Ask 'What does file_id 5 show?' or 'Summarize patient MRN-001 case' "
                "(you must have access to that file).\n\n"
                "**Tip:** Set GROQ_API_KEY in your .env file for full AI-powered responses."
            )

    return {"response": response, "sources": sources}


# ── REPORT ENDPOINTS (unchanged) ─────────────────────────────────────────────
@router.post("/report")
def generate_report(body: dict, user=Depends(get_current_user), db: Session = Depends(get_db)):
    from jinja2 import Template
    from datetime import datetime

    patient_id = body.get("patient_id") or "Unknown"
    doctor_name = body.get("doctor_name") or (user.get("full_name") or "Doctor")
    summary = body.get("summary") or ""
    classification = body.get("classification") or {}
    segmentation = body.get("segmentation") or {}
    notes = body.get("notes") or ""
    predicted_type = classification.get("type") or classification.get("tumor_type") or "N/A"

    tmpl = Template(
        "Patient ID: {{ patient_id }}\nAssessing Clinician: {{ doctor_name }}\nDate: {{ date }}\n\n"
        "Clinical Summary:\n{{ summary }}\n\nAI Classification:\n"
        "- Predicted Type: {{ predicted_type }}\n- Confidence: {{ classification.confidence or 'N/A' }}\n\n"
        "Segmentation Metrics:\n- Tumor Volume (approx): {{ segmentation.volume or 'N/A' }}\n"
        "- Dice Score: {{ segmentation.dice or 'N/A' }}\n\nDoctor Notes:\n{{ notes }}\n\n"
        "Disclaimer: AI-generated insights for clinical support only. Verify with qualified clinician."
    )
    return {"report_text": tmpl.render(
        patient_id=patient_id, doctor_name=doctor_name, summary=summary,
        classification=classification, predicted_type=predicted_type,
        segmentation=segmentation, notes=notes,
        date=datetime.utcnow().strftime("%Y-%m-%d")
    )}


@router.get("/cases/{case_id}/similar")
def similar_cases(case_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        target = db.query(AnalysisResult).filter(AnalysisResult.file_id == case_id).first()
        if target and target.classification_type:
            matches = db.query(AnalysisResult).filter(
                AnalysisResult.classification_type == target.classification_type,
                AnalysisResult.file_id != case_id,
            ).order_by(AnalysisResult.analysis_date.desc()).limit(5).all()
        else:
            matches = db.query(AnalysisResult).order_by(AnalysisResult.analysis_date.desc()).limit(5).all()
        result = []
        for m in matches:
            f = db.query(DBFile).filter(DBFile.file_id == m.file_id).first()
            result.append({"file_id": m.file_id, "patient_id": f.patient_id if f else None,
                           "classification_type": m.classification_type,
                           "analysis_date": m.analysis_date.isoformat() if m.analysis_date else None})
        return {"similar": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report/pdf")
def generate_pdf_report(body: dict, user=Depends(get_current_user), db: Session = Depends(get_db)):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import inch
    from io import BytesIO
    import base64
    from datetime import datetime

    patient_id = body.get("patient_id") or "Unknown"
    doctor_name = body.get("doctor_name") or (user.get("full_name") or "Doctor")
    summary = body.get("summary") or ""
    classification = body.get("classification") or {}
    segmentation = body.get("segmentation") or {}
    notes = body.get("notes") or ""
    predicted_type = classification.get("type") or classification.get("tumor_type") or "N/A"

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=18,
                                  textColor=colors.HexColor('#003366'), spaceAfter=12)
    story.append(Paragraph("Seg-Mind Clinical Report", title_style))
    story.append(Spacer(1, 0.2*inch))

    info = [["Patient ID:", patient_id], ["Clinician:", doctor_name],
            ["Date:", datetime.utcnow().strftime("%Y-%m-%d")]]
    t = Table(info, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([('FONTNAME', (0,0),(0,-1),'Helvetica-Bold'),
                           ('FONTSIZE',(0,0),(-1,-1),10), ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>AI Classification</b>", styles['Heading2']))
    story.append(Paragraph(f"Type: {predicted_type} | Confidence: {classification.get('confidence','N/A')}", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Clinical Summary</b>", styles['Heading2']))
    story.append(Paragraph(summary or "N/A", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Notes</b>", styles['Heading2']))
    story.append(Paragraph(notes or "N/A", styles['BodyText']))

    doc.build(story)
    return {"pdf_base64": base64.b64encode(buffer.getvalue()).decode('utf-8')}


@router.get("/report/pdf/{file_id}")
def generate_report_for_file(file_id: int, user=Depends(get_current_user), db: Session = Depends(get_db)):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import inch
    from fastapi.responses import StreamingResponse
    from datetime import datetime
    from io import BytesIO

    user_id = int(user.get("user_id"))
    user_role = user.get("role")

    file_record = db.query(DBFile).filter(DBFile.file_id == file_id).first()
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    if user_role == "patient":
        db_user = db.query(User).filter(User.user_id == user_id).first()
        if file_record.patient_id != (db_user.medical_record_number if db_user else None):
            raise HTTPException(status_code=403, detail="Access denied")

    analysis = db.query(AnalysisResult).filter(
        AnalysisResult.file_id == file_id
    ).order_by(AnalysisResult.analysis_date.desc()).first()

    predicted_type = analysis.classification_type if analysis else "N/A"
    confidence = f"{analysis.classification_confidence:.1f}%" if (analysis and analysis.classification_confidence) else "N/A"
    volume = f"{analysis.tumor_volume:.2f} mm³" if (analysis and analysis.tumor_volume) else "N/A"
    doctor_name = "N/A"
    if analysis and analysis.assessed_by:
        doc_user = db.query(User).filter(User.user_id == analysis.assessed_by).first()
        doctor_name = doc_user.full_name if doc_user else "N/A"

    buffer = BytesIO()
    doc_pdf = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=18,
                                  textColor=colors.HexColor('#003366'), spaceAfter=12)
    story.append(Paragraph("Seg-Mind Clinical Report", title_style))
    story.append(Spacer(1, 0.15*inch))
    info = [["Patient ID:", file_record.patient_id or "Unknown"],
            ["File:", file_record.filename],
            ["Date:", datetime.utcnow().strftime("%Y-%m-%d")],
            ["Clinician:", doctor_name]]
    t = Table(info, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),
                           ('FONTSIZE',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>AI Classification Results</b>", styles['Heading2']))
    story.append(Paragraph(f"Predicted Type: {predicted_type} | Confidence: {confidence}", styles['BodyText']))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"Tumor Volume: {volume}", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    if analysis and analysis.doctor_interpretation:
        story.append(Paragraph("<b>Clinical Assessment</b>", styles['Heading2']))
        story.append(Paragraph(analysis.doctor_interpretation, styles['BodyText']))
    disc = ParagraphStyle('D', parent=styles['BodyText'], fontSize=8, textColor=colors.grey)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "<b>Disclaimer:</b> AI-generated insights for clinical support only. Verify with qualified clinician.", disc))
    doc_pdf.build(story)
    safe_pid = (file_record.patient_id or "report").replace(" ", "_")
    return StreamingResponse(BytesIO(buffer.getvalue()), media_type="application/pdf",
                             headers={"Content-Disposition": f'attachment; filename="report_{safe_pid}.pdf"'})
