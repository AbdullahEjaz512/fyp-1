from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pathlib import Path
import os

from app.database import get_db, AnalysisResult, File as DBFile
from app.dependencies.auth import get_current_user

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


# Semantic RAG over project docs using sentence-transformers + FAISS
_DOC_INDEX = {
    "ready": False,
    "documents": [],
    "paths": [],
    "model": None,
    "index": None,
}


def _collect_docs() -> None:
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
    except Exception:
        return  # dependencies not available

    # assistant.py is at backend/app/routers; go 3 levels up to project root
    root = Path(__file__).resolve().parents[3]
    md_files: List[Path] = []
    # Collect top-level markdown docs and key READMEs
    for p in root.glob("*.md"):
        md_files.append(p)
    # Frontend and backend READMEs if present
    for p in [root / "frontend" / "README.md", root / "backend" / "README.md"]:
        if p.exists():
            md_files.append(p)

    documents = []
    paths = []
    for p in md_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            documents.append(text)
            paths.append(str(p.relative_to(root)))
        except Exception:
            continue

    if not documents:
        return

    try:
        # Use a lightweight sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(documents, show_progress_bar=False)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        _DOC_INDEX.update({
            "ready": True,
            "documents": documents,
            "paths": paths,
            "model": model,
            "index": index,
        })
    except Exception as e:
        # fail silently; assistant will still respond without RAG
        print(f"Warning: Could not initialize doc index: {e}")
        return


def _search_docs(query: str, k: int = 3) -> List[dict]:
    if not _DOC_INDEX.get("ready"):
        _collect_docs()
    if not _DOC_INDEX.get("ready"):
        return []
    try:
        import numpy as np
        model = _DOC_INDEX["model"]
        index = _DOC_INDEX["index"]
        
        # Encode query
        query_vec = model.encode([query])
        
        # Search
        distances, indices = index.search(query_vec.astype('float32'), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(_DOC_INDEX["documents"]):
                results.append({
                    "path": _DOC_INDEX["paths"][idx],
                    "score": float(1.0 / (1.0 + dist)),  # convert distance to similarity
                    "snippet": _DOC_INDEX["documents"][idx][:500],
                })
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []


@router.post("/chat")
def chat_assistant(
    body: dict,
    user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Conversational assistant endpoint.
    Request: { "message": str, "conversation_history": list }
    Response: { "response": str, "sources": [ {path, score} ] }
    """
    message: Optional[str] = body.get("message")
    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' in request body")

    # Normalize message for pattern matching
    msg_lower = message.lower()
    
    # Enhanced response patterns for common medical questions
    default_responses = {
        "tumor classification": "Our platform uses a ResNet-based deep learning model to classify brain tumors into four categories: Glioma, Meningioma, Pituitary tumor, and No tumor. The model achieves 96%+ accuracy and provides confidence scores with each prediction. You can view the classification results in the Results tab after uploading a scan.",
        
        "segmentation": "The segmentation module uses a U-Net architecture trained on medical imaging data to precisely identify tumor boundaries. It provides volumetric measurements, 3D visualization, and generates masks that help in surgical planning and radiation therapy targeting.",
        
        "growth prediction": "Our LSTM-based growth prediction model analyzes sequential scans to forecast tumor growth over time. It predicts future tumor volume with MAE of ~1.45 cc and helps in treatment planning by showing expected growth trajectories.",
        
        "upload scan": "To upload a scan: 1) Go to the Upload tab, 2) Select your MRI files (NIfTI format supported), 3) Fill in patient details, 4) Click upload. The system will automatically run segmentation and classification analysis. Results typically appear within 2-3 minutes.",
        
        "visualization": "We offer two visualization modes: 2D Visualization (slice-by-slice view with adjustable windowing) and 3D Reconstruction (interactive volumetric rendering). Both include tumor highlighting and measurement tools.",
        
        "explainability": "Our XAI (Explainable AI) module uses Grad-CAM and SHAP to show which brain regions influenced the classification decision. This helps clinicians understand and validate the AI's reasoning.",
        
        "collaboration": "Doctors can share cases with colleagues for second opinions. Use the Share button on any case to invite other clinicians. All shared cases maintain full audit trails.",
        
        "report generation": "The platform can auto-generate clinical reports including patient info, AI findings, measurements, and doctor notes. Access this via the Generate Report button on the Results page.",
    }
    
    # Check for pattern matches
    response = None
    for keyword, answer in default_responses.items():
        if keyword in msg_lower:
            response = answer
            break
    
    # If no pattern match, search docs
    if not response:
        sources = _search_docs(message, k=3)
        if sources:
            top = sources[0]
            snippet = top["snippet"][:300] + "..."
            response = (
                f"Based on the documentation: {snippet}\n\n"
                "Would you like more specific information about any feature?"
            )
        else:
            response = (
                "I can help you with:\n\n"
                "• Tumor classification and results interpretation\n"
                "• Segmentation analysis and volumetric measurements\n"
                "• Growth prediction insights\n"
                "• How to upload and process scans\n"
                "• 2D/3D visualization features\n"
                "• Explainable AI (Grad-CAM, SHAP)\n"
                "• Collaboration and case sharing\n"
                "• Report generation\n\n"
                "What would you like to know more about?"
            )
    
    # Get sources if any
    sources = _search_docs(message, k=3)
    
    return {"response": response, "sources": [{"path": s["path"], "score": s["score"]} for s in sources]}



@router.post("/report")
def generate_report(
    body: dict,
    user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate a structured clinical-style report from provided case data.
    Expected body keys: patient_id, doctor_name, summary, classification, segmentation, notes
    Returns: { report_text: str }
    """
    from jinja2 import Template

    patient_id = body.get("patient_id") or "Unknown"
    doctor_name = body.get("doctor_name") or (user.get("full_name") or "Doctor")
    summary = body.get("summary") or ""
    classification = body.get("classification") or {}
    segmentation = body.get("segmentation") or {}
    notes = body.get("notes") or ""

    tmpl = Template(
        (
            "Patient ID: {{ patient_id }}\n"
            "Assessing Clinician: {{ doctor_name }}\n"
            "Date: {{ date }}\n\n"
            "Clinical Summary:\n{{ summary }}\n\n"
            "AI Classification:\n"
            "- Predicted Type: {{ classification.type or 'N/A' }}\n"
            "- Confidence: {{ classification.confidence or 'N/A' }}\n\n"
            "Segmentation Metrics:\n"
            "- Tumor Volume (approx): {{ segmentation.volume or 'N/A' }}\n"
            "- Dice Score: {{ segmentation.dice or 'N/A' }}\n\n"
            "Doctor Notes:\n{{ notes }}\n\n"
            "Disclaimer: This report leverages AI-generated insights and is intended to\n"
            "support clinical decision-making. It does not substitute for professional\n"
            "medical judgment. Findings should be verified by a qualified clinician."
        )
    )

    from datetime import datetime
    report_text = tmpl.render(
        patient_id=patient_id,
        doctor_name=doctor_name,
        summary=summary,
        classification=classification,
        segmentation=segmentation,
        notes=notes,
        date=datetime.utcnow().strftime("%Y-%m-%d"),
    )

    return {"report_text": report_text}


@router.get("/cases/{case_id}/similar")
def similar_cases(
    case_id: int,
    user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return similar cases by matching classification type when available.
    Fallback to recent analyses.
    """
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
            file_info = db.query(DBFile).filter(DBFile.file_id == m.file_id).first()
            result.append({
                "file_id": m.file_id,
                "patient_id": file_info.patient_id if file_info else None,
                "classification_type": m.classification_type,
                "analysis_date": m.analysis_date.isoformat() if m.analysis_date else None,
            })
        return {"similar": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve similar cases: {e}")


@router.post("/report/pdf")
def generate_pdf_report(
    body: dict,
    user = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate a PDF report from case data.
    Returns: { "pdf_base64": str } (downloadable via frontend)
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.units import inch
    from io import BytesIO
    import base64

    patient_id = body.get("patient_id") or "Unknown"
    doctor_name = body.get("doctor_name") or (user.get("full_name") or "Doctor")
    summary = body.get("summary") or ""
    classification = body.get("classification") or {}
    segmentation = body.get("segmentation") or {}
    notes = body.get("notes") or ""

    # Create PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#003366'),
        spaceAfter=12,
    )
    story.append(Paragraph("Seg-Mind Clinical Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient info table
    from datetime import datetime
    data = [
        ["Patient ID:", patient_id],
        ["Assessing Clinician:", doctor_name],
        ["Report Date:", datetime.utcnow().strftime("%Y-%m-%d")],
    ]
    t = Table(data, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))
    
    # Clinical Summary
    story.append(Paragraph("<b>Clinical Summary:</b>", styles['Heading2']))
    story.append(Paragraph(summary or "N/A", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # AI Classification
    story.append(Paragraph("<b>AI Classification Results:</b>", styles['Heading2']))
    class_data = [
        ["Predicted Type:", classification.get("type", "N/A")],
        ["Confidence:", str(classification.get("confidence", "N/A"))],
    ]
    ct = Table(class_data, colWidths=[2*inch, 4*inch])
    ct.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(ct)
    story.append(Spacer(1, 0.2*inch))
    
    # Segmentation
    story.append(Paragraph("<b>Segmentation Metrics:</b>", styles['Heading2']))
    seg_data = [
        ["Tumor Volume:", str(segmentation.get("volume", "N/A"))],
        ["Dice Score:", str(segmentation.get("dice", "N/A"))],
    ]
    st = Table(seg_data, colWidths=[2*inch, 4*inch])
    st.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.2*inch))
    
    # Doctor notes
    story.append(Paragraph("<b>Clinical Notes:</b>", styles['Heading2']))
    story.append(Paragraph(notes or "N/A", styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['BodyText'],
        fontSize=8,
        textColor=colors.grey,
        borderColor=colors.grey,
        borderWidth=1,
        borderPadding=8,
    )
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report leverages AI-generated insights and is intended to "
        "support clinical decision-making. It does not substitute for professional "
        "medical judgment. Findings should be verified by a qualified clinician.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    # Encode as base64
    pdf_b64 = base64.b64encode(pdf_bytes).decode('utf-8')
    
    return {"pdf_base64": pdf_b64}
