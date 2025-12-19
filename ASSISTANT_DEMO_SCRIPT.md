# Seg-Mind AI Assistant - Demo Script (60-90 seconds)

## Setup (5 seconds)
**Prerequisites:** Backend running, logged in as Doctor

## Demo Flow

### 1. Navigate to Assistant (10 sec)
- **Action:** Click "Assistant" in navbar
- **Show:** Clean chat UI with header "Ask about setup, features, or generate reports"

### 2. Query Documentation (20 sec)
- **Type:** "What is validation Dice?"
- **Show:** 
  - Response appears with semantic search results
  - Sources listed with doc paths (e.g., `RESNET_TRAINING_COMPLETE.md`)
  - Relevant snippet included in answer
- **Narrate:** "The assistant uses sentence-transformers + FAISS to search project docs, providing instant contextual help without LLM API costs."

### 3. Generate Clinical Report (30 sec)
- **Type:** "Generate report for PT-2025-00001"
- **Show in Browser Console (or use Postman):**
  ```javascript
  // POST to /api/v1/assistant/report
  {
    "patient_id": "PT-2025-00001",
    "doctor_name": "Dr. Demo",
    "summary": "62yo male with headaches. MRI shows enhancing lesion in right frontal lobe.",
    "classification": { "type": "Glioma", "confidence": "0.87" },
    "segmentation": { "volume": "28.3 cc", "dice": "0.81" },
    "notes": "Recommend biopsy and correlate with histopathology. Consider adjuvant therapy."
  }
  ```
- **Show:** Structured report with patient info, AI predictions, metrics, and disclaimer
- **Narrate:** "Templated report generation with Jinja2 turns ML outputs into clinical-ready documents. Add the PDF endpoint to enable one-click downloads."

### 4. Similar Cases Search (15 sec)
- **Type in console or describe:**
  ```
  GET /api/v1/assistant/cases/123/similar
  ```
- **Show:** Returns 5 similar cases by classification type
- **Narrate:** "Similar cases help doctors reference prior analyses. Currently matches by tumor type; can be upgraded to embedding-based similarity with volumetrics."

### 5. PDF Export (10 sec)
- **Show (in console or Postman):**
  ```javascript
  // POST to /api/v1/assistant/report/pdf
  // Same payload as text report
  ```
- **Show:** Returns `pdf_base64` field
- **Narrate:** "PDF export via reportlab creates downloadable, archivable reports with proper clinical formatting and disclaimers."

## Key Talking Points
- **Practical AI:** RAG help, auto-reports, and similar cases directly support clinical workflows
- **Cost-Effective:** Uses open-source embeddings (no OpenAI calls) and lightweight FAISS indexing
- **Recruiter-Friendly:** Demonstrates real-world ML integration, responsible AI (disclaimers), and full-stack skills
- **Scalable:** Built with FastAPI routers, can add advanced LLM chains, agent orchestration, or multimodal inputs later

## Why This Matters
- **Differentiator:** Most FYPs stop at inference; this adds clinical decision support tools
- **Hireability:** Shows understanding of RAG, embeddings, responsible AI, API design, and user needs
- **Production-Ready:** Proper error handling, auth-gated, and extensible architecture

## Optional Extensions to Mention
- Upgrade to LangChain/LangGraph for multi-step reasoning (e.g., "suggest follow-up based on findings")
- Add multimodal embeddings for scan images + text
- Integrate drift detection and MLOps logging
- Deploy with Docker + HTTPS for real clinical pilot

---

**Total Time:** ~90 seconds  
**Impact:** Transforms FYP from "ML project" to "AI-powered clinical decision support system"
