import base64
import io
import json
from openai import OpenAI
from config import Config
from utils.logger import logger
from services.medical_entity_service import MedicalEntityResolver

class LLMService:
    def __init__(self):
        if not Config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is missing")
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)

    @staticmethod
    def _parse_json(content: str) -> dict:
        if not content:
            return {}
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.lower().startswith("json"):
                content = content[4:].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start >= 0 and end > start:
                return json.loads(content[start:end + 1])
            raise

    def extract_structured_data(self, fused_ocr_text: str, ocr_result: dict = None) -> dict:
        """
        Sends fused OCR text to GPT-4o for structured extraction.
        Uses multi-engine confidence scores and grounding information.
        Enforces JSON Mode for accuracy.
        """
        system_prompt = """
        You are an expert Medical Compliance Auditor for Kenya Social Health Authority (SHA).
        Analyze the provided OCR text and extract specific fields.
        
        CONFIDENCE INTERPRETATION:
        - HIGH (≥95%): Multiple OCR engines agree. Very reliable.
        - MEDIUM (70-95%): Some engine agreement. Generally reliable.
        - LOW (<70%): Poor agreement or single engine. Verify manually.
        
        RULES:
        1. If a field is missing or illegible, return null.
        1b. Never guess values. If not explicitly visible in OCR text, return null.
        2. Normalize dates to YYYY-MM-DD.
        3. Normalize currency to numbers only (e.g., "KES 5,000" -> 5000).
        4. Assign confidence (0.0 to 1.0) for EVERY field based on OCR agreement and clarity.
        5. STRONGLY prefer HIGH confidence items (≥95%).
        6. Mark MEDIUM and LOW confidence items with appropriate caution.
        7. Preserve evidence_source as the engine/section that supports the value.
        8. Use grounding information to locate fields on the page.
        """

        # Build enhanced context with confidence levels and grounding
        ocr_context = fused_ocr_text
        
        if ocr_result:
            confidence_stats = ocr_result.get("confidence_stats", {})
            annotations = ocr_result.get("annotations", [])
            
            ocr_context += f"\n\n=== MULTI-ENGINE CONFIDENCE ANALYSIS ===\n"
            ocr_context += f"High Confidence Items: {confidence_stats.get('high', 0)}\n"
            ocr_context += f"Medium Confidence Items: {confidence_stats.get('medium', 0)}\n"
            ocr_context += f"Low Confidence Items: {confidence_stats.get('low', 0)}\n"
            ocr_context += f"Average Confidence Score: {confidence_stats.get('average_confidence', 0):.4f}\n"
            
            # Add grounding information for high-confidence items
            ocr_context += f"\n=== GROUNDING INFORMATION (Bounding Boxes) ===\n"
            for ann in annotations:
                if ann.get('confidence', 0) >= 0.85:
                    text = ann.get('text', '')[:50]  # Truncate long text
                    bbox = ann.get('bbox', [])
                    engines = ann.get('engines_agreeing', [])
                    ocr_context += f"• '{text}' @ bbox{bbox} (engines: {', '.join(engines)})\n"

        user_prompt = f"""
        OCR TEXT CONTEXT (Multi-Engine Fusion with Confidence Scores):
        {ocr_context}
        
        Extract the following fields in valid JSON format:
        {{
            "patient_name": {{"value": "...", "confidence": 0.9, "grounding": "location hint"}},
            "patient_id": {{"value": "...", "confidence": 0.8, "grounding": "location hint"}},
            "date_of_birth": {{"value": "YYYY-MM-DD", "confidence": 0.9, "grounding": "location hint"}},
            "service_date": {{"value": "YYYY-MM-DD", "confidence": 0.9, "grounding": "location hint"}},
            "facility_name": {{"value": "...", "confidence": 0.9, "grounding": "location hint"}},
            "physician_name": {{"value": "...", "confidence": 0.7, "grounding": "location hint"}},
            "diagnosis": {{"value": "...", "confidence": 0.9, "grounding": "location hint"}},
            "icd_code": {{"value": "...", "confidence": 0.9, "grounding": "location hint"}},
            "total_cost": {{"value": 0, "confidence": 0.9, "grounding": "location hint"}},
            "clinical_summary": "Brief 2-sentence summary",
            "document_quality": {{"value": "clear|mixed|poor", "confidence": 0.9}},
            "handwriting_present": {{"value": true, "confidence": 0.8}},
            "ocr_confidence_avg": {confidence_stats.get('average_confidence', 0) if ocr_result else 0}
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=Config.OPENAI_TEMPERATURE
            )
            
            content = response.choices[0].message.content
            extracted_data = self._parse_json(content)
            
            # Apply medical entity resolution to validate and normalize medical codes
            extracted_data = self._apply_entity_resolution(extracted_data)
            
            return extracted_data
        except Exception as e:
            logger.error(f"OpenAI Extraction Error: {e}")
            return {}

    def _verify_with_vision_single_page(self, image_pil, extracted_data: dict, annotations=None, page_number=None) -> dict:
        """
        Uses GPT-4o Vision to double-check critical fields (Name, Cost, ID) directly from the image.
        This catches OCR misinterpretations (e.g., S vs 5).
        """
        # Convert PIL to Base64
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        page_hint = f"This is page {page_number}." if page_number else "Page number unknown."
        prompt = f"""
        Verify the following data extracted by OCR against the original medical document image.
        Read both printed and handwritten content. Use the OCR annotations as hints only;
        if the image contradicts OCR, trust the image.
        {page_hint}

        Focus on:
        - Patient Name
        - Patient ID
        - Date of Birth
        - Service Date
        - Facility Name
        - Physician Name / Signature text if readable
        - Diagnosis / ICD Code
        - Total Cost
        
        Extracted Data:
        {json.dumps(extracted_data, indent=2)}

        OCR Annotation Samples:
        {json.dumps((annotations or [])[:80], indent=2)}
        
        Return valid JSON only:
        {{
            "status": "verified|corrected|uncertain",
            "corrections": {{
                "patient_name": "Correct Name or null",
                "patient_id": "Correct ID or null",
                "date_of_birth": "YYYY-MM-DD or null",
                "service_date": "YYYY-MM-DD or null",
                "facility_name": "Correct facility or null",
                "physician_name": "Correct physician or null",
                "diagnosis": "Correct diagnosis or null",
                "icd_code": "Correct ICD or null",
                "total_cost": 5000
            }},
            "visual_findings": ["short notes about unreadable handwriting, missing stamps, altered fields, or mismatches"],
            "annotation_targets": [
                {{"field": "patient_name", "text": "seen text", "reason": "why it matters"}}
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_VISION_MODEL,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]}
                ],
                temperature=0.0
            )
            
            # Parse the verification result
            parsed = self._parse_json(response.choices[0].message.content)
            if page_number is not None and isinstance(parsed, dict):
                parsed["page"] = page_number
            return parsed
        except Exception as e:
            logger.error(f"OpenAI Vision Error: {e}")
            return {"status": "verification_failed", "page": page_number}

    def verify_with_vision(self, image_pil, extracted_data: dict, annotations=None) -> dict:
        """
        Supports single-page PIL image or list of PIL images.
        For multi-page docs, runs verification per page and aggregates corrections.
        """
        if isinstance(image_pil, list):
            page_results = []
            for idx, page_image in enumerate(image_pil, start=1):
                page_annotations = [
                    ann for ann in (annotations or [])
                    if ann.get("page") == idx
                ]
                page_results.append(
                    self._verify_with_vision_single_page(
                        page_image,
                        extracted_data,
                        page_annotations,
                        page_number=idx,
                    )
                )

            correction_fields = [
                "patient_name",
                "patient_id",
                "date_of_birth",
                "service_date",
                "facility_name",
                "physician_name",
                "diagnosis",
                "icd_code",
                "total_cost",
            ]
            corrections = {}
            statuses = []
            visual_findings = []
            annotation_targets = []
            for result in page_results:
                statuses.append(result.get("status"))
                visual_findings.extend(result.get("visual_findings", []) or [])
                annotation_targets.extend(result.get("annotation_targets", []) or [])

            for field in correction_fields:
                selected = None
                for result in page_results:
                    value = (result.get("corrections", {}) or {}).get(field)
                    if value not in (None, "", "null"):
                        selected = value
                        break
                corrections[field] = selected

            if any(status == "corrected" for status in statuses):
                status = "corrected"
            elif any(status == "uncertain" for status in statuses):
                status = "uncertain"
            elif all(status == "verified" for status in statuses if status):
                status = "verified"
            else:
                status = "verification_failed"

            return {
                "status": status,
                "corrections": corrections,
                "visual_findings": visual_findings[:40],
                "annotation_targets": annotation_targets[:120],
                "per_page_checks": page_results,
            }

        return self._verify_with_vision_single_page(image_pil, extracted_data, annotations=annotations, page_number=1)

    def generate_reasoning(self, extracted_data: dict, validation: dict, vision_check: dict) -> str:
        """
        Generates AI reasoning about what fields are missing, invalid, or suspicious,
        and provides recommendations for compliance.
        """
        prompt = f"""
        You are a compliance assistant for ClaimFlow. An insurance claim has been processed by OCR and vision models.
        Review the following results and provide a brief, professional, and clear AI reasoning summary (maximum 3-4 sentences) explaining:
        1. What key information is missing or invalid according to the compliance rules.
        2. Any discrepancy found (e.g., between OCR and Vision).
        3. Recommendations on how the user can fix it (e.g. "Please upload a clearer scan to resolve the physician signature" or "Please fill in the missing ICD code manually").
        
        Extracted Fields:
        {json.dumps(extracted_data, indent=2)}
        
        Validation Results:
        {json.dumps(validation, indent=2)}
        
        Vision Findings:
        {json.dumps(vision_check.get("visual_findings", []), indent=2)}
        
        Return a clean text summary. Do not include markdown code block syntax. Just return the raw text.
        """
        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful medical claim compliance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating AI reasoning: {e}")
            return "Unable to generate AI reasoning at this time."

    def _apply_entity_resolution(self, extracted_data: dict) -> dict:
        """
        Apply medical entity resolution to validate and normalize medical codes.
        
        This fixes issues like:
        - "o111" → validated to "O11.1" (Pre-existing hypertension with pre-eclampsia)
        - "2ZIN FANNED" → flagged as invalid, suggests manual review
        - Procedure names normalized to standard terminology
        
        Adds resolution metadata without overwriting LLM extraction.
        """
        if not extracted_data:
            return extracted_data
        
        try:
            resolver = MedicalEntityResolver()
            
            # Prepare fields for resolution
            fields_to_resolve = {
                'diagnosis_icd': extracted_data.get('icd_code', {}).get('value'),
                'procedure': extracted_data.get('diagnosis', {}).get('value'),  # Often contains procedure info
            }
            
            # Run entity resolution
            resolution_result = resolver.resolve_extracted_fields(fields_to_resolve)
            
            # Add resolution metadata to the extracted data
            if resolution_result['anomalies']:
                extracted_data['_entity_resolution'] = {
                    'status': 'VALIDATION_ISSUES_DETECTED',
                    'anomalies': resolution_result['anomalies'],
                    'confidence_score': resolution_result['confidence_score'],
                    'requires_manual_review': resolution_result['requires_manual_review']
                }
                
                # Log anomalies for monitoring
                logger.warning(f"Entity resolution flagged {len(resolution_result['anomalies'])} anomalies")
                for anomaly in resolution_result['anomalies']:
                    logger.warning(f"  - {anomaly['field']}: {anomaly['issue']} (severity: {anomaly['severity']})")
            else:
                # All resolved successfully
                extracted_data['_entity_resolution'] = {
                    'status': 'VALIDATED',
                    'confidence_score': resolution_result['confidence_score'],
                    'requires_manual_review': False
                }
            
            # Update ICD code if better resolution found
            if resolution_result['resolved_fields'].get('diagnosis_icd'):
                icd_resolution = resolution_result['resolved_fields']['diagnosis_icd']
                if icd_resolution.get('resolved') and icd_resolution['resolved'] != fields_to_resolve.get('diagnosis_icd'):
                    if 'icd_code' in extracted_data and isinstance(extracted_data['icd_code'], dict):
                        extracted_data['icd_code']['resolved_value'] = icd_resolution['resolved']
                        extracted_data['icd_code']['resolution_confidence'] = icd_resolution['confidence']
                        extracted_data['icd_code']['resolution_description'] = icd_resolution['description']
            
            return extracted_data
        
        except Exception as e:
            logger.error(f"Error in entity resolution: {e}")
            # Return original data if resolution fails
            return extracted_data
    
    def chat(self, message: str, history: list = None, document_context: dict = None) -> str:
        """
        Interacts with the user on any topic. If document_context is provided,
        helps answer questions about the specific claim/document.
        """
        if history is None:
            history = []
            
        system_prompt = """
        You are ClaimFlow AI, a helpful, intelligent assistant for ClaimFlow, a professional document validation system for Kenya Social Health Authority (SHA) medical claims.
        You can chat about any topic, answer general knowledge questions, and help users with their tasks.
        
        If the user is asking about the current document, use the provided Document Context to answer accurately.
        For document questions, always prioritize:
        1) what is present,
        2) what is missing,
        3) page-level evidence,
        4) practical next actions to fix missing requirements.
        Be professional, concise, and helpful. You can format your output with simple markdown (bold, lists, etc.) where appropriate.
        """
        
        if document_context:
            system_prompt += f"\n\nCURRENT DOCUMENT CONTEXT:\n{json.dumps(document_context, indent=2)}"
            
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history
        # History format: [{"sender": "user"|"bot", "text": "..."}]
        for msg in history:
            role = "user" if msg.get("sender") == "user" else "assistant"
            messages.append({
                "role": role,
                "content": msg.get("text") or msg.get("content") or ""
            })
            
        # Add current message
        messages.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Chatbot Error: {e}")
            return "I apologize, but I encountered an error processing that message."
