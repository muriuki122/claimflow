import re
from collections import Counter


class RequirementsService:
    BASE_REQUIRED_FIELDS = {
        "patient_name": "Patient Name",
        "patient_id": "Patient ID",
        "date_of_birth": "Date of Birth",
        "service_date": "Service Date",
        "facility_name": "Facility Name",
        "physician_name": "Physician Name",
        "diagnosis": "Diagnosis",
        "icd_code": "ICD Code",
        "total_cost": "Total Cost",
    }

    STOPWORDS = {
        "the", "and", "for", "with", "from", "that", "this", "form", "page", "claim",
        "medical", "health", "sha", "name", "date", "code", "cost", "total", "patient",
        "facility", "doctor", "physician", "address", "number", "amount", "of", "to",
        "in", "on", "is", "are", "be", "at", "or", "by", "as", "an", "a", "it", "id",
    }

    DOC_PROFILES = {
        "medical_claim": [
            "patient_name", "patient_id", "service_date", "facility_name",
            "physician_name", "diagnosis", "icd_code", "total_cost"
        ],
        "birth_notification": [
            "patient_name", "date_of_birth", "facility_name", "service_date"
        ],
        "generic": ["patient_name", "service_date"],
    }

    REQUIREMENT_CATALOG = {
        "patient_name": ["patient name", "name", "surname", "other names", "first name"],
        "patient_id": ["patient id", "id number", "file no", "serial no", "membership number"],
        "date_of_birth": ["date of birth", "dob", "birth date", "day month year"],
        "service_date": ["service date", "date of service", "visit date", "admission date", "discharge date"],
        "facility_name": ["facility", "hospital", "clinic", "health center"],
        "physician_name": ["physician", "doctor", "attending", "provider", "consultant"],
        "diagnosis": ["diagnosis", "assessment", "impression"],
        "icd_code": ["icd", "diagnosis code", "icd-10", "icd-11"],
        "total_cost": ["total cost", "amount", "bill", "charge", "fee", "ksh", "kes"],
    }

    @staticmethod
    def _flat_value(value):
        if isinstance(value, dict):
            return value.get("value")
        return value

    @staticmethod
    def _normalize_text(text):
        return re.sub(r"\s+", " ", (text or "")).strip()

    @staticmethod
    def _tokenize_words(text):
        return re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text or "")

    def _extract_noun_candidates(self, text, limit=18):
        words = self._tokenize_words(text.lower())
        filtered = [w for w in words if w not in self.STOPWORDS]
        counts = Counter(filtered)
        return [w for w, _ in counts.most_common(limit)]

    @staticmethod
    def _extract_numeric_candidates(text):
        content = text or ""
        patterns = {
            "dates": r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
            "currency_values": r"\b(?:KES|KSH|USD)?\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b",
            "icd_codes": r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b",
            "ids": r"\b[A-Z0-9]{5,}\b",
        }
        extracted = {}
        for key, pattern in patterns.items():
            matches = re.findall(pattern, content, flags=re.IGNORECASE)
            cleaned = []
            seen = set()
            for match in matches:
                candidate = re.sub(r"\s+", " ", str(match)).strip()
                low = candidate.lower()
                if low and low not in seen:
                    seen.add(low)
                    cleaned.append(candidate)
            extracted[key] = cleaned[:20]
        return extracted

    @staticmethod
    def _find_value_pages(page_text_map, value):
        if value is None:
            return []
        value_str = str(value).strip().lower()
        if not value_str:
            return []
        hits = []
        for page, text in page_text_map.items():
            if value_str in (text or "").lower():
                hits.append(page)
        return hits

    @staticmethod
    def _is_filled_value(value):
        if value is None:
            return False
        text = str(value).strip().lower()
        if not text:
            return False
        return text not in {"n/a", "na", "none", "null", "unknown", "-", "--", "not provided"}

    @classmethod
    def _printed_presence_score(cls, page_text, field_key):
        text = (page_text or "").lower()
        hits = sum(1 for kw in cls.REQUIREMENT_CATALOG.get(field_key, []) if kw in text)
        if hits >= 2:
            return 1.0
        if hits == 1:
            return 0.65
        return 0.0

    def _detect_document_profile(self, page_results, extracted_data):
        all_text = " ".join((p.get("ocr_text") or "").lower() for p in page_results)
        if any(k in all_text for k in ("birth notification", "acknowledgement of birth", "form b1")):
            return "birth_notification"
        if any(self._is_filled_value(self._flat_value(extracted_data.get(k))) for k in ("diagnosis", "icd_code", "total_cost")):
            return "medical_claim"
        return "generic"

    def analyze(self, page_results, extracted_data, validation_dict):
        page_text_map = {
            page_result.get("page"): self._normalize_text(page_result.get("ocr_text", ""))
            for page_result in page_results
        }
        profile = self._detect_document_profile(page_results, extracted_data)
        # Dynamic requirements: detect what this document actually asks for via printed labels.
        dynamically_detected = []
        for field_key in self.REQUIREMENT_CATALOG.keys():
            if any(self._printed_presence_score(page_text_map.get(p.get("page"), ""), field_key) > 0 for p in page_results):
                dynamically_detected.append(field_key)

        if dynamically_detected:
            active_requirements = dynamically_detected
        else:
            # fallback profile when label detection is weak
            active_requirements = self.DOC_PROFILES.get(profile, self.DOC_PROFILES["generic"])

        page_requirements = []
        for page_result in page_results:
            page_no = page_result.get("page")
            page_text = self._normalize_text(page_result.get("ocr_text", ""))
            page_field_detection = {}
            for field_key in active_requirements:
                value = self._flat_value(extracted_data.get(field_key))
                filled = self._is_filled_value(value)
                printed_score = self._printed_presence_score(page_text, field_key)
                value_in_page = str(value).strip().lower() in page_text.lower() if filled else False
                extraction_confidence = 1.0 if value_in_page and filled else (0.45 if printed_score > 0 else 0.0)
                page_field_detection[field_key] = {
                    "printed_requirement_detected": printed_score > 0,
                    "filled_value_detected": bool(value_in_page),
                    "extracted_value": str(value) if filled and value_in_page else None,
                    "confidence": round(extraction_confidence, 4),
                }
            page_requirements.append({
                "page": page_no,
                "noun_candidates": self._extract_noun_candidates(page_text),
                "numeric_candidates": self._extract_numeric_candidates(page_text),
                "annotation_count": len(page_result.get("annotations", [])),
                "field_detection": page_field_detection,
            })

        missing_fields = set(validation_dict.get("missing_fields", []))
        field_presence = []
        present_fields = []
        missing_field_details = []
        requirement_scores = []
        requirement_breakdown = []
        total_requirements = max(1, len(active_requirements))
        points_per_requirement = 100.0 / total_requirements

        for field_key in active_requirements:
            label = self.BASE_REQUIRED_FIELDS.get(field_key, field_key.replace("_", " ").title())
            raw_value = self._flat_value(extracted_data.get(field_key))
            is_present = self._is_filled_value(raw_value)
            evidence_pages = self._find_value_pages(page_text_map, raw_value)
            field_conf = 0.0
            extracted_conf = 0.0
            source = ""
            if isinstance(extracted_data.get(field_key), dict):
                try:
                    extracted_conf = float(extracted_data.get(field_key, {}).get("confidence", 0.0))
                except (TypeError, ValueError):
                    extracted_conf = 0.0
                source = str(extracted_data.get(field_key, {}).get("source", "")).lower()

            printed_presence_avg = 0.0
            printed_samples = [
                self._printed_presence_score(page_text_map.get(p.get("page"), ""), field_key)
                for p in page_results
            ]
            if printed_samples:
                printed_presence_avg = sum(printed_samples) / len(printed_samples)

            if is_present and field_key not in missing_fields:
                # Blend extracted confidence + page evidence + printed-field detection
                evidence_score = min(len(evidence_pages) / max(1, len(page_results)), 1.0)
                field_conf = (0.60 * extracted_conf) + (0.25 * evidence_score) + (0.15 * printed_presence_avg)
                # Manual verified edits can earn full confidence if values are filled.
                if source == "user_edit":
                    field_conf = max(field_conf, 1.0 if is_present else 0.0)
            else:
                field_conf = 0.15 * printed_presence_avg

            entry = {
                "field": field_key,
                "label": label,
                "present": is_present and field_key not in missing_fields,
                "value": None if raw_value is None else str(raw_value),
                "evidence_pages": evidence_pages,
                "confidence": round(max(0.0, min(field_conf, 1.0)), 4),
            }
            field_presence.append(entry)
            requirement_scores.append(entry["confidence"])

            if entry["present"]:
                present_fields.append(field_key)
                requirement_breakdown.append({
                    "field": field_key,
                    "label": label,
                    "met": True,
                    "points_earned": round(points_per_requirement, 4),
                    "points_possible": round(points_per_requirement, 4),
                    "confidence": entry["confidence"],
                })
            else:
                requirement_breakdown.append({
                    "field": field_key,
                    "label": label,
                    "met": False,
                    "points_earned": 0.0,
                    "points_possible": round(points_per_requirement, 4),
                    "confidence": entry["confidence"],
                })
                missing_field_details.append({
                    "field": field_key,
                    "label": label,
                    "status": "missing",
                    "confidence": entry["confidence"],
                    "hint": f"{label} not found clearly. Please provide or re-upload a clearer page with this field visible.",
                })

        human_annotation_summary = [
            {
                "type": "present",
                "message": f"{item['label']} extracted{' on page(s) ' + str(item['evidence_pages']) if item['evidence_pages'] else ''} with confidence {round(item['confidence'] * 100, 1)}%.",
                "field": item["field"],
                "confidence": item["confidence"],
            }
            for item in field_presence
            if item["present"]
        ] + [
            {
                "type": "missing",
                "message": detail["hint"],
                "field": detail["field"],
                "confidence": detail.get("confidence", 0.0),
            }
            for detail in missing_field_details
        ]

        met_count = len(present_fields)
        final_requirement_score = round(met_count * points_per_requirement, 2)
        # avoid floating residue above 100
        final_requirement_score = min(100.0, final_requirement_score)

        return {
            "document_profile": profile,
            "active_requirements": active_requirements,
            "requirements_total": len(active_requirements),
            "requirements_met_count": met_count,
            "points_per_requirement": round(points_per_requirement, 4),
            "page_requirements": page_requirements,
            "field_presence": field_presence,
            "requirement_breakdown": requirement_breakdown,
            "present_fields": present_fields,
            "missing_fields_detailed": missing_field_details,
            "human_annotation_summary": human_annotation_summary,
            "final_requirement_score": final_requirement_score,
        }
