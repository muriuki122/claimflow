import os
import json
import re
from difflib import SequenceMatcher


class EvaluationService:
    FIELD_NAMES = [
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

    @staticmethod
    def normalize(value):
        if value is None:
            return ""
        value = str(value).strip().lower()
        value = re.sub(r"\s+", " ", value)
        value = re.sub(r"[^\w\s.-]", "", value)
        return value

    @staticmethod
    def value_similarity(expected, actual):
        expected_norm = EvaluationService.normalize(expected)
        actual_norm = EvaluationService.normalize(actual)
        if not expected_norm and not actual_norm:
            return 1.0
        if not expected_norm or not actual_norm:
            return 0.0
        if expected_norm == actual_norm:
            return 1.0
        return SequenceMatcher(None, expected_norm, actual_norm).ratio()

    @staticmethod
    def load_ground_truth(document_path):
        base, _ = os.path.splitext(document_path)
        expected_path = f"{base}.json"
        if not os.path.exists(expected_path):
            return None
        with open(expected_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare(self, expected, extracted_fields):
        field_results = {}
        scores = []

        for field in self.FIELD_NAMES:
            expected_value = expected.get(field)
            actual_field = extracted_fields.get(field, {})
            actual_value = actual_field.get("value") if isinstance(actual_field, dict) else actual_field
            similarity = self.value_similarity(expected_value, actual_value)
            passed = similarity >= 0.9
            field_results[field] = {
                "expected": expected_value,
                "actual": actual_value,
                "similarity": round(similarity, 4),
                "passed": passed,
            }
            if expected_value not in (None, ""):
                scores.append(similarity)

        accuracy = (sum(scores) / len(scores) * 100) if scores else None
        return {
            "has_ground_truth": True,
            "accuracy_score": None if accuracy is None else round(accuracy, 2),
            "field_results": field_results,
        }

    def estimate_quality(self, extracted_fields, validation, vision_check, engines_used, annotations):
        confidences = []
        for field in extracted_fields.values():
            if isinstance(field, dict):
                try:
                    confidences.append(float(field.get("confidence", 0)))
                except (TypeError, ValueError):
                    pass

        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0
        validation_score = validation.get("score", 0) / 100 if isinstance(validation, dict) else 0
        engine_score = min(len(engines_used or []) / 3, 1)
        annotation_score = min(len(annotations or []) / 25, 1)
        vision_status = vision_check.get("status") if isinstance(vision_check, dict) else None
        vision_score = {"verified": 1, "corrected": 0.85, "uncertain": 0.55}.get(vision_status, 0.4)

        score = (
            avg_confidence * 0.35
            + validation_score * 0.25
            + vision_score * 0.25
            + engine_score * 0.10
            + annotation_score * 0.05
        ) * 100

        return {
            "estimated_quality_score": round(score, 2),
            "average_llm_confidence": round(avg_confidence, 4),
            "validation_score": round(validation_score * 100, 2),
            "vision_status": vision_status,
            "engine_count": len(engines_used or []),
            "annotation_count": len(annotations or []),
            "note": "This is a confidence estimate. True accuracy requires a matching ground-truth JSON file.",
        }
