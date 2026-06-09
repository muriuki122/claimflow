import re
from datetime import datetime
from models.schemas import ValidationResult

class ValidatorService:
    @staticmethod
    def _to_number(value):
        if value is None or isinstance(value, (int, float)):
            return value
        cleaned = re.sub(r'[^\d.-]', '', str(value))
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    @staticmethod
    def _is_filled(value):
        if value is None:
            return False
        text = str(value).strip().lower()
        if not text:
            return False
        return text not in {"n/a", "na", "none", "null", "-", "--", "unknown"}

    def validate(self, data: dict) -> ValidationResult:
        """
        Applies SHA Compliance Logic Rules.
        """
        missing = []
        inconsistencies = []
        score = 100

        # 1. Check Required Fields
        required = [
            'patient_name',
            'patient_id',
            'date_of_birth',
            'service_date',
            'facility_name',
            'diagnosis',
            'icd_code',
            'total_cost',
        ]
        # Extract flat values from the nested dict returned by LLM
        flat_data = {k: v.get('value') if isinstance(v, dict) else v for k, v in data.items()}

        for field in required:
            if not self._is_filled(flat_data.get(field)):
                missing.append(field)
                score -= 10

        # 2. Date Logic (Chronology)
        dob = flat_data.get('date_of_birth')
        service_date = flat_data.get('service_date')
        
        if dob and service_date:
            try:
                d1 = datetime.strptime(dob, "%Y-%m-%d")
                d2 = datetime.strptime(service_date, "%Y-%m-%d")
                if d1 > d2:
                    inconsistencies.append("Service Date cannot be before Date of Birth.")
                    score -= 20
            except Exception:
                inconsistencies.append("Invalid date format.")

        # 3. ICD-10 Regex
        icd = flat_data.get('icd_code')
        if icd and not re.match(r'^[A-Z]\d{2}(\.\d{1,2})?$', icd, re.IGNORECASE):
            inconsistencies.append(f"ICD Code '{icd}' format is invalid.")
            score -= 10

        # 4. Cost Logic
        cost = self._to_number(flat_data.get('total_cost'))
        if cost is not None and (cost < 0 or cost > 1000000):
             inconsistencies.append("Total cost is outside logical range.")
             score -= 10

        # 5. Patient ID should be alphanumeric and at least 5 chars when present
        patient_id = flat_data.get('patient_id')
        if patient_id:
            pid = str(patient_id).strip()
            if not re.match(r'^[A-Za-z0-9\-\/]{5,}$', pid):
                inconsistencies.append("Patient ID format appears invalid or incomplete.")
                score -= 8

        return ValidationResult(
            is_compliant=score >= 75,
            score=max(0, score),
            missing_fields=missing,
            inconsistencies=inconsistencies
        )
