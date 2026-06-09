"""
Medical Entity Resolution Service

Maps extracted medical codes/terms to valid reference databases:
- ICD-10-CM diagnosis codes
- Medical procedure terminology
- Facility types
- Provider specialties

This prevents garbled OCR from propagating through the system.
Example: "o111" → detected as invalid → suggests "O11.1" (Pre-existing hypertension)
"""

import re
from typing import Dict, List, Tuple, Optional
from utils.logger import logger
from difflib import SequenceMatcher

# Reference databases (in production, load from external sources)
ICD10_REFERENCE = {
    "O11": "Pre-existing hypertension complicating pregnancy, childbirth and puerperium",
    "O11.1": "Pre-existing hypertension with pre-eclampsia",
    "O11.2": "Pre-existing hypertension with eclampsia",
    "O11.3": "Pre-existing hypertension with HELLP",
    "O12": "Gestational (pregnancy-induced) hypertension",
    "O13": "Gestational (pregnancy-induced) hypertension with proteinuria",
    "O14": "Pre-eclampsia",
    "O14.0": "Mild or moderate pre-eclampsia",
    "O14.1": "Severe pre-eclampsia",
    "O14.2": "HELLP syndrome (hemolysis, elevated liver enzymes and low platelet count)",
    "O15": "Eclampsia",
    "O16": "Unspecified maternal hypertension",
    "O21": "Excessive vomiting in pregnancy",
    "O21.0": "Mild hyperemesis gravidarum",
    "O21.1": "Hyperemesis gravidarum with metabolic disturbance",
    "O21.2": "Late vomiting of pregnancy",
    "O22": "Venous complications and hemorrhoids in pregnancy",
    "O23": "Infections of genitourinary tract in pregnancy",
    "O24": "Diabetes mellitus in pregnancy, childbirth, and the puerperium",
    "O25": "Malnutrition in pregnancy",
    "O26": "Maternal care for other conditions predominantly related to pregnancy",
    "Z12": "Encounter for screening for malignant neoplasms",
    "Z13": "Encounter for screening for other diseases and disorders",
}

MEDICAL_PROCEDURES = {
    "cesarean": "Cesarean delivery",
    "vaginal": "Vaginal delivery",
    "episiotomy": "Episiotomy",
    "vacuum": "Vacuum extraction",
    "forceps": "Forceps delivery",
    "induction": "Labor induction",
    "augmentation": "Labor augmentation",
    "amniotomy": "Artificial rupture of membranes",
    "epidural": "Epidural anesthesia",
    "spinal": "Spinal anesthesia",
}

FACILITY_TYPES = {
    "hospital": "Hospital",
    "clinic": "Clinic/Primary Health Center",
    "health_center": "Health Center",
    "dispensary": "Dispensary",
    "nursing_home": "Nursing Home",
    "maternity": "Maternity Hospital",
    "private": "Private Hospital",
    "public": "Public Hospital",
}

PROVIDER_SPECIALTIES = {
    "obstetrician": "Obstetrics & Gynecology",
    "obgyn": "Obstetrics & Gynecology",
    "midwife": "Midwifery",
    "pediatrician": "Pediatrics",
    "general": "General Practice",
    "surgeon": "Surgery",
    "cardiologist": "Cardiology",
    "internist": "Internal Medicine",
}


class MedicalEntityResolver:
    """Resolves extracted medical entities against reference databases"""
    
    def __init__(self):
        self.icd10_map = ICD10_REFERENCE
        self.procedures = MEDICAL_PROCEDURES
        self.facilities = FACILITY_TYPES
        self.specialties = PROVIDER_SPECIALTIES
        self.similarity_threshold = 0.7  # 70% similarity for fuzzy matching
    
    def resolve_icd_code(self, extracted_code: str) -> Dict[str, any]:
        """
        Resolve potentially garbled ICD code to valid code.
        
        Examples:
        - "o111" → "O11.1" (pre-existing hypertension with pre-eclampsia)
        - "2ZIN FANNED" → no match (flag for manual review)
        - "O14" → "O14" (already valid)
        
        Returns: {
            'original': str,
            'resolved': str or None,
            'description': str or None,
            'confidence': float (0-1),
            'requires_review': bool
        }
        """
        if not extracted_code or not isinstance(extracted_code, str):
            return {
                'original': extracted_code,
                'resolved': None,
                'description': None,
                'confidence': 0.0,
                'requires_review': True
            }
        
        # Clean the code
        clean_code = extracted_code.upper().strip()
        
        # Remove common OCR artifacts
        clean_code = re.sub(r'[^A-Z0-9\.]', '', clean_code)
        
        # Exact match
        if clean_code in self.icd10_map:
            return {
                'original': extracted_code,
                'resolved': clean_code,
                'description': self.icd10_map[clean_code],
                'confidence': 1.0,
                'requires_review': False
            }
        
        # Fuzzy matching for OCR errors
        best_match = None
        best_score = 0
        
        for valid_code in self.icd10_map.keys():
            # Calculate similarity
            score = SequenceMatcher(None, clean_code, valid_code).ratio()
            if score > best_score:
                best_score = score
                best_match = valid_code
        
        if best_score >= self.similarity_threshold:
            return {
                'original': extracted_code,
                'resolved': best_match,
                'description': self.icd10_map[best_match],
                'confidence': best_score,
                'requires_review': best_score < 0.95  # High confidence doesn't need review
            }
        
        # No match found
        return {
            'original': extracted_code,
            'resolved': None,
            'description': None,
            'confidence': 0.0,
            'requires_review': True
        }
    
    def resolve_procedure(self, extracted_proc: str) -> Dict[str, any]:
        """Resolve procedure terminology to standard terms"""
        if not extracted_proc:
            return {
                'original': extracted_proc,
                'resolved': None,
                'confidence': 0.0,
                'requires_review': True
            }
        
        clean_proc = extracted_proc.lower().strip()
        
        # Exact match in key
        for key, value in self.procedures.items():
            if key in clean_proc:
                return {
                    'original': extracted_proc,
                    'resolved': value,
                    'confidence': 1.0,
                    'requires_review': False
                }
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for key, value in self.procedures.items():
            score = SequenceMatcher(None, clean_proc, key).ratio()
            if score > best_score:
                best_score = score
                best_match = value
        
        if best_score >= self.similarity_threshold:
            return {
                'original': extracted_proc,
                'resolved': best_match,
                'confidence': best_score,
                'requires_review': best_score < 0.9
            }
        
        return {
            'original': extracted_proc,
            'resolved': None,
            'confidence': 0.0,
            'requires_review': True
        }
    
    def resolve_facility(self, extracted_facility: str) -> Dict[str, any]:
        """Resolve facility type to standard category"""
        if not extracted_facility:
            return {
                'original': extracted_facility,
                'resolved': None,
                'confidence': 0.0,
                'requires_review': True
            }
        
        clean_facility = extracted_facility.lower().strip()
        
        # Keyword match
        for key, value in self.facilities.items():
            if key in clean_facility:
                return {
                    'original': extracted_facility,
                    'resolved': value,
                    'confidence': 1.0,
                    'requires_review': False
                }
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for key, value in self.facilities.items():
            score = SequenceMatcher(None, clean_facility, key).ratio()
            if score > best_score:
                best_score = score
                best_match = value
        
        if best_score >= self.similarity_threshold:
            return {
                'original': extracted_facility,
                'resolved': best_match,
                'confidence': best_score,
                'requires_review': best_score < 0.9
            }
        
        return {
            'original': extracted_facility,
            'resolved': None,
            'confidence': 0.0,
            'requires_review': True
        }
    
    def resolve_specialty(self, extracted_spec: str) -> Dict[str, any]:
        """Resolve provider specialty to standard terms"""
        if not extracted_spec:
            return {
                'original': extracted_spec,
                'resolved': None,
                'confidence': 0.0,
                'requires_review': True
            }
        
        clean_spec = extracted_spec.lower().strip()
        
        # Keyword match
        for key, value in self.specialties.items():
            if key in clean_spec:
                return {
                    'original': extracted_spec,
                    'resolved': value,
                    'confidence': 1.0,
                    'requires_review': False
                }
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for key, value in self.specialties.items():
            score = SequenceMatcher(None, clean_spec, key).ratio()
            if score > best_score:
                best_score = score
                best_match = value
        
        if best_score >= self.similarity_threshold:
            return {
                'original': extracted_spec,
                'resolved': best_match,
                'confidence': best_score,
                'requires_review': best_score < 0.9
            }
        
        return {
            'original': extracted_spec,
            'resolved': None,
            'confidence': 0.0,
            'requires_review': True
        }
    
    def resolve_extracted_fields(self, fields: Dict[str, any]) -> Dict[str, any]:
        """
        Batch resolve multiple extracted fields.
        
        Input: {
            'diagnosis_icd': 'o111',
            'procedure': 'cesarean',
            'facility': 'government hospital',
            'physician_specialty': 'obgyn'
        }
        
        Output: {
            'diagnosis_icd': {...resolved...},
            'procedure': {...resolved...},
            'facility': {...resolved...},
            'physician_specialty': {...resolved...},
            'anomalies_detected': [...],
            'confidence_score': 0.85
        }
        """
        resolved = {}
        anomalies = []
        confidences = []
        
        # Resolve diagnosis codes
        if 'diagnosis_icd' in fields and fields['diagnosis_icd']:
            resolved_icd = self.resolve_icd_code(fields['diagnosis_icd'])
            resolved['diagnosis_icd'] = resolved_icd
            confidences.append(resolved_icd.get('confidence', 0))
            
            if resolved_icd.get('requires_review'):
                anomalies.append({
                    'field': 'diagnosis_icd',
                    'issue': 'Invalid or unrecognized ICD code',
                    'original': fields['diagnosis_icd'],
                    'suggestion': resolved_icd.get('resolved'),
                    'severity': 'HIGH'
                })
        
        # Resolve procedures
        if 'procedure' in fields and fields['procedure']:
            resolved_proc = self.resolve_procedure(fields['procedure'])
            resolved['procedure'] = resolved_proc
            confidences.append(resolved_proc.get('confidence', 0))
            
            if resolved_proc.get('requires_review'):
                anomalies.append({
                    'field': 'procedure',
                    'issue': 'Unrecognized procedure terminology',
                    'original': fields['procedure'],
                    'suggestion': resolved_proc.get('resolved'),
                    'severity': 'MEDIUM'
                })
        
        # Resolve facility
        if 'facility' in fields and fields['facility']:
            resolved_fac = self.resolve_facility(fields['facility'])
            resolved['facility'] = resolved_fac
            confidences.append(resolved_fac.get('confidence', 0))
        
        # Resolve specialty
        if 'physician_specialty' in fields and fields['physician_specialty']:
            resolved_spec = self.resolve_specialty(fields['physician_specialty'])
            resolved['physician_specialty'] = resolved_spec
            confidences.append(resolved_spec.get('confidence', 0))
        
        # Calculate overall confidence
        avg_confidence = sum(confidences) / max(len(confidences), 1) if confidences else 0
        
        return {
            'resolved_fields': resolved,
            'anomalies': anomalies,
            'confidence_score': avg_confidence,
            'requires_manual_review': len(anomalies) > 0
        }
    
    def get_reference_database_summary(self) -> Dict[str, int]:
        """Return summary of reference databases loaded"""
        return {
            'icd10_codes': len(self.icd10_map),
            'procedures': len(self.procedures),
            'facilities': len(self.facilities),
            'specialties': len(self.specialties),
            'total_entities': len(self.icd10_map) + len(self.procedures) + 
                              len(self.facilities) + len(self.specialties)
        }
