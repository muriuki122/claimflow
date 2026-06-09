from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class ExtractedField(BaseModel):
    field_name: str
    value: Optional[str]
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # 'ocr_fusion', 'llm_vision', 'rule'

class ValidationResult(BaseModel):
    is_compliant: bool
    score: float = Field(ge=0.0, le=100.0)
    missing_fields: List[str]
    inconsistencies: List[str]

class AnalysisResponse(BaseModel):
    success: bool
    document_id: str
    timestamp: datetime
    document_type: str
    extracted_fields: Dict[str, ExtractedField]
    validation: ValidationResult
    ocr_engines_used: List[str]
    raw_text_summary: str