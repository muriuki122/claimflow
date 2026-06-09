"""
Advanced OCR Fusion module.
Provides a thin wrapper around multiple OCR engines.
This stub implements a minimal interface required by the application.
"""

class AdvancedOCRFusion:
    """Simple placeholder for the AdvancedOCRFusion class.

    The real implementation would combine results from multiple OCR engines
    according to the project's fusion strategy. For now we expose a minimal
    API so that the backend can be imported and started.
    """

    def __init__(self, ocr_engines, config):
        self.ocr_engines = ocr_engines
        self.config = config

    def fuse(self, image):
        """Return fused OCR text for *image*.

        This placeholder simply delegates to the existing OCRService's
        ``get_fused_text`` method, which already implements a multi‑engine
        fusion workflow. If that fails, an empty string is returned.
        """
        try:
            from services.ocr_service import OCRService
            ocr_service = OCRService()
            text, _ = ocr_service.get_fused_text(image)
            return text
        except Exception:
            # In case OCRService is unavailable, fall back to an empty result.
            return ""

    def fuse_ocr_results(self, image):
        """Return OCR fusion result dict compatible with app expectations.

        Uses the existing ``fuse`` method for text extraction and provides
        placeholder metadata for the fusion process.
        """
        text = self.fuse(image)
        return {
            "fusion_strategy": "fallback_fuse",
            "confidence": 1.0,
            "engines_used": ["fallback"],
            "text": text,
            "engine_results": {}
        }


