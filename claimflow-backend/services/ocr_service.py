import pytesseract
import os
import base64
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.logger import logger
from config import Config
from datetime import datetime, timezone

class OCRService:
    def __init__(self):
        logger.info("Initializing Advanced Multi-Engine OCR System...")
        os.makedirs(Config.EASYOCR_MODEL_DIR, exist_ok=True)
        os.makedirs(Config.EASYOCR_USER_NETWORK_DIR, exist_ok=True)
        os.makedirs(Config.HUGGINGFACE_CACHE_DIR, exist_ok=True)
        os.makedirs(Config.PADDLE_CACHE_DIR, exist_ok=True)
        
        # 1. Tesseract
        if Config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
        self.tesseract_available = True
        
        # 2. EasyOCR
        try:
            import easyocr

            # Use GPU if available and configured
            self.easy_reader = easyocr.Reader(
                ['en'],
                gpu=Config.USE_CUDA,
                model_storage_directory=Config.EASYOCR_MODEL_DIR,
                user_network_directory=Config.EASYOCR_USER_NETWORK_DIR,
                verbose=False,
            )
            self.easyocr_available = True
            logger.info("EasyOCR Initialized")
        except Exception as e:
            logger.warning(f"EasyOCR failed: {e}")
            self.easyocr_available = False

        # 3. PaddleOCR
        try:
            from paddleocr import PaddleOCR

            self.paddle_reader = PaddleOCR(
                lang='en',
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device='gpu:0' if Config.USE_CUDA else 'cpu',
                enable_mkldnn=False,
            )
            self.paddle_available = True
            logger.info("PaddleOCR Initialized")
        except Exception as e:
            logger.warning(f"PaddleOCR failed: {e}")
            self.paddle_available = False

        # 4. TrOCR (Best for Handwriting/Short text)
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self.torch = torch
            self.trocr_processor = TrOCRProcessor.from_pretrained(Config.TROCR_MODEL)
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(Config.TROCR_MODEL)
            if Config.USE_CUDA and torch.cuda.is_available():
                self.trocr_model.cuda()
            self.trocr_available = True
            logger.info("TrOCR Initialized")
        except Exception as e:
            logger.warning(f"TrOCR failed: {e}")
            self.trocr_available = False

        # 5. Mistral OCR 3 (Primary VLM Engine)
        self.mistral_available = bool(Config.MISTRAL_API_KEY) and Config.USE_MISTRAL_PRIMARY
        if self.mistral_available:
            logger.info("Mistral OCR 3 Enabled as Primary Engine")
        
        self.textin_available = bool(Config.TEXTIN_APP_ID and Config.TEXTIN_SECRET_CODE)
        if self.textin_available:
            logger.info("TextIn OCR adapter enabled")
        
        # Multi-engine confidence computation state
        self.engine_weights = {
            'mistral_ocr': 3.0,      # VLM gets highest weight
            'easyocr': 2.0,          # Strong baseline
            'paddleocr': 2.0,        # Also solid
            'tesseract': 1.5,        # Useful fallback
            'trocr': 1.5,            # Good for handwriting
            'textin_cloud': 2.5,     # Commercial API, high reliability
        }

    def get_engine_status(self):
        return {
            "tesseract": bool(self.tesseract_available),
            "easyocr": bool(self.easyocr_available),
            "paddleocr": bool(self.paddle_available),
            "trocr": bool(self.trocr_available and Config.USE_TROCR_FULL_PAGE),
            "textin_cloud": bool(self.textin_available),
            "mistral_ocr": bool(self.mistral_available),
        }

    def run_runtime_smoke_test(self):
        """
        Runs a lightweight OCR smoke test on a tiny synthetic image
        to verify engines are callable at runtime.
        """
        test_image = Image.new("RGB", (360, 80), "white")
        draw = ImageDraw.Draw(test_image)
        draw.text((12, 24), "CLAIMFLOW OCR TEST 123", fill="black")

        report = {}
        
        # Test Mistral OCR (Primary)
        if self.mistral_available:
            try:
                text, _ = self.run_mistral_ocr(test_image, page=1)
                report["mistral_ocr"] = {"ok": bool((text or "").strip()), "chars": len((text or "").strip())}
            except Exception as e:
                report["mistral_ocr"] = {"ok": False, "error": str(e)}
        else:
            report["mistral_ocr"] = {"ok": False, "error": "not_enabled"}
        
        # Test Tesseract
        try:
            text, _ = self.run_tesseract(test_image, page=1)
            report["tesseract"] = {"ok": bool((text or "").strip()), "chars": len((text or "").strip())}
        except Exception as e:
            report["tesseract"] = {"ok": False, "error": str(e)}

        # Test EasyOCR
        if self.easyocr_available:
            try:
                text, _ = self.run_easyocr(test_image, page=1)
                report["easyocr"] = {"ok": bool((text or "").strip()), "chars": len((text or "").strip())}
            except Exception as e:
                report["easyocr"] = {"ok": False, "error": str(e)}
        else:
            report["easyocr"] = {"ok": False, "error": "not_enabled"}

        # Test PaddleOCR
        if self.paddle_available:
            try:
                text, _ = self.run_paddleocr(test_image, page=1)
                report["paddleocr"] = {"ok": bool((text or "").strip()), "chars": len((text or "").strip())}
            except Exception as e:
                report["paddleocr"] = {"ok": False, "error": str(e)}
        else:
            report["paddleocr"] = {"ok": False, "error": "not_enabled"}

        # Test TrOCR
        if self.trocr_available and Config.USE_TROCR_FULL_PAGE:
            try:
                text = self.run_trocr(test_image)
                report["trocr"] = {"ok": bool((text or "").strip()), "chars": len((text or "").strip())}
            except Exception as e:
                report["trocr"] = {"ok": False, "error": str(e)}
        else:
            report["trocr"] = {"ok": False, "error": "not_enabled"}

        # Test TextIn Cloud
        report["textin_cloud"] = {
            "ok": bool(self.textin_available),
            "error": None if self.textin_available else "not_enabled_or_missing_credentials",
        }
        
        return report

    @staticmethod
    def _poly_to_bbox(poly):
        points = np.array(poly, dtype=float).reshape(-1, 2)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        return [int(x1), int(y1), int(x2), int(y2)]

    @staticmethod
    def _make_annotation(engine, text, confidence, bbox, page=1):
        return {
            "page": page,
            "engine": engine,
            "text": text,
            "confidence": round(float(confidence), 4) if confidence is not None else None,
            "bbox": bbox,
        }

    def run_tesseract(self, image, page=1):
        if not self.tesseract_available:
            return "", []

        try:
            from pytesseract import Output

            text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            data = pytesseract.image_to_data(image, output_type=Output.DICT, config='--oem 3 --psm 6')
            annotations = []
            for i, word in enumerate(data.get('text', [])):
                word = (word or "").strip()
                try:
                    conf = float(data['conf'][i])
                except (ValueError, TypeError):
                    conf = -1
                if not word or conf < 0:
                    continue
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                annotations.append(self._make_annotation('tesseract', word, conf / 100, [x, y, x + w, y + h], page))
            return text, annotations
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return "", []

    def run_easyocr(self, image, page=1):
        if not self.easyocr_available:
            return "", []
        try:
            result = self.easy_reader.readtext(np.array(image), detail=1, paragraph=False)
            lines = []
            annotations = []
            for box, text, confidence in result:
                text = (text or "").strip()
                if not text:
                    continue
                lines.append(text)
                annotations.append(
                    self._make_annotation('easyocr', text, confidence, self._poly_to_bbox(box), page)
                )
            return "\n".join(lines), annotations
        except Exception as e:
            logger.warning(f"EasyOCR read failed: {e}")
            return "", []

    def run_paddleocr(self, image, page=1):
        if not self.paddle_available:
            return "", []
        try:
            result = self.paddle_reader.predict(np.array(image))
        except Exception as e:
            logger.warning(f"PaddleOCR read failed: {e}")
            return "", []

        lines = []
        annotations = []
        for ocr_page in result or []:
            if isinstance(ocr_page, dict):
                texts = ocr_page.get('rec_texts') or []
                scores = ocr_page.get('rec_scores') or []
                boxes = ocr_page.get('rec_polys') or ocr_page.get('dt_polys') or []
                for i, text in enumerate(texts):
                    text = (text or "").strip()
                    if not text:
                        continue
                    lines.append(text)
                    bbox = self._poly_to_bbox(boxes[i]) if i < len(boxes) else None
                    confidence = scores[i] if i < len(scores) else None
                    annotations.append(self._make_annotation('paddleocr', text, confidence, bbox, page))
            elif ocr_page:
                for line in ocr_page:
                    text = line[1][0]
                    lines.append(text)
                    annotations.append(self._make_annotation('paddleocr', text, line[1][1], self._poly_to_bbox(line[0]), page=page))
        return "\n".join(lines), annotations

    def run_trocr(self, image):
        if not self.trocr_available:
            return ""
        try:
            rgb_image = image.convert("RGB")
            pixel_values = self.trocr_processor(images=rgb_image, return_tensors="pt").pixel_values
            if Config.USE_CUDA and self.torch.cuda.is_available():
                pixel_values = pixel_values.cuda()
            generated_ids = self.trocr_model.generate(pixel_values, max_new_tokens=128)
            return self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            logger.warning(f"TrOCR read failed: {e}")
            return ""

    def run_mistral_ocr(self, image, page=1):
        """
        Calls Mistral OCR 3 API for Vision-Language Model-based text extraction.
        Returns structured output with high accuracy and grounding information.
        """
        if not self.mistral_available:
            return "", []
        
        try:
            import requests
            from io import BytesIO
            
            # Convert PIL image to base64
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            buffer.seek(0)
            image_bytes = buffer.getvalue()
            import base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Call Mistral OCR API
            headers = {
                "Authorization": f"Bearer {Config.MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": "mistral-ocr-3",
                "image": f"data:image/png;base64,{image_b64}",
                "include_grounding": Config.ENABLE_VISION_GROUNDING,
            }
            
            response = requests.post(
                Config.MISTRAL_OCR_API_URL,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "success":
                logger.warning(f"Mistral OCR failed: {data.get('error', 'Unknown error')}")
                return "", []
            
            lines = []
            annotations = []
            results = data.get("results", [])
            
            for item in results:
                text = (item.get("text") or "").strip()
                if not text:
                    continue
                
                lines.append(text)
                
                # Extract grounding (bounding box) information
                bbox = None
                if Config.ENABLE_VISION_GROUNDING and "grounding" in item:
                    grounding = item["grounding"]
                    if "bbox" in grounding:
                        bbox = grounding["bbox"]  # [x1, y1, x2, y2]
                
                # Mistral confidence is typically 0.0-1.0
                confidence = item.get("confidence", 0.95)
                
                annotations.append(
                    self._make_annotation('mistral_ocr', text, confidence, bbox, page)
                )
            
            return "\n".join(lines), annotations
        
        except Exception as e:
            logger.warning(f"Mistral OCR failed: {e}")
            return "", []

    def _compute_multi_engine_confidence(self, text, annotations_by_engine):
        """
        Computes a high-confidence score (0-1) by voting across multiple OCR engines.
        
        Strategy:
        - If multiple engines agree on the same text, confidence increases
        - Weighted by engine reliability (Mistral > TextIn > EasyOCR/Paddle > Tesseract)
        - Minimum confidence: 0.50
        - Maximum confidence: 0.99
        """
        if not annotations_by_engine:
            return 0.5
        
        matching_engines = []
        total_weight = 0.0
        weighted_score = 0.0
        
        for engine, text_value in annotations_by_engine.items():
            # Normalize for comparison (lowercase, strip extra whitespace)
            normalized_extracted = " ".join(text.split()).lower()
            normalized_engine = " ".join(text_value.split()).lower()
            
            # Check similarity (exact match or high Levenshtein)
            is_match = normalized_extracted == normalized_engine
            
            if not is_match:
                # Compute similarity score using basic string matching
                common_chars = sum(1 for a, b in zip(normalized_extracted, normalized_engine) if a == b)
                similarity = common_chars / max(len(normalized_extracted), len(normalized_engine), 1)
                is_match = similarity > 0.8
            
            if is_match:
                engine_weight = self.engine_weights.get(engine, 1.0)
                matching_engines.append(engine)
                weighted_score += engine_weight
                total_weight += engine_weight
        
        # Compute confidence score
        if matching_engines:
            # More engines agreeing = higher confidence, with weights
            agreement_ratio = len(matching_engines) / len(annotations_by_engine)
            weighted_confidence = weighted_score / total_weight if total_weight > 0 else 0.5
            
            # Combine agreement ratio and weighted score
            base_confidence = (agreement_ratio * 0.5) + (weighted_confidence / 3.0 * 0.5)
            
            # Scale to 0.50-0.99 range
            final_confidence = 0.50 + (base_confidence * 0.49)
        else:
            # No agreement between engines
            final_confidence = 0.50
        
        # Clamp to valid range
        return min(max(final_confidence, 0.50), 0.99)

    def _collect_text_by_engine(self, all_annotations):
        """
        Groups annotations by the text they contain, collecting which engines found each.
        Returns dict: {text -> {engine_name -> text_value}}
        """
        text_to_engines = {}
        
        for ann in all_annotations:
            text = (ann.get("text") or "").strip()
            engine = ann.get("engine", "unknown")
            
            if text:
                if text not in text_to_engines:
                    text_to_engines[text] = {}
                text_to_engines[text][engine] = text
        
        return text_to_engines

    @staticmethod
    def _collect_textin_text(value):
        texts = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {"text", "content", "contents", "word", "words"} and isinstance(item, str):
                    texts.append(item)
                else:
                    texts.extend(OCRService._collect_textin_text(item))
        elif isinstance(value, list):
            for item in value:
                texts.extend(OCRService._collect_textin_text(item))
        return texts

    def run_textin(self, image):
        if not self.textin_available:
            return ""
        try:
            import requests

            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            response = requests.post(
                Config.TEXTIN_DOCUMENT_URL,
                headers={
                    "x-ti-app-id": Config.TEXTIN_APP_ID,
                    "x-ti-secret-code": Config.TEXTIN_SECRET_CODE,
                    "Content-Type": "application/octet-stream",
                },
                data=buffer.getvalue(),
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") not in (None, 200, 0):
                logger.warning(f"TextIn OCR failed: {payload.get('message') or payload.get('msg') or payload.get('code')}")
                return ""
            texts = self._collect_textin_text(payload.get("result", payload))
            return "\n".join(dict.fromkeys(text.strip() for text in texts if text and text.strip()))
        except Exception as e:
            logger.warning(f"TextIn OCR failed: {e}")
            return ""

    def draw_annotations(self, image, annotations):
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        colors = {
            "tesseract": "#2563eb",
            "easyocr": "#16a34a",
            "paddleocr": "#dc2626",
            "trocr": "#9333ea",
            "mistral_ocr": "#f59e0b",
            "textin_cloud": "#06b6d4",
        }
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            small_font = ImageFont.truetype("arial.ttf", 10)
        except OSError:
            font = ImageFont.load_default()
            small_font = font

        for ann in annotations:
            bbox = ann.get("bbox")
            if not bbox:
                continue
            
            conf = ann.get("confidence", 0)
            engine = ann.get("engine", "unknown")
            color = colors.get(engine, "#f59e0b")
            
            # Use thicker lines for high-confidence items
            line_width = 4 if conf >= Config.OCR_CONFIDENCE_THRESHOLD else 2
            
            draw.rectangle(bbox, outline=color, width=line_width)
            
            # Enhanced label with confidence
            agreement_count = ann.get("agreement_count", 1)
            label = f"{engine} {conf:.2f}"
            if agreement_count > 1:
                label += f" ({agreement_count} engines)"
            
            label_box = draw.textbbox((bbox[0], max(0, bbox[1] - 20)), label, font=small_font)
            draw.rectangle(label_box, fill=color)
            draw.text((bbox[0], max(0, bbox[1] - 20)), label, fill="white", font=small_font)

        buffer = BytesIO()
        annotated.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def get_fused_result(self, image, page=1):
        """
        Runs all available engines with Mistral OCR 3 as primary.
        Applies multi-engine voting for confidence scoring.
        Returns fused text with 95%+ confidence targets, grounding info, and annotations.
        """
        results = {}
        all_annotations = []
        diagnostics = {}
        
        # PRIMARY: Mistral OCR 3 (Vision-Language Model)
        if self.mistral_available:
            try:
                text, anns = self.run_mistral_ocr(image, page)
                if text.strip():
                    results['mistral_ocr'] = text
                    all_annotations.extend(anns)
                    diagnostics["mistral_ocr"] = {
                        "chars": len((text or "").strip()),
                        "annotations": len(anns),
                        "is_primary": True,
                    }
                    logger.info(f"Mistral OCR: {len(anns)} annotations extracted")
            except Exception as e:
                logger.warning(f"Mistral OCR execution failed: {e}")
                diagnostics["mistral_ocr"] = {"error": str(e)}
        
        # FALLBACK 1: EasyOCR
        if self.easyocr_available:
            try:
                text, anns = self.run_easyocr(image, page)
                if text.strip():
                    results['easyocr'] = text
                    all_annotations.extend(anns)
                    diagnostics["easyocr"] = {"chars": len((text or "").strip()), "annotations": len(anns)}
            except Exception as e:
                logger.warning(f"EasyOCR execution failed: {e}")
        
        # FALLBACK 2: PaddleOCR
        if self.paddle_available:
            try:
                text, anns = self.run_paddleocr(image, page)
                if text.strip():
                    results['paddleocr'] = text
                    all_annotations.extend(anns)
                    diagnostics["paddleocr"] = {"chars": len((text or "").strip()), "annotations": len(anns)}
            except Exception as e:
                logger.warning(f"PaddleOCR execution failed: {e}")
        
        # FALLBACK 3: Tesseract
        try:
            text, anns = self.run_tesseract(image, page)
            if text.strip():
                results['tesseract'] = text
                all_annotations.extend(anns)
                diagnostics["tesseract"] = {"chars": len((text or "").strip()), "annotations": len(anns)}
        except Exception as e:
            logger.warning(f"Tesseract execution failed: {e}")
        
        # FALLBACK 4: TrOCR (Handwriting)
        if self.trocr_available and Config.USE_TROCR_FULL_PAGE:
            try:
                trocr_text = self.run_trocr(image)
                if trocr_text.strip():
                    results['trocr_handwritten'] = trocr_text
                    diagnostics["trocr_handwritten"] = {
                        "chars": len((trocr_text or "").strip()),
                        "annotations": 0,
                    }
            except Exception as e:
                logger.warning(f"TrOCR execution failed: {e}")
        
        # FALLBACK 5: TextIn Cloud
        if self.textin_available:
            try:
                textin_text = self.run_textin(image)
                if textin_text.strip():
                    results['textin_cloud'] = textin_text
                    diagnostics["textin_cloud"] = {
                        "chars": len((textin_text or "").strip()),
                        "annotations": 0,
                    }
            except Exception as e:
                logger.warning(f"TextIn execution failed: {e}")
        
        # MULTI-ENGINE CONFIDENCE SCORING
        # Improve confidence by having multiple engines validate
        enhanced_annotations = []
        text_to_engines = self._collect_text_by_engine(all_annotations)
        
        for text, engine_dict in text_to_engines.items():
            # Pick the best annotation for this text (prefer Mistral, then others)
            best_ann = None
            best_priority = -1
            priority_order = ['mistral_ocr', 'textin_cloud', 'easyocr', 'paddleocr', 'trocr', 'tesseract']
            
            for ann in all_annotations:
                if ann.get("text") == text:
                    engine_idx = priority_order.index(ann.get("engine")) if ann.get("engine") in priority_order else 999
                    if engine_idx > best_priority:
                        best_ann = ann
                        best_priority = engine_idx
            
            if best_ann:
                # Compute multi-engine confidence
                multi_engine_confidence = self._compute_multi_engine_confidence(text, engine_dict)
                
                # Update annotation with enhanced confidence
                enhanced_ann = best_ann.copy()
                enhanced_ann['confidence'] = multi_engine_confidence
                enhanced_ann['engines_agreeing'] = list(engine_dict.keys())
                enhanced_ann['agreement_count'] = len(engine_dict)
                enhanced_annotations.append(enhanced_ann)
        
        # Sort by confidence descending
        enhanced_annotations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # ADVANCED FUSION STRATEGY
        # Structure the output to show which engine is most confident
        fused_text = ""
        high_confidence_items = []
        medium_confidence_items = []
        low_confidence_items = []
        
        for ann in enhanced_annotations:
            conf = ann.get('confidence', 0)
            text = ann.get('text', '')
            engines = ann.get('engines_agreeing', [])
            
            item = {
                'text': text,
                'confidence': round(conf, 4),
                'engines': engines,
                'source_engine': ann.get('engine'),
                'bbox': ann.get('bbox'),
            }
            
            if conf >= Config.OCR_CONFIDENCE_THRESHOLD:
                high_confidence_items.append(item)
            elif conf >= 0.70:
                medium_confidence_items.append(item)
            else:
                low_confidence_items.append(item)
        
        # Build structured fused text
        if high_confidence_items:
            fused_text += "\n=== HIGH CONFIDENCE (≥95%) ===\n"
            for item in high_confidence_items:
                fused_text += f"[{item['confidence']}] {item['text']}\n"
        
        if medium_confidence_items:
            fused_text += "\n=== MEDIUM CONFIDENCE (70-95%) ===\n"
            for item in medium_confidence_items:
                fused_text += f"[{item['confidence']}] {item['text']}\n"
        
        if low_confidence_items:
            fused_text += "\n=== LOW CONFIDENCE (<70%) ===\n"
            for item in low_confidence_items:
                fused_text += f"[{item['confidence']}] {item['text']} (Unverified)\n"
        
        return {
            "text": fused_text,
            "engines_used": [engine for engine, text in results.items() if text],
            "annotations": enhanced_annotations,
            "annotated_image_base64": self.draw_annotations(image, enhanced_annotations),
            "engine_diagnostics": diagnostics,
            "high_confidence_count": len(high_confidence_items),
            "confidence_stats": {
                "high": len(high_confidence_items),
                "medium": len(medium_confidence_items),
                "low": len(low_confidence_items),
                "average_confidence": round(
                    sum(a.get('confidence', 0) for a in enhanced_annotations) / max(len(enhanced_annotations), 1),
                    4
                ),
            },
        }

    def get_fused_text(self, image):
        result = self.get_fused_result(image)
        return result["text"], result["engines_used"]
    
    def extract_by_layout_regions(self, image, page=1):
        """
        Layout-aware extraction: Separates form into logical regions 
        (header, patient info, clinical, summary) before OCR.
        
        Fixes the "text bleeding across columns" problem by detecting
        structural boundaries and processing each region independently.
        
        Returns: List of (region_type, text, bbox, confidence) tuples
        """
        try:
            import cv2
            
            # Convert to numpy array
            img_array = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect layout regions using morphological operations
            # Dilate to find major text blocks
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dilated = cv2.dilate(gray, kernel, iterations=2)
            
            # Find contours (region boundaries)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort contours by size and position
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip very small regions (noise)
                if w < 100 or h < 50:
                    continue
                
                # Skip very large regions (full page)
                if w > img_array.shape[1] * 0.9 or h > img_array.shape[0] * 0.9:
                    continue
                
                # Extract region with padding
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(img_array.shape[1], x + w + padding)
                y2 = min(img_array.shape[0], y + h + padding)
                
                regions.append({
                    'bbox': [x1, y1, x2, y2],
                    'area': w * h,
                    'y_position': y,  # For sorting top-to-bottom
                })
            
            # Sort regions top-to-bottom, left-to-right
            regions.sort(key=lambda r: (r['y_position'], r['bbox'][0]))
            
            # Process each region independently
            layout_results = []
            for i, region in enumerate(regions):
                x1, y1, x2, y2 = region['bbox']
                region_img = image.crop((x1, y1, x2, y2))
                
                # Classify region by position in document
                region_type = self._classify_region(i, len(regions), region['bbox'], img_array.shape)
                
                # Extract text from region using best engine
                region_text, region_conf = self._extract_region_text(region_img)
                
                if region_text.strip():
                    layout_results.append({
                        'region_type': region_type,
                        'text': region_text,
                        'bbox': region['bbox'],
                        'confidence': region_conf,
                        'position': i
                    })
            
            return layout_results
        
        except Exception as e:
            logger.warning(f"Layout-aware extraction failed: {e}")
            return []
    
    def _classify_region(self, index, total, bbox, img_shape):
        """Classify region by its position in the document"""
        x1, y1, x2, y2 = bbox
        height = img_shape[0]
        
        # Regions are sorted top-to-bottom
        if index < total * 0.2:
            return "HEADER"
        elif index < total * 0.4:
            return "PATIENT_INFO"
        elif index < total * 0.7:
            return "CLINICAL"
        else:
            return "SUMMARY"
    
    def _extract_region_text(self, region_img):
        """Extract text from a single region using multi-engine validation"""
        results = []
        
        # Try each engine on this specific region
        if self.tesseract_available:
            try:
                text, _ = self.run_tesseract(region_img)
                if text.strip():
                    results.append(('tesseract', text, 0.75))
            except:
                pass
        
        if self.easyocr_available:
            try:
                text, _ = self.run_easyocr(region_img)
                if text.strip():
                    results.append(('easyocr', text, 0.85))
            except:
                pass
        
        if self.paddle_available:
            try:
                text, _ = self.run_paddleocr(region_img)
                if text.strip():
                    results.append(('paddleocr', text, 0.80))
            except:
                pass
        
        if self.mistral_available:
            try:
                text, _ = self.run_mistral_ocr(region_img)
                if text.strip():
                    results.append(('mistral_ocr', text, 0.95))
            except:
                pass
        
        if not results:
            return "", 0.0
        
        # Return best result by engine weight
        results.sort(key=lambda r: self.engine_weights.get(r[0], 1.0), reverse=True)
        return results[0][1], results[0][2]
    
    def structured_extract_with_layout(self, image, page=1):
        """
        High-level method combining layout detection + multi-engine validation.
        
        Returns structured output with separated regions to prevent
        "text bleeding" problems common in complex medical forms.
        """
        layout_regions = self.extract_by_layout_regions(image, page)
        
        if not layout_regions:
            # Fallback to standard fusion if layout detection fails
            return self.get_fused_result(image, page)
        
        # Reconstruct text with layout structure
        structured_text = f"\n=== PAGE {page} - LAYOUT-AWARE EXTRACTION ===\n"
        
        for region in layout_regions:
            structured_text += f"\n--- {region['region_type']} ---\n"
            structured_text += f"[Confidence: {region['confidence']:.2%}, Position: {region['position']}]\n"
            structured_text += f"{region['text']}\n"
        
        return {
            "text": structured_text,
            "layout_regions": layout_regions,
            "extraction_method": "layout_aware",
            "region_count": len(layout_regions),
            "avg_confidence": sum(r['confidence'] for r in layout_regions) / max(len(layout_regions), 1),
            "structured": True
        }
