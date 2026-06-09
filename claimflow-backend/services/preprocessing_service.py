import cv2
import numpy as np
from PIL import Image

class Preprocessor:
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """
        Advanced pipeline: Grayscale -> Denoise -> Contrast -> Threshold -> Deskew
        """
        try:
            img = np.array(image)
            
            # 1. Grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # 2. Denoise
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

            # 3. Contrast (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)

            # 4. Thresholding (Adaptive)
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )

            # 5. Deskew
            coords = np.column_stack(np.where(thresh > 0))
            if coords.size > 0:
                angle = cv2.minAreaRect(coords)[-1]
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle
                
                if abs(angle) > 0.5:
                    (h, w) = thresh.shape[:2]
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return Image.fromarray(rotated)
            
            return Image.fromarray(thresh)
        except Exception as e:
            print(f"Preprocessing Warning: {e}")
            return image
