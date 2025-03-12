import numpy as np
import cv2
from pathlib import Path

class CuneiformFingerprintAnalyzer:
    def __init__(self):
        self.fingerprint_db = {}
        self.threshold = 40

    def preprocess_image(self, image_path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(img)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        return sharpened

    def _calculate_lbp(self, region):
        """Fixed Local Binary Pattern calculation"""
        lbp = np.zeros_like(region)
        for i in range(1, region.shape[0] - 1):
            for j in range(1, region.shape[1] - 1):
                center = region[i, j]
                neighborhood = region[i-1:i+2, j-1:j+2].flatten()
                # Remove center pixel from neighborhood
                neighborhood = np.concatenate([neighborhood[:4], neighborhood[5:]])
                binary = neighborhood >= center
                lbp[i, j] = np.sum(binary * (2**np.arange(8)))
        return cv2.calcHist([lbp.astype(np.uint8)], [0], None, [256], [0, 256])

    def _calculate_gradient_coherence(self, region):
        gx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
        orientation = np.arctan2(gy, gx)
        coherence = np.mean(np.cos(2 * (orientation - np.mean(orientation))))
        return abs(coherence)

    def _is_potential_fingerprint(self, region):
        lbp = self._calculate_lbp(region)
        gradient_coherence = self._calculate_gradient_coherence(region)
        return (np.std(lbp) > 20 and gradient_coherence > 0.4)

    def analyze_tablet(self, tablet_image_path, region_size=(200, 200), overlap=100):
        preprocessed = self.preprocess_image(tablet_image_path)
        height, width = preprocessed.shape
        
        detected_regions = []
        
        for y in range(0, height - region_size[0], overlap):
            for x in range(0, width - region_size[1], overlap):
                region = preprocessed[y:y + region_size[0], x:x + region_size[1]]
                
                if self._is_potential_fingerprint(region):
                    # Save the region image
                    region_filename = f"region_{len(detected_regions)+1}.png"
                    cv2.imwrite(region_filename, region)
                    
                    detected_regions.append({
                        'coordinates': (x, y, x + region_size[1], y + region_size[0]),
                        'region_file': region_filename
                    })
        
        return detected_regions

    def visualize_results(self, original_image_path, detected_regions):
        img = cv2.imread(str(original_image_path))
        if img is None:
            raise ValueError(f"Could not read image at {original_image_path}")
            
        for i, region in enumerate(detected_regions):
            x1, y1, x2, y2 = region['coordinates']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Region {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        output_path = "detected_regions.png"
        cv2.imwrite(output_path, img)
        return output_path

def main():
    analyzer = CuneiformFingerprintAnalyzer()
    
    # Update this path to your image path
    tablet_image = r"E:\fingerprints\SM_036475_GMO_r1.00_n4_v256.volume_04_ha_right.png"
    
    try:
        print("Analyzing tablet...")
        results = analyzer.analyze_tablet(tablet_image)
        
        print(f"\nFound {len(results)} potential fingerprint regions")
        print("Saving individual regions as separate files...")
        
        output_image = analyzer.visualize_results(tablet_image, results)
        print(f"\nVisualization saved as: {output_image}")
        
        for i, region in enumerate(results):
            print(f"Region {i+1} saved as: {region['region_file']}")
            print(f"Coordinates: {region['coordinates']}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()