from sourceafis import CuneiformFingerprintAnalyzer
from pathlib import Path

def main():
    # Create the analyzer
    analyzer = CuneiformFingerprintAnalyzer()
    
    # Path to your tablet image
    # Replace this with the actual path to your image
    tablet_image = r"E:\fingerprints\SM_037321_GMO_r1.00_n4_v256.volume.png"
    
    # Make sure the image exists
    if not Path(tablet_image).exists():
        print(f"Error: Image file {tablet_image} not found!")
        return
    
    try:
        # Analyze the tablet
        print("Analyzing tablet...")
        results = analyzer.analyze_tablet(tablet_image)
        
        # Print results
        print(f"\nFound {len(results)} potential fingerprint regions")
        for i, region in enumerate(results):
            print(f"Region {i+1}: {region['coordinates']}")
        
        # Save to database
        database_file = r"C:\Users\luiss\mesopotamian-_forencis\fingerprint_database.json"
        analyzer.save_database(database_file)
        print(f"\nResults saved to {database_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()