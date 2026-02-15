import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../project')))

try:
    from project.utils.mock_data import create_mock_data
    from project.speech_pipeline.train import train_speech_model
    from project.text_pipeline.train import train_text_model
    from project.fusion_pipeline.train import train_fusion_model
except ImportError as e:
    print(f"Import Error: {e}")
    # Try alternate import path if running from root
    from utils.mock_data import create_mock_data
    from speech_pipeline.train import train_speech_model
    from text_pipeline.train import train_text_model
    from fusion_pipeline.train import train_fusion_model

def run_verification():
    mock_data_path = 'project/data_mock'
    
    print("=== 1. Creating Mock Data ===")
    create_mock_data(mock_data_path, num_samples=5)
    
    print("\n=== 2. Testing Speech Pipeline ===")
    try:
        train_speech_model(data_path=mock_data_path, epochs=1)
        print("Speech Training: SUCCESS")
    except Exception as e:
        print(f"Speech Training: FAILED - {e}")
        import traceback
        traceback.print_exc()

    print("\n=== 3. Testing Text Pipeline ===")
    try:
        train_text_model(data_path=mock_data_path, epochs=1)
        print("Text Training: SUCCESS")
    except Exception as e:
        print(f"Text Training: FAILED - {e}")
        import traceback
        traceback.print_exc()

    print("\n=== 4. Testing Fusion Pipeline ===")
    try:
        train_fusion_model(data_path=mock_data_path, epochs=1)
        print("Fusion Training: SUCCESS")
    except Exception as e:
        print(f"Fusion Training: FAILED - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
