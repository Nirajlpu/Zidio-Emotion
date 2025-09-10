"""
Zidio-Emotion Demo Script
Simple command-line demo to test core functionality without the full Streamlit interface.
"""

import os
import sys
import pandas as pd
import joblib
from datetime import datetime

def test_text_emotion():
    """Test text emotion detection functionality."""
    print("🔍 Testing Text Emotion Detection...")
    
    # Check if model exists
    if not os.path.exists("svm_model1.pkl"):
        print("❌ svm_model1.pkl not found. Please ensure the model file is in the directory.")
        return False
    
    try:
        # Load the model
        model = joblib.load("svm_model1.pkl")
        
        # Test samples
        test_texts = [
            "I am very happy today!",
            "This makes me so angry",
            "I feel scared about the future",
            "Everything is going well",
            "I am sad and lonely"
        ]
        
        print("\n📝 Sample Text Emotion Analysis:")
        print("-" * 50)
        
        for text in test_texts:
            try:
                prediction = model.predict([text])[0]
                probabilities = model.predict_proba([text])[0]
                confidence = max(probabilities) * 100
                
                print(f"Text: '{text}'")
                print(f"Emotion: {prediction.upper()} (Confidence: {confidence:.1f}%)")
                print()
            except Exception as e:
                print(f"Error processing text '{text}': {e}")
        
        print("✅ Text emotion detection working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading text emotion model: {e}")
        return False

def test_data_files():
    """Test if required data files exist and are readable."""
    print("\n📊 Testing Data Files...")
    
    files_to_check = [
        "emotion_dataset_balanced.csv",
        "HR_alert.csv", 
        "stress_history.csv"
    ]
    
    all_good = True
    
    for file_name in files_to_check:
        if os.path.exists(file_name):
            try:
                df = pd.read_csv(file_name)
                print(f"✅ {file_name}: {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"❌ {file_name}: Error reading file - {e}")
                all_good = False
        else:
            print(f"⚠️  {file_name}: File not found")
            all_good = False
    
    return all_good

def test_facial_model():
    """Test if facial emotion model exists."""
    print("\n🎭 Testing Facial Emotion Model...")
    
    if os.path.exists("emotion_detection_model.h5"):
        try:
            # Try to import tensorflow and load model
            import tensorflow as tf
            model = tf.keras.models.load_model("emotion_detection_model.h5")
            print("✅ Facial emotion model loaded successfully!")
            print(f"   Model input shape: {model.input_shape}")
            print(f"   Model output shape: {model.output_shape}")
            return True
        except ImportError:
            print("⚠️  TensorFlow not available, but model file exists")
            return False
        except Exception as e:
            print(f"❌ Error loading facial model: {e}")
            return False
    else:
        print("❌ emotion_detection_model.h5 not found")
        return False

def main():
    """Run the demo."""
    print("🧠 Zidio-Emotion System Demo")
    print("=" * 40)
    
    # Test components
    text_ok = test_text_emotion()
    data_ok = test_data_files()
    facial_ok = test_facial_model()
    
    print("\n" + "=" * 40)
    print("📋 DEMO SUMMARY")
    print("=" * 40)
    
    print(f"Text Emotion Detection: {'✅ Working' if text_ok else '❌ Issues'}")
    print(f"Data Files: {'✅ Working' if data_ok else '❌ Issues'}")
    print(f"Facial Model: {'✅ Working' if facial_ok else '❌ Issues'}")
    
    if text_ok and data_ok:
        print("\n🎉 Core functionality is working!")
        print("You can now run: streamlit run app1.py")
    else:
        print("\n⚠️  Some components have issues. Check the error messages above.")
        print("Make sure all required files are present and dependencies are installed.")
    
    print("\n📝 To install dependencies:")
    print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()