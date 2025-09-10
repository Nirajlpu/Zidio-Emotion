# 🧠 Zidio-Emotion: Multi-Modal Emotion and Stress Detection System

A comprehensive emotion and stress detection application that analyzes human emotions through multiple channels: text analysis, facial expression recognition, and voice tone analysis. Built with machine learning models and an intuitive Streamlit interface.

## 🌟 Features

### 🎭 Multi-Modal Emotion Detection
- **Text Emotion Analysis**: Analyze emotions in written text using trained SVM models
- **Facial Expression Recognition**: Real-time emotion detection from facial expressions using deep learning
- **Voice Tone Analysis**: Emotion recognition from speech patterns using pre-trained transformers

### 📊 Stress Level Analysis
- Comprehensive stress level assessment based on text input
- Personal stress history tracking and visualization
- Stress trend analysis over time

### 🚨 HR Alert System
- Automated alerts for concerning emotional states
- Critical and warning level notifications
- Dashboard for HR professionals to monitor employee wellbeing

### 💻 Interactive Web Interface
- User-friendly Streamlit dashboard
- Real-time webcam integration
- Voice recording and file upload capabilities
- Historical data visualization and charts

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam (for facial emotion detection)
- Microphone (for voice emotion detection)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nirajlpu/Zidio-Emotion.git
   cd Zidio-Emotion
   ```

2. **Install required packages**
   ```bash
   pip install streamlit pandas numpy scikit-learn tensorflow opencv-python pillow
   pip install librosa torch transformers altair joblib audio-recorder-streamlit
   ```

3. **Test the installation (optional)**
   ```bash
   python demo.py
   ```

4. **Run the application**
   ```bash
   streamlit run app1.py
   ```

5. **Open your browser**
   - The application will automatically open at `http://localhost:8501`

### Alternative Setup with Script
You can also use the provided setup script:
```bash
chmod +x setup.sh
./setup.sh
```

This script will:
- Check Python installation
- Optionally create a virtual environment  
- Install all dependencies
- Verify model files exist

### Required Dependencies

The `requirements.txt` file includes:

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
opencv-python>=4.8.0
Pillow>=10.0.0
librosa>=0.10.0
torch>=2.0.0
transformers>=4.30.0
altair>=5.0.0
joblib>=1.3.0
audio-recorder-streamlit>=0.0.8
```

## 🎯 Usage Guide

### 1. Text Emotion Analysis
- Navigate to "Emotion Detection" in the sidebar
- Enter your text in the text area
- Click "Analyze Emotion" to get results
- View emotion probabilities and recommendations

### 2. Facial Expression Detection
- Go to "Live Detection" → "Facial Expression" tab
- Click "Start Webcam" to activate your camera
- Capture an image for analysis
- View detected emotions and confidence scores

### 3. Voice Emotion Analysis
- Navigate to "Live Detection" → "Voice Tone" tab
- Choose between recording or uploading audio
- Record your voice or upload a WAV file
- Analyze voice patterns for emotional content

### 4. Stress Analysis
- Use "Stress Analysis" section
- Enter your current thoughts or feelings
- Get stress level assessment (No Stress/Mild/Moderate/Severe)
- View personalized recommendations

### 5. Monitor Your History
- Check "My Stress History" for personal analytics
- View stress trends over time
- Track emotional patterns and progress

## 🏗️ Project Structure

```
Zidio-Emotion/
├── app1.py                           # Main Streamlit application
├── emotion_detection_model.h5        # Trained facial emotion CNN model
├── svm_model1.pkl                   # Trained SVM model for text emotions
├── emotion_dataset_balanced.csv      # Text emotion training dataset
├── fer2013_part_*.csv               # FER2013 facial expression dataset (split)
├── FER2013_Emotion_Detection.ipynb  # Facial emotion model training notebook
├── Text Emotion Detection.ipynb     # Text emotion model training notebook
├── HR_alert.csv                     # HR alert storage
├── stress_history.csv               # User stress history storage
├── demo.py                          # Demo script to test core functionality
├── setup.sh                         # Automated setup script  
├── requirements.txt                  # Python dependencies
├── .gitignore                       # Git ignore file
└── README.md                        # This file
```

## 🔧 Technical Details

### Models Used

1. **Text Emotion Detection**
   - Algorithm: Support Vector Machine (SVM)
   - Features: TF-IDF vectorization
   - Emotions: anger, disgust, fear, happy, sad, neutral, surprise

2. **Facial Emotion Recognition**
   - Architecture: Convolutional Neural Network (CNN)
   - Dataset: FER2013
   - Input: 48x48 grayscale facial images
   - Emotions: angry, disgust, fear, happy, neutral, sad, surprise

3. **Voice Emotion Analysis**
   - Model: Pre-trained Wav2Vec2 transformer
   - Repository: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
   - Emotions: angry, calm, happy, sad

### Data Storage
- CSV files for persistent storage
- Session state management for real-time data
- Automatic data backup and recovery

## 📊 Emotion Categories

The system recognizes these emotional states:

| Emotion | Description | Alert Level |
|---------|-------------|-------------|
| 😊 Happy | Positive, joyful state | None |
| 😔 Sad | Melancholic, down mood | Warning |
| 😠 Angry | Frustrated, irritated | Critical |
| 😨 Fear | Anxious, worried | Critical |
| 🤮 Disgust | Repulsion, distaste | Moderate |
| 😮 Surprise | Unexpected reaction | None |
| 😐 Neutral | Balanced emotional state | None |
| 😌 Calm | Peaceful, relaxed | None |

## 🚨 HR Alert System

### Alert Types
- **Critical**: High anger, severe stress, concerning facial expressions
- **Warning**: Moderate stress levels, sad voice tones
- **Informational**: General emotional state updates

### Alert Features
- Real-time notifications
- Historical alert tracking
- Bulk alert management
- Custom alert messages

## 🎨 User Interface

### Dashboard Sections
1. **Sidebar**: Quick actions, navigation, user ID management
2. **Main Area**: Primary analysis interfaces
3. **Alert Panel**: HR notifications and warnings
4. **History View**: Personal analytics and trends

### Visualization Components
- Real-time emotion probability charts
- Stress level trend graphs
- Historical data visualizations
- Interactive alert dashboard

## 🛠️ Development

### Training New Models

1. **Text Emotion Model**
   ```python
   # Open Text Emotion Detection.ipynb
   # Follow the notebook to retrain with new data
   ```

2. **Facial Emotion Model**
   ```python
   # Open FER2013_Emotion_Detection.ipynb
   # Use the notebook to train on FER2013 or custom dataset
   ```

### Customization
- Modify emotion categories in `emotions_emoji_dict`
- Adjust stress classification rules in `classify_stress()`
- Update recommendation systems in `recommend_task()`
- Customize UI themes in the CSS section

## 📈 Performance Metrics

### Model Accuracy
- Text SVM Model: ~85% accuracy on balanced dataset
- Facial CNN Model: ~70% accuracy on FER2013
- Voice Model: Pre-trained transformer with high accuracy

### System Requirements
- RAM: Minimum 4GB, Recommended 8GB
- CPU: Multi-core processor recommended
- Storage: ~1GB for models and datasets
- Bandwidth: Minimal (local processing)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Niraj** - *Initial work* - [Nirajlpu](https://github.com/Nirajlpu)

## 🙏 Acknowledgments

- FER2013 dataset contributors
- Hugging Face transformers library
- Streamlit community
- Open source emotion recognition research

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   ```

2. **Model files not found**
   - Ensure `emotion_detection_model.h5` and `svm_model1.pkl` are in the project directory
   - If missing, train the models using the provided Jupyter notebooks

3. **Webcam not working**
   - Check camera permissions in your browser
   - Ensure no other applications are using the camera
   - Try refreshing the page

4. **Audio recording issues**
   - Check microphone permissions
   - Ensure audio-recorder-streamlit is properly installed
   - Try using file upload instead

5. **TensorFlow warnings**
   ```bash
   # Set environment variable to suppress warnings
   export TF_CPP_MIN_LOG_LEVEL=2
   ```

6. **Streamlit app won't start**
   ```bash
   # Check if port 8501 is available
   streamlit run app1.py --server.port 8502
   ```

### Test Your Installation
Run the demo script to verify everything is working:
```bash
python demo.py
```

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation and usage guide above
- Review the Jupyter notebooks for model training details

## 🔮 Future Enhancements

- [ ] Multi-language support for text analysis
- [ ] Mobile application development
- [ ] Advanced analytics dashboard
- [ ] Integration with popular HR systems
- [ ] Real-time group emotion monitoring
- [ ] API endpoints for external integration
- [ ] Enhanced privacy and security features

---

**Happy Emotion Detection! 🎭✨**