# lip-reading
A Lip Reading System, also known as Visual Speech Recognition (VSR), is an advanced application of Computer Vision and Machine Learning that interprets spoken words by analyzing the movements of a speaker’s lips, without relying on audio input.  

AI-Powered Visual Speech Recognition (Lip Reading)
📌 Overview

This project presents an AI-based Visual Speech Recognition (VSR) system, commonly known as lip reading, which predicts spoken words by analyzing lip movements from video input—without using audio.

The system is designed to assist deaf and hearing-impaired individuals, enable silent communication, and improve speech recognition in noisy environments.

🚀 Features
🎥 Video-to-text prediction using lip movements
🧠 Deep learning-based architecture
🔍 Mouth region detection and preprocessing
🔄 Spatiotemporal modeling (spatial + sequence learning)
📊 Word-level recognition (Completed)
🔜 Sentence-level recognition (Ongoing)
🏗️ System Architecture

The system follows a structured pipeline:

Video Input
Frame Extraction
Mouth Region Localization
Feature Extraction (CNN)
Sequence Modeling (LSTM/GRU)
Prediction & Decoding
⚙️ Technologies Used
Python
OpenCV
TensorFlow / PyTorch
NumPy
Deep Learning Models (CNN + RNN)
📂 Dataset
GRID Audiovisual Sentence Corpus
Structured dataset with labeled video sequences
Contains multiple speakers and fixed-format sentences
🧪 Working Principle
Extract frames from input video
Detect and crop mouth region
Normalize and resize frames
Extract spatial features using CNN
Model temporal dependencies using LSTM/GRU
Predict spoken word or sentence
📈 Results
Achieved ~86% accuracy in word-level recognition
Stable training convergence over multiple epochs
Effective feature learning and temporal modeling
Minor errors due to visually similar words (homophenes)
⚠️ Challenges
Homophene ambiguity (similar lip movements)
Speaker variability
Lighting and video quality issues
Complexity in sentence-level decoding
🔮 Future Work
Implement sentence-level recognition using CTC
Improve model generalization across speakers
Optimize decoding using beam search
Enhance robustness in real-world conditions
🎯 Applications
Assistive technology for hearing-impaired individuals
Silent communication systems
Surveillance and security
Speech recognition in noisy environments
📊 Evaluation Metrics
Accuracy
Character Error Rate (CER)
Word Error Rate (WER)
📜 Conclusion

This project demonstrates the feasibility of visual-only speech recognition using deep learning. The system successfully performs word-level prediction and provides a strong foundation for future sentence-level decoding.

📚 References
GRID Audiovisual Sentence Corpus
Research papers on CNN, LSTM, and CTC
Deep learning and speech recognition literature
