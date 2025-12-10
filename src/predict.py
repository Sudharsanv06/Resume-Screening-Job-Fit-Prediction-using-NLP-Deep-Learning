"""
predict.py
==========
Prediction module for Resume Screening & Job Fit Prediction

Functions:
    - load_model(): Load trained Keras model
    - load_preprocessors(): Load tokenizer and label encoder
    - predict_resume(): Predict job role from resume text
    - predict_with_details(): Get detailed predictions with all class probabilities
"""

import numpy as np
import pickle
from tensorflow import keras
from preprocess import clean_text
import warnings
warnings.filterwarnings('ignore')


def load_model(model_path='../models/resume_classifier.keras'):
    """
    Load the trained Keras model.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        keras.Model: Loaded Keras model
    """
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def load_preprocessors(tokenizer_path='../models/tokenizer.pkl', 
                      label_encoder_path='../models/label_encoder.pkl'):
    """
    Load tokenizer and label encoder.
    
    Args:
        tokenizer_path (str): Path to tokenizer pickle file
        label_encoder_path (str): Path to label encoder pickle file
        
    Returns:
        tuple: (tokenizer, label_encoder)
    """
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        print(f"✅ Preprocessors loaded successfully!")
        print(f"   - Vocabulary size: {len(tokenizer.word_index) + 1}")
        print(f"   - Number of classes: {len(label_encoder.classes_)}")
        
        return tokenizer, label_encoder
    
    except Exception as e:
        print(f"❌ Error loading preprocessors: {e}")
        raise


def predict_resume(resume_text, model, tokenizer, label_encoder, max_length=100):
    """
    Predict job role from resume text.
    
    Args:
        resume_text (str): Raw resume text
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        label_encoder: Fitted label encoder
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Prediction results containing:
            - predicted_role (str): Predicted job category
            - confidence (float): Confidence score (0-100)
            - top_3_predictions (list): Top 3 predictions with scores
    """
    try:
        # Step 1: Clean the resume text
        cleaned_text = clean_text(resume_text, remove_stopwords=True)
        
        # Step 2: Convert to sequence
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        
        # Step 3: Pad sequence
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        padded_sequence = pad_sequences(sequence, maxlen=max_length, 
                                       padding='post', truncating='post')
        
        # Step 4: Predict
        predictions = model.predict(padded_sequence, verbose=0)[0]
        
        # Step 5: Get top prediction
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class] * 100
        predicted_role = label_encoder.classes_[predicted_class]
        
        # Step 6: Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [
            {
                'role': label_encoder.classes_[idx],
                'confidence': float(predictions[idx] * 100)
            }
            for idx in top_3_indices
        ]
        
        return {
            'predicted_role': predicted_role,
            'confidence': float(confidence),
            'top_3_predictions': top_3_predictions
        }
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise


def predict_with_details(resume_text, model, tokenizer, label_encoder, max_length=100):
    """
    Get detailed predictions with all class probabilities.
    
    Args:
        resume_text (str): Raw resume text
        model: Trained Keras model
        tokenizer: Fitted tokenizer
        label_encoder: Fitted label encoder
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Detailed prediction results
    """
    try:
        # Get basic prediction
        result = predict_resume(resume_text, model, tokenizer, label_encoder, max_length)
        
        # Clean text for word count
        cleaned_text = clean_text(resume_text, remove_stopwords=False)
        word_count = len(cleaned_text.split())
        
        # Add additional details
        result['resume_word_count'] = word_count
        result['cleaned_text_preview'] = cleaned_text[:200] + '...' if len(cleaned_text) > 200 else cleaned_text
        
        return result
    
    except Exception as e:
        print(f"❌ Error during detailed prediction: {e}")
        raise


def format_prediction_output(prediction_result):
    """
    Format prediction results for display.
    
    Args:
        prediction_result (dict): Output from predict_resume() or predict_with_details()
        
    Returns:
        str: Formatted output string
    """
    output = "\n" + "="*60 + "\n"
    output += "RESUME CLASSIFICATION RESULT\n"
    output += "="*60 + "\n"
    
    output += f"\n🎯 Predicted Role: {prediction_result['predicted_role']}\n"
    output += f"📊 Confidence: {prediction_result['confidence']:.2f}%\n"
    
    output += f"\n🏆 Top 3 Predictions:\n"
    for i, pred in enumerate(prediction_result['top_3_predictions'], 1):
        output += f"   {i}. {pred['role']:25s} - {pred['confidence']:5.2f}%\n"
    
    if 'resume_word_count' in prediction_result:
        output += f"\n📝 Resume Word Count: {prediction_result['resume_word_count']}\n"
    
    output += "\n" + "="*60 + "\n"
    
    return output


# Example usage
if __name__ == "__main__":
    print("Loading model and preprocessors...")
    
    # Load model and preprocessors
    model = load_model()
    tokenizer, label_encoder = load_preprocessors()
    
    # Sample resume text
    sample_resume = """
    Experienced Data Scientist with 5 years of expertise in machine learning, 
    deep learning, and statistical analysis. Proficient in Python, TensorFlow, 
    scikit-learn, and pandas. Strong background in natural language processing, 
    computer vision, and predictive modeling. Successfully deployed multiple 
    ML models in production environments.
    """
    
    print("\n" + "="*60)
    print("SAMPLE PREDICTION")
    print("="*60)
    print("\nSample Resume:")
    print(sample_resume.strip())
    
    # Make prediction
    print("\n🔍 Analyzing resume...")
    result = predict_with_details(sample_resume, model, tokenizer, label_encoder)
    
    # Display formatted output
    print(format_prediction_output(result))
