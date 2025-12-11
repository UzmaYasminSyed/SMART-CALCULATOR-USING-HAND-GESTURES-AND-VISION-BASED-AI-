import cv2
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class ImageSimilarityClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset_features = {}
        self.load_dataset()
    
    def extract_features(self, image):
        """Extract color histogram features from image"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        
        return hist.flatten()
    
    def load_dataset(self):
        """Load and extract features from dataset images"""
        print("Loading dataset...")
        
        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            
            if os.path.isdir(class_path):
                self.dataset_features[class_name] = []
                
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        image = cv2.imread(img_path)
                        
                        if image is not None:
                            features = self.extract_features(image)
                            self.dataset_features[class_name].append(features)
        
        print(f"Dataset loaded with classes: {list(self.dataset_features.keys())}")
    
    def compare_similarity(self, features1, features2):
        """Compare similarity between two feature vectors"""
        # Reshape for cosine similarity
        features1 = features1.reshape(1, -1)
        features2 = features2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(features1, features2)[0][0]
        return similarity
    
    def predict_class(self, query_image):
        """Find the most similar class for the query image"""
        query_features = self.extract_features(query_image)
        best_similarity = -1
        best_class = "Unknown"
        
        for class_name, features_list in self.dataset_features.items():
            for features in features_list:
                similarity = self.compare_similarity(query_features, features)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_class = class_name
        
        return best_class, best_similarity

def capture_and_classify():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize classifier
    dataset_path = "Dataset"  # Change this to your dataset path
    classifier = ImageSimilarityClassifier(dataset_path)
    
    print("Press 'c' to capture image, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('Camera - Press c to capture, q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            # Capture and classify
            predicted_class, similarity = classifier.predict_class(frame)
            
            # Display result on image
            result_text = f"Class: {predicted_class} ({similarity:.2f})"
            result_frame = frame.copy()
            cv2.putText(result_frame, result_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Result', result_frame)
            print(f"Predicted Class: {predicted_class}, Similarity: {similarity:.2f}")
            
            # Wait for key press to continue
            cv2.waitKey(0)
            cv2.destroyWindow('Result')
        
        elif key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_classify()