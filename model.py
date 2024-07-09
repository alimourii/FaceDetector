import os
import numpy as np
import cv2 as cv
import joblib
import dlib
from sklearn.svm import SVC

class Model:
    def __init__(self, shape_predictor_path='pretrainedModels/shape_predictor_68_face_landmarks.dat', face_rec_model_path='pretrainedModels/dlib_face_recognition_resnet_model_v1.dat'):
        self.model = SVC(kernel='linear', probability=True)
        self.classNames = []
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)

    def _detect_faces(self, img):
        dets = self.detector(img, 1)
        return dets

    def _predict_landmarks(self, img, face):
        shape = self.predictor(img, face)
        return shape

    def _extract_face_features(self, img, shape):
        if shape is None:
            return None
        face_descriptor = self.facerec.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)

    def train_model(self, photos_dir):
        features = []
        labels = []
        for class_label, class_name in enumerate(os.listdir(photos_dir), start=1):
            class_dir = os.path.join(photos_dir, class_name)
            self.classNames.append(class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        img_path = os.path.join(class_dir, filename)
                        img = cv.imread(img_path)
                        if img is not None:
                            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                            faces = self._detect_faces(img_gray)
                            if faces:
                                for face in faces:
                                    shape = self._predict_landmarks(img_gray, face)
                                    face_features = self._extract_face_features(img, shape)
                                    if face_features is not None:
                                        features.append(face_features)
                                        labels.append(class_label)
        
        features = np.array(features)
        labels = np.array(labels)
        self.model.fit(features, labels)
        print("Model successfully trained!")

    def predict(self, frame):
        img_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self._detect_faces(img_gray)
        if not faces:
            return "No face detected"
        
        results = []
        probabilities = []
        for face in faces:
            shape = self._predict_landmarks(img_gray, face)
            face_features = self._extract_face_features(frame, shape)
            if face_features is not None:
                # Predict probabilities for each class
                probabilities_per_class = self.model.predict_proba([face_features])[0]
                prediction = self.model.predict([face_features])[0]
                index = int(prediction - 1)
                results.append(self.classNames[index])
                probabilities.append(probabilities_per_class[index])
        
        return results, probabilities

    def save_model(self, model_path='./SavedModel/trained_model.pkl', class_names_path='./SavedModel/class_names.txt'):
        joblib.dump(self.model, model_path)
        with open(class_names_path, 'w') as f:
            for class_name in self.classNames:
                f.write(f"{class_name}\n")
        print("Model and class names saved successfully at ./SavedModel")

    def load_model(self, model_path='./SavedModel/trained_model.pkl', class_names_path='./SavedModel/class_names.txt'):
        self.model = joblib.load(model_path)
        with open(class_names_path, 'r') as f:
            self.classNames = [line.strip() for line in f]
        print("Model and class names loaded successfully!")
