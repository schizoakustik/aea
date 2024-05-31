import tensorflow as tf
import keras
from time import sleep
from mtcnn import MTCNN
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pandas as pd
import skimage as ski
# from console_progressbar import ProgressBar
import time
from datetime import timedelta
import av
import os
from .report import Report

emotion_labels = list(map(str.lower, ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']))

class Analyzer:
    """ Audience Emotion Analyzer class """
    def __init__(self, detector = 'mtcnn') -> None:
        self.emo_classifier = keras.models.load_model('./modelv1.keras')
        if detector == 'mtcnn':
            self.detector = MTCNN()
        else:
            self.detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        self.results = pd.DataFrame(columns=["frame"] + emotion_labels + ["x", "y", "width", "height"])

    def get_detector(self):# -> MTCNN | CascadeClassifier | Any:
        return self.detector
    
    def set_detector(self, detector) -> None:
        self.detector = detector

    def analyze(self, 
        file = None, 
        outfile = "output.mp4",
        show_video=False, 
        save_video = False, 
        skip = False, 
        save_results = True, 
        confidence = .5,
        callback = False) -> None:
        
        """ Analyze video in file """
        # Disable logging for MTCNN
        keras.utils.disable_interactive_logging()

        # Check for file
        if file is None:
            raise ValueError('Must have a file to analyze.')

        # Start timer for report
        start = time.perf_counter()

        # Load video and get properties
        container = av.open(file)
        stream = container.streams.video[0]

        number_of_frames = int(stream.frames)

        random_frame = np.random.randint(number_of_frames)

        if number_of_frames < 0:
            number_of_frames = False

        framerate = int(stream.average_rate)

        if not skip:
            tick = 1
        else:
            try:
                tick = framerate // skip
            except:
                tick = 10

        # TODO: Implement save_video in pyav
        # if save_video:   
            # pass

        i = 0
        for i, frame in enumerate(container.decode(stream)):
            # Grayscale for cv
            gray = frame.to_ndarray()[:360, :640]
            frame = frame.to_rgb().to_ndarray()[:360, :640]

            # Detect faces
            if type(self.detector) == MTCNN:
                faces = self.detector.detect_faces(frame)
            else:
                faces = self.detector.detectMultiScale(gray)

            # Label data for before an emotion has been detected
            label = ""
            label_position = (0, 0)

            # Classify emotions in detected faces
            for face in faces:
                if type(self.detector) == MTCNN:
                    x, y, width, height = face['box']
                else:
                    x, y, width, height = face
                # if show_video:
                #     cv2.rectangle(frame, (x, y), (x+width, y+height), (64, 139, 0), 2)

                # Preprocess detected face pt. I 
                roi_gray = gray[y:y+height, x:x+width]
                roi_gray = ski.transform.resize(roi_gray, (48, 48))

                if (i % skip == 0) & (np.sum([roi_gray]) != 0):

                    # Preprocess detected face pt. II
                    roi = roi_gray.astype('float') / 255.0
                    roi = keras.utils.img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    roi = tf.convert_to_tensor(roi)

                    # Predict emotion
                    prediction = self.emo_classifier.predict(roi, verbose = 0)[0] > confidence
                    pred_idx = prediction.argmax()

                    # Check for confident prediction, add to results
                    if sum(prediction > 0):
                        frame_results = pd.Series(
                            np.concatenate([[i],  
                                            prediction, 
                                            [x, y, width, height]]), 
                                            index=self.results.columns)
                        self.results = pd.concat([self.results, frame_results.to_frame().T])

                    # if show_video:
                    #     label = emotion_labels[pred_idx]
                    #     label_position = (x,y)
                        # cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, .7, (212, 194, 106), 2)


                # Print frame information
                # cv2.putText(frame, f'{i}/{number_of_frames}', (int(fr_width) - 150, int(fr_height) - 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (212, 194, 106), 2)
            
            if i == random_frame:
                self.random_frame_cap = frame

            # if save_video:
                # out.write(frame)

            # if show_video:
                # cv2.imshow('Audience Emotion Analyser', frame)
            
            # Update streamlit progress bar
            if callback:
                callback(i)
            i += 1

        container.close()
        # if save_video:
        #     out.release


        end = time.perf_counter()    
        outfile = outfile if save_video else 'Video not saved'
        skip = skip if skip else 'No frames skipped'
        self.report = Report(file.name, outfile, self.detector, confidence, number_of_frames, skip, end-start, self.results, self.random_frame_cap)
        self.report.generate_report()
        if save_results:
            self.results.to_csv(f'report/{file.name}_results.csv', index = False)
        keras.utils.enable_interactive_logging()
        return True