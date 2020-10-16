# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from pyqtgraph.Qt import QtGui, QtCore
from numpy import *
import pyqtgraph as pg
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import argparse
import imutils
import time
import dlib
import cv2
import time
import csv

def updateGraph(x, y):
    global curve, ptr, Xm    
    Xm[:-1] = Xm[1:]                      # shift data in the temporal mean 1 sample left
    value = y[len(y) - 1]					          # read line (single value) from the serial port
    Xm[-1] = float(value)                 # vector containing the instantaneous values      
    ptr += 1                              # update x position for displaying the curve
    curve.setData(Xm)                     # set the curve with this data
    curve.setPos(ptr,0)                   # set x position in the graph to 0
    QtGui.QApplication.processEvents()    # you MUST process the plot now

def graphEye(x, y):
	plt.xticks(rotation=45, ha='right')
	plt.subplots_adjust(bottom=0.30)
	plt.title('Time')
	plt.ylabel('Eye Aspect Ratio')
	plt.plot(x, y, linewidth=5, color='r')
	plt.pause(0.01)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
	
def drawPartLine(frame, part):
	PartHull = cv2.convexHull(part)
	cv2.drawContours(frame, [PartHull], -1, (0, 255, 0), 1)


def videoReading():
	# define two constants, one for the eye aspect ratio to indicate
	# blink and then a second constant for the number of consecutive
	# frames the eye must be below the threshold
	EYE_AR_THRESH = 0.23 # 0.23 was the default.
	EYE_AR_CONSEC_FRAMES = 3

	# initialize the frame counters and the total number of blinks
	COUNTER = 0
	TOTAL = 0

	# Count Time
	start_time = time.time()

	# Graph Axis Initializing
	x, y = list(), list()

	# loop over frames from the video stream
	while True:
		# if this is a file video stream, then we need to check if
		# there any more frames left in the buffer to process
		if fileStream and not vs.more():
			break

		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale
		# channels)
		frame = vs.read()

		##frame = imutils.resize(frame, width=450)
		
		## Rotation IF NEEDED
		frame = imutils.rotate(frame, 270, scale=1)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)


		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0
			elapsed = time.time() - start_time
			x.append(elapsed)
			y.append(ear)

			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
	#		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
	#		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < EYE_AR_THRESH:
				COUNTER += 1

			# otherwise, the eye aspect ratio is not below the blink
			# threshold
			else:
				# if the eyes were closed for a sufficient number of
				# then increment the total number of blinks
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					TOTAL += 1
					plt.scatter(elapsed, ear, alpha = 0.3, s = 200)
				# reset the eye frame counter
				COUNTER = 0

			
			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Time: {}".format(elapsed), (10, 250),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


		# cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		updateGraph(x, y)

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
			# do a bit of cleanup

	cv2.destroyAllWindows()
	vs.stop()

def faceReading():
	# Count Time
	start_time = time.time()

	# loop over frames from the video stream
	while True:
		# if this is a file video stream, then we need to check if
		# there any more frames left in the buffer to process
		if fileStream and not vs.more():
			break

		# grab the frame from the threaded video file stream, resize
		# it, and convert it to grayscale channels)
		frame = vs.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = detector(gray, 0)

		#If find the square, resize the image
		if rects:
			if rects[0].left() < 0:
				xCorrection = rects[0].left() - 0
			elif rects[0].right() > 450:
				xCorrection = rects[0].right() - 450
			else:
				xCorrection = 0
			if rects[0].top() < 0:
				yCorrection = rects[0].top() - 0
			elif rects[0].bottom() > 450:
				yCorrection = rects[0].bottom() - 450
			else:
				yCorrection = 0
			print(rects[0])
			frame = frame[rects[0].top() - yCorrection: rects[0].bottom() - yCorrection,
				rects[0].left() - xCorrection: rects[0].right() - xCorrection]
			frame = cv2.resize(frame, dsize=(200,200), interpolation=cv2.INTER_LINEAR)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rects = detector(gray, 0)

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the face component coordinates, 
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			mouth = shape[mouth_S:mouth_E]
			innerMouth = shape[inmouth_S:inmouth_E]
			rightEyebrow = shape[eyebrowR_S:eyebrowR_E]
			leftEyebrow = shape[eyebrowL_S:eyebrowL_E]
			nose = shape[nose_S:nose_E]
			jaw = shape[jaw_S:jaw_E]

			#print(str(lStart) + " " + str(lEnd))
			# average the eye aspect ratio together for both eyes
			elapsed = time.time() - start_time
			with open('test.csv', "a") as f:
				writer = csv.writer(f)
				writer.writerow(([elapsed] + [item for sublist in shape for item in sublist]))
			# compute the convex hull for the left and right eye, then
			# visualize each of the eyes
			drawPartLine(frame, leftEye)
			drawPartLine(frame, rightEye)
			drawPartLine(frame, mouth)
			drawPartLine(frame, innerMouth)
			drawPartLine(frame, rightEyebrow)
			drawPartLine(frame, leftEyebrow)
			drawPartLine(frame, nose)
			drawPartLine(frame, jaw)
			cv2.rectangle(frame, (rect.left(), rect.top()),
				(rect.right(), rect.bottom()), (0,0,255), 2)
			
			# draw the total number of blinks on the frame along with
			# the computed eye aspect ratio for the frame
			cv2.putText(frame, "Time: {}".format(elapsed), (10, 250),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
			# do a bit of cleanup

	cv2.destroyAllWindows()
	vs.stop()

if __name__ == "__main__":

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True,
		help="path to facial landmark predictor")
	ap.add_argument("-v", "--video", type=str, default="",
		help="path to input video file")
	args = vars(ap.parse_args())

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	print("[INFO] loading facial landmark predictor...")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])

	# grab the indexes of the facial landmarks
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	(mouth_S, mouth_E) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
	(inmouth_S, inmouth_E) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
	(eyebrowR_S, eyebrowR_E) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
	(eyebrowL_S, eyebrowL_E) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
	(nose_S, nose_E) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
	(jaw_S, jaw_E) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

	# start the video stream thread
	print("[INFO] starting video stream thread...")
	
	# Purpose for reading the video file
	# vs = FileVideoStream(args["video"]).start()
	# fileStream = True

	# Purpose for streaming the video from the camera
	vs = VideoStream(args["video"]).start()
	# vs = VideoStream(usePiCamera=True).start()
	fileStream = False
	time.sleep(1.0)

	# Execute the function
	faceReading()

	# Print the result
