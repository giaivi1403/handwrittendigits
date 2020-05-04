import argparse
import cv2
from keras.models import load_model
import numpy as np
from LibaryMe.datasets import simpledatasetloader
from LibaryMe.preprocessing import imagetoarraypreprocessor,simplepreprocessor,simpledatasetconvertgray
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model" , required = True)
ap.add_argument("-d" , "--dataset" , required = True)
args = vars(ap.parse_args())

class_labels = ["0","1","2","3","4","5","6","7","8","9","10"]
imagePaths = np.array(list(paths.list_images(args["dataset"])))

sp = simplepreprocessor.SimplePreProcessor(28,28)
ip = imagetoarraypreprocessor.ImageToArrayPreprocessor()
spcvtgray = simpledatasetconvertgray.SimpleDatasetConvertGray()
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors = [sp])

(data,labels) = sdl.load(imagePaths,verbose = 1)
image_data = data.copy()
data = data.reshape(data.shape[0],28,28,1)


model = load_model(args["model"])

preds = model.predict(data,batch_size = 32).argmax(axis = 1)
print(preds)

for (i,image) in enumerate(imagePaths) :
	#cv2.namedWindow("Predicted" ,cv2.WINDOW_NORMAL)
	img = cv2.imread(image)
	img_resize = cv2.resize(img,(450,500))
	cv2.putText(img_resize,"label :{}".format(class_labels[preds[i]]), (10,30) , cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) , 2)
	cv2.imshow("Predicted" ,img_resize)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
