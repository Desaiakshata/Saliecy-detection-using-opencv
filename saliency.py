import cv2
import argparse
a=argparse.ArgumentParser()
a.add_argument("-i","--image", required=True, help="input image path")
args=vars(a.parse_args())

image=cv2.imread(args["image"])

# two methods

# Static spectral saliency
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")

# Fine grained saliency
saliency=cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap)=saliency.computeSaliency(image)

threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Image",saliencyMap)
cv2.imshow("Thresh",threshMap)
cv2.waitKey(0)
