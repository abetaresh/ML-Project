import cv2 as cv
import pathlib
import random

data_dir = pathlib.Path("../data/")
images = data_dir.glob("**/*.jpg")
sample = str(random.choice(list(images)))
sample

img = cv.imread(sample)
img.shape
img = cv.resize(img, (480, 360), interpolation=cv.INTER_AREA)
img.shape
# cv.imshow("Image", img)

cv.waitKey(0)

# %%

capture = cv.VideoCapture("")

while True:
    isTrue, frame = capture.read()
    cv.imshow("Video", frame)

    if cv.waitKey(20) & 0xFF==ord("d"):
        break

capture.release()
cv.destroyAllWindows()

# %%

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

cv.waitKey(0)
