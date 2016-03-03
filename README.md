Simple Digit Recognition OCR in OpenCV Python
=============================================

This code is originally based on Abid Rahman K's answer in [Simple Digit Recognition OCR in OpenCV-Python](http://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python/9620295).

Several tiny modification has been made to adapt OpenCV's new API.

Hope it helps for OpenCV newbies.


### Progress
Run `train.py`, label the number surrounded with red rectangle **manually**. The labeled result will seems like 

![Train Result](/data/train_result.png)

Run `test.py`, the program will load the test image and automatically recognize digits using `KNearNeighbour` Algorithm. The results will seem like

Labeled Test Image

![Labeled Test Image](/data/in.png)

Recognized Digits

![Recognized Digits](/data/out.png)






### Development Environment Info
```python
>>> sys.version
'2.7.10 |Anaconda 2.3.0 (64-bit)| (default, May 28 2015, 16:44:52) [MSC v.1500 64 bit (AMD64)]'

>>> numpy.__version__
'1.9.2'

>>> cv2.__version__
'3.0.0'
```
