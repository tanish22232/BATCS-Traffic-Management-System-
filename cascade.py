import urllib.request
import os

# Download Haar cascade for license plate detection
haar_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml"
haar_path = "haarcascade_russian_plate_number.xml"

if not os.path.exists(haar_path):
    print("Downloading Haar cascade file...")
    urllib.request.urlretrieve(haar_url, haar_path)
    print("Download completed!")