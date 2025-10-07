from scanner import *


test = preprocess_for_ocr(capture_image())

print(extract_text(test))

