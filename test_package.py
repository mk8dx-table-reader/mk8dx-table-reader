from mk8dx_table_reader.fullreader import Fullreader
import PIL

# test the Fullreader with a sample image
img = PIL.Image.open("test_img.png")  # Replace with your image path
reader = Fullreader()
names, scores = reader.fullOCR(img)
print(names, scores)