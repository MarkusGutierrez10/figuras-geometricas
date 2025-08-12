import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

# ------------------------- Función para mostrar imágenes -------------------------

def plt_imshow(title, image):
    #Muestra una imagen en una ventana de Matplotlib.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.axis('off')
    plt.show()

# ------------------------- Clase para detectar formas -------------------------

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        #Detecta la forma de un contorno dado.
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 6:
            shape = "hexagon"
        else:
            shape = "circle"

        return shape

# ------------------------- Lectura y preprocesamiento de imagen -------------------------

# Ruta de la imagen
image_path = "img/img2.jpeg"
image = cv2.imread(image_path)

# Redimensionar imagen
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# Escala de grises + desenfoque
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# ------------------------- Detección del fondo -------------------------

# Calcular cantidad de píxeles blancos y negros
white_pixels = np.sum(blurred > 200)
black_pixels = np.sum(blurred < 50)

# Determinar tipo de fondo y aplicar umbral adecuado
inverted = False
if white_pixels > black_pixels:
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )
    inverted = True
else:
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

# Color del texto según el fondo
text_color = (0, 0, 0) if inverted else (255, 255, 255)

# ------------------------- Detección de contornos y formas -------------------------

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

sd = ShapeDetector()

for c in cnts:
    M = cv2.moments(c)
    if M["m00"] == 0:
        continue

    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)

    shape = sd.detect(c)

    c = c.astype("float")
    c *= ratio
    c = c.astype("int")

    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, text_color, 2)

# ------------------------- Mostrar imagen final -------------------------

plt_imshow("Image", image)