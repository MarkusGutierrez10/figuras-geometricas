import cv2
import imutils
import numpy as np
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame
from PIL import Image, ImageTk

class ShapeDetector:
    def detect(self, c):
        shape = "Indefinido"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 3:
            shape = "Triangulo"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            shape = "Cuadrado" if 0.95 <= ar <= 1.05 else "Rectangulo"
        elif len(approx) > 4:
            shape = "Circulo"
        return shape

    def obtener_lados(self, c):
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        lados = []
        for i in range(len(approx)):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            distancia = np.linalg.norm(p1 - p2)
            lados.append(int(distancia))
        return lados

    def obtener_radio(self, c):
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        return int(radius)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Figuras")
        self.root.configure(bg="#1e1e2f")
        self.root.resizable(False, False)

        self.frame_imagen = Frame(root, bg="#2e2e3e", bd=2, relief="ridge")
        self.frame_imagen.pack(padx=20, pady=20)

        self.label = Label(self.frame_imagen, bg="#2e2e3e")
        self.label.pack(padx=10, pady=10)

        style = {"font": ("Helvetica", 11, "bold"), "bg": "#4e9aff", "fg": "white", "activebackground": "#316ac5"}

        Button(root, text="📂 Cargar Imagen", command=self.cargar_imagen, **style).pack(pady=5)
        Button(root, text="📸 Cámara en Tiempo Real", command=self.usar_camara, **style).pack(pady=5)
        Button(root, text="🎥 Video en Tiempo Real", command=self.figurasVideos, **style).pack(pady=5)
        Button(root, text="❌ Salir", command=root.quit, bg="#ff4e4e", fg="white", activebackground="#c93d3d", font=("Helvetica", 11, "bold")).pack(pady=15)


    def cargar_imagen(self):
        path = filedialog.askopenfilename(filetypes=[("Imagen", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            image = cv2.imread(path)
            self.procesar_imagen(image)

    def usar_camara(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "No se detectó cámara.")
            return
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.procesar_imagen(frame)
        else:
            messagebox.showerror("Error", "No se pudo capturar imagen desde la cámara.")

    def procesar_imagen(self, image):
        sd = ShapeDetector()
        resized = imutils.resize(image, width=300)
        ratio = image.shape[0] / float(resized.shape[0])

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        resumen = {"Triangulo": 0, "Cuadrado": 0, "Rectangulo": 0, "Circulo": 0}

        for c in cnts:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)

            c_scaled = c.astype("float")
            c_scaled *= ratio
            c_scaled = c_scaled.astype("int")

            shape = sd.detect(c_scaled)
            resumen[shape] += 1

            cv2.drawContours(image, [c_scaled], -1, (0, 255, 0), 2)

            if shape == "Circulo":
                radio = sd.obtener_radio(c_scaled)
                texto = f"{shape} (r={radio})"
            else:
                lados = sd.obtener_lados(c_scaled)
                texto = f"{shape} L:{lados}"

            cv2.putText(image, texto, (cX - 50, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Agregar resumen a la imagen (esquina superior izquierda)
        y0 = 20
        for forma, cantidad in resumen.items():
            txt = f"{forma}: {cantidad}"
            cv2.putText(image, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y0 += 25

        self.mostrar_resultado(image)

    def mostrar_resultado(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        img = img.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)

        self.label.configure(image=img_tk)
        self.label.image = img_tk

    def figurasVideos(self):
        print("hola mundo")
        #de aqui
        cap = cv2.VideoCapture(0)  # 0 es la cámara por defecto
        sd = ShapeDetector()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Redimensionar
            resized = imutils.resize(frame, width=400)
            ratio = frame.shape[0] / float(resized.shape[0])

            # Escala de grises + desenfoque
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detección del fondo
            white_pixels = np.sum(blurred > 200)
            black_pixels = np.sum(blurred < 50)
            inverted = False

            if white_pixels > black_pixels:
                thresh = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 3
                )
                inverted = True
            else:
                _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

            text_color = (0, 0, 0) if inverted else (255, 255, 255)

            # Contornos
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

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

                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
                cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

            # Mostrar la imagen
            cv2.imshow("Detección de Figuras", frame)

            # Salir con la tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # ------------------------- Cierre -------------------------
        #cap.release()
        #cv2.destroyAllWindows()
        #hasta aqui

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.geometry("700x700")
    root.mainloop()

