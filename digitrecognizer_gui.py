import numpy as np
import tkinter as tk
from keras.models import load_model
from PIL import ImageGrab, ImageOps, Image
import ctypes

# to solve the issue with high DPI that caused incorrectly captured images...
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # for Windows >= 8.1
except:
    ctypes.windll.user32.SetProcessDPIAware() # for Windows <= 8.0

model = load_model('digitrecognizer_model.h5')

def img_preproc(img):
    img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28,28), Image.ANTIALIAS)
    img = np.asarray(img, 'float32')
    img = img.reshape(1,28,28,1) 
    img /= 255
    img[img < 0.2] = 0
    return img

def digit_recognizer(img):
    pred = model.predict(img)[0]
    return np.argmax(pred), max(pred)

class RecognizerApp(tk.Frame):
    def __init__(self, parent=None):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.pack()
        self.make_widgets()
    def make_widgets(self):
        self.winfo_toplevel().title('Digit Recognizer')
        self.winfo_toplevel().resizable(False, False)
        self.label = tk.Label(self, text='Draw a digit on the canvas!', font=('Helvetica',12))
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white', cursor='dot')
        self.btn_clear = tk.Button(self, text='Clear canvas', command=self.clear_canvas)
        self.btn_recognize = tk.Button(self, text='Recognize', command=self.classify_digit) 
        self.label.pack(padx=30, pady=30)
        self.canvas.pack(padx=30, pady=30)
        self.btn_clear.pack(side='right', padx=30, pady=30)
        self.btn_recognize.pack(side='right')
        self.canvas.bind('<B1-Motion>', self.draw_digit)
    def clear_canvas(self):
        self.canvas.delete('all')
        self.label.configure(text='Draw a digit on the canvas!')
    def draw_digit(self, event):
        self.x, self.y = event.x, event.y
        r = 7
        self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill='black')
    def classify_digit(self):
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()
        img = ImageGrab.grab((x0,y0,x1,y1))
        img = img_preproc(img)
        digit, prob = digit_recognizer(img)
        self.label.configure(text = 'I am '+str(int(round(prob*100)))+'%'+' certain this is '+str(digit)+'.')

root = tk.Tk()
app = RecognizerApp(root)
root.mainloop()