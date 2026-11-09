import tkinter as tk
import numpy as np
import os
from model import AI
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift


class App(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pack(fill=tk.BOTH, expand=True)

        self.model = AI()
        if os.path.exists("model.h5"):
            self.model.load_model("model.h5")
        else:
            print("No saved model found. Starting with a fresh model.")

        # --- INTERFACE ---
        labels_text = [str(i) for i in range(10)] + ["NOT"]
        self.pct_labels = []

        for i, text in enumerate(labels_text):
            btn_correct = tk.Button(self, text="✓", command=lambda idx=i: self.train_action(idx))
            btn_correct.place(x=5, y=10 + i * 20, width=20, height=20)

            lbl_name = tk.Label(self, text=f"{text}:", font=("Arial", 10, "bold"))
            lbl_name.place(x=35, y=10 + i * 20)

            lbl_pct = tk.Label(self, text=f"{0.0:07.4f}%", font=("Arial", 10))
            lbl_pct.place(x=70, y=10 + i * 20)

            self.pct_labels.append(lbl_pct)

        run_button = tk.Button(self, text="Run", command=self.predict_action)
        run_button.place(x=10, y=240, width=60, height=30)

        clear_button = tk.Button(self, text="Clear", command=self.clear_board)
        clear_button.place(x=80, y=240, width=60, height=30)

        save_button = tk.Button(self, text="Save Model", command=self.save_model_action)
        save_button.place(x=10, y=280, width=120, height=30)

        # --- DRAWING ---
        self.canvas = tk.Canvas(self, width=300, height=300, bg='white')
        self.canvas.place(x=150, y=0, width=300, height=300)

        self.show_instruction()

        self.canvas.bind('<Button-1>', self.on_left_press)
        self.canvas.bind('<ButtonRelease-1>', self.on_left_release)
        self.canvas.bind('<Motion>', self.mouse_move)

        self.left_button_held = False
        self.pixels = np.zeros((300, 300), dtype=np.uint8)

    def show_instruction(self):
        self.canvas.create_text(
            150, 20,
            text="Hold left click to draw",
            fill="gray",
            font=("Arial", 11, "italic"),
            tags="instruction"
        )

    def predict_action(self):
        arr = self.get_downscaled_pixels()
        predictions = self.model.predict(arr)

        for i, val in enumerate(predictions):
            percentage = val * 100
            self.pct_labels[i].config(text=f"{percentage:07.4f}%")

    def train_action(self, correct_index):
        arr = self.get_downscaled_pixels()
        self.model.train(arr, correct_index)
        self.clear_board()
        print(f"Trained with correct answer: {correct_index}")

    def save_model_action(self):
        self.model.save_model("model.h5")

    def clear_board(self):
        self.pixels.fill(0)
        self.canvas.delete("all")
        self.show_instruction()

        for lbl in self.pct_labels:
            lbl.config(text=f"{0.0:07.4f}%")

    def get_downscaled_pixels(self) -> np.ndarray:
        img = Image.fromarray((self.pixels * 255).astype(np.uint8))

        bbox = img.getbbox()
        if not bbox:
            return np.zeros((28, 28), dtype=np.float64)
        img = img.crop(bbox)

        size = max(img.width, img.height)
        pad_w = int(np.subtract(size, img.width)) // 2
        pad_h = int(np.subtract(size, img.height)) // 2
        img = ImageOps.expand(img, border=(pad_w, pad_h, pad_w, pad_h), fill=0)

        img = img.resize((20, 20), Image.Resampling.BILINEAR)
        img = ImageOps.expand(img, border=4, fill=0)

        arr = np.array(img, dtype=np.float64) / 255.0
        cy, cx = center_of_mass(arr)
        arr = shift(arr, shift=(np.subtract(13.5, cy), np.subtract(13.5, cx)), mode='constant',
                    cval=0.0)
        return arr

    def mouse_move(self, event):
        if self.left_button_held:
            if 10 <= event.x < 290 and 10 <= event.y < 290:
                self.canvas.delete("instruction")

                y1 = int(np.subtract(event.y, 10))
                x1 = int(np.subtract(event.x, 10))
                self.pixels[y1:event.y + 10, x1:event.x + 10] = 1

                self.canvas.create_oval(
                    x1, y1, event.x + 10, event.y + 10,
                    fill="black", outline="black"
                )

    def on_left_press(self, event):
        self.left_button_held = True
        self.canvas.delete("instruction")

    def on_left_release(self, event):
        self.left_button_held = False


root = tk.Tk()
root.geometry("460x320")
root.title("Deep Learning Digit Recognizer")
myapp = App(root)
myapp.mainloop()