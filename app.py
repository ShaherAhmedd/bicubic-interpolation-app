import streamlit as st
import cv2
import numpy as np
import math
import time



# BICUBIC FUNCTIONS

def cubic_weight(x):
    a = -0.5
    x = abs(x)

    if x <= 1:
        return (a + 2) * (x ** 3) - (a + 3) * (x ** 2) + 1
    elif x < 2:
        return a * (x ** 3) - 5 * a * (x ** 2) + 8 * a * x - 4 * a
    else:
        return 0


def manual_bicubic_cpu(image, scale_factor):
    old_height, old_width, channels = image.shape

    new_height = int(old_height * scale_factor)
    new_width = int(old_width * scale_factor)

    output = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for y_new in range(new_height):
        for x_new in range(new_width):
            x_old = x_new / scale_factor
            y_old = y_new / scale_factor

            x_base = math.floor(x_old)
            y_base = math.floor(y_old)

            for c in range(channels):
                pixel_value = 0.0

                for m in range(-1, 3):
                    for n in range(-1, 3):
                        y_neighbor = min(max(y_base + m, 0), old_height - 1)
                        x_neighbor = min(max(x_base + n, 0), old_width - 1)

                        weight_y = cubic_weight(y_old - (y_base + m))
                        weight_x = cubic_weight(x_old - (x_base + n))

                        pixel_value += (
                            image[y_neighbor, x_neighbor, c]
                            * weight_x
                            * weight_y
                        )

                output[y_new, x_new, c] = np.clip(pixel_value, 0, 255)

    return output



# GUI APP


class BicubicApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bicubic Interpolation Image Scaling System")
        self.root.geometry("1200x750")

        self.image_path = None
        self.original_image = None

        self.create_widgets()

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="Bicubic Interpolation Image Scaling System",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=10)

        controls_frame = tk.Frame(self.root)
        controls_frame.pack(pady=10)

        self.choose_btn = tk.Button(
            controls_frame,
            text="Choose Image",
            font=("Arial", 12),
            command=self.choose_image
        )
        self.choose_btn.grid(row=0, column=0, padx=10)

        tk.Label(
            controls_frame,
            text="Low Resolution Size:",
            font=("Arial", 12)
        ).grid(row=0, column=1, padx=5)

        self.low_res_entry = tk.Entry(controls_frame, width=8)
        self.low_res_entry.insert(0, "40")
        self.low_res_entry.grid(row=0, column=2, padx=5)

        tk.Label(
            controls_frame,
            text="Scale Factor:",
            font=("Arial", 12)
        ).grid(row=0, column=3, padx=5)

        self.scale_entry = tk.Entry(controls_frame, width=8)
        self.scale_entry.insert(0, "8")
        self.scale_entry.grid(row=0, column=4, padx=5)

        self.run_btn = tk.Button(
            controls_frame,
            text="Run Interpolation",
            font=("Arial", 12, "bold"),
            command=self.run_interpolation
        )
        self.run_btn.grid(row=0, column=5, padx=10)

        self.status_label = tk.Label(
            self.root,
            text="Choose an image to start.",
            font=("Arial", 11),
            fg="blue"
        )
        self.status_label.pack(pady=5)

        self.images_frame = tk.Frame(self.root)
        self.images_frame.pack(pady=10)

        self.labels = {}

        names = [
            "Low Resolution Input",
            "Nearest Neighbor",
            "Bilinear",
            "OpenCV Bicubic",
            "Manual Bicubic"
        ]

        for i, name in enumerate(names):
            frame = tk.Frame(self.images_frame, bd=2, relief="groove")
            frame.grid(row=i // 3, column=i % 3, padx=10, pady=10)

            title = tk.Label(frame, text=name, font=("Arial", 12, "bold"))
            title.pack()

            img_label = tk.Label(frame, width=300, height=220)
            img_label.pack()

            self.labels[name] = img_label

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All Files", "*.*")
            ]
        )

        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)

            if self.original_image is None:
                messagebox.showerror("Error", "Could not read selected image.")
                return

            self.status_label.config(
                text=f"Selected image: {os.path.basename(file_path)}"
            )

    def display_image(self, image, label):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image_rgb)
        pil_image.thumbnail((300, 220))

        tk_image = ImageTk.PhotoImage(pil_image)

        label.config(image=tk_image)
        label.image = tk_image

    def run_interpolation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please choose an image first.")
            return

        try:
            low_res_size = int(self.low_res_entry.get())
            scale_factor = int(self.scale_entry.get())
        except:
            messagebox.showerror("Error", "Low resolution size and scale factor must be numbers.")
            return

        if low_res_size <= 0 or scale_factor <= 0:
            messagebox.showerror("Error", "Values must be greater than zero.")
            return

        self.status_label.config(text="Processing... please wait.")
        self.root.update()

        image = self.original_image

        low_res = cv2.resize(
            image,
            (low_res_size, low_res_size),
            interpolation=cv2.INTER_AREA
        )

        nearest = cv2.resize(
            low_res,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST
        )

        bilinear = cv2.resize(
            low_res,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR
        )

        opencv_bicubic = cv2.resize(
            low_res,
            None,
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_CUBIC
        )

        start_time = time.time()
        manual_bicubic = manual_bicubic_cpu(low_res, scale_factor)
        end_time = time.time()

        output_folder = os.path.join(os.path.dirname(self.image_path), "bicubic_outputs")
        os.makedirs(output_folder, exist_ok=True)

        cv2.imwrite(os.path.join(output_folder, "01_low_resolution_input.png"), low_res)
        cv2.imwrite(os.path.join(output_folder, "02_nearest_neighbor.png"), nearest)
        cv2.imwrite(os.path.join(output_folder, "03_bilinear.png"), bilinear)
        cv2.imwrite(os.path.join(output_folder, "04_opencv_bicubic.png"), opencv_bicubic)
        cv2.imwrite(os.path.join(output_folder, "05_manual_bicubic.png"), manual_bicubic)

        self.display_image(low_res, self.labels["Low Resolution Input"])
        self.display_image(nearest, self.labels["Nearest Neighbor"])
        self.display_image(bilinear, self.labels["Bilinear"])
        self.display_image(opencv_bicubic, self.labels["OpenCV Bicubic"])
        self.display_image(manual_bicubic, self.labels["Manual Bicubic"])

        self.status_label.config(
            text=f"Done. Manual bicubic time: {round(end_time - start_time, 2)} seconds. Outputs saved in bicubic_outputs folder."
        )


# RUN APP

root = tk.Tk()
app = BicubicApp(root)
root.mainloop()