import tkinter as tk
import cv2
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Underwater Circle Detection")

        # Set the background color
        self.root.configure(bg='#2C3E50')  # Use your preferred color code

        # Variables for file selection
        self.file_path = tk.StringVar()

        # Flags for video processing
        self.is_video = False
        self.playing_video = False
        self.cap = None  # Initialize cap attribute here
        self.video_update_id = None  # Store the update_id for video processing

        # Create a frame to contain the image label and buttons
        self.frame = tk.Frame(root, bg='#2C3E50')  # Use your preferred color code
        self.frame.pack(side=tk.BOTTOM, padx=10, pady=10)

        # Create image label
        self.image_label = tk.Label(self.frame, bg='black')  # Set image label background color to black
        self.image_label.pack()

        # Load default image
        self.load_image()

        # Button for selecting file
        select_file_button = tk.Button(self.frame, text="Select File", command=self.select_file, bg='#3498DB', fg='white')  # Use your preferred color codes
        select_file_button.pack(side=tk.LEFT, padx=10)

        # Button for rewinding
        rewind_button = tk.Button(self.frame, text="Rewind", command=self.rewind, bg='#E74C3C', fg='white')  # Use your preferred color codes
        rewind_button.pack(side=tk.LEFT, padx=10)

        # Button for pausing and resuming video
        pause_resume_button = tk.Button(self.frame, text="Pause/Resume", command=self.pause_resume_video, bg='#2ECC71', fg='white')  # Use your preferred color codes
        pause_resume_button.pack(side=tk.LEFT, padx=10)

    def load_image(self):
        # Release the previous video capture object
        if self.cap is not None:
            self.stop_video()

        if self.file_path.get().endswith((".mp4", ".avi", ".mkv")):
            self.is_video = True
            self.cap = cv2.VideoCapture(self.file_path.get())
            self.playing_video = True
            self.process_video()
        else:
            self.is_video = False
            self.original_image = np.zeros((600, 600, 3), dtype=np.uint8)
            # Convert to PIL ImageTk format
            self.pil_image = Image.fromarray(self.original_image)
            self.photo = ImageTk.PhotoImage(self.pil_image)
            self.image_label.config(image=self.photo)

    def process_video(self):
        
        if self.is_video and self.playing_video:
            ret, frame = self.cap.read()
            if ret:
                self.original_image = cv2.resize(frame, (600, 600))
                self.cv_image = self.original_image.copy()

                # Convert image to grayscale
                gray = cv2.cvtColor(self.cv_image, cv2.COLOR_RGB2GRAY)

                # Apply GaussianBlur to reduce noise
                blur = cv2.GaussianBlur(gray, (7, 7), 0)

                # Apply adaptive thresholding
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

                # Perform Hough Circle Transform
                circles = cv2.HoughCircles(thresh, method=cv2.HOUGH_GRADIENT_ALT, dp=2.5, minDist=50, param1=20, param2=0.1, minRadius=100, maxRadius=700)

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        # Draw the outer circle
                        cv2.circle(self.original_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        # Draw the center of the circle
                        cv2.circle(self.original_image, (i[0], i[1]), 2, (0, 0, 255), 3)

                # Update the displayed image
                self.pil_image = Image.fromarray(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
                self.photo = ImageTk.PhotoImage(self.pil_image)
                self.image_label.config(image=self.photo)

                self.video_update_id = self.root.after(10, self.process_video)  # Schedule the next frame update
                
    def select_file(self):
        if self.is_video:
            self.stop_video()
        file_path = filedialog.askopenfilename(filetypes=[("Files", "*.mp4;*.avi;*.mkv")])
        if file_path:
            self.file_path.set(file_path)
            self.load_image()

    def rewind(self):
        if self.is_video:
            self.stop_video()
            self.cap = cv2.VideoCapture(self.file_path.get())  # Reinitialize the video capture object
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.playing_video = True  # Set the flag to indicate that the video is playing
            self.process_video()  # Restart the video processing loop after rewinding

    def pause_resume_video(self):
        if self.is_video:
            if self.playing_video:
                self.playing_video = False
                if self.video_update_id is not None:
                    self.root.after_cancel(self.video_update_id)  # Cancel any scheduled video updates
                    self.video_update_id = None
            else:
                self.playing_video = True
                self.process_video()  # Restart the video processing loop after resuming

    def stop_video(self):
        self.playing_video = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None  # Set cap to None after releasing
        if self.video_update_id is not None:
            self.root.after_cancel(self.video_update_id)  # Cancel any scheduled video updates
            self.video_update_id = None

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
