import os
import tkinter
import glob
import threading

import customtkinter
import numpy as np
from PIL import Image

from config.DTO import PadimADConfig
from data.dataset import PadimDataset
from data.transform import DataTransform
from PadimAD import PadimAnomalyDetector


class App(customtkinter.CTk):
    app_mode = None
    assets_path = None
    # training frame attributes
    whistle_image = None
    train_im_paths = None
    train_images = None
    n_sample_images = 5
    train_header = None
    train_header_stringvar = None
    n_train_columns = 7
    start_train_button = None
    train_progressbar = None
    train_thread = None
    train_config = None
    ad_detector = None
    # validation attributes within the training frame
    val_path_button = None
    val_path_stringvar = None
    val_path_label = None
    val_im_paths = None
    val_images = None
    val_progressbar = None
    start_val_calibration_button = None
    calibrate_image = None
    # inspection frame attributes
    inspection_frame = None
    inspect_header_stringvar = None
    inspect_header = None
    inspection_path_button = None
    inspection_im_path = None
    inspection_ctk_im = None
    inspection_im_panel = None
    detect_anomalies_button = None
    insp_result_ctk_im = None
    insp_result_im_panel = None

    def __init__(self):
        super().__init__()

        # variables instantiated later on
        self.train_path_stringvar = None
        self.train_path_label = None
        self.train_path_button = None

        # constructor code
        self.title("anomaly_detection_ui.py")
        self.geometry("1400x800")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        self.assets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images")
        self.logo_image = customtkinter.CTkImage(Image.open(os.path.join(self.assets_path, "logo.png")), size=(26, 26))
        self.large_test_image = customtkinter.CTkImage(
            Image.open(os.path.join(self.assets_path, "logo.png")), size=(500, 150))
        self.image_icon_image = customtkinter.CTkImage(
            Image.open(os.path.join(self.assets_path, "logo.png")), size=(20, 20))
        self.file_folder_image = customtkinter.CTkImage(
            Image.open(os.path.join(self.assets_path, "file-and-folder.png")), size=(20, 20))

        self.home_image = customtkinter.CTkImage(
            light_image=Image.open(os.path.join(self.assets_path, "home.png")),
            dark_image=Image.open(os.path.join(self.assets_path, "home.png")),
            size=(20, 20),
        )
        self.training_image = customtkinter.CTkImage(
            light_image=Image.open(os.path.join(self.assets_path, "ai.png")),
            dark_image=Image.open(os.path.join(self.assets_path, "ai.png")),
            size=(20, 20),
        )
        self.inspect = customtkinter.CTkImage(
            light_image=Image.open(os.path.join(self.assets_path, "analysis.png")),
            dark_image=Image.open(os.path.join(self.assets_path, "analysis.png")),
            size=(20, 20),
        )

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(
            self.navigation_frame,
            text="  Image Example",
            image=self.logo_image,
            compound="left",
            font=customtkinter.CTkFont(size=15, weight="bold"),
        )
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(
            self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
            image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.frame_2_button = customtkinter.CTkButton(
            self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Train",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
            image=self.training_image, anchor="w", command=self.frame_2_button_event)
        self.frame_2_button.grid(row=2, column=0, sticky="ew")

        self.frame_3_button = customtkinter.CTkButton(
            self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Inspect",
            fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
            image=self.inspect, anchor="w", command=self.frame_3_button_event)
        self.frame_3_button.grid(row=3, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(
            self.navigation_frame, values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)

        self.home_frame_large_image_label = customtkinter.CTkLabel(
            self.home_frame, text="", image=self.large_test_image)
        self.home_frame_large_image_label.grid(row=0, column=0, padx=20, pady=10)

        self.home_frame_button_1 = customtkinter.CTkButton(self.home_frame, text="", image=self.image_icon_image)
        self.home_frame_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.home_frame_button_2 = customtkinter.CTkButton(
            self.home_frame, text="CTkButton", image=self.image_icon_image, compound="right")
        self.home_frame_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.home_frame_button_3 = customtkinter.CTkButton(
            self.home_frame, text="CTkButton", image=self.image_icon_image, compound="top")
        self.home_frame_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.home_frame_button_4 = customtkinter.CTkButton(
            self.home_frame, text="CTkButton", image=self.image_icon_image, compound="bottom", anchor="w")
        self.home_frame_button_4.grid(row=4, column=0, padx=20, pady=10)

        # create second frame
        self.train_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.train_frame.grid_columnconfigure(0, weight=1)
        self.create_train_frame()

        # create third frame
        self.inspection_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.create_inspection_frame()

        # select default frame
        self.select_frame_by_name("home")
        # TODO debug - set back to home, for implementation reasons
        self.select_frame_by_name("frame_2")

    def create_train_frame(self):
        for i in range(self.n_train_columns):
            self.train_frame.columnconfigure(i, weight=1)

        self.train_header_stringvar = tkinter.StringVar(value="Train Data")
        self.train_header = customtkinter.CTkLabel(
            master=self.train_frame,
            textvariable=self.train_header_stringvar,
            width=250,
            height=50,
            bg_color="dodger blue",
            corner_radius=16
        )
        self.train_header.grid(row=0, column=0, padx=20, pady=10)

        self.train_path_button = customtkinter.CTkButton(
            self.train_frame,
            text="Set Train Path",
            image=self.file_folder_image,
            compound="right",
            command=self.select_train_path,
        )
        self.train_path_button.grid(row=1, column=0, padx=20, pady=10)

        self.train_path_stringvar = tkinter.StringVar(value="Not Set")

        self.train_path_label = customtkinter.CTkLabel(
            master=self.train_frame,
            textvariable=self.train_path_stringvar,
            # width=500,
            # height=25,
            bg_color="dodger blue",
            corner_radius=16
        )
        self.train_path_label.grid(row=1, column=1, padx=20, pady=10)
        # create the button that starts training
        self.whistle_image = customtkinter.CTkImage(
            Image.open(os.path.join(self.assets_path, "start_training.png")), size=(20, 20))

        self.start_train_button = customtkinter.CTkButton(
            self.train_frame,
            text="Train",
            image=self.whistle_image,
            compound="right",
            command=self.start_training,
        )
        self.start_train_button.grid(row=2, column=0, padx=10, pady=5)
        self.train_progressbar = customtkinter.CTkProgressBar(
            master=self.train_frame, width=800, corner_radius=5, height=50)
        self.train_progressbar.set(0)
        self.train_progressbar.grid(row=2, column=1, padx=10, pady=5, columnspan=6)
        self.create_validation_frame()

    def create_validation_frame(self):
        """
        creates the validation frame within the training frame
        :return:
        """
        self.val_path_button = customtkinter.CTkButton(
            self.train_frame,
            text="Set Val Path",
            image=self.file_folder_image,
            compound="right",
            command=self.select_val_path,
        )
        self.val_path_button.grid(row=3, column=0, padx=20, pady=10)
        self.val_path_stringvar = tkinter.StringVar(value="Not Set")

        self.val_path_label = customtkinter.CTkLabel(
            master=self.train_frame,
            textvariable=self.val_path_stringvar,
            # width=500,
            # height=25,
            bg_color="dodger blue",
            corner_radius=16
        )
        self.val_path_label.grid(row=3, column=1, padx=20, pady=10)
        self.calibrate_image = customtkinter.CTkImage(
            Image.open(os.path.join(self.assets_path, "kalibrieren.png")), size=(20, 20))
        self.start_val_calibration_button = customtkinter.CTkButton(
            self.train_frame,
            text="Calibrate",
            image=self.calibrate_image,
            compound="right",
            command=self.calibrate_validation_data,
        )
        self.start_val_calibration_button.grid(row=4, column=0, padx=10, pady=5)
        self.val_progressbar = customtkinter.CTkProgressBar(
            master=self.train_frame, width=800, corner_radius=5, height=50)
        self.val_progressbar.set(0)
        self.val_progressbar.grid(row=4, column=1, padx=10, pady=5, columnspan=6)

    def calibrate_validation_data(self):
        """
        calibrate the anomaly detector on the validation data.
        :return:
        """
        val_dataset = PadimDataset(
            data_path=self.val_path_stringvar.get(),
            transform=DataTransform.get_train_transform()
        )
        self.ad_detector.calibrate_anomalies_on_dataset(dataset=val_dataset)
        print(self.ad_detector.cal_min_score)
        print(self.ad_detector.cal_max_score)

    def select_val_path(self):
        filepath = tkinter.filedialog.askdirectory()
        self.val_path_stringvar.set(filepath)
        self.init_val_data()

    def init_val_data(self):
        """
        initialize the validation data.
        :return:
        """
        im_paths = glob.glob(os.path.join(self.val_path_stringvar.get(), '**/*.png'), recursive=True)
        print('[INFO] found {} images for training'.format(len(im_paths)))
        self.val_im_paths = im_paths
        self.val_images = []
        for i in range(self.n_sample_images):
            tmp_im = customtkinter.CTkImage(
                light_image=Image.open(self.val_im_paths[i]),
                dark_image=Image.open(self.val_im_paths[i]),
                size=(100, 100),
            )
            im_panel = customtkinter.CTkLabel(
                self.train_frame,
                image=tmp_im,
                text='',
            )
            im_panel.grid(row=3, column=i + 2, padx=20, pady=10)
            self.val_images.append(im_panel)

    def create_inspection_frame(self):
        """
        create inspection frame
        :return:
        """
        self.inspect_header_stringvar = tkinter.StringVar(value="Inspection")
        self.inspect_header = customtkinter.CTkLabel(
            master=self.inspection_frame,
            textvariable=self.inspect_header_stringvar,
            width=250,
            height=50,
            bg_color="dodger blue",
            corner_radius=16
        )
        self.inspect_header.grid(row=0, column=0, padx=20, pady=10)
        self.inspection_path_button = customtkinter.CTkButton(
            self.inspection_frame,
            text="Load Image",
            image=self.file_folder_image,
            compound="right",
            command=self.select_inspection_image,
        )
        self.inspection_path_button.grid(row=1, column=0, padx=20, pady=10)
        # inspection button
        self.detect_anomalies_button = customtkinter.CTkButton(
            self.inspection_frame,
            text="Detect Anomalies",
            image=self.inspect,
            compound="right",
            command=self.detect_anomalies,
        )
        self.detect_anomalies_button.grid(row=2, column=0, padx=20, pady=10)

    def detect_anomalies(self):
        """
        detect anomalies of an image selected in the ui
        :return:
        """
        src_im = Image.open(self.inspection_im_path).convert('RGB')
        anom_score = self.ad_detector.detect_anomaly(im=src_im, transform=DataTransform.get_test_transform())
        anomaly_im = Image.fromarray(anom_score.astype(np.uint8))

        self.insp_result_ctk_im = customtkinter.CTkImage(
            light_image=anomaly_im,
            dark_image=anomaly_im,
            size=(256, 256),
        )
        self.insp_result_im_panel = customtkinter.CTkLabel(
            self.inspection_frame,
            image=self.insp_result_ctk_im,
            text='',
        )
        self.insp_result_im_panel.grid(row=2, column=1, padx=20, pady=10)

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "frame_2":
            self.train_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.train_frame.grid_forget()
        if name == "frame_3":
            self.inspection_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.inspection_frame.grid_forget()

    def home_button_event(self):
        self.select_frame_by_name("home")

    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")

    def change_appearance_mode_event(self, new_appearance_mode):
        self.app_mode = new_appearance_mode
        customtkinter.set_appearance_mode(new_appearance_mode)

    def select_train_path(self):
        """
        select the training path and display it in the GUI, also extract images
        :return:
        """
        filepath = tkinter.filedialog.askdirectory()
        self.train_path_stringvar.set(filepath)
        self.init_train_data()

    def select_inspection_image(self):
        """
        select an inspection image and load it into the GUI
        :return:
        """
        # path_anomalous_data = r'C:\data\mvtec\bottle\test'
        # ad_dataset = PadimDataset(data_path=path_anomalous_data, transform=DataTransform.get_test_transform())
        # self.ad_detector.detect_anomalies_on_dataset(dataset=ad_dataset, path=r'C:\data\padim\anom')
        # print('Finished on the anomalous dataset')

        self.inspection_im_path = r'C:\data\mvtec\bottle\test\broken_large\003.png'
        file = tkinter.filedialog.askopenfile()
        self.inspection_im_path = os.path.abspath(file.name)
        print('[INFO] selected inspection image: {}'.format(self.inspection_im_path))
        self.inspection_ctk_im = customtkinter.CTkImage(
            light_image=Image.open(self.inspection_im_path),
            dark_image=Image.open(self.inspection_im_path),
            size=(256, 256),
        )
        self.inspection_im_panel = customtkinter.CTkLabel(
            self.inspection_frame,
            image=self.inspection_ctk_im,
            text='',
        )
        self.inspection_im_panel.grid(row=1, column=1, padx=20, pady=10)

    def init_train_data(self):
        """
        whenever a training path is selected the data is initialized again
        :return:
        """
        im_paths = glob.glob(os.path.join(self.train_path_stringvar.get(), '**/*.png'), recursive=True)
        print('[INFO] found {} images for training'.format(len(im_paths)))
        self.train_im_paths = im_paths
        self.train_images = []
        for i in range(self.n_sample_images):
            tmp_im = customtkinter.CTkImage(
                light_image=Image.open(self.train_im_paths[i]),
                dark_image=Image.open(self.train_im_paths[i]),
                size=(100, 100),
            )
            im_panel = customtkinter.CTkLabel(
                self.train_frame,
                image=tmp_im,
                text='',
            )
            im_panel.grid(row=1, column=i + 2, padx=20, pady=10)
            self.train_images.append(im_panel)

    def run_training_thread(self):
        """

        :return:
        """
        self.train_config = PadimADConfig(
            model_name='wide_resnet50_2',
            device='cuda',
            batch_size=8
        )
        self.ad_detector = PadimAnomalyDetector(config=self.train_config)
        good_dataset = PadimDataset(
            data_path=self.train_path_stringvar.get(),
            transform=DataTransform.get_train_transform()
        )
        self.ad_detector.train_anomaly_detection(dataset=good_dataset, progress_bar=self.train_progressbar)

    def start_training(self):
        self.train_thread = threading.Thread(target=self.run_training_thread)
        self.train_thread.start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
