import tkinter
import customtkinter
from customtkinter import *


class TrainParametersWidget(CTkFrame):
    """
    padim widget to control certain parameters of the application
    """
    def __init__(self, parent, corner_radius, fg_color):
        super().__init__(parent, corner_radius=corner_radius, fg_color=fg_color)
        # self.base_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.im_size_label_stringvar = tkinter.StringVar(value="Image Size")
        self.im_size_label = customtkinter.CTkLabel(
            master=self,
            textvariable=self.im_size_label_stringvar,
            width=80,
            height=30,
            bg_color="dodger blue",
            corner_radius=16
        )
        self.im_size_label.grid(row=0, column=0, padx=20, pady=10)
        self.im_size_entry = CTkEntry(self)
        self.im_size_entry.insert(END, '224')
        self.im_size_entry.grid(row=0, column=1, padx=20, pady=10)

    @property
    def im_size(self) -> int:
        return int(self.im_size_entry.get())

        # self.im_w_stringvar = tkinter.StringVar(value="Image Width")
        # self.im_w_label = customtkinter.CTkLabel(
        #     master=self,
        #     textvariable=self.im_w_stringvar,
        #     width=80,
        #     height=30,
        #     bg_color="dodger blue",
        #     corner_radius=16,
        # )
        # self.im_w_label.grid(row=1, column=0, padx=20, pady=10)
        # self.im_w_entry = CTkEntry(self)
        # self.im_w_entry.insert(END, '224')
        # self.im_w_entry.grid(row=1, column=1, padx=20, pady=10)

