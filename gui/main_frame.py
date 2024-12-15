import customtkinter
import tkinter as tk
from tkinter import filedialog


class PadimMainFrame(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
        self.n_rows = 5
        self.n_columns = 10

        self.app = customtkinter.CTk()  # create CTk window like you do with the Tk window
        self.app.geometry("1200x800")
        # self.app.wm_attributes('-fullscreen', True)
        # self.app.state('normal')
        self._self_create_layout()

    def browse_button(self):
        """
        allow user to get a filedialog and select
        :return:
        """
        filename = filedialog.askdirectory()
        # self.train_path.set(filename)
        print(filename)
        self.lbl1.configure(text=filename)

    def _self_create_layout(self):
        self.select_train_button = customtkinter.CTkButton(
            master=self.app,
            text="train path",
            command=self.browse_button
        )
        self.select_train_button.grid(row=0, column=0)

        # self.train_path = customtkinter.StringVar()
        # self.train_path.set('undefined')
        self.lbl1 = customtkinter.CTkLabel(master=self.app, text='undefined')
        # self.lbl1 = tk.Label(master=self.app, text=self.train_path.get())
        self.lbl1.grid(row=0, column=1)

    def mainloop(self):
        self.app.mainloop()
