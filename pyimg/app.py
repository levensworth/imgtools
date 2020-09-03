from tkinter import Menu, messagebox, ttk

from pyimg.config.interface_info import InterfaceInfo
from pyimg.menus.contrast_menu import FunctionMenu
from pyimg.menus.io_menu import ImageMenu
from pyimg.menus.point_operators import PointOperatorMenu


class App:
    def __init__(self):
        interface = InterfaceInfo.get_instance()
        root = interface.get_root()
        interface.configure()
        interface.load_frames()
        self.load_footer_buttons(interface)
        self.load_menu(root)

    def load_footer_buttons(self, interface):
        exit_program_btn = ttk.Button(
            interface.footer_frame,
            text="Exit Program",
            command=lambda: self.ask_quit(root),
        )
        exit_program_btn.grid(column=0, row=0)
        clean_window_btn = ttk.Button(
            interface.footer_frame,
            text="Clean buttons",
            command=lambda: interface.delete_widgets(interface.buttons_frame),
        )
        clean_window_btn.grid(column=1, row=0)
        clean_window_btn = ttk.Button(
            interface.footer_frame,
            text="Clean image",
            command=lambda: interface.delete_widgets(interface.image_frame),
        )
        clean_window_btn.grid(column=2, row=0)

    def ask_quit(self, root):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            root.destroy()

    def load_menu(self, root):
        menubar = Menu(root)
        root.config(menu=menubar)
        ImageMenu(menubar=menubar)
        PointOperatorMenu(menubar=menubar)
        FunctionMenu(menubar=menubar)


app = App()
root = InterfaceInfo.get_instance().get_root()
# main loop
root.mainloop()
