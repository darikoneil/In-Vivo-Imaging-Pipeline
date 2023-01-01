# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 15:25:14 2021

@author: sp3660
"""

from tkinter import *


def manually_get_some_metadata():
    plane_number = ''
    frame_volume_number = ''
    resolution = ''

    def retrieve_input():
        nonlocal plane_number
        nonlocal frame_volume_number
        nonlocal resolution
        plane_number = veh_reg_text_box.get()
        frame_volume_number = time_text_box.get()
        resolution = distance_text_box.get()
        root.destroy()

    root = Tk()
    root.title("Give Me metadata")
    root.geometry("450x165")

    veh_reg_label = Label(root, text="'Plane Number':")
    veh_reg_label.pack()

    veh_reg_text_box = Entry(root, bd=1)
    veh_reg_text_box.pack()

    distance_label = Label(root, text="Frame/Volume Number")
    distance_label.pack()

    distance_text_box = Entry(root, bd=1)
    distance_text_box.pack()
    time_label = Label(root, text="Resolution")
    time_label.pack()

    time_text_box = Entry(root, bd=1)
    time_text_box.pack()

    enter_button = Button(root, text="Enter", command=retrieve_input)
    enter_button.pack()

    root.mainloop()
    return [plane_number, frame_volume_number, resolution]