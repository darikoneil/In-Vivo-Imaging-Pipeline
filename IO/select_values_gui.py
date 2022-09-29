# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:52:01 2021

@author: sp3660
"""
from tkinter import *
from tkinter.ttk import Combobox


def select_values_gui(values, title):
    c = ''

    def check_cbox(event):
        nonlocal c
        for val in values:
            if cb.get() == val:
                c = cb.get()  # this will assign the variable c the value of cbox
        # if cb.get() == values[1]:
        #     c = cb.get()

    def close_window():
        top.destroy()

    top = Tk()
    top.title(title)
    top.geometry("300x50")  # Add a title
    var = StringVar()
    langs = values
    cb = Combobox(top, values=langs, textvariable=var)
    cb.pack(side=LEFT)
    cb.bind("<<ComboboxSelected>>", check_cbox)
    b = Button(top, text='ok', command=close_window)

    b.pack()
    top.mainloop()
    print('this should print after mainloop is ended')
    return c

