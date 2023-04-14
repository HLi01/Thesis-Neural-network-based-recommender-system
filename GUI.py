import tkinter as tk
from tkinter import filedialog, Text
import neural_network
import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import requests

nn=neural_network.NeuralNetwork()
root=tk.Tk()
root.configure(bg='#FAD6A5')
#root.attributes('-fullscreen', True)
root.state("zoomed")
root.title("Neurális hálózat alapú film- és sorozatajánló rendszer")
img = tk.PhotoImage(file='./images/video.png')
root.iconphoto(True, img)
frame=tk.Frame(root, bg='#ABC4AA')
frame.place(relx=0.025, rely=0.025, relwidth=0.95, relheight=0.95)
frame.columnconfigure((0,2),weight=4)
frame.columnconfigure(1,weight=1)
frame.rowconfigure((0),weight=4)
frame.rowconfigure((1,2,3,4),weight=1)


def LoadData():
    try:
        filename=filedialog.askopenfilename( title="Select File", filetypes=(("tsv","*.tsv"), ("all files", "*.*")))
        print("File (",filename,") loaded.")
        nn.loadCsv(csv_path=filename)
    except:
        print("File not opened.")
    button_edit_ratings["state"]=tk.ACTIVE

def NewData():
    df=nn.whole

def EditRatings():
    #nn.editRatings(csv_path="asd.tsv")
    df=nn.whole
    for index, row in (df[df['score'].isna()]).iterrows():
        image = ImageTk.PhotoImage(Image.open(requests.get(row['poster'], stream=True).raw))
        label_poster = tk.Label(image=image)
        label_poster.image = image
        label_poster.grid()
        label_title=tk.Label(frame, text=row['primaryTitle']).grid()
        label_year=tk.Label(frame, text=row['startYear']).grid()

def Help():
    pass

def ExitApp():
    exit_answer=tk.messagebox.askquestion(title="Exit?", message="Do you want to exit the application?", icon="warning")
    if exit_answer == 'yes':
        root.destroy()

button_new=tk.Button(frame, text="New dataset", command=NewData, font=('Helvetica 30')).grid(row=1, column=1 ,sticky='nsew', pady=30)
button_load=tk.Button(frame, text="Load data", command=LoadData,font=('Helvetica 30')).grid(row=2, column=1, sticky='nsew', pady=30)
button_edit_ratings=tk.Button(frame, text="Edit ratings", command=EditRatings, state='disabled')
#button_edit_ratings.grid()
button_help=tk.Button(frame, text="Help", command=Help, font=('Helvetica 30')).grid(row=3, column=1 ,sticky='nsew', pady=30)
button_exit=tk.Button(frame, text="Exit application", command=ExitApp,font=('Helvetica 30')).grid(row=4, column=1 ,sticky='nsew', pady=30)


root.protocol('WM_DELETE_WINDOW', ExitApp)
root.mainloop()