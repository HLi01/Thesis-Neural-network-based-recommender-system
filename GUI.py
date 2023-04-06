import tkinter as tk

def LoadData():
    return 0


root=tk.Tk()

canvas=tk.Canvas(root, height=700, width=800)
canvas.pack()

frame=tk.Frame(root, bg='#57ebd4')
frame.place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

label=tk.Label(frame, text="Neural network-based movie and TV series recommendation system")
label.pack()
button=tk.Button(frame, text="Load data", command=LoadData)
button.pack()


root.mainloop()