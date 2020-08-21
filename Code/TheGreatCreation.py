## There are multiple ways to create a panda dataframe
## One can create a panda using a dictionary
## One can also import from either an excel file or a csv
## Here are some examples of how to create the same panda dataframe from a variety of different methods
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk





#Create the window and the mainframe
class TheGreatApp(tk.Tk):
    def __init__(self, **kwargs):
        super(TheGreatApp, self).__init__(**kwargs)
        mainframe = ttk.Frame(self)
        mainframe.pack(expand=True, fill=tk.BOTH)
        self.Food = None
        self.creationType = tk.StringVar()
        dict = ttk.Radiobutton(mainframe, text='Dictionary', variable=self.creationType, value='dictionary')
        dict.pack()
        ex = ttk.Radiobutton(mainframe, text='Excel', variable=self.creationType, value='excel')
        ex.pack()
        csv = ttk.Radiobutton(mainframe, text='CSV', variable=self.creationType, value='csv')
        csv.pack()

        foodDF = ttk.Button(mainframe, text='CREATE!', command=self.createFrame)
        foodDF.pack()

    def createFrame(self):
        creationType = self.creationType.get()
        if creationType == 'dictionary':
            Stuffs = {
                'Food': ['Banana', 'Strawberry', 'Wine', 'Zucchini', 'Sweet Potato', 'Potato', 'Brussel Sprouts',
                         'Milk',
                         'Cheese', 'Beer'],
                'Category': ['Fruit', 'Fruit', 'Alcohol', 'Veggie', 'Starch', 'Starch', 'Veggie', 'Dairy', 'Dairy',
                             'Alcohol'],
                'Price': [0.79, 2.49, 13.99, 2.07, 0.81, 0.53, 3.86, None, 4.83, 11.97],
                'Danielle Likes?': [True, None, False, False, True, True, True, False, True, True]}

            Foooooood = pd.DataFrame(Stuffs)

        elif creationType == 'excel':
            fileName = '.\TheGreatCreationExample.xlsx'
            Foooooood = pd.read_excel(fileName, true_values='True', false_values='False')

        elif creationType == 'csv':
            fileName = '.\TheGreatCreationExample.csv'
            Foooooood = pd.read_csv(fileName)
        self.Food = Foooooood

        print(self.Food.head(10))
        self.destroy()


root = TheGreatApp()
root.mainloop()



