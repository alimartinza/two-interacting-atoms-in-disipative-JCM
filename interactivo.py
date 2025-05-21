import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import jcm_lib as jcm

# Define external functions
def f1(w0, g, p, gamma, x, d, k, J, ax):
    ax.clear()
    ax.set_title("f1")
    print(f"f1 called with: {w0=}, {g=}, {p=}, {gamma=}, {x=}, {d=}, {k=}, {J=}")

def f2(w0, g, p, gamma, x, d, k, J, ax):
    ax.clear()
    ax.set_title("f2")
    print(f"f2 called with: {w0=}, {g=}, {p=}, {gamma=}, {x=}, {d=}, {k=}, {J=}")

def f3(w0, g, p, gamma, x, d, k, J, ax):
    ax.clear()
    ax.set_title("f3")
    print(f"f3 called with: {w0=}, {g=}, {p=}, {gamma=}, {x=}, {d=}, {k=}, {J=}")

def f4(w0, g, p, gamma, x, d, k, J, ax):
    ax.clear()
    ax.set_title("f4")
    print(f"f4 called with: {w0=}, {g=}, {p=}, {gamma=}, {x=}, {d=}, {k=}, {J=}")

# GUI class
class InteractivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Plot Window")

        # Parameter names
        self.param_names = ['w0', 'g', 'p', 'gamma', 'x', 'd', 'k', 'J']

        # Dictionary to store the variables
        self.params = {name: tk.DoubleVar(value=0.0) for name in self.param_names}

        # Setup PanedWindow
        self.main_pane = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Left pane
        self.left_pane = ttk.PanedWindow(self.main_pane, orient=tk.VERTICAL)
        self.main_pane.add(self.left_pane, weight=1)

        self.create_param_entries()
        self.create_function_buttons()

        # Right pane
        self.right_frame = ttk.Frame(self.main_pane)
        self.main_pane.add(self.right_frame, weight=3)
        self.create_plot_area()

    def create_param_entries(self):
        param_frame = ttk.Frame(self.left_pane)
        self.left_pane.add(param_frame, weight=1)

        for i, name in enumerate(self.param_names):
            label = ttk.Label(param_frame, text=name)
            label.grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            entry = ttk.Entry(param_frame, textvariable=self.params[name])
            entry.grid(row=i, column=1, padx=5, pady=2)

    def create_function_buttons(self):
        button_frame = ttk.Frame(self.left_pane)
        self.left_pane.add(button_frame, weight=1)

        button_defs = [
            ("f1", jcm.fases),
            ("f2", jcm.concurrence),
            ("f3", f3),
            ("f4", f4)
        ]

        for i, (label, func) in enumerate(button_defs):
            button = ttk.Button(button_frame, text=label, command=lambda func=func: self.call_function(func))
            button.grid(row=i, column=0, padx=5, pady=5)

    def create_plot_area(self):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Save button
        save_button = ttk.Button(self.right_frame, text="Save Plot", command=self.save_plot)
        save_button.pack(side=tk.BOTTOM, pady=5)

    def call_function(self, func):
        # Extract float values from param vars
        values = [self.params[name].get() for name in self.param_names]
        func(*values, self.ax)
        self.canvas.draw()

    def save_plot(self):
        self.fig.savefig("saved_plot.png")
        print("Plot saved as 'saved_plot.png'")

if __name__ == "__main__":
    root = tk.Tk()
    app = InteractivePlotApp(root)
    root.mainloop()
