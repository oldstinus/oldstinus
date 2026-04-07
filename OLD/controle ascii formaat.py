import tkinter as tk
from tkinter import filedialog, messagebox
import codecs

def is_utf8(filename):
    try:
        with codecs.open(filename, 'r', encoding='utf-8') as f:
            f.read()
        return True
    except UnicodeDecodeError:
        return False

def select_file():
    # Open een bestandselectiedialoog
    file_path = filedialog.askopenfilename(title="Selecteer een bestand")
    if file_path:
        if is_utf8(file_path):
            messagebox.showinfo("Bestandscodering", f"{file_path} is UTF-8 gecodeerd.")
        else:
            messagebox.showinfo("Bestandscodering", f"{file_path} is niet UTF-8 gecodeerd.")
    else:
        messagebox.showwarning("Selectie geannuleerd", "Geen bestand geselecteerd.")

# Hoofdvenster instellen
root = tk.Tk()
root.title("UTF-8 Check")
root.geometry("300x150")

# Label en knop voor het selecteren van een bestand
label = tk.Label(root, text="Selecteer een bestand om te controleren:", padx=10, pady=10)
label.pack()

select_button = tk.Button(root, text="Bestand selecteren", command=select_file)
select_button.pack(pady=20)

# Voer de GUI uit
root.mainloop()
