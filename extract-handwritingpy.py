from tkinter import Tk, Label, Button, filedialog, StringVar, OptionMenu
from PIL import Image
import pytesseract
import os

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\claeysst\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def select_file():
    file_path = filedialog.askopenfilename(
        title='Select Image File',
        filetypes=[('Image Files', '*.jpg;*.jpeg;*.png;*.tiff;*.bmp')]
    )
    if file_path:
        file_label.config(text=file_path)
        global selected_file
        selected_file = file_path

def process_image():
    if selected_file:
        try:
            img = Image.open(selected_file)

            # Perform OCR with the selected language
            selected_lang = language_var.get()
            text = pytesseract.image_to_string(img, lang=selected_lang)

            # Save the text to a file in the same directory
            dir_name = os.path.dirname(selected_file)
            base_name = os.path.splitext(os.path.basename(selected_file))[0]
            output_file = os.path.join(dir_name, f'{base_name}_output.txt')

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)

            result_label.config(text=f'Extracted text saved to: {output_file}')

        except Exception as e:
            result_label.config(text=f'An error occurred: {e}')
    else:
        result_label.config(text='Please select an image file.')

# Create the main window
root = Tk()
root.title('Image OCR Extractor')

selected_file = ''

# File selection label and button
file_label = Label(root, text='No file selected.')
file_label.pack(pady=5)

select_button = Button(root, text='Select Image File', command=select_file)
select_button.pack(pady=5)

# Language selection
language_var = StringVar(root)
language_var.set('eng')  # Default language

languages = {
    'English': 'eng',
    'Dutch': 'nld',
    'French': 'fra',
    'German': 'deu',
    'Spanish': 'spa'
}

language_label = Label(root, text='Select Language:')
language_label.pack(pady=5)

language_menu = OptionMenu(root, language_var, *languages.values())
language_menu.pack(pady=5)

# Process button
process_button = Button(root, text='Extract Text', command=process_image)
process_button.pack(pady=10)

# Result label
result_label = Label(root, text='')
result_label.pack(pady=5)

# Start the GUI event loop
root.mainloop()
