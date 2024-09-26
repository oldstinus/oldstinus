from textgetter.gettxt import img_txt_extract
from textgetter.gettxt import tif_txt_extract
from textgetter.gettxt import pdf_txt_extract

if __name__ == "__main__":
    
    # use img_txt_extract for extracting text from images like jpg,png etc
    img_txt_extract('/home/user/test', '/home/user/output', ['jpeg','png'],ocr_path='C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',
                    verbose=True)
    # use tif_txt_extract for extracting text from tif files
    tif_txt_extract('/home/user/test', '/home/user/output', verbose=True)
    # use pdf_txt_extract for extracting text from pdf files
    pdf_txt_extract('/home/user/test', '/home/user/output', verbose=True)
    
