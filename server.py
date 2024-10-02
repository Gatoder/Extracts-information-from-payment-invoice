from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import torch
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
from PIL import Image
import pytesseract
import io
import os
import re
import pathlib
import sqlite3
temp = pathlib.PosixPath
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
from ultralytics import YOLO
pathlib.PosixPath = pathlib.WindowsPath
import logging

logging.basicConfig(level=logging.DEBUG)

#Load model YOLO
model = YOLO("E:\project\yolo8.pt")  # Adjust the path to your YOLOv8 model


app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\VietOCR\VietOCR.exe'
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
PRE_FOLDER = 'pre'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)




# Xử lý hình ảnh
def process_image(image_path, filename):
    
    #kết nối db
    conn = sqlite3.connect('./database/detections.db')
    c = conn.cursor()

    

    # Đọc ảnh
    img = cv2.imread(image_path)
    
    # Chuyển đổi sang định dạng RGB cho model YOLO
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb)
    
    # Lấy bounding boxes từ kết quả YOLO
    
    for r in results:
        annotator = Annotator(img_rgb)
        bboxes = r.boxes
        logging.debug(bboxes)
        extracted_text = ""
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox.xyxy[0]
            b = bbox.xyxy[0]
            cls = bbox.cls
            conf = bbox.conf 
            xmin, ymin, xmax, ymax, cls = int(xmin), int(ymin), int(xmax), int(ymax), int (cls)
            
            annotator.box_label(b, model.names[cls])

            cropped_img = img_rgb[ymin:ymax, xmin:xmax]
            
            
            #Xử lý text
            if cls == 0:
                text = pytesseract.image_to_string(cropped_img, lang='vie')
                clean_text = cleanning_text(text, cls)
                extracted_text += "Item: " + clean_text + ";"
                c.execute("INSERT INTO detections (image_name, detected_class, detected_text) VALUES (?, ?, ?)", (filename, "item", clean_text))
            elif cls == 1:
                text = pytesseract.image_to_string(cropped_img, lang='vie')
                clean_text = cleanning_text(text, cls)
                extracted_text += "Store name: " + clean_text + ";"
                c.execute("INSERT INTO detections (image_name, detected_class, detected_text) VALUES (?, ?, ?)", (filename, "store", clean_text))
            elif cls == 2:
                num_quan = pytesseract.image_to_string(cropped_img, config='-c tessedit_char_whitelist=0123456789')
                clean_num = cleanning_num(num_quan,  cls)
                extracted_text += "Price: " + clean_num + ";"
                c.execute("INSERT INTO detections (image_name, detected_class, detected_text) VALUES (?, ?, ?)", (filename, "price", clean_num))
            elif cls == 3:
                num_quan = pytesseract.image_to_string(cropped_img, config='--psm 10 tessedit_char_whitelist=0123456789')
                clean_num = cleanning_num(num_quan,  cls)
                extracted_text += "Quantity: " + clean_num + ";"
                c.execute("INSERT INTO detections (image_name, detected_class, detected_text) VALUES (?, ?, ?)", (filename, "quantity", clean_num))
            else:
                extracted_text += "EROR"
    
    # bboxes = results.boxes  # assuming single image input
    # logging.debug(bboxes)
    # extracted_text = ""
    
    # for bbox in bboxes:
    #     xmin, ymin, xmax, ymax, conf, cls = bbox
    #     xmin, ymin, xmax, ymax, cls = int(xmin), int(ymin), int(xmax), int(ymax), int (cls)
    #     logging.debug(cls)
    #     cropped_img = img_rgb[ymin:ymax, xmin:xmax]
        
    #     # Sử dụng pytesseract để OCR
    #     text = pytesseract.image_to_string(cropped_img, lang='vie')
        
    #     # Xử lý đữ liệu
    #     clean_text = cleanning_text(text, cls)
        
    #     # Draw bounding box
    #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    #     cv2.putText(img, clean_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    #     # Process text based on class
        
    #     #Xử lý text
    #     if cls == 0:
    #         extracted_text += "Item: " + clean_text + ";"
    #         c.execute("INSERT INTO detections (image_name, detected_class, detected_text) VALUES (?, ?, ?)", (filename, "item", clean_text))
    #     elif cls == 1:
    #         extracted_text += "Store name: " + clean_text + ";"
    #         c.execute("INSERT INTO detections (image_name, detected_class, detected_text) VALUES (?, ?, ?)", (filename, "store", clean_text))
    #     elif cls == 2:
    #         extracted_text += "Price: " + clean_text + ";"
    #         c.execute("INSERT INTO detections (image_name, detected_class, detected_text) VALUES (?, ?, ?)", (filename, "price", clean_text))
    #     elif cls == 3:
    #         extracted_text += "Quantity: " + clean_text + ";"
    #         c.execute("INSERT INTO detections (image_name, detected_class, detected_text) VALUES (?, ?, ?)", (filename, "quantity", clean_text))
    #     else:
    #         extracted_text += "EROR"
            
    conn.commit()
    conn.close()
    
    # Save the image with bounding boxes
    processed_image_path = os.path.join(RESULT_FOLDER, filename)
    img = annotator.result()  
    cv2.imwrite(processed_image_path, img)

    
    return extracted_text, processed_image_path

def cleanning_text(text, cls):
    clean_text = ""
    #Xoá ký tự xuống dòng
    text_without_space = text.replace('\n', ' ').strip()
    
    #Lower chữ
    final_lower = text_without_space.lower()
    clean_text = re.sub(r'[^a-zA-ZâấầẩẫậăắằẳẵặáàảãạăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĂÂĐÊÔƠƯƵÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ ]+', '', final_lower)

    return clean_text.strip()

def cleanning_num(num, cls) :
    clean_num = ""
    #Xoá ký tự xuống dòng
    text_without_space = num.replace('\n', ' ').strip()
    
    #Lower chữ
    final_lower = text_without_space.lower()
    clean_num = re.sub(r'[^0-9]+', '', final_lower)
        
    return clean_num.strip()



@app.route('/')
def hello_world():
    return render_template('index.html', name='World')

@app.route('/', methods=['POST', 'GET'])
def predict_img():
    if request.method == 'POST':
        if 'file' in request.files:
            f = request.files['file']
            filename = secure_filename(f.filename)
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print ("Upload folder is", filepath)
            f.save(filepath)
            global imgpath
            predict_img.imgpath = f.filename
            print("printing predict_img ::::::", predict_img)
            
            extracted_text, processed_image_path = process_image(filepath, filename)
            
            # Lưu kết quả vào file văn bản
            result_txt_path = os.path.join(RESULT_FOLDER, f.filename.rsplit('.', 1)[0] + '.txt')
            with open(result_txt_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
        
        return send_file(result_txt_path, as_attachment=True)
    
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)
            
# #The display function is used to serve the image or video from the folder_path directory.
# @app.route('/<path:filename>')
# def display(filename):
#     folder_path = 'runs/detect'
#     subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
#     latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
#     directory = folder_path+'/'+latest_subfolder    
#     print("printing directory: ",directory) 
#     files = os.listdir(directory)
#     latest_file = files[0]
    
#     print(latest_file)

#     filename = os.path.join(folder_path, latest_subfolder, latest_file)

#     file_extension = filename.rsplit('.', 1)[1].lower()

#     environ = request.environ
#     if file_extension == 'jpg':      
#         return send_from_directory(directory,latest_file,environ) #shows the result in seperate tab

#     else:
#         return "Invalid file format"
        




            

if __name__ == '__main__':
    app.run(debug=True)
