from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import shutil
import openpyxl
from datetime import datetime

app = Flask(__name__)

# Load your model
model = load_model('modelfinal2.h5')

# Updated Class mapping with descriptive names
classes = {
    4: ('nv', 'Melanocytic Nevi'),
    6: ('mel', 'Melanoma'),
    2: ('bkl', 'Benign Keratosis-like Lesions'),
    1: ('bcc', 'Basal Cell Carcinoma'),
    5: ('vasc', 'Pyogenic Granulomas and Hemorrhage'),
    0: ('akiec', 'Actinic Keratoses and Intraepithelial Carcinomae'),
    3: ('df', 'Dermatofibroma')
}
classes2={
    4: ('nv', '''A congenital melanocytic nevus (CMN) is a skin lesion characterized by benign proliferations of nevomelanocytes and presents at birth or develops within the first few weeks. 
    [1] These lesions may also be referred to as giant hairy nevi, 
    the term conveying the frequent clinical presence of excess hair growth.'''),
    6: ('mel', '''Melanoma is a kind of skin cancer that starts in the melanocytes.
     Melanocytes are cells that make the pigment that gives skin its color. The pigment is called melanin. 
     Melanoma typically starts on skin that's often exposed to the sun.'''),
    2: ('bkl', '''Seborrheic keratoses are a common, noncancerous skin growth that can look like benign lesions. They 
    are often brown, black, or light tan, and can look waxy or scaly. They can be scaly, greasy plaques that vary in 
    size and thickness. They can sometimes appear to be stuck onto the skin surface. Seborrheic keratoses are caused 
    by rapid multiplication of skin cells called keratinocytes. They can occur in people with a family history of the 
    condition, or it may affect people who have spent a significant amount of time in the sun. Seborrheic keratoses 
    can be removed by: Freezing the growth Scraping (curettage) or shaving the skin's surface Burning with an 
    electric current (electrocautery) Seborrheic keratoses can sometimes resemble skin cancer, such as basal cell 
    carcinoma, squamous cell carcinoma or melanoma.'''),
    1: ('bcc', '''A type of skin cancer that begins in the basal cells.
Basal cells produce new skin cells as old ones die. Limiting sun exposure can help prevent these cells from becoming cancerous.
This cancer typically appears as a white, waxy lump or a brown, 
scaly patch on sun-exposed areas, such as the face and neck.
Treatments include prescription creams or surgery to remove the cancer. In some cases radiation therapy may be required.'''),
    5: ('vasc', '''Pyogenic granulomas are skin growths that are small, round, and usually bloody red in color.
                   They tend to bleed because they contain a large number of blood vessels. 
                   They're also known as lobular capillary hemangioma or granuloma telangiectaticum.'''),
    0: ('akiec', '''Actinic keratoses are premalignant cutaneous lesions that may progress to squamous cell carcinoma.
     These lesions commonly appear on sun-exposed areas of the skin in individuals with a history of cumulative sun exposure. 
     Diagnosing actinic keratosis promptly and providing appropriate treatment to mitigate the risk of malignant transformation is crucial. 
     Additionally, implementing preventive strategies is essential to minimize the occurrence of actinic keratosis. 
         This activity aims to discuss the evaluation and treatment of actinic keratosis, 
    emphasizing the role of the interprofessional team in managing patients with this condition.'''),
    3: ('df', '''Dermatofibroma is a commonly occurring cutaneous entity usually centered within the skin's dermis.
     Dermatofibromas are referred to as benign fibrous histiocytomas of the skin, superficial/cutaneous benign fibrous histiocytomas, 
     or common fibrous histiocytoma.''')
}

def archive_existing_files():
    upload_dir = 'uploads'
    archive_dir = 'archived_uploads'

    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)

    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(archive_dir, filename))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    archive_existing_files()

    file_path = 'uploads/' + file.filename
    file.save(file_path)

    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (28, 28))  # Resize to the expected input shape
    img_array = np.array(img_resized).reshape((1, 28, 28, 3))

    result = model.predict(img_array)

    max_prob = np.max(result)
    class_ind = np.argmax(result)
    class_name, class_description = classes[class_ind]
    detailed_description = classes2[class_ind]
    wb=openpyxl.load_workbook('audit.xlsx')
    sheet = wb['Sheet1']
    next_row = sheet.max_row +1
    data=[[class_description, class_name,datetime.now()]]
    for row in data:
        sheet.append(row)
    wb.save('audit.xlsx')


    return render_template('result.html',
                       prediction=f"Prediction: {class_name} ({class_description})",
                       probability=f"Probability: {max_prob:.2f}",
                       detailed_description=detailed_description)
if __name__ == '__main__':
    app.run(debug=True)
