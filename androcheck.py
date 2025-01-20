from flask import Flask, request, jsonify,send_from_directory,make_response
import MySQLdb
from flask_mysqldb import MySQL
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import re
import bcrypt
import pandas as pd
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
from werkzeug.utils import secure_filename
from datetime import datetime
app = Flask(__name__)
app.secret_key = 'xyzsdfg'


def get_mysql_connection():
    try:
        connection = MySQLdb.connect(
            host='localhost',
            user='root',
            password='k9kishore',
            db='spermdetect',
            cursorclass=MySQLdb.cursors.DictCursor,
            connect_timeout=28800
        )
        return connection
    except MySQLdb.Error as e:
        print(f"Error connecting to MySQL: {str(e)}")
        return None

# Upload Folder Configuration
app.config['UPLOAD_FOLDER'] = '/Users/kishore/Desktop/app/uploads/'
app.config['PROCESSED_FOLDER'] ='/Users/kishore/Desktop/app/processed_images/'
# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)


# Define the sperm detection model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),  # Adding dropout for regularization
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

@app.route('/uploadsample/<int:uid>/<int:sample_number>', methods=['POST'])
def upload_sample(uid, sample_number):
    connection = get_mysql_connection()
    cursor = connection.cursor()

    # Validate sample_number (should be between 1 and 6)
    if sample_number not in range(1, 7):
        return jsonify({'message': 'Invalid sample number'}), 400

    # Collect files from request
    files = []
    for i in range(1, 17):
        file_key = f'file{i}'
        if file_key in request.files:
            files.append(request.files[file_key])

    # Validate number of uploaded files
    if len(files) != 16:
        return jsonify({'message': 'Please upload exactly 16 images'}), 400

    total_count = 0  # Initialize total active sperm count for the sample

    # Process each file and save its path and count in the database
    for i, file in enumerate(files):
        if file.filename == '':
            return jsonify({'message': 'One of the files is empty'}), 400

        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_ext = filename.split('.')[-1]
        image_filename = f"sample{sample_number}_image{i + 1}.{file_ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        file.save(file_path)

        # Detect sperm in the image
        try:
            sperm_count, dead_count, processed_filepath  = detect_sperm(file_path, sample_number, uid)
            print("sperm:-",sperm_count)
        except Exception as e:
            return jsonify({'message': f'Error processing image {i + 1}: {str(e)}'}), 500

        # Save the image data to the database
        cursor.execute(f'''INSERT INTO sample_images (uid, sample_number, image_number, image_path, sperm_count, date_uploaded) VALUES (%s, %s, %s, %s, %s, NOW())''', (uid, sample_number, i + 1, processed_filepath, sperm_count))
        #connection.commit()
        # Update the total count for the sample
        total_count += sperm_count
    print("total_count:-",total_count)
    total_count=total_count*200000
    # Dynamically construct the column name
    try:
        print("sample_number",sample_number)
        sample_column = f"sample{sample_number}_count"
        sample_date = f"sample{sample_number}_date"
        cursor.execute('SELECT COUNT(*) FROM result WHERE uid = %s', (uid,))
        record_exists = cursor.fetchone()['COUNT(*)']
        if record_exists >0:
            query = f"UPDATE result SET {sample_column} = %s, current_sample = %s, {sample_date} = NOW() WHERE uid = %s"
            cursor.execute(query, (total_count, sample_number, uid))
            connection.commit()
        else:
            print(123)
            query=f"insert into result (uid,{sample_column},{sample_date}) values(%s,%s,NOW())"
            cursor.execute(query,(uid,total_count))
            print("paased")
            connection.commit()
        return jsonify({
        'message': 'Sample uploaded and processed successfully',
        'active_count': total_count
    }), 200
    except Exception as e:
        connection.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()

# Image processing function for detecting sperm
def detect_sperm(filepath,sample_type, uid):
    active_count = 0
    dead_count = 0
    img = cv2.imread(filepath)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.array(img_resized).reshape(-1, 224, 224, 3)
    img_array = img_array / 255.0  # Rescale

    # Predict using the model
    prediction = model.predict(img_array)
    detected = prediction[0][0] > 0.5  # Threshold
    file_ext = filepath.split('.')[-1]
    processed_filename = f'sample{sample_type}_{uid}_processed.{file_ext}'
    # Process the image and save
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(filepath))

    if detected:
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise and improve contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the grayscale image
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the contours (sperms)
        marked_img = img.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            marked_img = cv2.rectangle(marked_img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green rectangle
        active_count = len(contours)  # Number of contours detected
        dead_count = 0  # Update as needed based on your criteria
        cv2.imwrite(processed_filepath, marked_img)
    else:
        cv2.imwrite(processed_filepath, img)

    return active_count, dead_count, processed_filepath
# Route for user signup
@app.route('/Admin_page', methods=['POST'])
def admin():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    try:
        data = request.get_json()
        name = data.get('name')
        password = data.get('password')

        if not name or not password:
            return jsonify({'message': 'Missing fields'}), 400

        
        cursor.execute('Select * from admin where name=%s',(name,))
        account = cursor.fetchone()
        print(account)
        if account:
            password1 = account['pass']
            if(password==password1):
                 return jsonify({'message': 'Login successful'}), 200
            else:
                return jsonify({'message': 'Invalid email or password'}), 401
        else:
            return jsonify({'message': 'No Data Found'}), 401
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()
    

# Route for user login
@app.route('/login_page', methods=['POST'])
def login():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        cursor.execute('SELECT * FROM signup WHERE email = %s', (email,))
        account = cursor.fetchone()
        print(account)
        if account:
            # Retrieve the hashed password from the database
            stored_hashed_password = account['pass']

            # Verify the provided password against the stored hash
            if bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
                return jsonify({'message': 'Login successful'}), 200
            else:
                return jsonify({'message': 'Invalid email or password'}), 401
        else:
            return jsonify({'message': 'Invalid email or password'}), 401
    except Exception as e:
        print("login",e)
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()
# Route for adding patient data
@app.route('/patient_page', methods=['POST'])
def patient():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    try:
        data = request.get_json()
        name = data.get('name')
        age = data.get('age')
        occupation = data.get('occupation')
        height = data.get('height')
        weight = data.get('weight')
        sexual_dysfunction = data.get('sexual_dysfunction')
        alcoholic = data.get('alcoholic')
        smoker = data.get('smoker')
        drugs = data.get('drugs')
        
        if None in [name, age, occupation, height, weight, sexual_dysfunction, alcoholic, smoker, drugs]:
            return jsonify({'message': 'Missing fields'}), 400

        bmi = weight / (height * height)
        
        cursor.execute('INSERT INTO patient (name, age, occupation, height, weight, bmi, sexual_dysfunction, alcoholic, smoker, drugs) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)',
                       (name, age, occupation, height, weight, bmi, sexual_dysfunction, alcoholic, smoker, drugs))
        connection.commit()
        uid = cursor.lastrowid

        return jsonify({'message': 'Patient data added successfully', 'Patient Unique Id': uid}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()

# Route to retrieve recent patients
@app.route('/recent_patients_page', methods=['GET'])
def recent_patients():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT name, uid AS patient_id, age, occupation, height, weight, bmi, sexual_dysfunction, alcoholic, smoker, drugs, date AS date FROM patient ORDER BY uid desc')
        patients = cursor.fetchall()
        return jsonify(patients), 200
    except Exception as e:
        print("recnet",e)
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()

# Route to retrieve patient results
@app.route('/result', methods=['GET'])
def get_result():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    uid = request.args.get('uid')
    print(type(uid))
    if uid:
        
        cursor.execute('SELECT * FROM patient WHERE uid = %s', (uid,))
        result = cursor.fetchone()

        if result:
            cursor.execute('SELECT * FROM result WHERE uid = %s', (uid,))
            sperm_counts = cursor.fetchone()
            cursor.close()
            return jsonify({'patient': result, 'sperm_counts': sperm_counts}), 200
        else:
            return jsonify({'message': 'No patient found with given ID'}), 404
    else:
        return jsonify({'message': 'Missing patient ID'}), 400

@app.route('/get_sample_status_page/<int:uid>', methods=['GET'])
def get_sample_status(uid):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    if uid:
        
        cursor.execute('SELECT current_sample FROM result WHERE uid = %s', (uid,))
        patient=cursor.fetchone()
        cursor.close()
        if patient:
            sample=patient['current_sample']
            return jsonify({
            'current_sample': sample,
        }), 200
        else:
            print("else")
            return jsonify({'error': 'Patient not found'}), 404
@app.route('/result/update_current_sample/<string:uid>', methods=['PUT'])
def update_current_sample(uid):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    data = request.get_json()
    current_sample = data.get('current_sample')
    current_sample+=1
    print(current_sample)

    # Validate input: Check if uid and current_sample are provided and correct
    if uid and isinstance(current_sample, int) and 1 <= current_sample <= 7:  
        

        # Fetch the paths of the samples for the specified uid
        cursor.execute("SELECT sample1_count, sample2_count, sample3_count, sample4_count, sample5_count, sample6_count FROM result WHERE uid=%s", (uid,))
        result = cursor.fetchone()
        print(result)
        # Check if the patient exists in the result
        if result:
            if current_sample-1==0:
                return jsonify({'Not uploaded the image'}),404
            # Get the path for the current sample

            key = f'sample{current_sample - 1}_count'
            print("key:-",key)
            sample_count = result[key]

            # Check if the sample path is valid (not None)
            if sample_count is not None:
                # Proceed to update the current_sample
                cursor.execute("UPDATE result SET current_sample=%s WHERE uid=%s", (current_sample, uid))
                connection.commit()
                cursor.close()
                
                return jsonify({'message': 'Current sample updated successfully.'}), 200
            else:
                cursor.close()
                return jsonify({'error': 'Image path for the specified sample does not exist.'}), 404
        else:
            cursor.close()
            return jsonify({'error': 'Patient not found.'}), 404

    return jsonify({'error': 'Invalid UID or current_sample.'}), 400

@app.route('/uploads_page/<path:filename>')
def get_uploaded_file(filename):
    filename = os.path.basename(filename)
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    print("path",filepath)
    print("name",filename)
    if os.path.exists(filepath):
        return send_from_directory(app.config['PROCESSED_FOLDER'], filename)
    else:
        return jsonify({'error': 'File not found'}), 404
# Route to retrieve the processed results for a specific sample
@app.route('/view_result_page', methods=['GET'])
def view_result():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    uid = request.args.get('uid')
    sample_type = request.args.get('sample_type')  # Expecting a sample number (1-6)

    if not uid or not sample_type:
        return jsonify({'message': 'Missing patient ID or sample type'}), 400

    # Validate sample type
    if sample_type not in ['1', '2', '3', '4', '5', '6']:
        return jsonify({'message': 'Invalid sample type'}), 400

    try:
        # Retrieve patient details
        cursor.execute('SELECT * FROM patient WHERE uid = %s', (uid,))
        patient = cursor.fetchone()
        name=patient['name']
        age=patient['age']
        sexual_dysfunction=patient['sexual_dysfunction']
        if not patient:
            return jsonify({'message': 'No patient found with given ID'}), 404

        # Retrieve sperm counts
        cursor.execute(
            '''
            SELECT sample_number, image_path, sperm_count, date_uploaded
            FROM sample_images
            WHERE uid = %s and sample_number = %s
            ''', (uid,sample_type)
        )
        images_data = cursor.fetchall()
        if not images_data:
            return jsonify({'message': 'No processed images found for the given patient ID'}), 404

        processed_images = []
        for image in images_data:
            sample_number=image['sample_number']
            image_path=image['image_path']
            sperm_count=image['sperm_count']
            date_uploaded=image['date_uploaded']
            # Check if date_uploaded is a datetime object or a string
            if isinstance(date_uploaded, datetime):
                formatted_date = date_uploaded.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(date_uploaded, str):
                formatted_date = date_uploaded  # If it's already a string, use it as is
            else:
                formatted_date = None  # Handle any other case (e.g., None)
            print(1221)
            processed_images.append({
                'sample_number': sample_number,
                'image_path': image_path,
                'sperm_count': sperm_count,
                'date_uploaded': formatted_date
             })
        key = f'sample{sample_type}_count'
        print(key)
        cursor.execute(f"SELECT {key} FROM result WHERE uid = %s", (uid,))
        result_counts = cursor.fetchone()
        if not result_counts:
            return jsonify({'message': 'No sperm counts found for the given patient ID'}), 404

        response_data = {
            'patient': {
                'name': name,
                'age': age,
                'sexual_dysfunction': sexual_dysfunction
            },
            'total_sperm_count':result_counts[key] ,
            'processed_images': processed_images
        }
        #print(response_data)
        return jsonify(response_data), 200

    except Exception as e:
        print(e)
        return jsonify({'message': str(e)}), 500

    finally:
        cursor.close()

@app.route('/get_profile_page', methods=['GET'])
def get_profile():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    email = request.args.get('email')
    print(email)
    print(type(email))
    if email:
        cursor.execute('select name from signup where email=%s',(email,))
        result = cursor.fetchone()
        print(result)
        if result:
            cursor.close()
            return jsonify({'username': result['name']})
        else:
            return jsonify({'error': 'User not found'}), 404
    return jsonify({'error': 'Email is required'}), 400
@app.route('/change_password_page', methods=['POST'])
def change_password():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    data = request.get_json()
    email = data.get('email')
    current_password = data.get('current_password')
    new_password = data.get('new_password')

    
    try:
        # Fetch the current hashed password from the database
        cursor.execute('SELECT pass FROM signup WHERE email = %s', (email,))
        user = cursor.fetchone()

        if user is None:
            return jsonify({"error": "User not found"}), 404

        # Check if the current password matches the stored hashed password
        stored_hashed_password = user['pass']
        if not bcrypt.checkpw(current_password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
            return jsonify({"error": "Incorrect current password"}), 400

        # Hash the new password
        hashed_new_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        # Update the password in the database
        cursor.execute("UPDATE signup SET pass = %s WHERE email = %s", (hashed_new_password, email))
        connection.commit()

        return jsonify({"message": "Password updated successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cursor.close()
        
@app.route('/get_sperm_count_page/<uid>', methods=['GET'])
def get_sperm_count(uid):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT sample1_count, sample2_count, sample3_count, sample4_count, sample5_count, sample6_count FROM result WHERE uid = %s', (uid,))
    result = cursor.fetchone()
    
    print(result)
    if result:
        sperm_counts = {
            'active_sperm1': result['sample1_count'],
            'active_sperm2': result['sample2_count'],
            'active_sperm3': result['sample3_count'],
            'active_sperm4': result['sample4_count'],
            'active_sperm5': result['sample5_count'],
            'active_sperm6': result['sample6_count']
        }
        cursor.close()
        return jsonify(sperm_counts), 200
    else:
        cursor.close()
        return jsonify({'message': 'No data found for this UID'}), 404
@app.route('/update_patient_page/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    data = request.get_json()
    bmi=data['weight']/(data['height']*data['height'])
    print(bmi)
    try:
        update_query = """
            UPDATE patient
            SET name=%s, age=%s, occupation=%s, height=%s, weight=%s, bmi=%s ,
                sexual_dysfunction=%s, alcoholic=%s, smoker=%s, drugs=%s
            WHERE uid=%s
        """
        cursor.execute(update_query, (
            data['name'], data['age'], data['occupation'], data['height'], data['weight'], bmi, data['sexual_dysfunction'], data['alcoholic'],
            data['smoker'], data['drugs'], patient_id
        ))
        connection.commit()
        return jsonify({'message': 'Patient details updated successfully!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        cursor.close()
        

@app.route('/get_patient_page/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    try:
        cursor.execute("select * from patient where uid=%s",(patient_id,))
        patient=cursor.fetchone()
        if patient:
            return jsonify(patient), 200
        else:
            return jsonify({'message': 'Patient not found'}), 404
    except Exception as e:
         return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()

@app.route('/add_doctor_page', methods=['POST'])
def add_doctor():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    data = request.get_json()
    username = data['username']
    email=data['email']
    password = data['password']
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO signup (name, pass,email) VALUES (%s, %s, %s)", (username, hashed_password,email))
        connection.commit()
        cursor.close()
        return jsonify({'message': 'Doctor ID added successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_doctor_page/<int:doctor_id>', methods=['DELETE'])
def delete_doctor(doctor_id):
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute("DELETE FROM signup WHERE uid = %s", (doctor_id,))
        connection.commit()
        cursor.close()
        return jsonify({'message': 'Doctor ID deleted successfully'}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/delete_patient_page/<int:patient_id>',methods=['DELETE'])
def delete_patient(patient_id):
    try:
        connection=get_mysql_connection()
        cursor=connection.cursor()
        cursor.execute("DELETE from patient where uid=%s",(patient_id,))
        connection.commit()
        try:
            cursor.execute("DELETE from count where uid=%s",(patient_id,))
            connection.commit()
            cursor.execute("DELETE from patient where uid=%s",(patient_id,))
            connection.commit()
        except:
            pass
        cursor.close()
        return jsonify({'message': 'Patient ID deleted successfully'}),200
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/doctors_page', methods=['GET'])
def get_doctors():
    try:
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT uid,name,email FROM signup")
        doctors = cursor.fetchall()
        print(doctors)
        cursor.close()
        doctor_list = [{'id': row['uid'], 'username': row['name'], 'email': row['email']} for row in doctors]
        return jsonify(doctor_list), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/download_report/<int:uid>', methods=['GET'])
def download_report(uid):
    connection = get_mysql_connection()
    cur = connection.cursor()
    cur.execute("SELECT name, age, sexual_dysfunction, bmi FROM patient WHERE uid = %s", (uid,))
    patient = cur.fetchone()

    if not patient:
        return jsonify({"error": "Patient not found"}), 404

    name=patient['name']
    age=patient['age']
    dysfunction=patient['sexual_dysfunction']
    bmi=patient['bmi']
    # Fetch sperm count and date details
    cur.execute("""
        SELECT *
        FROM result WHERE uid = %s
    """, (uid,))
    data = cur.fetchone()
    cur.execute("""
        SELECT *
        FROM result WHERE uid = %s
    """, (uid,))
    data1 = cur.fetchone()
    print(data1)
    cur.close()
    if not data:
        return jsonify({"error": "Sperm count details not found"}), 404

    # Separate counts and dates
    counts = [data.get(f'sample{i+1}_count', 'NIL') for i in range(6)]
    dates = [data1.get(f'sample{i+1}_date', 'NIL') for i in range(6)]
    counts = ["NIL" if count is None else count for count in counts]
    print(dates)
    dates = [
        "NIL" if date is None else datetime.strptime(str(date), '%Y-%m-%d %H:%M:%S').strftime('%d-%m-%Y %H:%M:%S')
        if date != "NIL" else "NIL"
        for date in dates
    ]
    # Generate PDF or Excel report
    report_type= request.args.get('format')

    if report_type == 'pdf':
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt="Patient Report", ln=True, align='C')
        pdf.ln(10)
        # Patient details
        pdf.cell(200, 10, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {age}", ln=True)
        pdf.cell(200, 10, txt=f"Sexual Dysfunction: {dysfunction}", ln=True)
        pdf.cell(200, 10, txt=f"BMI: {bmi}", ln=True)
        pdf.ln(10)

        # Sperm count and dates
        pdf.cell(200, 10, txt="Sperm Count and Dates:", ln=True)
        for i, (count, date) in enumerate(zip(counts, dates), 1):
            pdf.cell(200, 10, txt=f"Sample {i}: {count} (Date: {date})", ln=True)

        # Save the PDF to a BytesIO object
        pdf_buffer = BytesIO()
        pdf.output(dest="S").encode("latin1")  # Get the content as bytes
        pdf_buffer.write(pdf.output(dest='S').encode('latin1'))
        pdf_buffer.seek(0)  # Reset the pointer to the beginning of the stream

        # Return the PDF as a Flask response
        response = make_response(pdf_buffer.read())
        response.headers["Content-Type"] = "application/pdf"
        response.headers["Content-Disposition"] = f"attachment; filename=patient_report_{uid}.pdf"

        return response


    elif report_type == 'csv':
        # Prepare data for Excel
        data = {
    "Name": [name] + [''] * 5,  # Repeat only in the first row
    "Age": [age] + [''] * 5,
    "Dysfunction": [dysfunction] + [''] * 5,
    "BMI": [bmi] + [''] * 5,
    "Sample": [f"Sample {i+1}" for i in range(6)],
    "Sperm Count": counts,
    "Date": dates
}
        df = pd.DataFrame(data)

        # Save to BytesIO for Excel
        response = BytesIO()
        df.to_csv(response, index=False)
        response.seek(0)

        return make_response(response.getvalue(), {
            "Content-Type": "text/csv",
            "Content-Disposition": f"attachment; filename=patient_report_{uid}.csv"
        })

    return jsonify({"error": "Invalid report type"}), 400
        
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=8000)
