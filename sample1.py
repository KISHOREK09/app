from flask import Flask, request, jsonify,send_from_directory
import MySQLdb
from flask_mysqldb import MySQL
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import re
import bcrypt
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

# Route for uploading and processing sperm samples
@app.route('/upload_sample/<int:uid>/<string:sample_type>', methods=['POST'])
def upload_sample(uid, sample_type):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    sample_type = re.findall(r'\d+', sample_type)[0]  # Extract sample number from the sample_type
    if sample_type not in ['1', '2', '3', '4', '5', '6']:
        return jsonify({'message': 'Invalid sample type'}), 400

    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file detected'}), 400

    # Secure and format filename
    file_ext = file.filename.split('.')[-1]  # Get the file extension
    filename = f'sample{sample_type}_{uid}.{file_ext}'  # Custom filename format
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded file with the custom filename
    file.save(filepath)

    # Process the image and detect sperm
    active_count, dead_count, processed_filepath = detect_sperm(filepath, sample_type, uid)

    if active_count is None or dead_count is None:
        return jsonify({'message': 'Error processing the image'}), 500

    try:
        # Check if the patient already has a record in the 'count' table
        cursor.execute('SELECT COUNT(*) FROM result WHERE uid = %s', (uid,))
        record_exists1 = cursor.fetchone()['COUNT(*)']
        if record_exists1 > 0:
            cursor.execute(f'UPDATE count SET active_sperm{sample_type} = %s WHERE uid = %s',
                           (active_count, uid))
        else:
            cursor.execute(f'INSERT INTO count (uid, active_sperm{sample_type}) VALUES (%s, %s)',
                           (uid, active_count))

        # Check if there's already a record for the uid in the 'result' table
        cursor.execute('SELECT COUNT(*) FROM result WHERE uid = %s', (uid,))
        record_exists = cursor.fetchone()['COUNT(*)']

        if record_exists > 0:
            # Update the existing record
            cursor.execute(f'UPDATE result SET sample{sample_type} = %s WHERE uid = %s',
                           (processed_filepath, uid))
        else:
            # Insert a new record into the 'result' table
            cursor.execute(f'INSERT INTO result (uid, sample{sample_type}) VALUES (%s, %s)',
                           (uid, processed_filepath))

        connection.commit()
    except Exception as e:
        mysql.connection.rollback()
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()
        

    return jsonify({
        'message': 'Sample uploaded and processed successfully',
        'active_count': active_count,
    }), 200

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
@app.route('/signup', methods=['POST'])
def signup():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')

        if not name or not email or not password:
            return jsonify({'message': 'Missing fields'}), 400

        # Hash the password before saving it
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        
        cursor.execute('INSERT INTO signup (name, email, pass) VALUES (%s, %s, %s)', 
                       (name, email, hashed_password))
        connection.commit()
        cursor.close()

        return jsonify({'message': 'Signup successful'}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    

# Route for user login
@app.route('/login', methods=['POST'])
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
@app.route('/patient', methods=['POST'])
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
        foc = data.get('frequency_of_coitus')
        sexual_dysfunction = data.get('sexual_dysfunction')
        alcoholic = data.get('alcoholic')
        smoker = data.get('smoker')
        drugs = data.get('drugs')
        
        if None in [name, age, occupation, height, weight, foc, sexual_dysfunction, alcoholic, smoker, drugs]:
            return jsonify({'message': 'Missing fields'}), 400

        bmi = weight / (height * height)
        
        cursor.execute('INSERT INTO patient (name, age, occupation, height, weight, bmi, foc, sexual_dysfunction, alcoholic, smoker, drugs) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)',
                       (name, age, occupation, height, weight, bmi, foc, sexual_dysfunction, alcoholic, smoker, drugs))
        connection.commit()
        uid = cursor.lastrowid

        return jsonify({'message': 'Patient data added successfully', 'Patient Unique Id': uid}), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()

# Route to retrieve recent patients
@app.route('/recent_patients', methods=['GET'])
def recent_patients():
    connection = get_mysql_connection()
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT name, uid AS patient_id, age, occupation, height, weight, bmi, foc, sexual_dysfunction, alcoholic, smoker, drugs FROM patient ORDER BY uid desc')
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
            cursor.execute('SELECT * FROM count WHERE uid = %s', (uid,))
            sperm_counts = cursor.fetchone()
            cursor.close()
            return jsonify({'patient': result, 'sperm_counts': sperm_counts}), 200
        else:
            return jsonify({'message': 'No patient found with given ID'}), 404
    else:
        return jsonify({'message': 'Missing patient ID'}), 400

@app.route('/get_sample_status/<int:uid>', methods=['GET'])
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
    if uid and isinstance(current_sample, int) and 1 <= current_sample <= 6:  
        

        # Fetch the paths of the samples for the specified uid
        cursor.execute("SELECT sample1, sample2, sample3, sample4, sample5, sample6 FROM result WHERE uid=%s", (uid,))
        result = cursor.fetchone()
        print(result)
        # Check if the patient exists in the result
        if result:
            if current_sample-1==0:
                return jsonify({'Not uploaded the image'}),404
            # Get the path for the current sample
            key=f'sample{current_sample-1}'
            sample_path = result[key]  # Adjust for zero-based indexing

            # Check if the sample path is valid (not None)
            if sample_path is not None:
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

@app.route('/uploads/<path:filename>')
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
@app.route('/view_result', methods=['GET'])
def view_result():
    connection = get_mysql_connection()
    cursor=connection.cursor()
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

        if not patient:
            return jsonify({'message': 'No patient found with given ID'}), 404

        # Retrieve sperm counts
        cursor.execute('SELECT active_sperm{} FROM count WHERE uid = %s'.format(sample_type), (uid,))
        sperm_counts = cursor.fetchone()

        # Retrieve the processed image filepath
        cursor.execute('SELECT sample{} FROM result WHERE uid = %s'.format(sample_type), (uid,))
        processed_result = cursor.fetchone()
        print(sperm_counts)
        print(processed_result)
        if not sperm_counts or not processed_result:
            return jsonify({
            'patient': patient,
            'sperm_counts': {
                'active_sperm': None
            },
            'processed_image': None
        }), 404

        active_sperm = sperm_counts[f'active_sperm{sample_type}']
        processed_image = processed_result[f'sample{sample_type}']
        # Return the results in JSON format
        return jsonify({
            'patient': patient,
            'sperm_counts': {
                'active_sperm': active_sperm
            },
            'processed_image': processed_image
        }), 200
    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        cursor.close()
@app.route('/get_profile', methods=['GET'])
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
@app.route('/change_password', methods=['POST'])
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
        
@app.route('/get_sperm_count/<uid>', methods=['GET'])
def get_sperm_count(uid):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT active_sperm1, active_sperm2, active_sperm3, active_sperm4, active_sperm5, active_sperm6 FROM count WHERE uid = %s', (uid,))
    result = cursor.fetchone()
    
    print(result)
    if result:
        sperm_counts = {
            'active_sperm1': result['active_sperm1'],
            'active_sperm2': result['active_sperm2'],
            'active_sperm3': result['active_sperm3'],
            'active_sperm4': result['active_sperm4'],
            'active_sperm5': result['active_sperm5'],
            'active_sperm6': result['active_sperm6']
        }
        cursor.close()
        return jsonify(sperm_counts), 200
    else:
        cursor.close()
        return jsonify({'message': 'No data found for this UID'}), 404
@app.route('/update_patient/<int:patient_id>', methods=['PUT'])
def update_patient(patient_id):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    data = request.get_json()
    bmi=data['weight']/(data['height']*data['height'])
    print(bmi)
    try:
        update_query = """
            UPDATE patient
            SET name=%s, age=%s, occupation=%s, height=%s, weight=%s, bmi=%s ,foc=%s,
                sexual_dysfunction=%s, alcoholic=%s, smoker=%s, drugs=%s
            WHERE uid=%s
        """
        cursor.execute(update_query, (
            data['name'], data['age'], data['occupation'], data['height'], data['weight'], bmi,
            data['foc'], data['sexual_dysfunction'], data['alcoholic'],
            data['smoker'], data['drugs'], patient_id
        ))
        connection.commit()
        return jsonify({'message': 'Patient details updated successfully!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        cursor.close()
        

@app.route('/get_patient/<int:patient_id>', methods=['GET'])
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
        
if __name__ == '__main__':
    app.run(debug=True)
