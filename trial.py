@app.route('/Admin', methods=['POST'])
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

    
@app.route('/add_doctor', methods=['POST'])
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

@app.route('/delete_doctor/<int:doctor_id>', methods=['DELETE'])
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

@app.route('/doctors', methods=['GET'])
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
        return jsonify({'error': str(e)}), 500
