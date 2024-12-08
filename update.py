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
        FROM count WHERE uid = %s
    """, (uid,))
    data = cur.fetchone()
    cur.close()

    if not data:
        return jsonify({"error": "Sperm count details not found"}), 404

    # Separate counts and dates
    counts = [data.get(f'active_sperm{i+1}', 'NIL') for i in range(6)]
    dates = [data.get(f'datetime{i+1}', 'NIL') for i in range(6)]
    counts = ["NIL" if count is None else count for count in counts]
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