from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, redirect, send_from_directory, url_for, session, jsonify
import mysql.connector  # Using MySQL instead of SQLite
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
import pickle
from predict import blooddetection
from flask_cors import CORS, cross_origin

app = Flask(__name__, template_folder="templates")
app.secret_key = "1234567890."

# MySQL Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",  # Change if you have a different MySQL username
    "password": "",   # Set your MySQL password if applicable
    "database": "expense_tracker"
}

# Database Manager for MySQL
class DatabaseManager:
    def __enter__(self):
        self.conn = mysql.connector.connect(**DB_CONFIG)
        self.cursor = self.conn.cursor(dictionary=True)  # Return data as a dictionary
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        self.cursor.close()
        self.conn.close()

# Create Tables in MySQL
def create_tables():
    with DatabaseManager() as cursor:
        cursor.execute("CREATE DATABASE IF NOT EXISTS expense_tracker")
        cursor.execute("USE expense_tracker")

        # Users Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                address TEXT,
                phone VARCHAR(15) NOT NULL
            )
        ''')

# Register Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        address = request.form["address"]
        phone = request.form["phone"]

        with DatabaseManager() as cursor:
            try:
                cursor.execute("""
                    INSERT INTO users (username, email, password, address, phone) 
                    VALUES (%s, %s, %s, %s, %s)""",
                    (username, email, password, address, phone))
                return redirect("/login")  # Redirect to login
            except mysql.connector.IntegrityError:
                return "Email already registered!", 400  # Error if email exists

    return render_template("register.html")  # Ensure register.html exists in /templates

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        with DatabaseManager() as cursor:
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()

            if user and check_password_hash(user["password"], password):
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                return redirect("/upload")  # Redirect after login
            else:
                return "Invalid email or password!", 401  # Error message

    return render_template("login.html")  # Ensure login.html exists in /templates

# Prediction Route
@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
   if request.method == "POST":
        image_file = request.files['file']
        print(image_file)
        classifier = blooddetection() 
        result = classifier.bloodgroup(image_file)
        return result
   else:
       print('Loading Error')

# Logout Route
@app.route("/logout")
def logout():
    session.clear()  # Clear session
    return redirect("/login")  # Redirect to login

@app.route('/upload')
def upload():
    return render_template('upload.html')
@app.route('/Analysis')
def analysis():
    return render_template('Analysis.html')  # Ensure Analysis.html exists in /templates

@app.route('/')
def home():
    return render_template('home.html')
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/Safe')
def safe():
    return render_template('Safe.html')
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == "__main__":
    create_tables()
    app.run(debug=True)
