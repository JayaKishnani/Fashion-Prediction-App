from flask import Flask, render_template, flash, request, url_for, send_file, Response
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")