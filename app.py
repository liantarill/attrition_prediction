from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
