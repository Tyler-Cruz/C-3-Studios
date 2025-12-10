from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/index.html')
def index():
    css_=url_for('static', filename='style.css')
    return render_template('index.html', css_path=css_)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=49164, debug=True)