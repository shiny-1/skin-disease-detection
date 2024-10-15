import numpy as np
import os, cv2
from PIL import Image
from keras.models import load_model
import datetime
from flask import Flask,request, json, render_template, request, redirect, url_for
import mysql.connector
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)

def skin_detection(image_path, threshold_percentage=10):
    # Read image
    image = cv2.imread(image_path)
    argb = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    height, width, _ = argb.shape

    # Convert RGB to HSV
    hsv = cv2.cvtColor(argb, cv2.COLOR_BGR2HSV)

    # Define thresholds for skin detection
    min_hue = 0
    max_hue = 50
    min_saturation = 0.23
    max_saturation = 0.68
    min_red = 95
    min_green = 40
    min_blue = 20
    min_rg_diff = 15
    min_rb_diff = 15
    min_alpha = 15

    # Mask for skin detection
    skin_mask = (
        (hsv[:, :, 0] >= min_hue) & (hsv[:, :, 0] <= max_hue) &
        (hsv[:, :, 1] >= min_saturation * 255) & (hsv[:, :, 1] <= max_saturation * 255) &
        (argb[:, :, 2] > min_red) & (argb[:, :, 1] > min_green) & (argb[:, :, 0] > min_blue) &
        (argb[:, :, 2] > argb[:, :, 1]) & (argb[:, :, 2] > argb[:, :, 0]) &
        (np.abs(argb[:, :, 2] - argb[:, :, 1]) > min_rg_diff) &
        (np.abs(argb[:, :, 2] - argb[:, :, 0]) > min_rb_diff) &
        (argb[:, :, 3] > min_alpha)
    ).astype(np.uint8) * 255

    # Count skin pixels
    skin_pixels = cv2.countNonZero(skin_mask)

    # Calculate percentage
    total_pixels = height * width
    skin_percentage = (skin_pixels / total_pixels) * 100
    print(skin_percentage)
    # Determine if there is skin or not
    return skin_percentage >= threshold_percentage

# Connect to MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='shainy1234',
    database='project'
)

# Create a cursor
cursor = conn.cursor()

# Define the User table creation query
create_user_table_query = """CREATE TABLE IF NOT EXISTS User (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    password VARCHAR(80) NOT NULL
)"""

create_history_table_query="""CREATE TABLE IF NOT EXISTS UserUploads (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    image_filename VARCHAR(255) NOT NULL,
    image_filepath VARCHAR(255) NOT NULL,
    predicted_disease VARCHAR(255),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES User(id)
)"""


# Execute the query to create the table
cursor.execute(create_user_table_query)
cursor.execute(create_history_table_query)

class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# Define the function to fetch user from database
def get_user(username):
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM User WHERE username = %s", (username,))
    user_data = cursor.fetchone()
    cursor.close()
    if user_data:
        return User(user_data['id'], user_data['username'], user_data['password'])
    return None


def get_user_id():
    print(logged_user)
    if 'user_id' not in logged_user:
            return redirect(url_for('login'))
    return logged_user['user_id']

def insert_image_and_upload(user_id, filename, filepath, disease):
    cursor.execute("INSERT INTO UserUploads (image_filename, image_filepath, user_id, predicted_disease, uploaded_at) VALUES (%s, %s, %s, %s, %s)", (filename, filepath, user_id, disease, datetime.datetime.now()))
    conn.commit()

logged_user=dict()

app = Flask(__name__)
model = load_model("skin-effnet.h5")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username already exists
        cursor.execute("SELECT * FROM User WHERE username=%s", (username,))
        if cursor.fetchone():
            error = 'Username already exists'
        else:
            # Insert the new user into the database
            insert_query = "INSERT INTO User (username, password) VALUES (%s, %s)"
            cursor.execute(insert_query, (username, password))
            conn.commit()
            success_message = 'Successfully signed up!'
            return redirect(url_for('signup', success=success_message))
    return render_template('signup.html', error=error)

@app.route('/signin', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user(username)
        if user and user.password == password:
            print(logged_user)
            logged_user['user_id'] = user.id  # Set user_id in session
            print(logged_user)
            return redirect(url_for('UIndex'))
        else:
            return render_template('signin.html', error='Invalid username or password')
    return render_template('signin.html')


@app.route('/history')
def upload_history():
    user_id = get_user_id()
    if user_id:
        cursor.execute("SELECT * FROM UserUploads WHERE user_id = %s ORDER BY uploaded_at DESC", (user_id,))
        history = cursor.fetchall()
        return render_template('history.html', history=history)
    else:
        # Redirect to login if user is not logged in
        return redirect(url_for('signin'))

@app.route('/UIndex',methods=['POST',"GET"])
def UIndex():
    return render_template('UIndex.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        user_id=logged_user['user_id']
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'templates/uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)


        img = Image.open(filepath)

        print(skin_detection(filepath))
        if skin_detection(filepath):
            print("The provided image contains skin.")
        else:
            print("The provided image does not contain skin.")
            insert_image_and_upload(user_id,f.filename, filepath, "NOT SKIN IMAGE")
            return json.dumps({"text":"This is not a skin image. Please upload a skin image","link":None})
            
        
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(opencvImage,(150,150))
        img = img.reshape(1,150,150,3)
        x = img
        labels =["actinic keratosis", "dermatofibroma", "melanoma", "seborrheic keratosis", "squamous cell carcinoma","Acne_and_rosacea","Eczema","Tinea_Ringworm","healthy"]           


        preds = model.predict(x)[0]
        probabilities = np.exp(preds) / np.sum(np.exp(preds), axis=-1, keepdims=True)
        probabilities=[round(i*100) for i in probabilities]
        print(probabilities)

        if(all(True if(i<25) else False for i in probabilities)):
            insert_image_and_upload(user_id,f.filename, filepath,"DOES NOT SUPPORT THIS DISEASE TYPE")
            return json.dumps({"text":"The prediction results are too low. It may be that our model does not support this skin disease type. However it is recommended to check several things: 1.Make sure the image is good quality. 2.Make sure the wound is clearly visible. 3.Make sure the image is not dark","link":None})  
        
        index = np.argmax(preds, axis=-1)
        print(index)

        disease=labels[index]
        insert_image_and_upload(user_id,f.filename, filepath, disease)
        
        print("------------------------------------\n")
        print("\nprediction -> ",disease)
        print("\n-------------------------------------")
        text = "The predicted Disease is " + disease
        
        links={"Eczema":"https://nationaleczema.org/eczema/","Tinea_Ringworm":"https://www.cdc.gov/fungal/diseases/ringworm/index.html",
           "Acne_and_rosacea":"https://www.niams.nih.gov/health-topics/rosacea#:~:text=Rosacea%20is%20a%20long-term,emotional%20stress%2C%20bring%20th",
           "squamous cell carcinoma":"https://www.cancer.org/cancer/types/basal-and-squamous-cell-skin-cancer/about/what-is-basal-and-squamous-cell.html",
           "seborrheic keratosis":"https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878",
           "melanoma":"https://www.skincancer.org/skin-cancer-information/melanoma/",
           "dermatofibroma":"https://www.westlakedermatology.com/blog/dermatofibroma-treatment-options/#:~:text=Dermatofibromas%20are%20the%20result%20of,body%20attempts%20to%20repair%20itself.",
           "actinic keratosis":"https://www.aad.org/public/diseases/skin-cancer/actinic-keratosis-overview",
           "healthy": None}
        
        if disease=="healthy":
            text="There is no skin disease, your skin is healthy"
        link=links[disease]
        dictionary={"text":text,"link":link}
        jsondata=json.dumps(dictionary)    
    return jsondata
if __name__ == '__main__':
    app.run(debug = True, threaded = False)