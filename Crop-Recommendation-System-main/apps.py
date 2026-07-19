from flask import Flask,request,render_template
import numpy as np
import pickle

# importing model
model = pickle.load(open('model.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    template_name = "login.html"
    return render_template(template_name)

   

   #code for crop.html
@app.route("/recommend",methods=['post'])
def Result():
    return render_template("crop.html")
    

@app.route("/Login",methods=['POST'])
def Login():
    
    username = request.form.get('username')
    pwd = request.form.get('pwd')
    
    if username=="admin" and pwd=="admin":
        return render_template("crop.html")



@app.route('/cropinput',methods=['GET','POST'])
def cropinput():
    temp = request.form.get('temp')
    ph = request.form['ph']
    humidity = request.form['humidity']
    feature_list = [ ph,temp, humidity]
    
    single_pred = np.array(feature_list).reshape(1, -1)

  
    prediction = model.predict(single_pred)

    crop_dict = {"rice": "Rice", "maize": "maize", "jute": "Jute", "cotton": "Cotton", "coconut": "Coconut", "papaya": "Papaya", "orange": "Orange",
                 "apple": "Apple", "muskmelon": "Muskmelon", "watermelon": "Watermelon", "grapes": "Grapes", "mango": "Mango","banana": "Banana",
                 "pomegranate": "Pomegranate", "lentil": "Lentil", "blackgram": "Blackgram", "mungbean": "Mungbean", "mothbeans": "Mothbeans",
                 "pigeonopeas": "Pigeonpeas", "kidneybeans": "Kidneybeans", "chickpea": "Chickpea","coffee": "Coffee"}
    print(prediction)
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('result.html',result = result)


    



# python main
if __name__ == "__main__":
    app.run(debug=True)