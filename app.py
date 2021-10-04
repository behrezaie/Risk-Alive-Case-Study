from flask import Flask, render_template, url_for, request, redirect
import joblib


app = Flask('__name__')


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        proba_list = []
        classifier = joblib.load("best_pipeline.pkl")
        new_content = request.form["content"]
        predicted_probabilities = classifier.predict_proba([new_content])
        for prob in predicted_probabilities[0]:
            proba_list.append(round(prob, 2))
        predicted_category = classifier.predict([new_content])
        return render_template('results.html', new_content=new_content, proba_0=proba_list[0], proba_1=proba_list[1],
                               proba_2=proba_list[2], proba_3=proba_list[3], proba_4=proba_list[4], proba_5=proba_list[5],
                               proba_6=proba_list[6], proba_7=proba_list[7], proba_8=proba_list[8], proba_9=proba_list[9],
                               proba_10=proba_list[10], proba_11=proba_list[11], proba_12=proba_list[12], proba_13=proba_list[13],
                               predicted_category=predicted_category[0])
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
