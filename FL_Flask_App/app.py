from flask import Flask, request, render_template, jsonify
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

input_size = 6
learning_rate = 0.01
num_iterations = 20000

class LogisticRegression(torch.nn.Module):
			
	def __init__(self):
		super(LogisticRegression, self).__init__()
		self.linear = torch.nn.Linear(input_size, 1)

	def forward(self, x):
		return torch.sigmoid(self.linear(x))

@app.route('/', methods =["GET", "POST"])
def home():
	return render_template("index.html")


@app.route('/result', methods=["GET", "POST"])
def result():
	if request.method == "POST":
		req = request.form
		temperature = int(req.get('temperature'))
		name = req.get('fullname')
		if req.get('occ_n') == 'yes':
			occ_n = int(1)
		else:
			occ_n = int(0)

		if req.get('lum_p') == 'yes':
			lum_p = int(1)
		else:
			lum_p = int(0)

		if req.get('uri_p') == 'yes':
			uri_p = int(1)
		else:
			uri_p = int(0)

		if req.get('mig_p') == 'yes':
			mig_p = int(1)
		else:
			mig_p = int(0)

		if req.get('bur_u') == 'yes':
			bur_u = int(1)
		else:
			bur_u = int(0)
		
		pat = []
		pat.append(temperature)
		pat.append(occ_n)
		pat.append(lum_p)
		pat.append(uri_p)
		pat.append(mig_p)
		pat.append(bur_u)
		disease = req.get('disease')
		
		#return jsonify(pat)



		model_IUB = pickle.load(open('model_IUB.pkl','rb'))
		model_NRP = pickle.load(open('model_NRP.pkl','rb'))
		pt_pat = torch.FloatTensor(pat)

		if disease == 'Inflamation_of_urinary_bladder':
			dis_name = 'Inflamation of Urinary Bladder'
			pred_pat = model_IUB(pt_pat)
			a = pred_pat.tolist()
			res = "{:.2f}".format(a[0]*100)
			#print("Your chances of this disease are: " + "{:.2f}".format(a[0]*100) + "%")
			#res = model_IUB.predict(pat)
		else:
			dis_name = 'Nephritis of Renal Pelvis Origin'
			pred_pat = model_NRP(pt_pat)
			a = pred_pat.tolist()
			res = "{:.2f}".format(a[0]*100)
			#res = model_NRP.predict(pat)
		#print(model.predict([[1.8]]))
		#print(model.predict([[1.8]]))


	return render_template('result.html',output=res, disease=dis_name, name=name)



if __name__=='__main__':
   app.run()