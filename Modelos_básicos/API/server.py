import numpy as np
from flask import Flask, request, jsonify
import pickle
import os
from pathlib import Path

app = Flask(__name__)
#Modelos treinados
absolutepath = Path('Machine_Learning_Production/Modelos_Básicos/modelo/dados/modelo_preco_casas.pkl').absolute()
model_casas = pickle.load(open(r'{}'.format(absolutepath),'rb'))

absolutepath = Path('Machine_Learning_Production/Modelos_Básicos/modelo/dados/modelo_diabetes.pkl').absolute()
model_diabetes = pickle.load(open(r'{}'.format(absolutepath),'rb'))


colunas_diabetes = ['NUMERO_GRAVIDEZ','GLICOSE','PRESSAO_SANGUINEA','ESPESSURA_PELE_TRICEPS','INSULINA','IMC','FUNCAO_DIABETES','IDADE']
colunas_preco_casas = ['tamanho','ano','garagem']

@app.route("/modelos")
def verifica_api_online():
  return "API ONLINE v1.0", 200

@app.route('/modelos/diabetes/previsao', methods=['POST'])
def predict():
  dados = request.get_json()
  dados_preparadados = [dados[col] for col in colunas_diabetes]
  prediction = model_diabetes.predict([dados_preparadados])
  return jsonify(Diabetes = int(prediction[0]))

@app.route('/modelos/aluguel/previsao', methods=['POST'])
def aluguel():
    dados = request.get_json(force=True)
    dados_preparados = [dados[col] for col in colunas_preco_casas]
    preco = model_casas.predict([dados_preparados])
    return jsonify(preco = int(preco[0]))


if __name__ == "__main__": 
  port = int(os.environ.get("PORT", 5000))
  app.run(host='127.0.0.1', port=port)