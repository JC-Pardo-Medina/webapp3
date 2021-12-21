import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template

app = Flask(__name__)

MODEL_PATH = 'models/mobilenet_final.hd5'

model = tf.compat.v1.keras.experimental.load_from_saved_model(MODEL_PATH)
model.make_predict_function()

class_names=['Agaricus campestris',  'Amanita bisporigera',  'Amanita caesarea',  'Amanita muscaria',  'Boletus edulis',  'Boletus satanas',  'Cantharellus cibarius',  'Cortinarius orellanus',  'Gyromitra esculenta',
 'Hygrophorus russula',  'Lactarius deliciosus',  'Lycoperdon perlatum',  'Omphalotus olearius',  'Pleurotus eryngii',  'Russula mariae']
lista_textos=["\nLa Agaricus Campestris, también conocido como champiñon silvestre es\033[1m COMESTIBLE \033[0m.\n\033[1mHábitat\033[0m: Praderas, campo abierto, pastizales abonados por ganado, jardines y pinares.\n\033[1mTemporada\033[0m: Primavera, Abril-Mayo. Posible segundo brote a final de Verano y Otoño. Esta especie de champiñón, no se cultiva, pero es muy común. Sale en círculos o grupos numerosos.\n",
              "\nLa Amanita Bisporigera es conocida como el Ángel de la muerte, \033[1mNO ES COMESTIBLE, PUEDE SER MORTAL\033[0m.\n",
              "\nLa Amanita Caesarea es también conocida como Huevo del Rey es,\033[1m COMESTIBLE\033[0m.\n\033[1mHábitat\033[0m: Claros soleados y zonas orientadas al sur de bosques de frondosas: roble (sobre todo), encina, haya, castaño. También en matorral Mediterráneo.\n\033[1mTemporada\033[0m: Si hay humedad, puede aparecer ya a finales de Agosto. Dura todo el Otoño, aunque se trata de una especie más bien escasa. Sale en pequeños grupos.\n",
              "\nLa Amanita Muscaria también conocida como Matamoscas es una seta,\033[1m TÓXICA\033[0m.\n\033[1mHábitat\033[0m: Bosques de coníferas (alerces, pinos, abedules), encinares, castañares. También en hayedos, jarales y bosque de frondosas.\n\033[1mTemporada\033[0m: Otoño. Especie muy abundante.\n",
              "\nLos Boletus Edulis, también conocidos como Boletos de Calabaza son \033[1mCOMESTIBLES\033[0m. Son excelentes en múltiples preparaciones (incluso en crudo).\n\033[1mHábitat\033[0m: Pinares, castañares, hayedos y robledales.\n\033[1mTemporada\033[0m: Verano - Otoño, siendo muy abundante en Otoños húmedos.\n",
              "\nLos Boletus Satanás también conocidos como Corvall del Dimoni, son \033[1mTÓXICOS\033[0m, su ingesta produce Síndrome Gastrointestinal.\n\033[1mHábitat\033[0m: Prefiere la sombra de bosques caducifolios húmedos de castaños y robles. También crece entre alcornoques y madroños. Suelos calizos.\n\033[1mTemporada\033[0m:Final del Verano o inicio de Otoño.\n",
              "\nLa Cantharellus cibarius también conocida como Rebozuelo es,\033[1m COMESTIBLE\033[0m, y tiene alto valor culinario.\n\033[1mHábitat\033[0m: Hayas, castaños, robles, encinas, jaras, etc. También en campas, herbales, helechos… Normalmente en grupos numerosos.\n\033[1mTemporada\033[0m: Primavera a Otoño.\n",
              "\nLa Cortinarius orellanus también conocido como Cortinario de Montaña, es\033[1m TÓXICA\033[0m, pudiendo llegar a ser \033[1mMORTAL\033[0m.\n\033[1mHábitat\033[0m: Robledal, latifolios y más raramente en coníferas de montaña.\n\033[1mTemporada\033[0m: Verano - Otoño.\n",
              "<p>\nLa Gyromitra esculenta también conocida como falsa colmenilla, es <b>TÓXICA</b>.\nAunque esculenta=comestible, es mentira, produce Síndrome Giromitriano. Mortal en crudo. Para volatilizar la Giromitrina, debe ser desecada, esperar 6 meses hasta cocerla y retirar el agua.\nO hervirla mucho tiempo en agua. Pero quedan toxinas precancerígenas.\n\033[1mHábitat\033[0m: Coníferas sobre todo, aunque también puede verse en caducifolios y en matorrales. Busca suelos con humus y en altitud (> 800m.).\n\033[1mTemporada\033[0m: Primavera, exclusivamente.\n</p>",
              "\nLa Hygrophorus russula también conocido como Rovellón Escarlata, es\033[1m COMESTIBLE\033[0m pero tiene un sabor particularmente amargo. Apreciada en la cocina Catalana.\n\033[1mHábitat\033[0m: Bajo bosques caducifolios o mixtos. Roble melojo y brezo. Lugares húmedos y sombríos. Climas cálidos.\n\033[1mTemporada\033[0m: Verano - Otoño. Sale en grupo, bajo la hojarasca. Poco frecuente.\n",
              "\nEl Lactarius deliciosus conocido popularmente como Níscalo es \033[1m COMESTIBLE\033[0m.\nPuede confundirse con otros tipos de níscalos NO COMESTIBLES, para cerciorarte realiza un corte a la seta si su látex es anaranjado PUEDES comerla.\n\033[1mHábitat\033[0m: Se asocia con las raíces de los árboles en bosques de coníferas o mixtos, especialmente los pinares de Insignis, en el país vasco y litoral Cantábrico.\n\033[1mTemporada\033[0m: Verano-Otoño.\n",
              "\nEl Lycoperdon perlatum conocido popularmente como Pedo de Lobo es\033[1m COMESTIBLE CON PRECAUCIONES\033[0m, sólo consumible si es jóven y cocinada.\n\033[1mHábitat\033[0m: Todo tipo de bosques, con predilección por coníferas. En todos los ecosistemas.\n\033[1mTemporada\033[0m: En todas las estaciones.\n",
              "\nLa Omphalotus Olearios conocida también como Seta del Olivo es\033[1m TÓXICA\033[0m. Genera Síndrome Gastrointestinal Grave y alucinaciones.\n\033[1mHábitat\033[0m: Olivos. Pero también sobre jaras pringosas, encinas, alcornoques, acacias, etc. Especie lignícola.\n\033[1mTemporada\033[0m: Otoño.\n",
              "\nLa Pleurotus eryngii conocida como Seta de cardo es\033[1m COMESTIBLE\033[0m y de valor gastronómico.\n\033[1mHábitat\033[0m: Campos abiertos, praderas, barbechos, bordes de caminos, zonas planas desforestadas. Especie sometida a cultivo intensivo. Abunda en Castilla.\n\033[1mTemporada\033[0m: Primavera y Otoño.\n",
              "\nLa Russula mariae se considera\033[1m COMESTIBLE\033[0m, aunque de mediocre calidad.\n\033[1mHábitat\033[0m: Bosques mixtos de coníferas (pino piñonero) y planifolios (roble, encina).\n\033[1mTemporada\033[0m: Verano – Otoño. Ambiente mediterráneo.\n"
              ]
dict_setas = dict(zip(class_names, lista_textos))

@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def predict():
    image_path="./static/uploads/"
    carpeta = os.listdir(image_path)
    if len(os.listdir(image_path)) != 0:
        os.remove(os.path.join(image_path,carpeta[0]))
    setapredecible = request.files['imagefile']
    image_path="./static/uploads/"+setapredecible.filename
    setapredecible.save(image_path)
    
    img=tf.keras.preprocessing.image.load_img((image_path), target_size=(224, 224, 3))

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions= model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    fst_label= class_names[np.argmax(score)]
    fst_prob= round(np.max(score)*100,2)
    snd_label=class_names[np.argsort(score)[-2:][0]] 
    snd_prob=round(np.sort(score)[-2:][0]*100,2)
    fst_texto=dict_setas.get(fst_label)
    snd_texto=dict_setas.get(snd_label)
    if fst_prob < 30:
        classification = 'No puedo dar una estimación cierta, esta seta no esta en mi base de datos, pero podría ser una %s con un (%.2f%%) de probabilidad' % (fst_label, fst_prob)
    else:
        if fst_prob - snd_prob <15:
            classification = 'El resultado no está muy claro. Podría ser una %s con una probabilidad de (%.2f%%): %s También podría ser una %s con una probabilidad del (%.2f%%): %s' % (fst_label, fst_prob, fst_texto, snd_label, snd_prob, snd_texto)

        else:
            classification = 'El resultado deja muy poco lugar a la duda. Seguramente sea una %s con una probabilidad de (%.2f%%): %s La segunda opción más probable es una %s con una probabilidad del (%.2f%%).' % (fst_label, fst_prob, fst_texto, snd_label, snd_prob)



    return render_template("home.html", filename=image_path, prediction=classification)

if __name__ == '__main__':
    app.run(debug=True)