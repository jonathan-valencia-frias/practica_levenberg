import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tensorflow as tf
import time
import levenberg_marquardt as lm
import tensorflow_datasets as tfds
import math
import numpy as np
opciones_neurona=["sigmoid","tanh","relu","softmax"]
modelo = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)), #1 - blanco y negro
    ])

def importar_dataset_mnist():
    datos,metadatos = tfds.load('mnist',as_supervised=True,with_info=True)  # type: ignore
    datos_entrenamiento,datos_pruebas = datos['train'],datos['test']  # type: ignore
    nombres_clases = metadatos.features['label'].names  # type: ignore

    def normalizar(imagenes, etiquetas):
        imagenes = tf.cast(imagenes, tf.float32)
        imagenes /= 255  # type: ignore
        return imagenes, etiquetas

    datos_entrenamiento = datos_entrenamiento.map(normalizar)
    datos_pruebas = datos_pruebas.map(normalizar)

    datos_entrenamiento = datos_entrenamiento.cache()
    datos_pruebas = datos_pruebas.cache()
    return datos_entrenamiento,datos_pruebas,metadatos
    
def importar_dataset_rmnist():
    datos,metadatos = tfds.load('mnist',as_supervised=True,with_info=True)  # type: ignore
    datos_entrenamiento,datos_pruebas = datos['train'],datos['test']  # type: ignore
    nombres_clases = metadatos.features['label'].names  # type: ignore

    def normalizar(imagenes, etiquetas):
        imagenes = tf.cast(imagenes, tf.float32)
        imagenes /= 255  # type: ignore
        return imagenes, etiquetas

    datos_entrenamiento = datos_entrenamiento.map(normalizar)
    datos_pruebas = datos_pruebas.map(normalizar)

    datos_entrenamiento = datos_entrenamiento.cache()
    datos_pruebas = datos_pruebas.cache()
    
datos_entrenamiento,datos_pruebas,metadatos = importar_dataset_mnist()


ventana = tk.Tk()
ventana.geometry("900x800")

resultados = tk.Tk()
resultados.geometry("900x800")
fig=plt.figure(1,figsize=(3,3))
ax=fig.add_subplot(111)
plt.xlim(0,2)
plt.ylim(0,100)
#plt.autoscale(False)

fig2=plt.figure(2)
ax2=fig2.add_subplot(111)
plt.xlim(0,1)
plt.ylim(0,1)
plt.autoscale(False)

    
def graficar_datos(historial):
    fig=plt.figure(1)
    plt.xlabel("epoca")
    plt.ylabel("error")
    x = [[0,100],[1,historial.history["loss"][0]]]
    plt.plot(x)
    fig.canvas.draw()
    

def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
    global metadatos
    fig=plt.figure(2)
    arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
  
    plt.imshow(img[...,0], cmap=plt.cm.binary)  # type: ignore

    etiqueta_prediccion = np.argmax(arr_predicciones)
    if etiqueta_prediccion == etiqueta_real:
        color = 'blue'
    else:
        color = 'red'
        
    nombres_clases = metadatos.features['label'].names  # type: ignore
  
    plt.xlabel("{} {:2.0f}% ({})".format(nombres_clases[etiqueta_prediccion],
                                100*np.max(arr_predicciones),
                                nombres_clases[etiqueta_real]),
                                color=color)
    fig.canvas.draw()
  
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
    fig=plt.figure(2)
    arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    grafica = plt.bar(range(10), arr_predicciones, color="#777777")
    plt.ylim([0, 1]) 
    etiqueta_prediccion = np.argmax(arr_predicciones)
  
    grafica[etiqueta_prediccion].set_color('red')
    grafica[etiqueta_real].set_color('blue')
    fig.canvas.draw()

def prueba_muchos():
    global modelo
    global datos_pruebas
    
    imagenes_prueba = 1
    etiquetas_prueba = 1
    predicciones = 1
    datos_pruebas = datos_pruebas.batch(32)
    for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
        imagenes_prueba = imagenes_prueba.numpy()
        etiquetas_prueba = etiquetas_prueba.numpy()
        predicciones = modelo.predict(imagenes_prueba)
        
    filas = 5
    columnas = 5
    num_imagenes = filas*columnas
    fig = plt.figure(2,figsize=(2*2*columnas, 2*filas))
    fig.clear()
    for i in range(num_imagenes):
        plt.subplot(filas, 2*columnas, 2*i+1)
        graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
        plt.subplot(filas, 2*columnas, 2*i+2)
        graficar_valor_arreglo(i, predicciones, etiquetas_prueba)  
    fig.canvas.draw()

def prueba_solo():
    global modelo
    global datos_pruebas
    
    nombres_clases = metadatos.features['label'].names  # type: ignore
    imagenes_prueba = 1
    etiquetas_prueba = 1
    datos_pruebas = datos_pruebas.batch(32)
    for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
        imagenes_prueba = imagenes_prueba.numpy()
        etiquetas_prueba = etiquetas_prueba.numpy()
        
    imagen = imagenes_prueba[int(numero_prueba.get())] # type: ignore #AL ser la variable imagenes_prueba solo tiene lo que se le puso en el bloque anterior heheh
    imagen = np.array([imagen])
    prediccion = modelo.predict(imagen)
    
    filas = 1
    columnas = 1
    num_imagenes = filas*columnas
    fig = plt.figure(2,figsize=(2*2*columnas, 2*filas))
    fig.clear()
    for i in range(num_imagenes):
        plt.subplot(filas, 2*columnas, 2*i+1)
        graficar_imagen(i, prediccion, etiquetas_prueba, imagen)
        plt.subplot(filas, 2*columnas, 2*i+2)
        graficar_valor_arreglo(i, prediccion, etiquetas_prueba)
        
    fig.canvas.draw()

    print("Prediccion: " + nombres_clases[np.argmax(prediccion[0])])

def Entrenar():
    global datos_entrenamiento
    global metadatos
    global historial
    global modelo
    if(numero_capas.get()=="1"):
        modelo.add(tf.keras.layers.Dense(int(entry_neuronas1.get()), activation=valor_neurona1.get()))  # type: ignore
        modelo.add(tf.keras.layers.Dense(10, activation=valor_salida.get()))
    else:
        modelo.add(tf.keras.layers.Dense(int(entry_neuronas1.get()), activation=valor_neurona1.get()))
        modelo.add(tf.keras.layers.Dense(int(entry_neuronas2.get()), activation=valor_neurona2.get()))
        modelo.add(tf.keras.layers.Dense(10, activation=valor_salida.get()))
    
    modelo = lm.ModelWrapper(tf.keras.models.clone_model(modelo))
    modelo.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=float(entry_learning_rate.get())),
        loss=lm.SparseCategoricalCrossentropy(from_logits=True),  # type: ignore
        metrics=['accuracy'])

    #Los numeros de datos en entrenamiento y pruebas (60k y 10k)
    num_ej_entrenamiento = metadatos.splits["train"].num_examples
    TAMANO_LOTE = 32
    if(tamaño_batch_radio.get()=="1"):
        TAMANO_LOTE = 1
        datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
    if(tamaño_batch_radio.get()=="2"):
        TAMANO_LOTE = int(entry_tamaño_batch.get())
        datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
    
    #Entrenar
    historial = modelo.fit(datos_entrenamiento, epochs=int(entry_epocas_maximas.get()), steps_per_epoch=math.ceil(num_ej_entrenamiento/TAMANO_LOTE))
    entry_epoca.set("1")
    entry_error.set(""+str(historial.history["loss"][0])+"")
    graficar_datos(historial)

grafica = tk.Frame(ventana)
grafica.grid(column=0,row=0)

canvas = FigureCanvasTkAgg(fig,master=grafica)
canvas.get_tk_widget().grid(column=0,row=0)

tk.Button(grafica,text="muchos",command=prueba_muchos).grid(column=0,row=1)

numero_prueba = tk.StringVar()
tk.Entry(grafica,textvariable=numero_prueba).grid(column=1,row=1)

tk.Button(grafica,text="solo",command=prueba_solo).grid(column=2,row=1)

prueba_m = FigureCanvasTkAgg(fig2,master=resultados)
prueba_m.get_tk_widget().grid(column=0,row=2)

"""prueba_i = FigureCanvasTkAgg(fig3,master=grafica)
prueba_i.get_tk_widget().grid(column=0,row=2,padx=3,pady=3)"""

datos = tk.Frame(ventana)
datos.grid(column=1,row=0,padx=5,pady=5)
tk.Label(datos,text="Configuracion de la arquitectura",font="50").grid(column=0,row=0,columnspan=2,padx=5,pady=5)


numero_capas=tk.StringVar()
numero_capas.set("1")
tk.Radiobutton(datos,text="1ra capa neurona",variable=numero_capas,value=1).grid(column=0,row=1,padx=5,pady=5)
entry_neuronas1=tk.StringVar()
tk.Entry(datos,textvariable=entry_neuronas1).grid(column=1,row=1,padx=5,pady=5)
valor_neurona1=tk.StringVar()
valor_neurona1.set("sigmoid")
tk.OptionMenu(datos,valor_neurona1,*opciones_neurona).grid(column=2,row=1,padx=5,pady=5)

tk.Radiobutton(datos,text="2ra capa neurona",variable=numero_capas,value=2).grid(column=0,row=2,padx=5,pady=5)
entry_neuronas2 = tk.StringVar()
tk.Entry(datos,textvariable=entry_neuronas2).grid(column=1,row=2,padx=5,pady=5)
valor_neurona2=tk.StringVar()
valor_neurona2.set("sigmoid")
tk.OptionMenu(datos,valor_neurona2,*opciones_neurona).grid(column=2,row=2,padx=5,pady=5)

tk.Label(datos,text="Capa de Salida",font="60").grid(column=0,row=3,padx=5,pady=5)
valor_salida=tk.StringVar()
valor_salida.set("sigmoid")
tk.OptionMenu(datos,valor_salida,*opciones_neurona).grid(column=1,row=3,padx=5,pady=5)

#############      hiperparametros
tk.Label(datos,text="Hiperparametros",font="60").grid(column=0,row=4,padx=5,pady=5)
tk.Label(datos,text="Leaning Rate",font="30").grid(column=0,row=5,padx=5,pady=5)
entry_learning_rate = tk.StringVar()
tk.Entry(datos,textvariable=entry_learning_rate).grid(column=1,row=5,padx=5,pady=5)
tk.Label(datos,text="Epocas Maximas",font="30").grid(column=0,row=6,padx=5,pady=5)
entry_epocas_maximas = tk.StringVar()
tk.Entry(datos,textvariable=entry_epocas_maximas).grid(column=1,row=6,padx=5,pady=5)

####### Tamaño de batch
tk.Label(datos,text="Tamaño de Batch",font="60").grid(column=2,row=4,padx=5,pady=5)
tamaño_batch_radio=tk.StringVar()
tamaño_batch_radio.set("1")
tk.Radiobutton(datos,variable=tamaño_batch_radio,text="estocastico",value=1).grid(column=2,row=5,padx=5,pady=5)
tk.Radiobutton(datos,variable=tamaño_batch_radio,text="mini-batch",value=2).grid(column=2,row=6,padx=5,pady=5)
entry_tamaño_batch = tk.StringVar()
tk.Entry(datos,textvariable=entry_tamaño_batch).grid(column=3,row=6,padx=5,pady=5)
    
tk.Button(datos,text="Entrenamiento",command=Entrenar,width=40).grid(column=0,row=7,columnspan=4)
tk.Label(datos,text="Resultados",font="70").grid(column=0,row=8,padx=5,pady=5)
tk.Label(datos,text="Epoca",font="30").grid(column=0,row=9,padx=5,pady=5)
entry_epoca=tk.StringVar()
entry_epoca.set("0")
tk.Entry(datos,font="70",state="readonly",textvariable=entry_epoca).grid(column=1,row=9,padx=5,pady=5)
tk.Label(datos,text="Error",font="30").grid(column=0,row=10,padx=5,pady=5)
entry_error=tk.StringVar()
entry_error.set("0")
tk.Entry(datos,font="70",state="readonly",textvariable=entry_error).grid(column=1,row=10,padx=5,pady=5)



ventana.mainloop()
