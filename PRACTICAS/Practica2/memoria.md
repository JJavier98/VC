---
title: "Práctica 2"
author: [José Javier Alonso Ramos]
date: "Curso: 2019 - 2020"
subject: "Markdown"
keywords: [Markdown, Example]
subtitle: "Redes neuronales convolucionales"
lang: "es"
titlepage: true
titlepage-rule-height: 1
logo: "/home/jjavier/GII/logoUGR/3.png"
logo-width: 300
toc: TRUE
toc-own-page: TRUE
titlepage-color: e0fbff
titlepage-text-color: 110406
---

# Ejercicio 1

Implementamos un modelo base para trabajar con el conjunto de imágenes **CIFAR100** al que llamamos _BaseNet_.  
El conjunto inicial se ha modificado para que el número de **clases sea 25** y se ha dividido en **12500 imágenes para realizar el entrenamiento** del modelo y **2500 para el conjunto de prueba** (test). De esas 12500 imágenes de entrenamiento reservaremos un 10% **(1250) para realizar la validación** del modelo según lo vamos entrenando.  
El modelo _BaseNet_ viene definido por la siguente configuración de capas:  
![Capas](data-images/capas-BaseNet.png)  

Acontinuación mostraremos algunos resultados obtenidos con distintos parámetros en en modelo.  
- **Optimizador:** SGD
- **Épocas:** 20
- **Batch size:** 64
- **Precisión final en test:** 0.3176000118255615
- **Pérdida final en test:** 2.35547516746521
![Accuracy](data-images/SGD-20-acc.png)
![Loss](data-images/SGD-20-loss.png)
- **Optimizador:** SGD
- **Épocas:** 40
- **Batch size:** 64
- **Precisión final en test:** 0.38119998574256897
- **Pérdida final en test:** 2.1934714401245117
![Accuracy](data-images/SGD-40-acc.png)
![Loss](data-images/SGD-40-loss.png)
- **Optimizador:** Adam
- **Épocas:** 20
- **Batch size:** 64
- **Precisión final en test:** 0.45680001378059387
- **Pérdida final en test:** 1.9854614252090454
![Accuracy](data-images/Adam-20-acc.png)
![Loss](data-images/Adam-20-loss.png)
- **Optimizador:** Adam
- **Épocas:** 40
- **Batch size:** 64
- **Precisión final en test:** 0.4844000041484833
- **Pérdida final en test:** 1.8760112937927247
![Accuracy](data-images/Adam-40-acc.png)
![Loss](data-images/Adam-40-loss.png)