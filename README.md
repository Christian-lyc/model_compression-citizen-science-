# A Comparison of Deep Neural Network Compression for Citizen-Driven Tick and Mosquito Surveillance.

## Abstract
Citizen science has emerged as an effective approach for infectious disease surveillance. With advancements in machine learning, entomologists can now be relieved from the labor-intensive task of species identification. However, deploying machine learning models on mobile devices presents challenges due to constraints on battery life and memory capacity. In this study, we explore the potential of various model compression techniques for deploying machine learning models on resource-limited devices, enabling low-energy consumption and on-device processing for disease surveillance in remote or low-resource settings. We have compared two main-stream model compression techniques, pruning and quantization on various mobile devices. Our findings indicate that quantization methods outperform pruning methods in terms of efficiency. Furthermore, we propose to integrate structured and unstructured pruning to enhance model performance while addressing key constraints of mobile deployment.

Link to the research article: [to appear soon]

Keywords: deep learning, pruning, quantization, object detection, tick and mosquito citizen science
## Authors
Yichao Liu[1], Emmanuel Dufour[2],[3],[4], Peter Fransson[1], Joacim Rocklöv[1], [5], [6]

[1]: Interdisciplinary Center for Scientific Computing, Heidelberg University, 69120, Germany  
[2]: African Institute for Mathematical Sciences, Muizenberg, South Africa  
[3]: African Institute for Mathematical Sciences Research and Innovation Centre, Kigali, Rwanda  
[4]: Department of Mathematical Sciences, Stellenbosch University, Stellenbsoch, South Africa  
[5]: Heidelberg Institute of Global Health, Heidelberg University Hospital, 69120, Germany  
[6]: Department of Epidemiology and Global Health, Umeå University, Sweden  

## Demo

A quick 3 minute demo for Algorithm 1 is available here on <a href="https://colab.research.google.com/drive/1Lq1rPGPg3viidtC5pXP0AphMx-RxfTiA?usp=sharing">Google Colab demo</a>.

### Libraries:
Ultralytics==8.3.93  
Python>=3.10.18  
Torchinfo==1.8.0  
Torch-pruning==1.5.1  


Note: in order to run the code, some adaption need to be done for Ultralytics package, see modification folder.
