# studyingDL

## Overview
  This repository involves deep learning model which I studied. It will be updated new models in **model** directory.   
  There are several data in **dataset** directory.
## Paper Reviews
  TF-C
  
  
## Expected  
  **Scheduled to update during Feb/23**
  * CNN
    * Resnet
    * Imagenet
  * Transformer
  * seq2seq
  * attention 
  * NLP
    * Word2Vec
    * RNN, LSTM, GRU
    * Transformer
    * Bert
    * GPT

**Learning**
  * Supervised learning
  * Unsupervised learning  
  * Representation learning
    * Contrastive learning
    * Generative learning

**insights**
  * loss function should be de differentiable
  * optimizing means flow to the lowest non-convex function ( Don't know where to flow )
  * log transformation is powerful
    * because of np.inf, use log(i+x) in log transformation ( i > 0 )
    
**CNN-LSTM**
  * use CNN for encoder and use lstm for decoder
  * each step of convolution result becomes each time steps of lstm
