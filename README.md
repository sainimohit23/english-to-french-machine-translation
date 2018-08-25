# Neural Machine Translation (Attention Model)

In this project, I have built a language translation model called `seq2seq model or encoder-decoder model` model in TensorFlow. The objective of the model is translating English sentences to French sentences.
The model uses `attention mechanism` to learn. Model is written on latest tensorflow api.


# Brief Overview of the Contents
### Data preprocessing
In this section, you will see how to get the data, how to create `lookup table`, and how to `convert raw text to index based array` with the lookup table.
Use the file data_preprocess.py to preprocess data and save it on the disk.


![conversion](https://user-images.githubusercontent.com/26195811/44616221-799a7c00-a869-11e8-9c90-ee86233731e9.png)


### Build model
In short, this section will show how to `define the Seq2Seq model in TensorFlow`. The below steps (implementation) will be covered.
- __(1)__ define input parameters to the encoder model
  - `enc_dec_model_inputs`
- __(2)__ build encoder model
  - `encoding_layer`
- __(3)__ define input parameters to the decoder model
  - `enc_dec_model_inputs`, `process_decoder_input`, `decoding_layer`
- __(4)__ build decoder model for training
  - `decoding_layer_train`
- __(5)__ build decoder model for inference
  - `decoding_layer_infer`
- __(6)__ put (4) and (5) together
  - `decoding_layer`
- __(7)__ connect encoder and decoder models
  - `seq2seq_model`
- __(8)__ train and estimate loss and accuracy

Model structure:


![atten](https://user-images.githubusercontent.com/26195811/44616233-b8303680-a869-11e8-80e5-270d571228fe.png)


Graph created by tensorboard:


![a](https://user-images.githubusercontent.com/26195811/44616445-65587e00-a86d-11e8-8375-9c0cf985b938.png)



### Training
This section is about putting previously defined functions together to `build an actual instance of the model`. Furthermore, it will show how to `define cost function`, how to `apply optimizer` to the cost function, and how to modify the value of the gradients in the TensorFlow's optimizer module to perform `gradient clipping`.







## *************************************************************************************






This type of model takes too long to train on high end GPU and I don't have enough resources to train it for long. I have trained 
this model for 40 epochs on google colab GPU and you can download pre-trained variables from [here](https://drive.google.com/open?id=1wBfBJn1VKmOzQpP8W0vRa8Crc9QjePNd).


Here are some of the outputs that model gave while training:
![pred](https://user-images.githubusercontent.com/26195811/44616335-bcf5ea00-a86b-11e8-8cab-2ddb9e16c0a1.png)





(NOTE: Use .ipynb file to train model on google colab gpu)
