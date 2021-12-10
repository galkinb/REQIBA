##This is where we set up and train the IPNN based on the dataset we generated
rm(list=ls())
library(spatstat)
library(hypergeo)
library(keras)


##Load the dataset
load(file="IPNNDataset.RData")
    
    
#model generation and training
    
    model <- keras_model_sequential()
    model %>%
      layer_dense(units=10,input_shape = 3) %>%
     layer_dense(units = 20, activation = 'linear') %>%
      layer_dense(units = 20, activation = 'tanh') %>%
      layer_dense(units = 1, activation = 'linear')
    
    
    
    model %>% compile(
      optimizer = 'adamax', 
      loss = 'mse',
      metrics = c('mae')
    )
    
    #Take the training entries from the dataset and normalise them
    train_images = results[1:480000,1:3]
    train_images[,1]= train_images[,1]/1000
    train_images[,3]=train_images[,3]/300
    train_labels = log10(results[1:480000,c(4)])
    test_images = results[480001:500000,1:3]
    test_images[,1]= test_images[,1]/1000
    test_images[,3]=test_images[,3]/300
    test_labels = log10(results[480001:500000,c(4)])
  
    
    
    model %>% fit(train_images, train_labels, epochs = 100, validation_data=list(test_images, test_labels),batch_size=100)
    
    score <- model %>% evaluate(test_images, test_labels)
    
    cat('Test loss:', score$loss, "\n")
    cat('Test mean error:', score$mae, "\n")
    
    
    model %>% save_model_weights_hdf5("IPNNweights.h5")
    