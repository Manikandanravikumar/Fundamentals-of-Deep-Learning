# Fundamentals-of-Deep-Learning
  In this exercise, you will train a model to recognize fresh and rotten fruits. The dataset comes from Kaggle, a great place to go if you're interested in starting a project after this class. The dataset structure is in the data/fruits folder. There are 6 categories of fruits: fresh apples, fresh oranges, fresh bananas, rotten apples, rotten oranges, and rotten bananas. This will mean that your model will require an output layer of 6 neurons to do the categorization successfully. You'll also need to compile the model with categorical_crossentropy, as we have more than two categories.

# Steps in recognize fresh fruits and rotten fruits
  1.Load ImageNet Base Model
  2.Freeze Base Model
  3.Add Layers to Model
  4.Compile Model
  5.Augment the Data
  6.Load Dataset
  7.Train the Model
  8.Unfreeze Model for Fine Tuning
  9.Evaluate the Model
  10.Run the Assessment
  
# 1. Load ImageNet Base Model
      
      from tensorflow import keras base_model = keras.applications.VGG16(weights="imagenet", input_shape=(224, 224, 3), include_top=False)

# 2. Freeze Base Model
      
      base_model.trainable = False       

# 3. Add Layers to Model
      
      # Create inputs with correct shape
      inputs = base_model.input

      x = base_model(inputs, training=False)

      # Add pooling layer or flatten layer
      x = keras.layers.Flatten()(base_model.output)

      # Add final dense layer
      outputs = keras.layers.Dense(6, activation = 'softmax')(x)

      # Combine inputs and outputs to create model
      model = keras.Model(inputs=inputs, outputs=outputs)
      
      model.summary()

# 4. Compile Model
      
      model.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer='rmsprop')

# 5.Augment the Data

      from tensorflow.keras.preprocessing.image import ImageDataGenerator

      datagen = ImageDataGenerator( rotation_range=20,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      rescale=1./255,  
      preprocessing_function=keras.applications.vgg16.preprocess_input)
      
# 6. Load Dataset

      # load and iterate training dataset
      train_path='data/fruits/train/'  #on a travaillé sur le notebook local de Kaggle test_path='data/fruits/valid/'
      train_it = datagen.flow_from_directory(train_path, target_size=[224,224], color_mode='rgb', class_mode="categorical", batch_size = 32)
      
      # load and iterate test dataset
      test_it = datagen.flow_from_directory(test_path, target_size=[224,224], color_mode='rgb', class_mode="categorical", batch_size = 32)
      
# 7. Train the Model
  
      model.fit(train_it,
          validation_data=test_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=test_it.samples/test_it.batch_size,
          epochs=10)
# 8. Unfreeze Model for Fine Tuning

      # Unfreeze the base model
      base_model.trainable = True

      # Compile the model with a low learning rate
      model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),
              loss =keras.losses.BinaryCrossentropy(from_logits=True) , metrics =[keras.metrics.BinaryAccuracy()])
      
      model.fit(train_it,
          validation_data=test_it,
          steps_per_epoch=train_it.samples/train_it.batch_size,
          validation_steps=test_it.samples/test_it.batch_size,
          epochs=10)
          
# 9. Evaluate the Model

      model.evaluate(test_it, steps=test_it.samples/test_it.batch_size)

# 10. Run the Assessment
      
      from run_assessment import run_assessment
      run_assessment(model, test_it)
      
