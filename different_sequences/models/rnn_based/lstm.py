#imports
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Concatenate, Embedding, Flatten, Activation, Dropout
from sklearn.model_selection import KFold
from tensorflow.python.client import device_lib
from tensorflow import keras
warnings.filterwarnings('ignore')

#IN OUR EXPERIMENT WE SET MAXLENGTH TO 
#FOLLOWING VALUES 2, 5-15, 100, 200, 300, 400
MAXLENGTH = 100
EMBEDDING_DIM = 128
DENSE_NEURON = 16
LSTM_NEURON = 32
CHAPTER_SIZE = 38
SUB_CHAPTER_SIZE = 223
QUESTION_SIZE = 1069

# 5 fold cross validation
#HERE WE ONLY USE chapter
#sub_chapter and question
#without additional features
import torch
X = np.array(grouped_data.keys())
kfold = KFold(n_splits=5, shuffle=True)
train_losses = list()
train_aucs = list()
val_losses = list()
val_aucs = list()
train_eval = list()
test_eval = list()
for train, test in kfold.split(X):
    users_train, users_test =  X[train], X[test]
    n = len(users_test)//2
    users_test, users_val = users_test[:n], users_test[n: ]
    train_data_space = SPACE_DATASET(grouped_data[users_train], MAXLENGTH)
    val_data_space = SPACE_DATASET(grouped_data[users_val], MAXLENGTH)
    test_data_space = SPACE_DATASET(grouped_data[users_test], MAXLENGTH)
    
    #construct training input
    train_chapter=[]
    train_sub_chapter=[]
    train_question = []
    train_labels=[]
    for i in range(len(users_train)):
        user = train_data_space.__getitem__(i)
        train_chapter.append(user[0])
        train_sub_chapter.append(user[1]) 
        train_question.append(user[2])
        train_labels.append(user[3])
    train_chapter = np.array(train_chapter)
    train_sub_chapter = np.array(train_sub_chapter)
    train_question = np.array(train_question)
    train_labels= np.array(train_labels)[..., np.newaxis]

    #construct validation input
    val_chapter=[]
    val_sub_chapter=[]
    val_question = []
    val_labels=[]
    for i in range(len(users_val)):
        user = val_data_space.__getitem__(i)
        val_chapter.append(user[0])
        val_sub_chapter.append(user[1]) 
        val_question.append(user[2])
        val_labels.append(user[3])
    val_chapter = np.array(val_chapter)
    val_sub_chapter = np.array(val_sub_chapter)
    val_question = np.array(val_question)
    val_labels= np.array(val_labels)[..., np.newaxis]

    # construct test input
    test_chapter=[]
    test_sub_chapter=[]
    test_question=[]
    test_labels=[]
    for i in range(len(users_test)):
        user = test_data_space.__getitem__(i)
        test_chapter.append(user[0])
        test_sub_chapter.append(user[1]) 
        test_question.append(user[2])
        test_labels.append(user[3])
    test_chapter = np.array(test_chapter)
    test_sub_chapter = np.array(test_sub_chapter)
    test_question = np.array(test_question)
    test_labels= np.array(test_labels)[..., np.newaxis]

    # define loss function and evaluation metrics
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    acc = tf.keras.metrics.Accuracy()
    auc = tf.keras.metrics.AUC()

    #using masked metrics and loss 
    #since we make use of padding
    #when the number of interactions
    #is less than sequence size
    def masked_bce(y_true, y_pred):
        flat_pred = y_pred
        flat_ground_truth = y_true
        label_mask = tf.math.not_equal(flat_ground_truth, -1)
        return bce(flat_ground_truth, flat_pred, sample_weight=label_mask)

    def masked_acc(y_true, y_pred):
        flat_pred = y_pred
        flat_ground_truth = y_true
        flat_pred = (flat_pred >= 0.5)
        label_mask = tf.math.not_equal(flat_ground_truth, -1)
        return acc(flat_ground_truth, flat_pred, sample_weight=label_mask)

    def masked_auc(y_true, y_pred):
        flat_pred = y_pred
        flat_ground_truth = y_true
        label_mask = tf.math.not_equal(flat_ground_truth, -1)
        return auc(flat_ground_truth, flat_pred, sample_weight=label_mask)

    
    
    # input layer
    input_chap = tf.keras.Input(shape=(MAXLENGTH))
    input_sub_chap = tf.keras.Input(shape=(MAXLENGTH))
    input_ques =  tf.keras.Input(shape=(MAXLENGTH))

    # embedding layer for categorical features
    embedding_chap = Embedding(input_dim = CHAPTER_SIZE, output_dim = EMBEDDING_DIM)(input_chap)
    embedding_sub_chap = Embedding(input_dim = SUB_CHAPTER_SIZE, output_dim = EMBEDDING_DIM)(input_sub_chap) 
    embedding_ques = Embedding(input_dim = QUESTION_SIZE, output_dim = EMBEDDING_DIM)(input_ques)       
    
   
    # define LSTM layers
    LSTM_chap = LSTM(LSTM_NEURON, input_shape = (None, EMBEDDING_DIM),return_sequences = True)(embedding_chap)
    LSTM_sub_chap = LSTM(LSTM_NEURON, input_shape = (None, EMBEDDING_DIM),return_sequences = True)(embedding_sub_chap)
    LSTM_ques = LSTM(LSTM_NEURON, input_shape = (None, EMBEDDING_DIM),return_sequences = True)(embedding_ques)
    LSTM_output = tf.concat([LSTM_chap, LSTM_sub_chap, LSTM_ques], axis = 2)

    dense1 = Dense(256, input_shape = (None, 3*EMBEDDING_DIM), activation='relu')(LSTM_output)
    dropout1 = Dropout(0.1)(dense1)
    dense2 = Dense(64, input_shape = (None, 256), activation='relu')(dropout1)
    dropout2 = Dropout(0.1)(dense2)
    pred = Dense(1, input_shape = (None, 64), activation='sigmoid')(dropout2)

    model = tf.keras.Model(
        inputs=[input_chap, input_sub_chap,input_ques],
        outputs=pred,
        name='lstm_model'
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    opt_adam = Adam(learning_rate = 0.005)
    model.compile(
        optimizer=opt_adam,
        loss= masked_bce,
        metrics = [masked_acc, masked_auc]
    )

    history = model.fit(
      [train_chapter, train_sub_chapter, train_question],
      train_labels,
      batch_size = 64,
      epochs = 100,
      validation_data=([val_chapter, val_sub_chapter, val_question], val_labels),
      callbacks=[callback]
    )
    val_losses.append(list(history.history['val_loss']))
    train_losses.append(list(history.history['loss']))
    val_aucs.append(list(history.history['val_masked_auc']))
    train_aucs.append(list(history.history['masked_auc']))
    train_score = model.evaluate([train_chapter, train_sub_chapter, train_question], train_labels)
    train_eval.append(train_score)
    test_score = model.evaluate([test_chapter, test_sub_chapter, test_question], test_labels)
    test_eval.append(test_score)
    print("Test: ", test_score)
    
    def reset_weights(model):
        for layer in model.layers: 
            if isinstance(layer, tf.keras.Model):
                reset_weights(layer)
                continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
          # find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))
    reset_weights(model)
    

t_eval = np.array(test_eval)
print("test avg loss: ", np.mean(t_eval[:, 0]), "+/-" ,np.std(t_eval[:, 0]))
print("test avg acc: ", np.mean(t_eval[:, 1]),  "+/-" ,np.std(t_eval[:, 1]))
print("test avg auc: ", np.mean(t_eval[:, 2]), "+/-" ,np.std(t_eval[:, 2]))


t_eval = np.array(train_eval)
print("train avg loss: ", np.mean(t_eval[:, 0]), "+/-" ,np.std(t_eval[:, 0]))
print("train avg acc: ", np.mean(t_eval[:, 1]),  "+/-" ,np.std(t_eval[:, 1]))
print("train avg auc: ", np.mean(t_eval[:, 2]), "+/-" ,np.std(t_eval[:, 2]))