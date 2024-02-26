from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, concatenate
from tensorflow.keras.models import Model


def Multi_MLP(n_features, n_units, dropout): 
    input1 = Input(shape = n_features)
    shared = Dense(n_units, 'relu')(input1)
    shared = BatchNormalization()(shared)
    shared = Dropout(dropout)(shared)
    shared = Dense(n_units, 'relu')(shared)
    shared = BatchNormalization()(shared)
    shared = Dropout(dropout)(shared)
    
    out1 = Dense(n_units, 'relu')(shared)
    out1 = Dense(1, 'sigmoid', name = 'out1')(out1)
    out2 = Dense(n_units, 'relu')(shared)
    out2 = Dense(1, 'sigmoid', name = 'out2')(out2)
    out3 = Dense(n_units, 'relu')(shared)
    out3 = Dense(1, 'sigmoid', name = 'out3')(out3)
    out4 = Dense(n_units, 'relu')(shared)
    out4 = Dense(1, 'sigmoid', name = 'out4')(out4)
    out5 = Dense(n_units, 'relu')(shared)
    out5 = Dense(1, 'sigmoid', name = 'out5')(out5)
    
    combined1 = concatenate([out1, out2, out3, out4, out5])
    combined1 = Dense(5, 'relu')(combined1)
    
    out = Dense(1, 'sigmoid', name = 'out')(combined1)
    
    model = Model(inputs = input1, outputs = [out1, out2, out3, out4, out5, out])
    return model 