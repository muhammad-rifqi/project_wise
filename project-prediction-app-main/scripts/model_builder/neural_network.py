# scripts/model_builder/neural_network.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import lecun_normal
import numpy as np
from datetime import datetime

class NeuralNetworkTrainer:
    def __init__(self):
        self.model = None
        self.embedding_model = None
        self.history = None
        
    def _log(self, message):
        """Log internal dengan timestamp"""
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] NeuralNetwork: {message}")
        
    def build_model(self, input_dim):
        """Membangun model ANN dengan Functional API"""
        self._log("Membangun arsitektur ANN")
        
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Hidden layers
        x = Dense(128, activation='selu', kernel_initializer=lecun_normal())(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(64, activation='selu', kernel_initializer=lecun_normal())(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Dense(32, activation='selu', kernel_initializer=lecun_normal())(x)
        x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        # Embedding layer
        embedding = Dense(16, activation='selu', kernel_initializer=lecun_normal(), 
                         name='embedding_layer')(x)

        # Output layer
        outputs = Dense(1, activation='sigmoid')(embedding)

        # Buat model utama
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Buat model embedding terpisah
        self.embedding_model = Model(inputs=inputs, outputs=embedding)

        # Compile model utama
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self._log("Arsitektur ANN berhasil dibangun")
        return self.model, self.embedding_model
        
    def train_model(self, X_train, y_train, X_test, y_test):
        """Training ANN model"""
        self._log("Memulai training ANN")
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=0
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0
        )

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=300,
            batch_size=16,
            callbacks=[early_stopping, lr_scheduler],
            verbose=0
        )
        
        self._log(f"Training ANN selesai - {len(self.history.history['loss'])} epochs")
        return self.history
        
    def extract_embeddings(self, X_data):
        """Ekstrak embeddings dari ANN"""
        self._log("Ekstraksi embeddings dari ANN")
        
        embeddings = self.embedding_model.predict(X_data)
        self._log(f"Embeddings shape: {embeddings.shape}")
        
        return embeddings