import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import json

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator, **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
    
    def __len__(self):
        return len(self.generator)
    
    def __getitem__(self, idx):
        return self.generator[idx]

class TrainingManager:
    def __init__(self, save_dir='training_artifacts'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.state_path = os.path.join(save_dir, 'training_state.json')
        self.model_path = os.path.join(save_dir, 'best_model.keras')
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r') as f:
                    self.state = json.load(f)
                print(f"Loaded training state from epoch {self.state['current_epoch']}")
            except Exception as e:
                print(f"Error loading state: {e}")
                self.initialize_state()
        else:
            self.initialize_state()

    def initialize_state(self):
        self.state = {
            'current_epoch': 0,
            'best_val_accuracy': 0,
            'learning_rate': 0.001,
            'history': {
                'loss': [], 'accuracy': [], 'precision': [], 'recall': [],
                'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': []
            }
        }

    def save_state(self):
        with open(self.state_path, 'w') as f:
            json.dump(self.state, f)

    def create_model(self, input_shape, num_classes):
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        return model

    def load_or_create_model(self, input_shape, num_classes):
        if os.path.exists(self.model_path):
            try:
                print("Loading existing model...")
                return load_model(self.model_path)
            except Exception as e:
                print(f"Error loading model: {e}")
        
        print("Creating new model...")
        return self.create_model(input_shape, num_classes)

    def get_callbacks(self):
        class CustomCallback(tf.keras.callbacks.Callback):
            def __init__(self, manager):
                super().__init__()
                self.manager = manager

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                self.manager.state['current_epoch'] = epoch + 1
                
                # Update history
                for metric, value in logs.items():
                    if metric in self.manager.state['history']:
                        self.manager.state['history'][metric].append(float(value))
                
                # Save if improved
                val_accuracy = logs.get('val_accuracy', 0)
                if val_accuracy > self.manager.state['best_val_accuracy']:
                    self.manager.state['best_val_accuracy'] = val_accuracy
                    self.model.save(self.manager.model_path)
                    print(f"\nNew best model saved with val_accuracy: {val_accuracy:.4f}")
                
                # Save state
                self.manager.save_state()

        return [
            CustomCallback(self),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

def main():
    # Initialize training manager
    manager = TrainingManager()
    
    # Setup data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Setup directories and parameters
    train_dir = r'C:\Users\manup\Music\MAJOR PROJect - Copy\Skin Diseases\Skin Diseases\Skin Diseases\train'
    IMG_SIZE = 128
    batch_size = 32
    
    # Create data generators with custom wrapper
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Wrap generators
    train_dataset = DataGenerator(train_generator)
    validation_dataset = DataGenerator(validation_generator)
    
    # Get or create model
    model = manager.load_or_create_model(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=len(train_generator.class_indices)
    )
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=manager.state['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    try:
        print(f"\nStarting/Resuming training from epoch {manager.state['current_epoch']}")
        model.fit(
            train_dataset,
            epochs=100,
            initial_epoch=manager.state['current_epoch'],
            validation_data=validation_dataset,
            callbacks=manager.get_callbacks(),
            verbose=1
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Progress has been saved.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()      