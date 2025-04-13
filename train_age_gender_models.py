import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


IMAGE_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 50 



BASE_DATA_DIR = 'dataset/age_gender_dataset' 

MODEL_SAVE_DIR = 'saved_age_gender_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

AGE_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'age_group_model.h5')
GENDER_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, 'gender_model.h5')


AGE_BINS = [0, 18, 30, 45, 60, 120] 
AGE_LABELS = ['0-18', '19-30', '31-45', '46-60', '61+']
NUM_AGE_CLASSES = len(AGE_LABELS)

GENDER_LABELS = ['Male', 'Female'] 
NUM_GENDER_CLASSES = len(GENDER_LABELS) 



def load_data_from_directory(data_dir):
    image_paths = []
    ages = []
    genders = []
    print(f"Scanning directory: {data_dir}")
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        return [], [], []

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    count = 0
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(valid_extensions):
            try:
                parts = filename.split('_')
                if len(parts) >= 3:
                    age = int(parts[0])
                    gender = int(parts[1]) 
                    if age < 0 or age > 116 or gender not in [0, 1]:
                         continue

                    image_paths.append(os.path.join(data_dir, filename))
                    ages.append(age)
                    genders.append(gender)
                    count += 1
               
            except ValueError:
               
                continue
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    print(f"Found {count} valid images.")
    if count == 0:
         print("Error: No valid images found. Check directory path and file naming.")
    return image_paths, ages, genders


image_paths, ages_raw, genders_raw = load_data_from_directory(BASE_DATA_DIR)

if not image_paths:
    print("!!! CRITICAL ERROR: No image data loaded. Exiting training script.")
    print(f"!!! Please ensure the directory '{BASE_DATA_DIR}' exists and contains correctly named images (e.g., 'age_gender_...jpg').")
    exit()


age_groups = np.digitize(ages_raw, bins=AGE_BINS[1:], right=True) 


age_groups_onehot = tf.keras.utils.to_categorical(age_groups, num_classes=NUM_AGE_CLASSES)


genders_np = np.array(genders_raw)



from sklearn.model_selection import train_test_split


X_train_paths, X_val_paths, \
y_age_train, y_age_val, \
y_gender_train, y_gender_val = train_test_split(
    image_paths, age_groups_onehot, genders_np,
    test_size=0.2, random_state=42, stratify=genders_np 
)

print(f"Training samples: {len(X_train_paths)}, Validation samples: {len(X_val_paths)}")


class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, age_labels, gender_labels, batch_size, target_size, shuffle=True):
        self.image_paths = image_paths
        self.age_labels = age_labels
        self.gender_labels = gender_labels
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]

        X = np.empty((self.batch_size, *self.target_size, 3)) 
        y_age = np.empty((self.batch_size, NUM_AGE_CLASSES), dtype=int)
        y_gender = np.empty((self.batch_size, 1), dtype=int)

        for i, path in enumerate(batch_paths):
            try:
                img = tf.keras.preprocessing.image.load_img(path, target_size=self.target_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array /= 255.0 
                X[i,] = img_array
                y_age[i,] = self.age_labels[batch_indexes[i]]
                y_gender[i,] = self.gender_labels[batch_indexes[i]]
            except Exception as e:
                 print(f"Warning: Error loading image {path}, skipping: {e}")
              
                 X[i,] = np.zeros((*self.target_size, 3)) 
               
                 y_age[i,] = tf.keras.utils.to_categorical(0, num_classes=NUM_AGE_CLASSES)
                 y_gender[i,] = 0


        return X, {'age_output': y_age, 'gender_output': y_gender} 

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


train_generator = CustomDataGenerator(X_train_paths, y_age_train, y_gender_train, BATCH_SIZE, IMAGE_SIZE)
validation_generator = CustomDataGenerator(X_val_paths, y_age_val, y_gender_val, BATCH_SIZE, IMAGE_SIZE, shuffle=False)



def build_combined_model(input_shape, num_age_classes):
    inputs = Input(shape=input_shape)

   
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    

    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    
    age_output = Dense(num_age_classes, activation='softmax', name='age_output')(x)

    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x) 

    model = tf.keras.models.Model(inputs=inputs, outputs=[age_output, gender_output], name="AgeGenderModel")

   
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss={'age_output': 'categorical_crossentropy', 'gender_output': 'binary_crossentropy'},
                  metrics={'age_output': 'accuracy', 'gender_output': 'accuracy'},
                  loss_weights={'age_output': 1.0, 'gender_output': 1.0}) 
    return model


input_shape = IMAGE_SIZE + (3,)
model = build_combined_model(input_shape, NUM_AGE_CLASSES)
print(model.summary())



early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.00001, verbose=1)

print("\n--- Starting Model Training ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)



print("\n--- Evaluating Model Performance ---")

eval_results = model.evaluate(validation_generator)

print(f"\nValidation Loss (Total): {eval_results[0]:.4f}")
print(f"Validation Loss (Age): {eval_results[1]:.4f}")
print(f"Validation Loss (Gender): {eval_results[2]:.4f}")
print(f"Validation Accuracy (Age): {eval_results[3]:.4f}")
print(f"Validation Accuracy (Gender): {eval_results[4]:.4f}")


min_accuracy = 0.70
age_acc_ok = eval_results[3] >= min_accuracy
gender_acc_ok = eval_results[4] >= min_accuracy

if age_acc_ok:
    print(f"Age model meets the {min_accuracy*100}% accuracy target.")
else:
    print(f"WARNING: Age model accuracy ({eval_results[3]:.4f}) is BELOW the {min_accuracy*100}% target!")

if gender_acc_ok:
    print(f"Gender model meets the {min_accuracy*100}% accuracy target.")
else:
    print(f"WARNING: Gender model accuracy ({eval_results[4]:.4f}) is BELOW the {min_accuracy*100}% target!")


print("\nGenerating detailed reports (this may take a moment)...")
num_val_samples = len(X_val_paths)

pred_generator = CustomDataGenerator(X_val_paths, y_age_val, y_gender_val, BATCH_SIZE, IMAGE_SIZE, shuffle=False)


predictions = model.predict(pred_generator, steps=len(pred_generator))
age_preds_prob = predictions[0]
gender_preds_prob = predictions[1]


true_ages = []
true_genders = []
for i in range(len(pred_generator)):
     _, labels_dict = pred_generator[i]
     true_ages.extend(np.argmax(labels_dict['age_output'], axis=1))
     true_genders.extend(labels_dict['gender_output'].flatten())



age_preds_classes = np.argmax(age_preds_prob, axis=1)
gender_preds_classes = (gender_preds_prob > 0.5).astype(int).flatten()


num_preds = len(age_preds_classes)
true_ages = true_ages[:num_preds]
true_genders = true_genders[:num_preds]


print("\n--- Age Classification Report ---")
print(classification_report(true_ages, age_preds_classes, target_names=AGE_LABELS, zero_division=0))
print("Age Confusion Matrix:")
print(confusion_matrix(true_ages, age_preds_classes))

print("\n--- Gender Classification Report ---")
print(classification_report(true_genders, gender_preds_classes, target_names=GENDER_LABELS, zero_division=0))
print("Gender Confusion Matrix:")
print(confusion_matrix(true_genders, gender_preds_classes))



print(f"\nSaving combined model to {AGE_MODEL_PATH} (contains both age and gender)...")

model.save(AGE_MODEL_PATH)
print("Model saved successfully.")



print("\n--- Training Script Finished ---")
print("!!! IMPORTANT: Ensure the data loading section was correctly adapted for your dataset !!!")