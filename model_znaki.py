import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

base_dir = os.path.join(os.getcwd(), "dane")  
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(32, 32),
    batch_size=32
)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1): 
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  
        plt.title(train_ds.class_names[labels[i]])   
        plt.axis("off")
plt.tight_layout()
plt.show()

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(32, 32),
    batch_size=32
)

plt.figure(figsize=(10, 10))
for images, labels in test_ds.take(1): 
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))  
        plt.title(test_ds.class_names[labels[i]])    
        plt.axis("off")
plt.tight_layout()
plt.show()


klasy_opis = {
    "A-1": "Niebezpieczny zakręt w prawo",
    "A-2": "Niebezpieczny zakręt w lewo",
    "A-3": "Dwa niebezpieczne zakręty – pierwszy w prawo",
    "A-4": "Dwa niebezpieczne zakręty – pierwszy w lewo",
    "A-6a": "Skrzyżowanie z drogą podporządkowaną z obydwustron",
    "A-6b": "Skrzyżowanie z drogą podporządkowaną z prawej",
    "A-6c": "Skrzyżowanie z drogą podporządkowaną po obu stronach",
    "A-6d": "Wlot drogi jednokierunkowej z prawej",
    "A-6e": "Wlot drogi jednokierunkowej z lewej",
    "A-7": "Ustąp pierwszeństwa",
    "A-8": "Skrzyżowanie o ruchu okrężnym",
    "A-11": "Próg zwalniający",
    "A-11a": "Strefa ruchu powolnego",
    "A-12a": "Zwężenie jezdni – obustronne",
    "A-14": "Roboty na drodze",
    "A-15": "Śliska jezdnia",
    "A-16": "Przejście dla pieszych",
    "A-17": "Uwaga przechodzacy ludzie",
    "A-18b": "Zwierzęta dzikie",
    "A-20": "Odcinek jezdni o ruchu dwukierunkowym",
    "A-21": "Tramwaj",
    "A-23": "Rowerzyści",
    "A-24": "Roboty drogowe",
    "A-29": "Uwaga światła",
    "A-30": "Niebezpieceństwo",
    "A-32": "Zamieć śnieżna",
    "B-1": "Zakaz ruchu w obu kierunkach",
    "B-2": "Zakaz wjazdu",
    "B-5": "Zakaz wjazdu samochodów ciężarowych",
    "B-6-B-8-B-9": "Zakaz wjazdu ciągników i motocykli",
    "B-8": "Zakaz wjazdu wózków rowerowych",
    "B-9": "Zakaz wjazdu motorowerów",
    "B-18": "Zakaz wjazdu pojazdów o nacisku osi większym niż podany",
    "B-20": "STOP",
    "B-21": "Zakaz skrętu w lewo",
    "B-22": "Zakaz skrętu w prawo",
    "B-25": "Zakaz wyprzedzania",
    "B-26": "Zakaz wyprzedzania przez samochody ciężarowe",
    "B-27": "Koniec zakazu wyprzedzania",
    "B-33": "Ograniczenie prędkości",
    "B-34": "Koniec ograniczenia prędkości",
    "B-36": "Zakaz zatrzymywania się",
    "B-41": "Zakaz ruchu pieszych",
    "B-42": "Koniec zakazów",
    "B-43": "Strefa ograniczonej prędkości",
    "B-44": "Koniec strefy ograniczonej prędkości",
    "C-2": "Nakaz jazdy w prawo",
    "C-4": "Nakaz jazdy w lewo",
    "C-5": "Nakaz jazdy prosto",
    "C-6": "Nakaz jazdy prosto lub w prawo",
    "C-7": "Nakaz jazdy prosto lub w lewo",
    "C-9": "Nakaz jazdy z prawej strony przeszkody",
    "C-10": "Nakaz jazdy z lewej strony przeszkody",
    "C-12": "Ruch okrężny",
    "C-13": "Droga dla rowerów",
    "C-13a": "Koniec drogi dla rowerów",
    "C-13a-C-16a": "Koniec drogi dla rowerów i pieszych (rozdzielonej)",
    "C-13-C-16": "Droga dla pieszych i rowerów (wspólna)",
    "C-16": "Droga dla pieszych",
    "D-1": "Droga z pierwszeństwem",
    "D-2": "Koniec drogi z pierwszeństwem",
    "D-3": "Droga jednokierunkowa",
    "D-4a": "Droga bez przejazdu",
    "D-4b": "Droga bez przejazdu z lewej",
    "D-6": "Przejście dla pieszych",
    "D-6b": "Przejazd dla rowerzystów",
    "D-7": "Droga ekspresowa",
    "D-8": "Droga krajowa",
    "D-9": "Droga wojewódzka",
    "D-14": "Początek pasa ruchu",
    "D-15": "Przystanek autobusowy",
    "D-18": "Parking",
    "D-18b": "Parking zadaszony",
    "D-21": "Stacja paliw",
    "D-23": "Stacja paliw",
    "D-23a": "Miejsce obsługi podróżnych",
    "D-24": "Informacja drogowa",
    "D-26": "Informacja turystyczna",
    "D-26b": "Pomoc drogowa",
    "D-26c": "Toalety",
    "D-27": "Telefon",
    "D-28": "Jedzienie",
    "D-29": "Straż Pożarna",
    "D-40": "Strefa zamieszkania",
    "D-41": "Koniec strefy zamieszkania",
    "D-42": "Obszar zabudowany",
    "D-43": "Koniec obszaru zabudowanego",
    "D-51": "Kontrola graniczna",
    "D-52": "Punkt poboru opłat",
    "D-53": "Granica państwa",
    "D-tablica": "Tablica informacyjna",
    "G-1a": "Granica państwa",
    "G-3": "Miejsce obsługi podróżnych"
}

class_names = train_ds.class_names
num_classes = len(class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))



model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes)
])


model.summary()


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(train_ds, validation_data=test_ds, epochs=20)


plt.plot(history.history['accuracy'], label='Dokładność treningowa')
plt.plot(history.history['val_accuracy'], label='Dokładność testowa')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.legend()
plt.grid(True)
plt.show()

#model.save('CNNZnakiDrogowe.h5')

for images, labels in test_ds.take(1):  
    predictions = model.predict(images)
    predicted_labels = tf.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow((images[i].numpy() * 255).astype("uint8"))

        true_label = class_names[labels[i]]
        predicted_label = class_names[predicted_labels[i]]

       
        pred_text = f"{predicted_label} ({klasy_opis.get(predicted_label, 'Brak opisu')})"
        true_text = f"{true_label} ({klasy_opis.get(true_label, 'Brak opisu')})"

        color = "green" if predicted_label == true_label else "red"
        plt.title(f"P: {pred_text}\nT: {true_text}", color=color)
        plt.axis("off")

    plt.tight_layout()
    plt.show()




image_path = "dosprawdzenia/og30.png"  


img = Image.open(image_path).resize((32, 32))
img_array = np.array(img) / 255.0 


img_array = np.expand_dims(img_array, axis=0)


predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
predicted_class = class_names[predicted_index]
confidence = tf.nn.softmax(predictions[0])[predicted_index].numpy() * 100


print(f"🛑 Model przewiduje: {predicted_class} ({klasy_opis.get(predicted_class, 'Brak opisu')}) z pewnością {confidence:.2f}%")

plt.imshow((np.squeeze(img_array) * 255).astype("uint8"))

plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.axis("off")
plt.show()