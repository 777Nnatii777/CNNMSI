import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('CNNZnakiDrogowe.h5')






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
    "A-24": "Rowerzyści",
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
    
    "C-13-C-16": "Droga dla pieszych i rowerów (wspólna)",
"C-13a" :"Koniec ścieżki rowerowej",
    "C-13a-C-16a": "Koniec drogi dla rowerów i pieszych (rozdzielonej)",
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






def predict_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)

    lista_kluczy = sorted(klasy_opis.keys())  
    predicted_class = lista_kluczy[predicted_index]
    confidence = tf.nn.softmax(predictions[0])[predicted_index].numpy() * 100

    return predicted_class, klasy_opis.get(predicted_class, "Brak opisu"), confidence




def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predicted_class, opis, confidence = predict_image(file_path)
        result_label.config(text=f"Znak: {predicted_class}\nOpis: {opis}\nPewność: {confidence:.2f}%")

        img = Image.open(file_path)
        img = img.resize((150, 150))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img


root = tk.Tk()
root.title("Rozpoznawanie znaków drogowych 🚦")

open_button = tk.Button(root, text="Dodaj zdjęcie", command=open_file)
open_button.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack()

root.mainloop()
