import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
from keras.applications.resnet import preprocess_input, decode_predictions

# Classes
gender_class = ['Boys', 'Girls', 'Men', 'Unisex', 'Women']

type_class = ['Accessory Gift Set', 'Baby Dolls', 'Backpacks', 'Bangle', 'Basketballs', 'Bath Robe', 
              'Beauty Accessory', 'Belts', 'Blazers', 'Body Lotion', 'Body Wash and Scrub', 'Booties', 
              'Boxers', 'Bra', 'Bracelet', 'Briefs', 'Camisoles', 'Capris', 'Caps', 'Casual Shoes', 
              'Churidar', 'Clothing Set', 'Clutches', 'Compact', 'Concealer', 'Cufflinks', 'Cushion Covers', 
              'Deodorant', 'Dresses', 'Duffel Bag', 'Dupatta', 'Earrings', 'Eye Cream', 'Eyeshadow', 'Face Moisturisers', 
              'Face Scrub and Exfoliator', 'Face Serum and Gel', 'Face Wash and Cleanser', 'Flats', 'Flip Flops', 
              'Footballs', 'Formal Shoes', 'Foundation and Primer', 'Fragrance Gift Set', 'Free Gifts', 'Gloves', 
              'Hair Accessory', 'Hair Colour', 'Handbags', 'Hat', 'Headband', 'Heels', 'Highlighter and Blush', 
              'Innerwear Vests', 'Ipad', 'Jackets', 'Jeans', 'Jeggings', 'Jewellery Set', 'Jumpsuit', 'Kajal and Eyeliner', 
              'Key chain', 'Kurta Sets', 'Kurtas', 'Kurtis', 'Laptop Bag', 'Leggings', 'Lehenga Choli', 'Lip Care', 
              'Lip Gloss', 'Lip Liner', 'Lip Plumper', 'Lipstick', 'Lounge Pants', 'Lounge Shorts', 'Lounge Tshirts', 
              'Makeup Remover', 'Mascara', 'Mask and Peel', 'Mens Grooming Kit', 'Messenger Bag', 'Mobile Pouch', 'Mufflers', 
              'Nail Essentials', 'Nail Polish', 'Necklace and Chains', 'Nehru Jackets', 'Night suits', 'Nightdress', 
              'Patiala', 'Pendant', 'Perfume and Body Mist', 'Rain Jacket', 'Rain Trousers', 'Ring', 'Robe', 'Rompers', 
              'Rucksacks', 'Salwar', 'Salwar and Dupatta', 'Sandals', 'Sarees', 'Scarves', 'Shapewear', 'Shirts', 'Shoe Accessories', 
              'Shoe Laces', 'Shorts', 'Shrug', 'Skirts', 'Socks', 'Sports Sandals', 'Sports Shoes', 'Stockings', 'Stoles', 
              'Sunglasses', 'Sunscreen', 'Suspenders', 'Sweaters', 'Sweatshirts', 'Swimwear', 'Tablet Sleeve', 'Ties', 'Ties and Cufflinks', 
              'Tights', 'Toner', 'Tops', 'Track Pants', 'Tracksuits', 'Travel Accessory', 'Trolley Bag', 'Trousers', 'Trunk', 
              'Tshirts', 'Tunics', 'Umbrellas', 'Waist Pouch', 'Waistcoat', 'Wallets', 'Watches', 'Water Bottle', 'Wristbands']

season_class = ['Fall', 'Spring', 'Summer', 'Unknown', 'Winter']

color_class = ['Beige', 'Black', 'Blue', 'Bronze', 'Brown', 'Burgundy', 'Charcoal', 'Coffee Brown', 'Copper', 'Cream', 'Fluorescent Green', 
               'Gold', 'Green', 'Grey', 'Grey Melange', 'Khaki', 'Lavender', 'Lime Green', 'Magenta', 'Maroon', 'Mauve', 'Metallic', 'Multi',
                 'Mushroom Brown', 'Mustard', 'Navy Blue', 'Nude', 'Off White', 'Olive', 'Orange', 'Peach', 'Pink', 'Purple', 'Red', 'Rose', 
                 'Rust', 'Sea Green', 'Silver', 'Skin', 'Steel', 'Tan', 'Taupe', 'Teal', 'Turquoise Blue', 'Unknown', 'White', 'Yellow']

# Load the trained prediction models
gender_model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\assignment-codemonk\model_files\model_gender1.h5', compile = False)
gender_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

type_model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\assignment-codemonk\model_files\model_type.h5', compile = False)
type_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

season_model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\assignment-codemonk\model_files\model_season.h5', compile = False)
season_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

color_model = tf.keras.models.load_model(r'C:\Users\HP\Desktop\assignment-codemonk\model_files\model_color.h5', compile = False)
color_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Make predictions
def model_predictions(model, class_names, image):
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.resize(img, size = [224, 224])
    img = img/255.

    pred = model.predict(tf.expand_dims(img, axis=0))

    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    return pred_class

#GUI
st.title("Fashion Prediction App")
st.write("Upload an image to predict the fashion attributes.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    print(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Fashion Attributes"):
        gender = model_predictions(gender_model, gender_class, image)
        fashion_type = model_predictions(type_model, type_class, image)
        season = model_predictions(season_model, season_class, image)
        color = model_predictions(color_model, color_class, image)
        st.success(f"Predicted Gender: {gender}")
        st.success(f"Predicted Type: {fashion_type}")
        st.success(f"Predicted Season: {season}")
        st.success(f"Predicted Color: {color}")
