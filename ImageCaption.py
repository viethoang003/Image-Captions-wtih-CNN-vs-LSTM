
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import threading
from googletrans import Translator

from keras import Model
from keras.src.applications.vgg16 import VGG16, preprocess_input
from keras.src.utils import load_img, img_to_array

from XuLyAnh.BBTL.train import generate_caption, extract_features, load_and_clean_captions, tokenize_captions, \
    load_or_train_model, predict_caption


# Hàm để mở cửa sổ chọn ảnh và hiển thị chú thích
def select_image_and_generate_caption():
    # Mở cửa sổ chọn file ảnh
    file_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    if file_path:
        # Lấy tên ảnh và hiển thị ảnh
        image_name = os.path.basename(file_path)

        # In ra image_name để kiểm tra giá trị
        print(f"Image name: {image_name}")

        # Gọi hàm tạo chú thích trong một thread riêng
        caption_thread = threading.Thread(target=generate_caption_for_image, args=(file_path, image_name, model, tokenizer, max_length))
        caption_thread.start()


# Hàm để tạo chú thích cho ảnh trong thread riêng
def generate_caption_for_image(file_path, image_name, model, tokenizer, max_length):
    try:
        # Tạo chú thích cho ảnh
        vgg_model = VGG16()
        # restructure the model
        vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
        image = load_img(file_path, target_size=(224, 224))
        # convert image pixels to numpy array
        image = img_to_array(image)
        # reshape data for model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # preprocess image for vgg
        image = preprocess_input(image)
        feature = vgg_model.predict(image, verbose=0)
        caption = predict_caption(model, feature, tokenizer, max_length)

        # In ra caption để kiểm tra
        print(f"Generated caption (in English): {caption}")

        # Lấy ngôn ngữ đã chọn từ OptionMenu
        selected_language = language_var.get()

        # Dịch chú thích tùy theo ngôn ngữ chọn
        translator = Translator()
        if selected_language == 'Tiếng Việt':
            caption_translated = translator.translate(caption, src='en', dest='vi').text
        else:
            caption_translated = caption

        # Loại bỏ "startseq" và "endseq" trong caption
        caption_translated = remove_start_end(caption_translated)

        # In ra caption đã dịch
        print(f"Generated caption (in {selected_language}): {caption_translated}")

        # Hiển thị ảnh và caption
        img = Image.open(file_path)
        img = img.resize((300, 300))  # Resize ảnh nếu cần thiết
        img_tk = ImageTk.PhotoImage(img)

        # Cập nhật GUI với ảnh và caption đã dịch
        root.after(0, update_gui, img_tk, caption_translated)  # Update the GUI safely in the main thread
    except Exception as e:
        print(f"Error generating caption: {e}")


# Hàm để loại bỏ "startseq" và "endseq" trong caption
def remove_start_end(caption):
    caption = caption.lower()  # Đảm bảo chuỗi không phân biệt chữ hoa chữ thường
    caption = caption.replace('startseq', '').replace('endseq', '').strip()
    return caption


# Hàm cập nhật GUI chỉ với ảnh và caption
def update_gui(img_tk, caption):
    # Cập nhật ảnh
    panel.config(image=img_tk)
    panel.image = img_tk

    # Cập nhật chú thích
    caption_label.config(text=caption)


# Hàm thoát ứng dụng
def exit_app():
    root.quit()

# Tạo cửa sổ Tkinter và thay đổi kích thước cửa sổ
root = tk.Tk()
root.title("Chọn ảnh và hiển thị")
root.geometry("320x500")  # Đặt kích thước cửa sổ (width x height)

# Tạo panel để hiển thị ảnh
panel = tk.Label(root)
panel.place(x=10, y=80)  # Sử dụng grid để căn giữa ảnh

# Tạo label để hiển thị caption
caption_label = tk.Label(root, text="Captions: ", wraplength=300, justify="center")
caption_label.place(x=10, y=400)

# Tạo bảng chọn ngôn ngữ
language_var = tk.StringVar()
language_var.set('English')  # Mặc định là tiếng Anh

language_menu = tk.OptionMenu(root, language_var, 'English', 'Tiếng Việt')
language_menu.place(x=115, y=10, width=100, height=25)

# Nút để chọn ảnh
button = tk.Button(root, text="Chọn ảnh", command=select_image_and_generate_caption)
button.place(x=10, y=10, width=100, height=30)

# Nút thoát
exit_button = tk.Button(root, text="Thoát", command=exit_app)
exit_button.place(x=230, y=10, width=100, height=30)  # Nút nằm bên phải


# Các bước chuẩn bị dữ liệu và mô hình (giống như trong mã hiện tại của bạn)
try:
    features = extract_features()
    mapping = load_and_clean_captions()
    tokenizer, vocab_size, max_length = tokenize_captions(mapping)

    # Kiểm tra và tải model hoặc huấn luyện model nếu chưa có
    model, tokenizer = load_or_train_model(features, mapping, tokenizer, vocab_size, max_length)
except Exception as e:
    print(f"Error loading or training the model: {e}")

# Bắt đầu vòng lặp GUI của Tkinter
root.mainloop()
