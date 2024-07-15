from PIL import Image
import os
import csv 

source_dir = '/home/toe/Documents/Aladdin_Persson_Pytorch_Serie/07_Pytorch_How_to_build_custom_Datasets_for_Images/dataset/dogs'
target_dir = '/home/toe/Documents/Aladdin_Persson_Pytorch_Serie/07_Pytorch_How_to_build_custom_Datasets_for_Images/dataset/catsanddogs'

os.makedirs(target_dir, exist_ok=True)

def resize() : 
    for filename in os.listdir(source_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Kiểm tra định dạng tệp ảnh
            source_path = os.path.join(source_dir, filename)
            img = Image.open(source_path)
            img = img.resize((32, 32), Image.LANCZOS)
            target_path = os.path.join(target_dir, filename)
            img.save(target_path)

    print("Resize and save completed!")

def save_file_scv(): 
    files = os.listdir(target_dir)
    image = []
    label = [] 
    for file in files: 
        image.append(file)
        if file[:3] == 'dog': 
            label.append(0)
        else: 
            label.append(1) 

    data =  {
        'image': image,
        'label': label
    }

    csv_file = '/home/toe/Documents/Aladdin_Persson_Pytorch_Serie/07_Pytorch_How_to_build_custom_Datasets_for_Images/dataset/data.csv'

    # Lấy các khóa (keys) của dictionary làm tiêu đề (header) cho file CSV
    header = data.keys()

    # Ghi dữ liệu vào file CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for i in range(len(next(iter(data.values())))):
            row = {key: value[i] for key, value in data.items()}
            writer.writerow(row)

    print(f'Data has been written to {csv_file}')      

save_file_scv() 
