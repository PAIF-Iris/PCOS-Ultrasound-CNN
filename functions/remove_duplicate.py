import imagehash
import os
import shutil
from PIL import Image
#remove duplicate and save files into Infected_no_duplicate

def find_similar_images(userpaths, duplicates_folder, hashfunc=imagehash.average_hash):
    def is_image(filename):
        f = filename.lower()
        return f.endswith('.png') or f.endswith('.jpg') or \
            f.endswith('.jpeg') or f.endswith('.bmp') or \
            f.endswith('.gif') or '.jpg' in f or f.endswith('.svg')

    image_filenames = []
    for userpath in userpaths:
        image_filenames += [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
    images = {}
    duplicates = {}
    for img in sorted(image_filenames):
        try:
            hash_val = hashfunc(Image.open(img))
        except Exception as e:
            print('Problem:', e, 'with', img)
            continue
        if hash_val in images:
            basename = os.path.basename(img)
            target_path = os.path.join(duplicates_folder, basename)
            counter = 1
            while os.path.exists(target_path):
                name, ext = os.path.splitext(basename)
                target_path = os.path.join(duplicates_folder, f"{name}_{counter}{ext}")
                counter += 1

            shutil.move(img, target_path)
            print(f"   ✅ Moved to {target_path}")
            duplicates[hash_val] = duplicates.get(hash_val, []) + [img]
            
            #print(img, '  already exists as', ' '.join(images[hash]))
            if 'dupPictures' in img:
                print('rm -v', img)
        images[hash_val] = images.get(hash_val, []) + [img]

def remove_even_images(folder_path):
    for filename in os.listdir(folder_path):
        num_str = filename.split("_")[1].split(".")[0]
        num = int(num_str)

        if num % 2 == 0:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
