import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(source_dir, train_dir, val_dir, test_dir, train_size=0.7, val_size=0.15, test_size=0.15):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # print (f"Total files found: {len(all_files)}")
    
    train_files, temp_files = train_test_split(all_files, train_size=train_size, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=test_size/(test_size + val_size), random_state=42)

    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))
    
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))
    
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(test_dir, file))


split_data(
    source_dir='app/ai/datasets/dogs-vs-cats/raw',
    train_dir='app/ai/datasets/dogs-vs-cats/train',
    val_dir='app/ai/datasets/dogs-vs-cats/val',
    test_dir='app/ai/datasets/dogs-vs-cats/test'
)