import os

CHUNK_SIZE = 90 * 1024 * 1024 # 90MB (Safe limit)
file_path = 'best_face_model.keras'

with open(file_path, 'rb') as f:
    chunk_num = 0
    while True:
        chunk = f.read(CHUNK_SIZE)
        if not chunk:
            break
        with open(f'{file_path}.part{chunk_num}', 'wb') as chunk_file:
            chunk_file.write(chunk)
        chunk_num += 1

print("✅ Model split into parts! You can now delete the big .keras file and upload the .part files.")