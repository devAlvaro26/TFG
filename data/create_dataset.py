import zipfile
import gdown
import os

# ID del dataset
file_id = '1SFGqQuIdQWcQV7zS0vOE4td-8RZ575sS'

if not os.path.exists("data/dataset"):
    # Construir la URL de descarga
    url = f'https://drive.google.com/uc?id={file_id}'
    extract_file = "data/dataset.zip"

    # Descargar el zip
    gdown.download(url, extract_file, quiet=False)

    print("ZIP descargado")

    # Descomprimir
    with zipfile.ZipFile(extract_file, 'r') as zip_ref:
        zip_ref.extractall(os.path.splitext(extract_file)[0])

    os.remove(extract_file)
    print("ZIP descomprimido en:", os.path.splitext(extract_file)[0])

else:
    print("El dataset ya existe")