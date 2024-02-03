# scanslator

## How to run AI Service (Note: use python3 or pip3 accordingly)

### 1. set up environment

`cd scanslator`

`python3 -m venv .venv`

`. .venv/bin/activate`

`pip3 install -r requirements.txt`

### 2. get model weights

Get the weights file and the sample image from the shared drive: https://drive.google.com/drive/folders/1Zx8i9HUXlQg73lXnbyqlAITH71Ja_5mA?usp=share_link

Place them at the root of the directory

### 3. run app

`flask run`

### 4. invoke maks api endpoint (send raw image and receive mask image)

Try with a sample image in the `/testdata` directory

`POST: http://127.0.0.1:5000/mask`
