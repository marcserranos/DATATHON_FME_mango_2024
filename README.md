# DATATHON_FME_mango_2024

## Description
This project uses the CLIP model to identify clothing items in images. Users can upload an image, and the AI will predict the most likely type of clothing.

## Features
- Upload an image of clothing.
- Get predictions on what type of clothing the image contains (e.g., shirt, pants, jacket).
- Powered by the CLIP model for image and text understanding.

## Installation

### Requirements:
- Python 3.7+
- Streamlit
- Hugging Face Transformers
- PIL (Python Imaging Library)

### Setup:
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/ropa-identification.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open the app in your browser and upload an image to see the prediction.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- CLIP model from OpenAI (https://openai.com/research/clip)
- Streamlit for easy app deployment
