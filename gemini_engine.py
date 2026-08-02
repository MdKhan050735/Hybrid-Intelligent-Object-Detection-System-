import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv
import cv2

load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
# Using gemini-2.0-flash which is standard and highly capable for images
model = genai.GenerativeModel('gemini-2.0-flash')

def get_description(image_np):
    """
    Sends a cropped image to Google Gemini and returns a brief description.
    """
    try:
        # Convert BGR (OpenCV) to RGB (PIL)
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        img_pil = PIL.Image.fromarray(img_rgb)
        
        # Call Gemini with a more specific prompt
        prompt = (
            "Look closely at the object in the center of this crop. "
            "Identify it precisely, including its brand, material, and likely use-case "
            "(e.g., 'A metal Parker ballpoint pen used for writing'). "
            "Give a very brief 5-8 word description focusing on these specific features."
        )
        
        response = model.generate_content([prompt, img_pil])
        
        return response.text.strip()
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "Quota Exceeded. Waiting..."
        return f"Error: {error_msg}"
