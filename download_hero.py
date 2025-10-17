import requests
import os

def download_medical_image():
    """Download a medical/chest X-ray image for the hero section"""
    
    # Create a medical-themed hero image URL (from a free source)
    url = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1f?w=1200&h=800&fit=crop&crop=center"
    
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            with open('assets/medical_hero.jpg', 'wb') as f:
                f.write(response.content)
            print("Medical hero image downloaded successfully!")
            return True
    except Exception as e:
        print(f"Error downloading image: {e}")
    
    return False

if __name__ == "__main__":
    download_medical_image()