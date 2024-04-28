import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Define your Flickr API key
API_KEY = os.getenv('FLICKR_API_KEY')

# Define search parameters
search_params = {
    'api_key': API_KEY,
    'text': 'gharial reptile',
    'per_page': 20,  # Number of images per page
    'format': 'json',
    'nojsoncallback': 1,
    'license': '4,5,7'  # Filter for Creative Commons licenses and public domain (CC0)
}

# Flickr API endpoint for photo search
api_endpoint = 'https://www.flickr.com/services/rest/?method=flickr.photos.search'

# Make a request to the Flickr API
response = requests.get(api_endpoint, params=search_params)
data = response.json()

# Extract image URLs from the API response
photo_urls = []
for photo in data['photos']['photo']:
    photo_url = f"https://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}_b.jpg"
    photo_urls.append(photo_url)

# Directory to save downloaded images
save_dir = 'crocodilian_images'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Download images
for i, url in enumerate(photo_urls):
    image_response = requests.get(url)
    with open(os.path.join(save_dir, f'gharial_{i}.jpg'), 'wb') as f:
        f.write(image_response.content)