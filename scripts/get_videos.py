import requests

# Replace these with your actual playlist ID and API key
playlist_id = 'PLiDzi8ftbdotqNoWGZhF1fGnv7aqLyb3y'
api_key = '***************************************'
next_page_token = None
video_urls = []

while True:
    url = f"https://www.googleapis.com/youtube/v3/playlistItems?part=snippet&playlistId={playlist_id}&maxResults=50&key={api_key}{'' if next_page_token is None else '&pageToken=' + next_page_token}"
    response = requests.get(url)
    data = response.json()
    print(data)
    items = data.get('items', [])
    
    for item in items:
        video_id = item['snippet']['resourceId']['videoId']
        video_urls.append(f"https://www.youtube.com/watch?v={video_id}")
    
    next_page_token = data.get('nextPageToken')
    if not next_page_token:
        break

# Optionally, write the URLs to a file
with open('youtube_playlist_urls.txt', 'w') as file:
    for url in video_urls:
        file.write(url + '\n')

