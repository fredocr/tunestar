import tekore as tk
def authorize():
 CLIENT_ID = "your_id"
 CLIENT_SECRET = "your_secret"
 app_token = tk.request_client_token(CLIENT_ID, CLIENT_SECRET)
 return tk.Spotify(app_token)