"""
Utility functions shared across the video generator scripts.
"""

import re
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import pickle


def calculate_age_from_year_range(birth_year: int | None, year_range: str | int | None) -> int | None:
    """
    Calculate the approximate age of a person based on their birth year and a year range.
    
    Args:
        birth_year: Birth year of the person (e.g., 1879)
        year_range: Year range string (e.g., "1905-1910", "1831-1834", "around 1859") or int (e.g., 1905)
    
    Returns:
        Estimated age (int) or None if birth_year is None or year_range cannot be parsed
    """
    if birth_year is None or year_range is None:
        return None
    
    # Handle int input directly (e.g., year is just 1905)
    if isinstance(year_range, int):
        scene_year = year_range
        age = scene_year - birth_year
        return max(0, age)
    
    # Handle string input - parse it
    # Try to extract years from various formats:
    # "1905-1910" -> (1905, 1910)
    # "1905" -> (1905, 1905)
    # "around 1859" -> (1859, 1859)
    # "1831–1834" -> (1831, 1834) (em dash)
    # "1900s" -> (1900, 1909)
    
    # Convert to string if not already
    year_range_str = str(year_range)
    
    # Match year patterns (1800s, 1900s, 2000s)
    # Use non-capturing group so findall returns full matches, not just the group
    year_pattern = r'\b(?:18|19|20)\d{2}\b'
    years = re.findall(year_pattern, year_range_str)
    
    if not years:
        return None
    
    # Convert to integers (years are now full 4-digit strings like "1936")
    year_ints = [int(y) for y in years]
    
    if len(year_ints) == 1:
        # Single year
        scene_year = year_ints[0]
    elif len(year_ints) >= 2:
        # Range - use the middle or average
        scene_year = (min(year_ints) + max(year_ints)) // 2
    else:
        return None
    
    # Calculate age
    age = scene_year - birth_year
    
    # Ensure age is non-negative (cap at 0 if before birth)
    return max(0, age)


# YouTube API OAuth2 scopes
YOUTUBE_UPLOAD_SCOPE = ['https://www.googleapis.com/auth/youtube.upload']
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


def get_youtube_service(credentials_file: str = 'client_secrets.json', 
                       token_file: str = 'token.pickle') -> Any:
    """
    Authenticate and return a YouTube API service object.
    
    Args:
        credentials_file: Path to OAuth2 client secrets JSON file (downloaded from Google Cloud Console)
        token_file: Path to store the access token (will be created/updated automatically)
    
    Returns:
        YouTube API service object
    
    Raises:
        FileNotFoundError: If credentials_file doesn't exist
        Exception: If authentication fails
    """
    creds = None
    
    # Load existing token if available
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            # Refresh expired token
            creds.refresh(Request())
        else:
            # Run OAuth flow if credentials file exists
            if not os.path.exists(credentials_file):
                raise FileNotFoundError(
                    f"Credentials file '{credentials_file}' not found. "
                    f"Please download OAuth2 client secrets from Google Cloud Console "
                    f"and save as '{credentials_file}'"
                )
            
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_file, YOUTUBE_UPLOAD_SCOPE)
            creds = flow.run_local_server(port=0)
        
        # Save token for future use
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, credentials=creds)


def extract_metadata_from_script(script_path: str) -> Dict[str, Any]:
    """
    Extract video metadata from a JSON script file.
    
    Args:
        script_path: Path to the JSON script file
    
    Returns:
        Dictionary with 'title', 'description', 'tags' (as list), and 'category_id'
        Returns empty dict if script doesn't exist or doesn't have metadata
    """
    metadata = {}
    
    if not os.path.exists(script_path):
        return metadata
    
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if metadata exists (for shorts scripts)
        if isinstance(data, dict) and 'metadata' in data:
            meta = data['metadata']
            metadata['title'] = meta.get('title', '')
            metadata['description'] = meta.get('description', '')
            
            # Parse tags - could be comma-separated string or already a list
            tags_str = meta.get('tags', '')
            if isinstance(tags_str, str):
                metadata['tags'] = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            elif isinstance(tags_str, list):
                metadata['tags'] = tags_str
            else:
                metadata['tags'] = []
        
        # For full scripts, check if there's outline metadata
        elif isinstance(data, dict) and 'outline' in data:
            outline = data['outline']
            metadata['title'] = outline.get('title', '')
            metadata['description'] = outline.get('description', '')
            tags_str = outline.get('tags', '')
            if isinstance(tags_str, str):
                metadata['tags'] = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            elif isinstance(tags_str, list):
                metadata['tags'] = tags_str
            else:
                metadata['tags'] = []
    
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"Warning: Could not extract metadata from {script_path}: {e}")
    
    return metadata


def upload_video_to_youtube(
    video_path: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[list] = None,
    category_id: str = '22',  # '22' = People & Blogs, '27' = Education
    privacy_status: str = 'private',  # 'private', 'unlisted', or 'public'
    is_short: bool = False,
    script_path: Optional[str] = None,
    credentials_file: str = 'client_secrets.json',
    token_file: str = 'token.pickle'
) -> str:
    """
    Upload a video to YouTube using the YouTube Data API v3.
    
    Args:
        video_path: Path to the video file to upload
        title: Video title (if None, will try to extract from script_path or use filename)
        description: Video description (if None, will try to extract from script_path)
        tags: List of tags (if None, will try to extract from script_path)
        category_id: YouTube category ID ('22' = People & Blogs, '27' = Education, etc.)
        privacy_status: 'private', 'unlisted', or 'public'
        is_short: If True, marks video as YouTube Short (#Shorts will be added to title/description)
        script_path: Path to JSON script file to extract metadata from (optional)
        credentials_file: Path to OAuth2 client secrets JSON file
        token_file: Path to store OAuth2 token
    
    Returns:
        YouTube video ID (e.g., 'dQw4w9WgXcQ')
    
    Raises:
        FileNotFoundError: If video_path or credentials_file doesn't exist
        Exception: If upload fails
    
    Example:
        # Upload with metadata from script
        video_id = upload_video_to_youtube(
            'finished_shorts/alan_turing_short1.mp4',
            script_path='shorts_scripts/alan_turing_short1.json',
            privacy_status='private',
            is_short=True
        )
        print(f"Uploaded! Watch at: https://www.youtube.com/watch?v={video_id}")
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Try to extract metadata from script if provided
    metadata = {}
    if script_path:
        metadata = extract_metadata_from_script(script_path)
    
    # Use provided values or fall back to extracted metadata or defaults
    final_title = title or metadata.get('title', Path(video_path).stem)
    final_description = description or metadata.get('description', '')
    final_tags = tags or metadata.get('tags', [])
    
    # Handle YouTube Shorts
    if is_short:
        # Ensure title includes #Shorts if not already present
        if '#Shorts' not in final_title and '#shorts' not in final_title:
            final_title = f"{final_title} #Shorts"
        
        # Ensure description includes #Shorts tag for discoverability
        if '#Shorts' not in final_description and '#shorts' not in final_description:
            if final_description:
                final_description = f"{final_description}\n\n#Shorts"
            else:
                final_description = "#Shorts"
    
    # Build video metadata
    body = {
        'snippet': {
            'title': final_title,
            'description': final_description,
            'tags': final_tags[:500],  # YouTube limit is 500 tags
            'categoryId': category_id
        },
        'status': {
            'privacyStatus': privacy_status,
            'selfDeclaredMadeForKids': False
        }
    }
    
    # If it's a Short, add that to the status
    if is_short:
        body['status']['madeForKids'] = False
    
    # Authenticate and get YouTube service
    youtube = get_youtube_service(credentials_file, token_file)
    
    # Create media file upload object
    media = MediaFileUpload(
        video_path,
        chunksize=-1,
        resumable=True,
        mimetype='video/*'
    )
    
    # Insert video
    print(f"[YOUTUBE] Uploading '{final_title}'...")
    insert_request = youtube.videos().insert(
        part=','.join(body.keys()),
        body=body,
        media_body=media
    )
    
    # Execute upload with resumable upload support
    video_id = None
    response = None
    error = None
    retry = 0
    
    while response is None:
        try:
            status, response = insert_request.next_chunk()
            if response is not None:
                if 'id' in response:
                    video_id = response['id']
                    print(f"[YOUTUBE] Upload successful! Video ID: {video_id}")
                    print(f"[YOUTUBE] Watch at: https://www.youtube.com/watch?v={video_id}")
                else:
                    raise Exception(f"Upload failed: {response}")
        except Exception as e:
            error = e
            if retry < 3:
                retry += 1
                print(f"[YOUTUBE] Retry {retry}/3...")
            else:
                raise Exception(f"Upload failed after {retry} retries: {error}")
    
    return video_id
