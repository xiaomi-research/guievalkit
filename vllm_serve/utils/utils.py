import time
import base64
import io
import importlib.resources as res

from copy import deepcopy
from PIL import Image

# subsec internal
from vllm_serve.utils.message_types import StrictMessages, Messages

VLLM_SERVE_BASE = res.files('vllm_serve')
VLLM_SERVE_UTILS = (VLLM_SERVE_BASE / 'utils')


def generate_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def encode_image(image: Image.Image) -> bytes:
    '''
    Convert PIL Image to base64 encoded bytes
    '''

    if image.mode != 'RGB':
        image = image.convert('RGB')  # This is especially important for PNG images which often have RGBA mode

    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=98)
    image_bytes = buffer.getvalue()

    base64_bytes = base64.b64encode(image_bytes)

    return f"data:image/jpeg;base64,{base64_bytes.decode('utf-8')}"


def process_vision_content(messages: Messages, images: list[Image.Image]) -> StrictMessages:
    '''
    Process vision content in messages to StrictMessages, i.e. convert ImageContent to ImageUrlContent.
    Note that this function will modify the original messages in place.

    E.g.,
    ```python
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': 'path/to/image.jpg',
                 'min_pixels': 100, 'max_pixels': 1000, 'patch_size': 100},
                {'type': 'text', 'text': 'Here is a text'},
                {'type': 'image_url', 'image_url': {'url': 'path/to/image1.jpg',
                                                    'min_pixels': 100,
                                                    'max_pixels': 1000,
                                                    patch_size': 100}}
            ]
        }
    ]
    ```
    will be converted to:
    ```python
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAA...'}},
                {'type': 'text', 'text': 'Here is a text'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAA...'}}
            ]
        }
    ]
    ```

    Args:
        messages: Messages to process.
        images: loaded images.
    Returns:
        StrictMessages: Processed messages.
    Raises:
        FileNotFoundError: If image file not found.
        TypeError: If image or content field type is invalid.
        ValueError: If content type is invalid.
    '''
    messages = deepcopy(messages)
    images = list(reversed(images))
    for message in messages:
        if isinstance(message['content'], list):
            for content in message['content']:
                if content['type'] == 'image' or content['type'] == 'image_url':
                    content.clear()
                    content['type'] = 'image_url'
                    content['image_url'] = dict(url=encode_image(image=images.pop()))
                elif content['type'] == 'text':
                    pass
                else:
                    raise ValueError(f"Invalid content type: {content['type']}")
        elif isinstance(message['content'], str):
            pass
        else:
            raise TypeError(f"Invalid content field type: {type(message['content'])}. "
                            "`content` field must be a `str` or a list of `TextContent` "
                            "or `ImageContent` or `ImageUrlContent`")
    else:
        if images:
            raise ValueError('The images count doesn\'t match the image field count in the messages.')
    return messages
