"""Utils for the database module."""

import base64
from io import BytesIO
import logging

from openai import OpenAI
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def encode_image(image: np.ndarray) -> str:
    """Encodes an image to base64.

    Args:
        image: image to encode

    Returns:
        base64 string
    """
    image = Image.fromarray(image)
    # change mimetype to image/jpeg
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_frame_caption(frame_base64: str, model: str = "gpt-4o-mini") -> np.ndarray:
    """Generates a caption for a frame using OpenAI.

    Args:
        frame: frame to generate a caption for
        model: model to use for caption generation

    Returns:
        caption: generated caption
    """
    try:
        logger.info("generate_frame_caption | making request to OpenAI")
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_base64}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        logger.info("generate_frame_caption | response received")
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"generate_frame_caption | warning: {e}")
        return None
