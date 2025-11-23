from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes(image_path, results):
    """
    Draws bounding boxes and labels on an image.

    Args:
        image_path (str): The path to the image file.
        results (list): A list of dictionaries, where each dictionary
                        contains 'name' and 'location' of a detected face.

    Returns:
        PIL.Image.Image: The image with bounding boxes and labels drawn.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for result in results:
        name = result['name']
        top, right, bottom, left = result['location']
        
        color = "green" if name != "Unknown" else "red"

        # Draw bounding box
        draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)
        
        # Draw label
        bbox = draw.textbbox((0, 0), name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        label_background = [(left, bottom - text_height - 10), (left + text_width + 10, bottom)]
        draw.rectangle(label_background, fill=color)
        draw.text((left + 6, bottom - text_height - 5), name, fill="white", font=font)
        
    return image
