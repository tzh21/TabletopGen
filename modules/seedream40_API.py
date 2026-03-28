import io
import os
import tempfile

from PIL import Image

from modules.replicate_client import run_seedream4_to_file


def adjust_image_aspect_ratio(image_path, max_ratio=2.5):
    """
    Adjust the image aspect ratio to be within the API's required range (0.33-3.00)
    """
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height

    if 0.4 <= aspect_ratio <= max_ratio:
        return img

    if aspect_ratio > max_ratio:
        new_height = int(width / max_ratio)
        padding = (new_height - height) // 2
        new_img = Image.new("RGB", (width, new_height), (128, 128, 128))
        new_img.paste(img, (0, padding))
    else:
        new_width = int(height * 0.4)
        padding = (new_width - width) // 2
        new_img = Image.new("RGB", (new_width, height), (128, 128, 128))
        new_img.paste(img, (padding, 0))

    return new_img


def _write_adjusted_png_temp(image_path: str) -> str:
    """Write aspect-adjusted image to a temp PNG; return path for Replicate upload."""
    img = adjust_image_aspect_ratio(image_path)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="seedream_adj_")
    os.close(fd)
    with open(tmp, "wb") as f:
        f.write(buf.getvalue())
    return tmp


def generate_object_image(
    image_path1,
    image_path2,
    object_name,
    output_path,
    is_main_obj=False,
    is_multi=False,
    replicate_api_token=None,
):
    """
    Generate object images using Seedream 4 on Replicate (two reference images).
    """
    p1 = _write_adjusted_png_temp(image_path1)
    p2 = _write_adjusted_png_temp(image_path2)
    try:
        if is_main_obj:
            prompt = f'''Please redraw the "{object_name}" object in the first reference image and generate a high-definition image of it. Its proportions, shape, texture, and other appearance features must remain unchanged. Its current front perspective must be maintained, it cannot be rotated. If it is obscured or overlapped by other objects, remove the other objects and draw only the "{object_name}" itself. But do not remove objects that come with it, such as drawers and sinks that come with the table. Fill the background with a solid color.
    If the object is obscured or partly missing, you can use the second scene image as a reference to better identify the object. However, be sure not to mistake other objects in the scene for the object!'''
        elif is_multi:
            prompt = f"""Please redraw the object **in the first reference image** and generate a high-definition image of it. Its proportions, shape, texture, and other appearance features must remain unchanged. If it is obscured or overlapped by other objects(usually the transparent mask in the first reference image is the obstructing object), remove the obstructing objects and draw only the object itself. Fill the background with a solid color, however, do not project the light and shadow of the background color onto the object. The object must maintain its own color, especially transparent objects.
    You can use the second scene image as a reference to better identify the object. However, be sure not to mistake other objects in the scene for the object!"""
        else:
            prompt = f'Please redraw the "{object_name}" object in the first reference image and generate a high-definition image of it. Its proportions, shape, texture, and other appearance features must remain unchanged. If it is obscured or overlapped by other objects(usually the transparent mask in the first reference image is the obstructing object), remove the obstructing objects and draw only the "{object_name}" itself. Fill the background with a solid color, however, do not project the light and shadow of the background color onto the object. The object must maintain its own color, especially transparent objects.'

        print(f"Generating object: {object_name}")

        image_url = run_seedream4_to_file(
            prompt,
            output_path,
            image_paths=[p1, p2],
            replicate_api_token=replicate_api_token,
            size="2K",
            aspect_ratio="match_input_image",
            sequential_image_generation="disabled",
            max_images=1,
            enhance_prompt=True,
        )
        print(f" Image saved to: {output_path}")
        return image_url
    finally:
        for p in (p1, p2):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    image_path1 = "input_images/object_1.png"
    image_path2 = "input_images/scene_1.png"
    object_name = "vase"
    output_path = "output_images/object_1_redraw.png"

    generate_object_image(
        image_path1=image_path1,
        image_path2=image_path2,
        object_name=object_name,
        output_path=output_path,
        is_main_obj=True,
        is_multi=False,
        replicate_api_token=os.environ.get("REPLICATE_API_TOKEN"),
    )
