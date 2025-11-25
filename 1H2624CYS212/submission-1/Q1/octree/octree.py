from PIL import Image

def octree_quantize_image(input_path, output_path, max_colors):

    #Perform Octree color quantization on an image using Pillow's FASTOCTREE method.

    # Load the input image and convert to RGB mode.
    # Converting ensures the image uses standard red, green, blue channels,
    img = Image.open(input_path).convert('RGB')

    print(f"Quantizing image to max {max_colors} colors using FASTOCTREE method...")

    quantized_img = img.quantize(colors=max_colors, method=Image.Quantize.FASTOCTREE)

    # Save the quantized image in a file format supporting palette mode.
    quantized_img.save(output_path)
    print(f"Quantized image saved to: {output_path}")


max_colors = int(input("Enter maximum number of colors (e.g., 256): "))
octree_quantize_image("input.jpg","octree_out.png", max_colors)