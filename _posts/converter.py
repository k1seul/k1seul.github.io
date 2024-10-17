import re

def convert_markdown_images_to_centered_html(input_file, output_file):
    # Read the input Markdown file
    with open(input_file, 'r') as file:
        markdown_content = file.read()

    # Regular expression to find Markdown image syntax ![alt text](image-url)
    image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    # Function to replace the Markdown image with the centered HTML version
    def replace_image_with_centered_html(match):
        alt_text = match.group(1)  # Extract alt text
        image_url = match.group(2)  # Extract image URL
        # Return the centered HTML
        return f'<div style="text-align: center;">\n    <img src="{image_url}" alt="{alt_text}" />\n</div>\n'

    # Replace all Markdown image syntax with centered HTML using the regex
    centered_html_content = re.sub(image_pattern, replace_image_with_centered_html, markdown_content)

    # Write the transformed content to the output file
    with open(output_file, 'w') as file:
        file.write(centered_html_content)

    print(f"Converted Markdown images to centered HTML in {output_file}")

# Example usage
input_file = "2024-10-15-LIBERO_Review.md"
output_file = "2024-10-15-LIBERO_Review.md"

convert_markdown_images_to_centered_html(input_file, output_file)
