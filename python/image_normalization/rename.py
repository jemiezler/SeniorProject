import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
    force=True
)


def rename(input_dir,  output_dir):
    for filename in os.listdir(input_dir):
        logging.info('Renaming file: ' + filename)
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"{filename.split("'")[0]}.png")
            # logging.info(f"Renaming {input_path} to {output_path}")
        os.rename(input_path, output_path)
            
            
if __name__ ==  "__main__":
    input_dir = './output/color_based'
    output_dir = './output/color_based'
    rename(input_dir, output_dir)