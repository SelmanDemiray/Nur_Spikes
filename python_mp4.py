from moviepy.editor import VideoFileClip
import os

def get_paths():
    # Get desktop path
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    
    # Define input/output paths
    input_path = os.path.join(desktop, r"C:\Users\Selma\Downloads\test2222222.mp4")
    output_path = os.path.join(desktop, r"C:\Users\Selma\Downloads\gif3.gif")
    
    return input_path, output_path

def convert_to_gif(input_path, output_path, resize_factor=1.0):
    try:
        video = VideoFileClip(input_path)
        if resize_factor != 1.0:
            video = video.resize(resize_factor)
        
        print(f"Converting: {input_path}")
        print(f"Output to: {output_path}")
        video.write_gif(output_path, fps=15)
        video.close()
        print("Conversion complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    input_path, output_path = get_paths()
    
    if not os.path.exists(input_path):
        print(f"Please place your MP4 file at: {input_path}")
    else:
        convert_to_gif(input_path, output_path)