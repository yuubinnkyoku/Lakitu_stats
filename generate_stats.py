import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import io

# Constants
WIDTH = 1920
HEIGHT = 1080
BG_COLOR = (20, 20, 20)  # Dark gray/black
ACCENT_RED = (180, 20, 40)
ACCENT_DARK_RED = (100, 10, 20)
TEXT_COLOR = (255, 255, 255)

FONT_PATH = os.path.join('fonts', 'RussoOne-Regular.ttf')

def create_background():
    """Creates the geometric background."""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img, 'RGBA')

    # Draw geometric shapes (triangles/polygons) to mimic the style
    # Large dark triangles
    draw.polygon([(0, 0), (WIDTH // 2, 0), (0, HEIGHT)], fill=(15, 15, 15, 255))
    draw.polygon([(WIDTH, 0), (WIDTH, HEIGHT), (WIDTH // 2, HEIGHT)], fill=(25, 25, 25, 255))

    # Red accent lines/shapes
    # Top left diagonal
    draw.line([(0, HEIGHT * 0.6), (WIDTH * 0.4, 0)], fill=ACCENT_DARK_RED, width=3)
    
    # Bottom right diagonal
    draw.line([(WIDTH * 0.6, HEIGHT), (WIDTH, HEIGHT * 0.4)], fill=ACCENT_DARK_RED, width=3)

    # Subtle overlay triangles
    draw.polygon([(WIDTH * 0.3, 0), (WIDTH * 0.7, 0), (WIDTH * 0.5, HEIGHT * 0.4)], fill=(30, 30, 30, 100))
    
    # Bottom left red glow hint
    draw.polygon([(0, HEIGHT), (300, HEIGHT), (0, HEIGHT - 300)], fill=(50, 10, 10, 50))

    return img

def create_graph(data_points):
    """Creates the MMR graph using matplotlib."""
    # Set style to dark
    plt.style.use('dark_background')
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Plot data
    x = range(len(data_points))
    y = data_points
    
    # Create gradient line effect (simulated by plotting multiple colored segments or just a colored line)
    # For simplicity and style matching, we'll use a purple-to-red gradient look
    # Matplotlib doesn't support gradient lines natively easily, so we'll use a custom colored line
    # or just a solid color that matches the theme.
    # The image shows a line transitioning from purple/blue to red.
    
    # We can simulate this by plotting segments
    for i in range(len(x) - 1):
        # Interpolate color
        progress = i / len(x)
        # Start color (Purple/Blue): (100, 100, 255) -> (0.4, 0.4, 1.0)
        # End color (Red): (255, 50, 50) -> (1.0, 0.2, 0.2)
        r = 0.4 + (0.6 * progress)
        g = 0.4 - (0.2 * progress)
        b = 1.0 - (0.8 * progress)
        color = (r, g, b)
        
        ax.plot(x[i:i+2], y[i:i+2], color=color, linewidth=2)

    # Grid and Spines
    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color('white')
        spine.set_alpha(0.5)

    # Labels
    # Use standard font for graph labels for now, or could load custom font in matplotlib
    ax.set_ylabel('MMR', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Events Played', color='white', fontsize=12, fontweight='bold')
    
    # Tick params
    ax.tick_params(axis='both', colors='white', labelsize=10)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)

def draw_text(draw, text, position, font_size=20, color=TEXT_COLOR, anchor="la", font_path=None):
    """Draws text with a given font."""
    if font_path is None:
        font_path = FONT_PATH

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Could not load font at {font_path}, falling back to default.")
        font = ImageFont.load_default()
    except Exception as e:
        print(f"Error loading font: {e}")
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color, anchor=anchor)

def draw_gauge(draw, center, radius, value, max_value, color=ACCENT_RED):
    """Draws the circular MMR gauge."""
    # Background arc (darker)
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    draw.arc(bbox, start=135, end=405, fill=(50, 10, 20), width=30)
    
    # Progress arc
    # Calculate angle based on value (dummy logic for visual match)
    # The image shows a full arc, so we'll just draw the red arc
    draw.arc(bbox, start=135, end=405, fill=color, width=30)

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Generate background
    bg = create_background()
    draw = ImageDraw.Draw(bg)
    
    # Generate Graph
    # Simulate some data that looks like the image (rising trend)
    # Start around 10500, end around 14500, with some noise
    np.random.seed(42)
    x = np.linspace(0, 164, 164)
    # Base trend
    y = 10500 + (x * 25) 
    # Add noise/fluctuations
    noise = np.cumsum(np.random.randn(164) * 50)
    y = y + noise
    # Ensure it ends near 14558 as per image
    y = y - (y[-1] - 14558)
    
    graph_img = create_graph(y)
    
    # Paste graph onto background
    # Position roughly where it is in the image (right side)
    # Resize graph if needed to fit
    graph_width = 900
    graph_height = 600
    graph_img = graph_img.resize((graph_width, graph_height), Image.Resampling.LANCZOS)
    
    bg.paste(graph_img, (950, 350), graph_img)

    # --- Text and Stats ---
    
    # Header
    # "Mario Kart World" Logo Placeholder
    draw.rectangle([50, 50, 350, 150], fill=(0, 0, 0), outline=(255, 255, 255))
    draw_text(draw, "MARIO KART\nWORLD", (200, 100), 40, anchor="mm")
    
    # Player Name & Season
    draw_text(draw, "Kusaan", (WIDTH // 2, 80), font_size=60, anchor="mm")
    draw_text(draw, "Season 1", (WIDTH // 2, 140), font_size=40, anchor="mm")
    
    # Flag Placeholder
    draw.rectangle([WIDTH // 2 - 40, 170, WIDTH // 2 + 40, 230], fill=(255, 255, 255))
    draw.ellipse([WIDTH // 2 - 15, 185, WIDTH // 2 + 15, 215], fill=(200, 0, 0)) # Japan flag circle
    
    # Character Icon Placeholder (Top Right)
    draw.rectangle([WIDTH - 200, 50, WIDTH - 50, 200], fill=(0, 0, 0), outline=(255, 255, 255))
    draw_text(draw, "Lakitu", (WIDTH - 125, 125), font_size=30, anchor="mm")

    # MMR Gauge (Left Side)
    gauge_center = (400, 350)
    draw_gauge(draw, gauge_center, 120, 14558, 20000)
    draw_text(draw, "MMR", (400, 300), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "14558", (400, 350), font_size=80, anchor="mm")

    # Stats Columns
    # Left Column
    start_y = 500
    gap_y = 150
    
    # Avg (12P)
    draw_text(draw, "Avg (12P)", (200, start_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "87.5", (200, start_y + 50), font_size=60, anchor="mm")
    
    # Partner (12P)
    draw_text(draw, "Partner (12P)", (200, start_y + gap_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "83.0", (200, start_y + gap_y + 50), font_size=60, anchor="mm")
    
    # Largest Gain
    draw_text(draw, "Largest Gain", (200, start_y + gap_y * 2), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "+182", (200, start_y + gap_y * 2 + 50), font_size=60, anchor="mm", color=(100, 255, 100))

    # Middle Column
    mid_x = 500
    
    # Avg (24P)
    draw_text(draw, "Avg (24P)", (mid_x, start_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "87.4", (mid_x, start_y + 50), font_size=60, anchor="mm")
    
    # W-L
    draw_text(draw, "W-L", (mid_x, start_y + gap_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "118-48", (mid_x, start_y + gap_y + 50), font_size=60, anchor="mm")
    
    # Largest Loss
    draw_text(draw, "Largest Loss", (mid_x, start_y + gap_y * 2), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "-338", (mid_x, start_y + gap_y * 2 + 50), font_size=60, anchor="mm", color=(255, 80, 80))

    # Right Column (Center-Right)
    right_x = 800
    
    # Peak MMR
    draw_text(draw, "Peak MMR", (right_x, 300), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "14558", (right_x, 350), font_size=60, anchor="mm", color=ACCENT_RED)

    # Top Score
    draw_text(draw, "Top Score", (right_x, start_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "123", (right_x, start_y + 50), font_size=60, anchor="mm")
    
    # Events
    draw_text(draw, "Events", (right_x, start_y + gap_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "2-164", (right_x, start_y + gap_y + 50), font_size=60, anchor="mm")
    
    # Rank
    draw_text(draw, "Rank", (right_x, start_y + gap_y * 2), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "1", (right_x, start_y + gap_y * 2 + 50), font_size=60, anchor="mm")

    # Save final image
    bg.save('output/stats_card_final.png')
    print("Stats card generated and saved to output/stats_card_final.png")

if __name__ == "__main__":
    main()
