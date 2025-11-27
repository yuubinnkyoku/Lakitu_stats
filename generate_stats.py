import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import io
import math

# Constants
WIDTH = 1920
HEIGHT = 1080
BG_COLOR = (5, 5, 5)  # Darker black
ACCENT_RED = (180, 20, 40)
ACCENT_DARK_RED = (100, 10, 20)
TEXT_COLOR = (255, 255, 255)

FONT_PATH = os.path.join('fonts', 'RussoOne-Regular.ttf')

def create_background():
    """Creates the geometric background using an Organically Warped Triangular Grid."""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img, 'RGBA')

    import random

    # 1. Define Grid Topology (Regular Triangular Grid)
    # We generate a grid, but then we WARP the vertices organically.
    
    # Grid parameters
    # Fewer rows/cols for larger triangles
    rows = 6
    cols = 6
    
    # Base scale
    scale = 700
    
    # Random seeds for organic distortion
    seed_x = random.uniform(0, 100)
    seed_y = random.uniform(0, 100)
    seed_rot = random.uniform(0, 100)
    
    # Global rotation/offset to break alignment
    base_rotation = random.uniform(-0.3, 0.3)
    offset_x = WIDTH / 2
    offset_y = HEIGHT / 2
    
    def get_organic_offset(x, y):
        """Calculates a smooth, organic displacement for a point (x, y)."""
        # Combine multiple sine waves with different frequencies and phases
        # to simulate a "liquid" or "cloth-like" distortion.
        
        # Large low-frequency waves (The "Flow")
        dx = math.sin(x / 1200 + seed_x) * 200 + math.cos(y / 1500 + seed_y) * 200
        dy = math.cos(x / 1300 + seed_y) * 200 + math.sin(y / 1100 + seed_x) * 200
        
        # Medium frequency waves (The "Ripple")
        dx += math.sin(x / 600 + seed_y * 2) * 80
        dy += math.cos(y / 700 + seed_x * 2) * 80
        
        # Small jitter (The "Imperfection")
        dx += math.sin(x / 200 + y / 200) * 20
        dy += math.cos(x / 200 - y / 200) * 20
        
        return dx, dy

    def transform_point(i, j):
        # 1. Base Hexagonal/Triangular Grid Coordinates
        # Staggered rows
        x = (i + (0.5 if j % 2 else 0)) * scale
        y = j * (scale * math.sqrt(3) / 2)
        
        # Center the grid roughly before transform
        x -= (cols * scale) / 2
        y -= (rows * scale * 0.8) / 2
        
        # 2. Apply Rotation (Base orientation)
        xr = x * math.cos(base_rotation) - y * math.sin(base_rotation)
        yr = x * math.sin(base_rotation) + y * math.cos(base_rotation)
        
        # 3. Apply Organic Distortion
        # We use the rotated coordinates to sample the distortion field
        dx, dy = get_organic_offset(xr, yr)
        
        final_x = offset_x + xr + dx
        final_y = offset_y + yr + dy
        
        return (final_x, final_y)

    # Generate Polygons
    polygons = []
    
    # Generate a slightly larger grid to ensure coverage after distortion
    for j in range(-2, rows + 2):
        for i in range(-2, cols + 2):
            pt_tl = transform_point(i, j)     # Top-Left
            pt_tr = transform_point(i+1, j)   # Top-Right
            
            if j % 2 == 0:
                pt_bl = transform_point(i, j+1)
                pt_br = transform_point(i+1, j+1)
                poly1 = [pt_tl, pt_tr, pt_bl]
                poly2 = [pt_tr, pt_br, pt_bl]
            else:
                pt_bl = transform_point(i, j+1)
                pt_br = transform_point(i+1, j+1)
                poly1 = [pt_tl, pt_tr, pt_br]
                poly2 = [pt_tl, pt_br, pt_bl]

            polygons.append(poly1)
            polygons.append(poly2)

    # Filter polygons that are on screen
    visible_polygons = []
    margin = 200 # Large margin because distortion can pull things in
    
    for poly in polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Check intersection with screen rectangle
        if max_x < -margin or min_x > WIDTH + margin or max_y < -margin or min_y > HEIGHT + margin:
            continue
            
        visible_polygons.append(poly)

    # Helper to draw gradient polygon
    def draw_gradient_polygon(draw_obj, poly_coords, color_start, color_end, steps=40):
        # Choose a random vertex as the "center" (convergence point)
        target_vertex = random.choice(poly_coords)
        tx, ty = target_vertex
        
        for i in range(steps):
            ratio = i / steps
            # Interpolate color
            r = int(color_start[0] * (1 - ratio) + color_end[0] * ratio)
            g = int(color_start[1] * (1 - ratio) + color_end[1] * ratio)
            b = int(color_start[2] * (1 - ratio) + color_end[2] * ratio)
            color = (r, g, b)
            
            # Interpolate points towards target vertex
            current_points = []
            for px, py in poly_coords:
                nx = px * (1 - ratio) + tx * ratio
                ny = py * (1 - ratio) + ty * ratio
                current_points.append((nx, ny))
            
            if len(current_points) >= 3:
                draw_obj.polygon(current_points, fill=color)

    # 3. Draw Polygons
    c_start = (0, 0, 0)
    c_end = (64, 62, 63)
    
    for poly in visible_polygons:
        draw_gradient_polygon(draw, poly, c_start, c_end, steps=40)
        
        # Draw Tapered Edges
        for i in range(len(poly)):
            p_start = poly[i]
            p_end = poly[(i + 1) % len(poly)]
            
            # Vector
            vx = p_end[0] - p_start[0]
            vy = p_end[1] - p_start[1]
            length = math.sqrt(vx*vx + vy*vy)
            if length == 0: continue
            
            # Normal vector (normalized)
            nx = -vy / length
            ny = vx / length
            
            # Midpoint
            mx = (p_start[0] + p_end[0]) / 2
            my = (p_start[1] + p_end[1]) / 2
            
            # Max width at center
            max_width = 3.0
            
            poly_shape = [
                p_start,
                (mx + nx * max_width, my + ny * max_width),
                p_end,
                (mx - nx * max_width, my - ny * max_width)
            ]
            
            draw.polygon(poly_shape, fill=ACCENT_RED)

    # 4. Subtle Overlay/Vignette
    draw.polygon([(0, 0), (400, 0), (0, 400)], fill=(0, 0, 0, 120))
    draw.polygon([(WIDTH, 0), (WIDTH-400, 0), (WIDTH, 400)], fill=(0, 0, 0, 120))
    draw.polygon([(0, HEIGHT), (0, HEIGHT-400), (400, HEIGHT)], fill=(0, 0, 0, 120))
    draw.polygon([(WIDTH, HEIGHT), (WIDTH, HEIGHT-400), (WIDTH-400, HEIGHT)], fill=(0, 0, 0, 120))

    return img

def create_graph(data_points):
    """Creates the MMR graph using matplotlib."""
    # Set style to dark
    plt.style.use('dark_background')
    
    # Create figure with transparent background
    # Target size: 900x600
    width_px = 900
    height_px = 600
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Define margins to fix the plot area
    m_left, m_right = 0.12, 0.95
    m_top, m_bottom = 0.95, 0.15
    fig.subplots_adjust(left=m_left, right=m_right, top=m_top, bottom=m_bottom)
    
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
    # Load Custom Font
    font_prop = fm.FontProperties(fname=FONT_PATH)
    
    ax.set_ylabel('MMR', color='white', fontsize=12, fontproperties=font_prop)
    ax.set_xlabel('Events Played', color='white', fontsize=12, fontproperties=font_prop)
    
    # Tick params
    ax.tick_params(axis='both', colors='white', labelsize=10)
    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)
    
    # Save to buffer
    buf = io.BytesIO()
    # Remove bbox_inches='tight' to keep fixed size and positioning
    plt.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    plt.close(fig)
    
    # Calculate plot area in pixels for the blur effect
    # PIL y-axis is top-down, Matplotlib is bottom-up
    # Top pixel = (1 - top) * height
    # Bottom pixel = (1 - bottom) * height
    
    plot_box = (
        int(m_left * width_px),
        int((1 - m_top) * height_px),
        int(m_right * width_px),
        int((1 - m_bottom) * height_px)
    )
    
    return Image.open(buf), plot_box

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

def draw_gauge(img, center, radius, value, max_value, color=ACCENT_RED):
    """Draws the circular MMR gauge with anti-aliasing."""
    # Supersampling factor
    scale = 4
    
    # Calculate dimensions for the temporary image
    width = 30
    padding = 20
    size = int(radius * 2 + padding)
    
    # Create high-res temporary image
    temp_size = (size * scale, size * scale)
    temp_img = Image.new('RGBA', temp_size, (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Calculate center in temp image
    temp_center = (temp_size[0] // 2, temp_size[1] // 2)
    temp_radius = radius * scale
    temp_width = width * scale
    
    # Bounding box for arc
    bbox = [
        temp_center[0] - temp_radius,
        temp_center[1] - temp_radius,
        temp_center[0] + temp_radius,
        temp_center[1] + temp_radius
    ]
    
    # Background arc (darker/remaining)
    bg_color = (100, 40, 50) # Light red for remaining part
    temp_draw.arc(bbox, start=135, end=405, fill=bg_color, width=int(temp_width))
    
    # Progress arc
    start_angle = 135
    ratio = min(1.0, max(0.0, value / max_value))
    end_angle = start_angle + (405 - 135) * ratio
    
    if ratio > 0:
        temp_draw.arc(bbox, start=start_angle, end=end_angle, fill=color, width=int(temp_width))

    # Rounded caps
    def get_point(angle_deg):
        angle_rad = math.radians(angle_deg)
        # Match original logic: effective_radius = radius - width / 2
        effective_radius = temp_radius - temp_width / 2
        x = temp_center[0] + effective_radius * math.cos(angle_rad)
        y = temp_center[1] + effective_radius * math.sin(angle_rad)
        return (x, y)

    cap_radius = temp_width / 2

    # Background caps
    start_point = get_point(135)
    temp_draw.ellipse([start_point[0] - cap_radius, start_point[1] - cap_radius, 
                       start_point[0] + cap_radius, start_point[1] + cap_radius], fill=bg_color)
    
    bg_end_point = get_point(405)
    temp_draw.ellipse([bg_end_point[0] - cap_radius, bg_end_point[1] - cap_radius, 
                       bg_end_point[0] + cap_radius, bg_end_point[1] + cap_radius], fill=bg_color)

    # Progress caps
    if ratio > 0:
        # Start cap (overwrites background start cap)
        temp_draw.ellipse([start_point[0] - cap_radius, start_point[1] - cap_radius, 
                           start_point[0] + cap_radius, start_point[1] + cap_radius], fill=color)

        # End cap
        end_point = get_point(end_angle)
        temp_draw.ellipse([end_point[0] - cap_radius, end_point[1] - cap_radius, 
                           end_point[0] + cap_radius, end_point[1] + cap_radius], fill=color)
                       
    # Resize down
    target_size = (size, size)
    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS
        
    temp_img = temp_img.resize(target_size, resample=resample_method)
    
    # Paste onto main image
    paste_x = int(center[0] - size // 2)
    paste_y = int(center[1] - size // 2)
    
    img.paste(temp_img, (paste_x, paste_y), temp_img)

def draw_stats_card(config):
    """
    Generates the stats card image based on the provided configuration.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Generate background
    bg = create_background()
    draw = ImageDraw.Draw(bg)
    
    # Generate Graph
    graph_data = config.get('graph_data', [])
    if len(graph_data) == 0:
        # Default dummy data if none provided
        graph_data = np.linspace(10000, 15000, 100)

    graph_img, plot_box = create_graph(graph_data)
    
    # Position for the graph
    graph_x, graph_y = 950, 350
    
    # --- Frosted Glass Effect for Graph ---
    # Calculate the absolute coordinates for the blur on the background
    # plot_box is relative to the graph image (0,0 is top-left of graph image)
    # We need to offset it by graph_x, graph_y
    
    blur_box = (
        graph_x + plot_box[0],
        graph_y + plot_box[1],
        graph_x + plot_box[2],
        graph_y + plot_box[3]
    )
    
    # 1. Crop the background
    crop = bg.crop(blur_box)
    
    # 2. Apply Blur
    blur = crop.filter(ImageFilter.GaussianBlur(radius=15))
    
    # 3. Apply Dark Overlay (Semi-transparent)
    # Create a solid color image with alpha
    overlay = Image.new('RGBA', blur.size, (40, 40, 40, 100))
    
    # Composite overlay onto blur
    blur.paste(overlay, (0, 0), overlay)
    
    # 4. Paste back onto background
    bg.paste(blur, blur_box)
    
    # 5. Paste graph on top
    bg.paste(graph_img, (graph_x, graph_y), graph_img)

    # --- Text and Stats ---
    
    # Header
    # "Mario Kart World" Logo Placeholder
    draw.rectangle([50, 50, 350, 150], fill=(0, 0, 0), outline=(255, 255, 255))
    draw_text(draw, "MARIO KART\nWORLD", (200, 100), 40, anchor="mm")
    
    # Player Name & Season
    draw_text(draw, config.get('player_name', 'Player'), (WIDTH // 2, 80), font_size=60, anchor="mm")
    draw_text(draw, config.get('season', 'Season X'), (WIDTH // 2, 140), font_size=40, anchor="mm")
    
    # Flag Placeholder
    draw.rectangle([WIDTH // 2 - 40, 170, WIDTH // 2 + 40, 230], fill=(255, 255, 255))
    draw.ellipse([WIDTH // 2 - 15, 185, WIDTH // 2 + 15, 215], fill=(200, 0, 0)) # Japan flag circle
    
    # Character Icon Placeholder (Top Right)
    draw.rectangle([WIDTH - 200, 50, WIDTH - 50, 200], fill=(0, 0, 0), outline=(255, 255, 255))
    draw_text(draw, config.get('character_name', 'Character'), (WIDTH - 125, 125), font_size=30, anchor="mm")

    # MMR Gauge (Left Side)
    gauge_center = (400, 350)
    mmr = config.get('mmr', 0)
    max_mmr = config.get('max_mmr_gauge', 20000)
    draw_gauge(bg, gauge_center, 135, mmr, max_mmr)
    draw_text(draw, "MMR", (400, 290), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(mmr), (400, 360), font_size=70, anchor="mm")

    # Stats Columns
    # Left Column
    start_y = 500
    gap_y = 150
    
    # Avg (12P)
    draw_text(draw, "Avg (12P)", (200, start_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('avg_12p', '-')), (200, start_y + 50), font_size=60, anchor="mm")
    
    # Partner (12P)
    draw_text(draw, "Partner (12P)", (200, start_y + gap_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('partner_12p', '-')), (200, start_y + gap_y + 50), font_size=60, anchor="mm")
    
    # Largest Gain
    draw_text(draw, "Largest Gain", (200, start_y + gap_y * 2), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('largest_gain', '-')), (200, start_y + gap_y * 2 + 50), font_size=60, anchor="mm", color=(100, 255, 100))

    # Middle Column
    mid_x = 500
    
    # Avg (24P)
    draw_text(draw, "Avg (24P)", (mid_x, start_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('avg_24p', '-')), (mid_x, start_y + 50), font_size=60, anchor="mm")
    
    # W-L
    draw_text(draw, "W-L", (mid_x, start_y + gap_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('win_loss', '-')), (mid_x, start_y + gap_y + 50), font_size=60, anchor="mm")
    
    # Largest Loss
    draw_text(draw, "Largest Loss", (mid_x, start_y + gap_y * 2), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('largest_loss', '-')), (mid_x, start_y + gap_y * 2 + 50), font_size=60, anchor="mm", color=(255, 80, 80))

    # Right Column (Center-Right)
    right_x = 800
    
    # Peak MMR
    draw_text(draw, "Peak MMR", (right_x, 300), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('peak_mmr', '-')), (right_x, 350), font_size=60, anchor="mm", color=ACCENT_RED)

    # Top Score
    draw_text(draw, "Top Score", (right_x, start_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('top_score', '-')), (right_x, start_y + 50), font_size=60, anchor="mm")
    
    # Events
    draw_text(draw, "Events", (right_x, start_y + gap_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('events', '-')), (right_x, start_y + gap_y + 50), font_size=60, anchor="mm")
    
    # Rank
    draw_text(draw, "Rank", (right_x, start_y + gap_y * 2), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, str(config.get('rank', '-')), (right_x, start_y + gap_y * 2 + 50), font_size=60, anchor="mm")

    # Save final image
    output_path = config.get('output_path', 'output/stats_card_final.png')
    bg.save(output_path)
    print(f"Stats card generated and saved to {output_path}")

def main():
    # Example Configuration
    # You can modify these values to generate different cards
    config = {
        "player_name": "Kusaan",
        "season": "Season 1",
        "character_name": "Lakitu",
        
        # Stats
        "mmr": 14776,
        "max_mmr_gauge": 15000,
        "peak_mmr": 14776,
        "rank": 1,
        
        "avg_12p": "87.5",
        "avg_24p": "87.6",
        "partner_12p": "83.0",
        "win_loss": "120-49",
        "top_score": 123,
        "events": "2-167",
        "largest_gain": "+182",
        "largest_loss": "-338",
        
        # Graph Data (List of MMR values)
        "graph_data": np.linspace(10500, 14776, 167),
        
        # Output
        "output_path": "output/stats_card_final.png"
    }
    
    draw_stats_card(config)

if __name__ == "__main__":
    main()
