import os
from PIL import Image, ImageDraw, ImageFont
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
    """Creates the geometric background using Perturbed 3-Family Line Arrangement."""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img, 'RGBA')

    from shapely.geometry import LineString, Polygon, MultiPolygon, GeometryCollection
    from shapely.ops import unary_union, polygonize

    # 1. Generate Lines (3 Families)
    lines = []
    
    # Canvas bounds for clipping
    canvas_box = (0, 0, WIDTH, HEIGHT)
    canvas_poly = Polygon([(0, 0), (WIDTH, 0), (WIDTH, HEIGHT), (0, HEIGHT)])
    
    # Extended bounds to ensure lines cover the canvas
    margin = 600
    ext_min_x, ext_min_y = -margin, -margin
    ext_max_x, ext_max_y = WIDTH + margin, HEIGHT + margin
    
    # Parameters for "Organic" feel
    # Base angles (shifted from 0, 60, 120 to look less mechanical)
    base_rotation = np.random.uniform(-15, 15)
    angles = [0 + base_rotation, 60 + base_rotation, 120 + base_rotation]
    
    # Spacing
    avg_spacing = 250
    
    for base_angle in angles:
        # Convert to radians
        theta = math.radians(base_angle)
        
        # Normal vector
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        # Tangent vector (direction of line)
        tx = -ny
        ty = nx
        
        # Determine range of offsets to cover the extended canvas
        # Project corners to normal axis to find min/max offset
        corners = [
            (ext_min_x, ext_min_y), (ext_max_x, ext_min_y),
            (ext_max_x, ext_max_y), (ext_min_x, ext_max_y)
        ]
        projections = [x * nx + y * ny for x, y in corners]
        min_p = min(projections)
        max_p = max(projections)
        
        # Generate lines along the normal axis
        current_p = min_p
        while current_p < max_p:
            # Add random jitter to spacing
            spacing = avg_spacing * np.random.uniform(0.7, 1.3)
            current_p += spacing
            
            # Add random jitter to angle for this specific line
            line_angle_jitter = math.radians(np.random.uniform(-2, 2))
            line_theta = theta + line_angle_jitter
            lnx = math.cos(line_theta)
            lny = math.sin(line_theta)
            ltx = -lny
            lty = lnx
            
            # Point on line
            px = current_p * lnx
            py = current_p * lny
            
            # Create long line segment
            p1 = (px + ltx * 4000, py + lty * 4000)
            p2 = (px - ltx * 4000, py - lty * 4000)
            
            lines.append(LineString([p1, p2]))

    # 2. Polygonize
    # Union all lines to find intersections and split them
    all_lines = unary_union(lines)
    
    # Create polygons from the unioned lines
    polys = list(polygonize(all_lines))
    
    # 3. Filter and Triangulate
    final_polygons = []
    
    for poly in polys:
        # Intersect with canvas
        if not poly.intersects(canvas_poly):
            continue
            
        intersection = poly.intersection(canvas_poly)
        
        # Handle MultiPolygon or GeometryCollection results
        parts = []
        if isinstance(intersection, Polygon):
            parts.append(intersection)
        elif isinstance(intersection, (MultiPolygon, GeometryCollection)):
            for geom in intersection.geoms:
                if isinstance(geom, Polygon):
                    parts.append(geom)
                    
        for part in parts:
            if part.is_empty or part.area < 100: # Filter tiny slivers
                continue
                
            # Triangulate if not a triangle (approximate check by vertex count)
            # Simplify slightly to remove collinear points that might increase vertex count
            simplified = part.simplify(1.0)
            coords = list(simplified.exterior.coords)
            if coords[0] == coords[-1]:
                coords.pop()
                
            if len(coords) > 3:
                # Simple fan triangulation or ear clipping
                # Since these are convex polygons (from line arrangement), fan from first vertex works well enough
                # OR split by shortest diagonal for better shape
                
                # Let's just use a simple fan for robustness as they are convex
                p0 = coords[0]
                for i in range(1, len(coords) - 1):
                    p1 = coords[i]
                    p2 = coords[i+1]
                    triangle = Polygon([p0, p1, p2])
                    if not triangle.is_empty and triangle.area > 10:
                        final_polygons.append(triangle)
            else:
                final_polygons.append(part)

    # Helper to draw gradient polygon (Same as before)
    def draw_gradient_polygon(draw_obj, poly_coords, color_start, color_end, steps=20):
        # Centroid
        cx = sum(p[0] for p in poly_coords) / len(poly_coords)
        cy = sum(p[1] for p in poly_coords) / len(poly_coords)
        
        for i in range(steps):
            ratio = i / steps
            # Interpolate color
            r = int(color_start[0] * (1 - ratio) + color_end[0] * ratio)
            g = int(color_start[1] * (1 - ratio) + color_end[1] * ratio)
            b = int(color_start[2] * (1 - ratio) + color_end[2] * ratio)
            color = (r, g, b)
            
            # Interpolate points towards centroid
            current_points = []
            for px, py in poly_coords:
                nx = px * (1 - ratio) + cx * ratio
                ny = py * (1 - ratio) + cy * ratio
                current_points.append((nx, ny))
            
            if len(current_points) >= 3:
                draw_obj.polygon(current_points, fill=color)

    # 4. Draw Polygons
    c_start = (0, 0, 0)
    c_end = (34, 32, 33)
    
    for poly in final_polygons:
        coords = list(poly.exterior.coords)
        if coords[0] == coords[-1]:
            coords.pop()
            
        draw_gradient_polygon(draw, coords, c_start, c_end, steps=20)
        
        # Draw Tapered Edges
        for i in range(len(coords)):
            p_start = coords[i]
            p_end = coords[(i + 1) % len(coords)]
            
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
            max_width = 2.0
            
            poly_shape = [
                p_start,
                (mx + nx * max_width, my + ny * max_width),
                p_end,
                (mx - nx * max_width, my - ny * max_width)
            ]
            
            draw.polygon(poly_shape, fill=ACCENT_RED)

    # 5. Subtle Overlay/Vignette
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
    start_angle = 135
    end_angle = 405
    width = 30
    draw.arc(bbox, start=start_angle, end=end_angle, fill=color, width=width)

    # Rounded caps
    # Convert angles to radians
    # PIL angles are clockwise from 3 o'clock (0 degrees)
    # Math angles are usually counter-clockwise from 3 o'clock, but here we just need to match PIL's coordinate system
    # x = cx + r * cos(theta)
    # y = cy + r * sin(theta)
    
    def get_point(angle_deg):
        angle_rad = math.radians(angle_deg)
        # Pillow's arc with width is drawn *inside* the bbox, so the center of the stroke
        # is at radius - width / 2
        effective_radius = radius - width / 2
        x = center[0] + effective_radius * math.cos(angle_rad)
        y = center[1] + effective_radius * math.sin(angle_rad)
        return (x, y)

    # Start cap
    start_point = get_point(start_angle)
    draw.ellipse([start_point[0] - width/2, start_point[1] - width/2, 
                  start_point[0] + width/2, start_point[1] + width/2], fill=color)

    # End cap
    end_point = get_point(end_angle)
    draw.ellipse([end_point[0] - width/2, end_point[1] - width/2, 
                  end_point[0] + width/2, end_point[1] + width/2], fill=color)

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Generate background
    bg = create_background()
    draw = ImageDraw.Draw(bg)
    
    # Generate Graph
    # Simulate some data that looks like the image (rising trend)
    # Start around 10500, end around 14776, with some noise
    np.random.seed(42)
    # Events 2-167 means 166 events played? Or just 0 to 167 range. 
    # Image x-axis goes up to 160+, so let's use 167 points.
    x = np.linspace(0, 167, 167)
    # Base trend
    y = 10500 + (x * 25) 
    # Add noise/fluctuations
    noise = np.cumsum(np.random.randn(167) * 50)
    y = y + noise
    # Ensure it ends near 14776 as per image
    y = y - (y[-1] - 14776)
    
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
    draw_gauge(draw, gauge_center, 135, 14776, 20000)
    draw_text(draw, "MMR", (400, 290), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "14776", (400, 360), font_size=70, anchor="mm")

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
    draw_text(draw, "87.6", (mid_x, start_y + 50), font_size=60, anchor="mm")
    
    # W-L
    draw_text(draw, "W-L", (mid_x, start_y + gap_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "120-49", (mid_x, start_y + gap_y + 50), font_size=60, anchor="mm")
    
    # Largest Loss
    draw_text(draw, "Largest Loss", (mid_x, start_y + gap_y * 2), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "-338", (mid_x, start_y + gap_y * 2 + 50), font_size=60, anchor="mm", color=(255, 80, 80))

    # Right Column (Center-Right)
    right_x = 800
    
    # Peak MMR
    draw_text(draw, "Peak MMR", (right_x, 300), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "14776", (right_x, 350), font_size=60, anchor="mm", color=ACCENT_RED)

    # Top Score
    draw_text(draw, "Top Score", (right_x, start_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "123", (right_x, start_y + 50), font_size=60, anchor="mm")
    
    # Events
    draw_text(draw, "Events", (right_x, start_y + gap_y), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "2-167", (right_x, start_y + gap_y + 50), font_size=60, anchor="mm")
    
    # Rank
    draw_text(draw, "Rank", (right_x, start_y + gap_y * 2), font_size=30, anchor="mm", color=(200, 200, 200))
    draw_text(draw, "1", (right_x, start_y + gap_y * 2 + 50), font_size=60, anchor="mm")

    # Save final image
    bg.save('output/stats_card_final.png')
    print("Stats card generated and saved to output/stats_card_final.png")

if __name__ == "__main__":
    main()
