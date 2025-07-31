import pygame
import numpy as np
from typing import Dict
import os


class Renderer:
    """
    Pygame renderer for the world builder game.
    """
    
    def __init__(self, grid_size: int, tile_size: int = 50):
        self.grid_size = grid_size
        self.tile_size = tile_size
        
        # Calculate window size - more space for UI and buttons
        self.width = grid_size * tile_size + 320  # More space for UI
        self.height = grid_size * tile_size + 180  # More space for buttons at bottom
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("World Builder City")
        
        # Enhanced colors with better aesthetics
        self.colors = {
            0: (245, 245, 220),  # Empty - Beige (neutral ground)
            1: (34, 139, 34),    # Farm - Forest green (natural farm)
            2: (160, 82, 45),    # Lumberyard - Saddle brown
            3: (105, 105, 105),  # Quarry - Dim gray
            4: (255, 215, 0)     # House - Gold (premium look)
        }
        
        # Road and path colors
        self.road_color = (64, 64, 64)  # Dark gray for roads
        self.path_color = (139, 69, 19)  # Brown for paths
        self.ground_color = (245, 245, 220)  # Beige for empty ground
        
        # Load images for buildings - updated assets
        # Get the directory where this script is located and navigate to assets
        script_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(script_dir, "..", "assets", "Assets")
        
        # Load images with error handling for updated assets
        self.images = {}
        image_files = {
            1: "Farm.png",
            2: "Lumberyard.png", 
            3: "Quarry.png",
            4: "House.png"
        }
        
        for building_type, filename in image_files.items():
            try:
                image_path = os.path.join(assets_dir, filename)
                if os.path.exists(image_path):
                    self.images[building_type] = pygame.transform.scale(
                        pygame.image.load(image_path), 
                        (self.tile_size, self.tile_size)
                    )
                else:
                    print(f"Warning: {filename} not found, using color fallback")
                    self.images[building_type] = None
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                self.images[building_type] = None
        
        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Building names for display
        self.building_names = {
            0: "Empty",
            1: "Farm",
            2: "Lumberyard",
            3: "Quarry",
            4: "House"
        }
    
    def render(self, grid: np.ndarray, resources: Dict[str, float], 
               population: int, population_capacity: int, steps: int, win_steps: int):
        """Render the current game state with enhanced visuals."""
        # Clear screen with gradient background
        self._draw_background_gradient()
        
        # Draw grid with enhanced visuals
        self._draw_grid(grid)
        
        # Add city atmosphere effects
        self._draw_city_effects(grid)
        
        # Draw UI with enhanced styling
        self._draw_ui(resources, population, population_capacity, steps, win_steps)
        
        # Draw action buttons
        action = self._draw_action_buttons()
        
        # Check for failure condition and draw big red FAILED text
        if population <= 0:
            self._draw_failure_overlay()
        
        # Update display
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return -1  # Quit signal
        
        return action
    
    def _draw_background_gradient(self):
        """Draw a subtle gradient background."""
        for y in range(self.height):
            # Create a subtle gradient from top to bottom
            ratio = y / self.height
            r = int(240 + ratio * 15)  # Light blue to slightly darker
            g = int(248 + ratio * 7)
            b = int(255 + ratio * 0)
            color = (r, g, b)
            pygame.draw.line(self.screen, color, (0, y), (self.width, y))
    
    def _draw_city_effects(self, grid: np.ndarray):
        """Add atmospheric effects to the city."""
        # Count buildings for effects
        building_count = np.sum(grid > 0)
        
        # Add subtle particle effects around buildings
        if building_count > 0:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if grid[row, col] > 0:
                        x = col * self.tile_size
                        y = row * self.tile_size
                        
                        # Add subtle glow around buildings
                        glow_radius = 3
                        glow_color = (255, 255, 255, 30)
                        for i in range(glow_radius):
                            glow_rect = pygame.Rect(x - i, y - i, 
                                                  self.tile_size + 2*i, 
                                                  self.tile_size + 2*i)
                            pygame.draw.rect(self.screen, glow_color, glow_rect, 1)
        
        # Add city title overlay - position to avoid overlap
        if building_count >= 5:
            title_text = self.small_font.render("🏙️ Growing City", True, (70, 130, 180))
            # Position it above the grid, centered
            title_x = (self.grid_size * self.tile_size) // 2 - title_text.get_width() // 2
            title_y = 10
            title_rect = pygame.Rect(title_x, title_y, title_text.get_width(), title_text.get_height())
            
            # Semi-transparent background for title
            title_bg = pygame.Rect(title_rect.x - 5, title_rect.y - 2, 
                                  title_rect.width + 10, title_rect.height + 4)
            pygame.draw.rect(self.screen, (255, 255, 255, 200), title_bg)
            pygame.draw.rect(self.screen, (70, 130, 180), title_bg, 1)
            self.screen.blit(title_text, title_rect)
    
    def _draw_grid(self, grid: np.ndarray):
        """Draw the game grid with enhanced visuals including roads and paths."""
        # First, draw background ground
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.tile_size
                y = row * self.tile_size
                # Draw neutral ground background for all tiles
                pygame.draw.rect(self.screen, self.ground_color, (x, y, self.tile_size, self.tile_size))
        
        # Draw roads and paths
        self._draw_roads_and_paths(grid)
        
        # Draw buildings with enhanced visuals
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.tile_size
                y = row * self.tile_size
                building_type = grid[row, col]
                
                if building_type == 0:
                    # Draw empty tile with ground texture
                    self._draw_ground_tile(x, y)
                else:
                    # Draw building with shadow and glow effect
                    self._draw_building_with_effects(x, y, building_type)
                
                # Remove visual grid lines - keep only for system use
    
    def _draw_roads_and_paths(self, grid: np.ndarray):
        """Draw simple paths around farms only."""
        # Draw paths around farms
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.tile_size
                y = row * self.tile_size
                building_type = grid[row, col]
                
                if building_type == 1:  # Farm
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if (0 <= nr < self.grid_size and 0 <= nc < self.grid_size and 
                            grid[nr, nc] == 0):
                            # Draw farm path
                            if dr == 0:  # Horizontal path
                                path_y = y + self.tile_size // 2 - 2
                                pygame.draw.rect(self.screen, self.path_color, 
                                              (x, path_y, self.tile_size, 4))
                            else:  # Vertical path
                                path_x = x + self.tile_size // 2 - 2
                                pygame.draw.rect(self.screen, self.path_color, 
                                              (path_x, y, 4, self.tile_size))
    
    def _draw_ground_tile(self, x: int, y: int):
        """Draw a ground tile with subtle texture pattern."""
        # Base ground color (already drawn)
        # Add subtle ground texture with small dots
        for i in range(0, self.tile_size, 12):
            for j in range(0, self.tile_size, 12):
                if (i + j) % 24 == 0:  # Create subtle pattern
                    pygame.draw.circle(self.screen, (220, 220, 200), 
                                     (x + i + 6, y + j + 6), 1)
    
    def _draw_building_with_effects(self, x: int, y: int, building_type: int):
        """Draw building with shadow and glow effects."""
        # Draw shadow first
        shadow_offset = 3
        shadow_rect = (x + shadow_offset, y + shadow_offset, 
                      self.tile_size - shadow_offset, self.tile_size - shadow_offset)
        pygame.draw.rect(self.screen, (0, 0, 0, 100), shadow_rect)
        
        # Draw building
        img = self.images.get(building_type)
        if img:
            self.screen.blit(img, (x, y))
        else:
            # Draw colored rectangle with gradient effect
            color = self.colors[building_type]
            pygame.draw.rect(self.screen, color, (x, y, self.tile_size, self.tile_size))
            
            # Add highlight effect
            highlight_color = tuple(min(255, c + 50) for c in color)
            pygame.draw.rect(self.screen, highlight_color, 
                           (x + 2, y + 2, self.tile_size - 4, 4))
        
        # Add glow effect for houses
        if building_type == 4:  # House
            glow_rect = (x - 2, y - 2, self.tile_size + 4, self.tile_size + 4)
            pygame.draw.rect(self.screen, (255, 255, 0, 50), glow_rect, 2)
    
    def _draw_ui(self, resources: Dict[str, float], population: int, 
                  population_capacity: int, steps: int, win_steps: int):
        """Draw the enhanced UI elements with professional styling."""
        ui_x = self.grid_size * self.tile_size + 20
        ui_y = 20
        
        # Draw professional UI background panel
        panel_width = 300
        panel_height = self.height - 120  # Leave space for buttons
        panel_rect = pygame.Rect(ui_x - 10, ui_y - 10, panel_width, panel_height)
        
        # Professional dark theme background
        pygame.draw.rect(self.screen, (40, 44, 52), panel_rect)  # Dark background
        pygame.draw.rect(self.screen, (58, 64, 74), panel_rect, 2)  # Dark border
        
        # Professional title with dark theme
        title_bg = pygame.Rect(ui_x - 5, ui_y - 5, 270, 45)
        pygame.draw.rect(self.screen, (0, 122, 204), title_bg)  # Blue accent
        pygame.draw.rect(self.screen, (0, 102, 184), title_bg, 2)  # Darker border
        
        title = self.font.render("🏙️ World Builder City", True, (255, 255, 255))
        title_rect = title.get_rect(center=title_bg.center)
        self.screen.blit(title, title_rect)
        ui_y += 55
        
        # Resources section with professional styling
        resources_title = self.small_font.render("📦 RESOURCES", True, (255, 255, 255))
        self.screen.blit(resources_title, (ui_x, ui_y))
        ui_y += 30
        
        # Professional resource icons and colors
        resource_icons = {
            'food': ('🌾', (255, 193, 7)),    # Gold for food
            'wood': ('🪵', (139, 195, 74)),   # Green for wood
            'stone': ('🪨', (156, 39, 176))   # Purple for stone
        }
        
        for resource, amount in resources.items():
            icon, base_color = resource_icons.get(resource, ('📦', (255, 255, 255)))
            
            # Professional color determination
            if resource == 'food':
                if amount < population:
                    color = (244, 67, 54)  # Red if not enough food
                elif amount > population * 1.5:
                    color = (76, 175, 80)  # Green if excess food
                else:
                    color = (255, 193, 7)  # Gold for balanced
            else:
                color = base_color
            
            # Professional resource text ABOVE the bar
            text = self.small_font.render(f"{icon} {resource.upper()}: {amount:.1f}", True, (255, 255, 255))
            self.screen.blit(text, (ui_x, ui_y))
            ui_y += 20
            
            # Draw professional resource bar
            bar_width = 200
            bar_height = 15
            bar_x = ui_x + 10
            bar_y = ui_y
            
            # Professional background bar
            pygame.draw.rect(self.screen, (58, 64, 74), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (78, 84, 94), (bar_x, bar_y, bar_width, bar_height), 1)
            
            # Resource bar with professional styling
            max_amount = 50
            bar_fill = min(bar_width, (amount / max_amount) * bar_width)
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_fill, bar_height))
            
            ui_y += 25
        
        ui_y += 15
        
        # Professional population section
        pop_color = (76, 175, 80) if population > 0 else (244, 67, 54)  # Green/Red
        
        # Professional population text ABOVE the bar
        pop_text = self.small_font.render(f"👥 POPULATION: {population}/{population_capacity}", True, (255, 255, 255))
        self.screen.blit(pop_text, (ui_x, ui_y))
        ui_y += 20
        
        # Professional population progress bar
        if population_capacity > 0:
            pop_bar_width = 200
            pop_bar_height = 18
            pop_bar_x = ui_x + 10
            pop_bar_y = ui_y
            
            # Professional background
            pygame.draw.rect(self.screen, (58, 64, 74), (pop_bar_x, pop_bar_y, pop_bar_width, pop_bar_height))
            pygame.draw.rect(self.screen, (78, 84, 94), (pop_bar_x, pop_bar_y, pop_bar_width, pop_bar_height), 1)
            
            # Fill based on population
            pop_fill = (population / population_capacity) * pop_bar_width
            pygame.draw.rect(self.screen, pop_color, (pop_bar_x, pop_bar_y, pop_fill, pop_bar_height))
        
        ui_y += 25
        
        # Professional game progress section
        progress_text = self.small_font.render(f"⏱️ STEPS: {steps}", True, (255, 255, 255))
        self.screen.blit(progress_text, (ui_x, ui_y))
        ui_y += 25
        
        if win_steps > 0:
            win_progress = (win_steps / 50) * 100
            win_text = self.small_font.render(f"🏆 WIN PROGRESS: {win_steps}/50 ({win_progress:.0f}%)", True, (255, 193, 7))
            self.screen.blit(win_text, (ui_x, ui_y))
            ui_y += 30
        
        # Professional action buttons section
        ui_y += 20
        action_title = self.small_font.render("🔨 BUILDING ACTIONS", True, (255, 255, 255))
        self.screen.blit(action_title, (ui_x, ui_y))
        ui_y += 25
        
        actions = [
            ("⏭️ PASS", (158, 158, 158)),
            ("🌾 FARM (5 wood)", (76, 175, 80)),
            ("🪵 LUMBERYARD (3 stone)", (255, 152, 0)),
            ("🪨 QUARRY (5 wood)", (156, 39, 176)),
            ("🏠 HOUSE (10 wood + 5 stone)", (255, 193, 7))
        ]
        
        for i, (action, color) in enumerate(actions):
            # Draw professional action button background
            btn_rect = pygame.Rect(ui_x, ui_y, 240, 25)
            pygame.draw.rect(self.screen, color, btn_rect)
            pygame.draw.rect(self.screen, (78, 84, 94), btn_rect, 1)
            
            # Professional action text
            small_font = pygame.font.Font(None, 16)
            text = small_font.render(action, True, (255, 255, 255))
            self.screen.blit(text, (ui_x + 5, ui_y + 4))
            ui_y += 30
        
        # Professional win/lose conditions
        ui_y += 15
        conditions_bg = pygame.Rect(ui_x - 5, ui_y - 5, 240, 70)
        pygame.draw.rect(self.screen, (33, 33, 33), conditions_bg)  # Dark background
        pygame.draw.rect(self.screen, (255, 193, 7), conditions_bg, 2)  # Gold border
        
        conditions = [
            "🎯 WIN: Reach 20 population",
            "   and survive 50 steps",
            "💀 LOSE: Population = 0"
        ]
        
        tiny_font = pygame.font.Font(None, 14)
        for condition in conditions:
            text = tiny_font.render(condition, True, (255, 255, 255))
            self.screen.blit(text, (ui_x, ui_y))
            ui_y += 15
    
    def _draw_action_buttons(self):
        """Draw enhanced action buttons with better styling and hover effects."""
        button_y = self.height - 80  # Move buttons to bottom
        button_width = 95
        button_height = 35
        button_spacing = 8
        
        # Enhanced button labels with icons
        labels = ["⏭️ Pass", "🌾 Farm", "🪵 Lumber", "🪨 Quarry", "🏠 House"]
        colors = [(200, 200, 200), (50, 205, 50), (160, 82, 45), (105, 105, 105), (255, 215, 0)]
        hover_colors = [(220, 220, 220), (70, 225, 70), (180, 102, 65), (125, 125, 125), (255, 235, 20)]
        
        # Check for mouse clicks
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = pygame.mouse.get_pressed()[0]
        
        for i, (label, color, hover_color) in enumerate(zip(labels, colors, hover_colors)):
            button_x = 20 + i * (button_width + button_spacing)
            
            # Check if mouse is over button
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            is_hovered = button_rect.collidepoint(mouse_pos)
            
            # Draw button with enhanced styling
            if is_hovered:
                # Hover effect with glow
                glow_rect = pygame.Rect(button_x - 2, button_y - 2, button_width + 4, button_height + 4)
                pygame.draw.rect(self.screen, hover_color, glow_rect)
                pygame.draw.rect(self.screen, color, button_rect)
                pygame.draw.rect(self.screen, (255, 255, 255), button_rect, 2)
            else:
                # Normal button
                pygame.draw.rect(self.screen, color, button_rect)
                pygame.draw.rect(self.screen, (100, 100, 100), button_rect, 2)
            
            # Add shadow effect
            shadow_rect = pygame.Rect(button_x + 2, button_y + 2, button_width, button_height)
            pygame.draw.rect(self.screen, (0, 0, 0, 50), shadow_rect)
            
            # Draw label with better positioning
            text = self.small_font.render(label, True, (255, 255, 255))
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
            
            # Check for click
            if is_hovered and mouse_clicked:
                return i
        
        return None
    
    def _draw_failure_overlay(self):
        """Draw a big red FAILED overlay when episode fails."""
        # Create a semi-transparent dark overlay
        overlay = pygame.Surface((self.width, self.height))
        overlay.set_alpha(128)  # Semi-transparent
        overlay.fill((0, 0, 0))  # Black background
        self.screen.blit(overlay, (0, 0))
        
        # Create big red FAILED text
        big_font = pygame.font.Font(None, 120)  # Very large font
        failed_text = big_font.render("FAILED", True, (255, 0, 0))  # Bright red
        
        # Position the text in the center of the screen
        text_rect = failed_text.get_rect(center=(self.width // 2, self.height // 2))
        
        # Add a dark background behind the text for better visibility
        text_bg = pygame.Rect(text_rect.x - 20, text_rect.y - 10, 
                             text_rect.width + 40, text_rect.height + 20)
        pygame.draw.rect(self.screen, (0, 0, 0), text_bg)
        pygame.draw.rect(self.screen, (255, 0, 0), text_bg, 3)  # Red border
        
        # Draw the FAILED text
        self.screen.blit(failed_text, text_rect)
        
        # Add a smaller subtitle
        subtitle_font = pygame.font.Font(None, 36)
        subtitle_text = subtitle_font.render("Population reached zero!", True, (255, 255, 255))
        subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, self.height // 2 + 80))
        self.screen.blit(subtitle_text, subtitle_rect)
    
    def close(self):
        """Close the renderer."""
        pygame.quit() 