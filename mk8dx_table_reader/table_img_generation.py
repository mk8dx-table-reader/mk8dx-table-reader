"""
Mario Kart 8 Deluxe Results Table Image Generator

This module generates dynamic result table images for Mario Kart races,
displaying player nicknames, scores, and position changes from previous races.
"""

from PIL import Image, ImageDraw, ImageFont
import os
from typing import List, Dict, Optional, Tuple


class PlayerResult:
    """Represents a single player's race result."""
    
    def __init__(self, nickname: str, score: int, position_change: int = 0):
        """
        Initialize a player result.
        
        Args:
            nickname: Player's nickname
            score: Player's score for this race
            position_change: Change in position from previous race (positive = up, negative = down, 0 = same)
        """
        self.nickname = nickname
        self.score = score
        self.position_change = position_change


class MarioKartTableGenerator:
    """Generates Mario Kart results table images."""
    
    def __init__(self):
        """Initialize the table generator with default styling."""
        # Color scheme (Mario Kart inspired)
        self.colors = {
            'background': '#0f0f1c',  # Deep blue
            'header': "#565566",      # Blue
            'row_even': "#10002b",    # White
            'row_odd': "#000000",     # Light gray
            'text_dark': '#000000',   # Black
            'text_light': '#ffffff',  # White
            'border': '#9e9e9e',      # Gray
            'position_up': "#34aa3f", # Green
            'position_down': '#dd2f24', # Red
            'position_same': '#757575'  # Gray
        }
        
        # Dimensions and spacing
        self.cell_height = 45  # Reduced from 60 to make rows slimmer
        self.header_height = 80
        self.margin = 20
        self.border_width = 2
        self.row_padding = 8  # Padding between player rows
        
        # Column widths
        self.col_widths = {
            'position': 80,
            'nickname': 200,
            'score': 100,
            'change': 100
        }
    
    def _get_font(self, size: int, bold: bool = False) -> ImageFont.ImageFont:
        """
        Get a font with the specified size.
        
        Args:
            size: Font size
            bold: Whether to use bold font
            
        Returns:
            Font object
        """
        try:
            # Try to use a system font
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/Windows/Fonts/arial.ttf"
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    return ImageFont.truetype(font_path, size)
            
            # Fallback to default font
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()
    
    def _calculate_image_dimensions(self, num_players: int) -> Tuple[int, int]:
        """
        Calculate the required image dimensions based on number of players.
        
        Args:
            num_players: Number of players in the results
            
        Returns:
            Tuple of (width, height)
        """
        total_width = (
            self.col_widths['position'] + 
            self.col_widths['nickname'] + 
            self.col_widths['score'] + 
            self.col_widths['change'] + 
            self.margin * 2
        )
        
        total_height = (
            self.header_height + 
            (self.cell_height * (num_players - 1)) + 
            (self.row_padding * (num_players - 1)) +  # Padding between rows
            self.margin * 2 - 10
        )
        
        return total_width, total_height
    
    def _draw_header(self, draw: ImageDraw.ImageDraw, width: int) -> int:
        """
        Draw the table header.
        
        Args:
            draw: ImageDraw object
            width: Total image width
            
        Returns:
            Y position after header
        """
        header_y = self.margin
        
        # Header background
        draw.rectangle(
            [self.margin, header_y, width - self.margin, header_y + self.header_height],
            fill=self.colors['header'],
            outline=self.colors['border'],
            width=self.border_width
        )
        
        # Header text
        font = self._get_font(20, bold=True)
        
        # Column headers
        headers = ['Pos', 'Player', 'Score', 'Change']
        x_positions = [
            self.margin + self.col_widths['position'] // 2,
            self.margin + self.col_widths['position'] + self.col_widths['nickname'] // 4,
            self.margin + self.col_widths['position'] + self.col_widths['nickname'] + self.col_widths['score'] // 2,
            self.margin + self.col_widths['position'] + self.col_widths['nickname'] + self.col_widths['score'] + self.col_widths['change'] // 2
        ]
        
        for i, (header, x_pos) in enumerate(zip(headers, x_positions)):
            # Get text size for centering
            bbox = draw.textbbox((0, 0), header, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.text(
                (x_pos - text_width // 2, header_y + (self.header_height - text_height) // 2),
                header,
                fill=self.colors['text_light'],
                font=font
            )
        
        return header_y + self.header_height
    
    def _draw_player_row(self, draw: ImageDraw.ImageDraw, player: PlayerResult, 
                        position: int, y_pos: int, width: int, is_even_row: bool):
        """
        Draw a single player row.
        
        Args:
            draw: ImageDraw object
            player: Player result data
            position: Player's position (1-based)
            y_pos: Y position to draw the row
            width: Total image width
            is_even_row: Whether this is an even-numbered row
        """
        # Row background
        bg_color = self.colors['row_even'] if is_even_row else self.colors['row_odd']
        draw.rectangle(
            [self.margin, y_pos, width - self.margin, y_pos + self.cell_height],
            fill=bg_color,
            outline=self.colors['border'],
            width=1
        )
        
        # Fonts
        font_regular = self._get_font(18)
        font_bold = self._get_font(18, bold=True)
        
        # Column positions
        col_x_positions = [
            self.margin,
            self.margin + self.col_widths['position'],
            self.margin + self.col_widths['position'] + self.col_widths['nickname'],
            self.margin + self.col_widths['position'] + self.col_widths['nickname'] + self.col_widths['score']
        ]
        
        # Position
        pos_text = str(position)
        bbox = draw.textbbox((0, 0), pos_text, font=font_bold)
        text_height = bbox[3] - bbox[1]
        draw.text(
            (col_x_positions[0] + self.col_widths['position'] // 2 - (bbox[2] - bbox[0]) // 2,
             y_pos + (self.cell_height - text_height) // 2),
            pos_text,
            fill=self.colors['text_light'],
            font=font_bold
        )
        
        # Nickname
        nickname_text = player.nickname[:15] + "..." if len(player.nickname) > 15 else player.nickname
        bbox = draw.textbbox((0, 0), nickname_text, font=font_regular)
        text_height = bbox[3] - bbox[1]
        draw.text(
            (col_x_positions[1] + 10,
             y_pos + (self.cell_height - text_height) // 2),
            nickname_text,
            fill=self.colors['text_light'],
            font=font_regular
        )
        
        # Score
        score_text = str(player.score)
        bbox = draw.textbbox((0, 0), score_text, font=font_regular)
        text_height = bbox[3] - bbox[1]
        draw.text(
            (col_x_positions[2] + self.col_widths['score'] // 2 - (bbox[2] - bbox[0]) // 2,
             y_pos + (self.cell_height - text_height) // 2),
            score_text,
            fill=self.colors['text_light'],
            font=font_regular
        )
        
        # Position change
        if player.position_change > 0:
            change_text = f"↑{player.position_change}"
            change_color = self.colors['position_up']
        elif player.position_change < 0:
            change_text = f"↓{abs(player.position_change)}"
            change_color = self.colors['position_down']
        else:
            change_text = "="
            change_color = self.colors['position_same']
        
        bbox = draw.textbbox((0, 0), change_text, font=font_bold)
        text_height = bbox[3] - bbox[1]
        draw.text(
            (col_x_positions[3] + self.col_widths['change'] // 2 - (bbox[2] - bbox[0]) // 2,
             y_pos + (self.cell_height - text_height) // 2),
            change_text,
            fill=change_color,
            font=font_bold
        )
    
    def generate_results_table(self, players: List[PlayerResult], 
                             title: str = "Mario Kart 8 Deluxe Results",
                             output_path: str = "mario_kart_results.png") -> str:
        """
        Generate a results table image.
        
        Args:
            players: List of player results (should be sorted by final position)
            title: Title for the results table
            output_path: Path where to save the generated image
            
        Returns:
            Path to the generated image
        """
        if not players:
            raise ValueError("At least one player result is required")
        
        if len(players) > 12:
            raise ValueError("Maximum 12 players supported")
        
        # Calculate image dimensions
        width, height = self._calculate_image_dimensions(len(players))
        
        # Add space for title
        title_height = 60
        height += title_height
        
        # Create image
        img = Image.new('RGB', (width, height), self.colors['background'])
        draw = ImageDraw.Draw(img)
        
        # Draw title
        # title_font = self._get_font(32, bold=True)
        # bbox = draw.textbbox((0, 0), title, font=title_font)
        # title_width = bbox[2] - bbox[0]
        # title_text_height = bbox[3] - bbox[1]
        
        # draw.text(
        #     ((width - title_width) // 2, (title_height - title_text_height) // 2),
        #     title,
        #     fill=self.colors['text_light'],
        #     font=title_font
        # )
        
        # Adjust starting position for table
        current_y = title_height
        
        # Draw header
        # current_y = self._draw_header(draw, width) + title_height
        current_y = self._draw_header(draw, width)

        
        # Draw player rows
        current_y += self.row_padding
        for i, player in enumerate(players):
            self._draw_player_row(
                draw, player, i + 1, current_y, width, i % 2 == 0
            )
            current_y += self.cell_height + self.row_padding  # Add padding after each row
        
        # Save image
        img.save(output_path, 'PNG', quality=95)
        return output_path


def create_sample_results():
    """Create sample results for testing."""
    sample_players = [
        PlayerResult("Luigi", 458, 2),      # Moved up 2 positions
        PlayerResult("Mario", 445, -1),     # Moved down 1 position
        PlayerResult("Peach", 432, 0),      # Same position
        PlayerResult("Bowser", 420, 1),     # Moved up 1 position
        PlayerResult("Yoshi", 415, -2),     # Moved down 2 positions
        PlayerResult("Toad", 398, 3),       # Moved up 3 positions
        PlayerResult("Koopa", 385, 0),      # Same position
        PlayerResult("Shy Guy", 370, -1),   # Moved down 1 position
    ]
    return sample_players


def main():
    """Main function to demonstrate the table generator."""
    # Create generator instance
    generator = MarioKartTableGenerator()
    
    # Create sample results
    players = create_sample_results()
    
    # Generate the results table
    output_path = generator.generate_results_table(
        players=players,
        title="Mario Kart 8 Deluxe - Race Results",
        output_path="mario_kart_results.png"
    )
    
    print(f"Results table generated: {output_path}")
    
    # Example with different number of players
    small_race = players[:4]
    generator.generate_results_table(
        players=small_race,
        title="Mario Kart 8 Deluxe - Quick Race",
        output_path="mario_kart_results_small.png"
    )
    print("Small race results generated: mario_kart_results_small.png")


if __name__ == "__main__":
    main()