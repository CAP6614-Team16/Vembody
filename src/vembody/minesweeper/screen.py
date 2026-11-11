"""The Minesweeper window: draws the game and turns pixel clicks into moves.

Players and agents share this pixel-level interface: they see `screenshot()` and
act only through `click(px, py, button)`.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from ..errors import OffTargetInput
from .game import Minesweeper

CELL = 32
MARGIN = 12
HEADER = 52
FACE = 36

BACKGROUND = (192, 192, 192)
OPEN = (214, 214, 214)
LINE = (128, 128, 128)
NUMBER_COLORS = {
    1: (0, 0, 255),
    2: (0, 128, 0),
    3: (255, 0, 0),
    4: (0, 0, 128),
    5: (128, 0, 0),
    6: (0, 128, 128),
    7: (0, 0, 0),
    8: (128, 128, 128),
}


class MinesweeperScreen:
    """A playable Minesweeper window rendered to images."""

    def __init__(
        self,
        rows: int = 9,
        cols: int = 9,
        mines: int = 10,
        seed: int | None = None,
        *,
        benchmark: bool = False,
        grid_labels: bool = False,
        highlight: tuple[tuple[int, int], ...] = (),
    ):
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.seed = seed
        self.benchmark = benchmark
        self.grid_labels = grid_labels
        self.highlight = highlight
        self.grid_left = MARGIN + (24 if benchmark else 0)
        self.grid_top = HEADER + (24 if benchmark else 0)
        self.width = self.grid_left + MARGIN + cols * CELL
        self.height = self.grid_top + MARGIN + rows * CELL
        self.font = ImageFont.load_default(size=20)
        self.game = Minesweeper(rows, cols, mines, seed)

        # Face button that restarts the game, centered in the header
        left = (self.width - FACE) // 2
        top = (HEADER - FACE) // 2
        self.face = (left, top, left + FACE - 1, top + FACE - 1)

    def click(self, px: int, py: int, button: str) -> None:
        """Handle one mouse click; button is 'left' or 'right'."""
        if self.benchmark:
            cell = self.cell_at(px, py)
            if cell is None:
                raise OffTargetInput("Benchmark clicks must target a cell interior")
            if button == "left":
                if cell not in self.game.revealed:
                    self.game.reveal(cell)
            elif button == "right":
                self.game.toggle_flag(cell)
            else:
                raise ValueError("Unknown mouse button")
            return
        left, top, right, bottom = self.face
        if button == "left" and left <= px <= right and top <= py <= bottom:
            self.game = Minesweeper(self.rows, self.cols, self.mines, self.seed)
            return

        # Convert the pixel to a grid cell; clicks outside the grid do nothing
        row = (py - HEADER) // CELL
        col = (px - MARGIN) // CELL
        if px < MARGIN or py < HEADER or row >= self.rows or col >= self.cols:
            return
        if button == "left":
            self.game.reveal((row, col))
        else:
            self.game.toggle_flag((row, col))

    def cell_center(self, cell: tuple[int, int]) -> tuple[int, int]:
        row, col = cell
        return self.grid_left + col * CELL + CELL // 2, self.grid_top + row * CELL + CELL // 2

    def cell_at(self, px: int, py: int) -> tuple[int, int] | None:
        x, y = px - self.grid_left, py - self.grid_top
        if not (0 <= x < self.cols * CELL and 0 <= y < self.rows * CELL):
            return None
        if self.benchmark and not (2 <= x % CELL < CELL - 2 and 2 <= y % CELL < CELL - 2):
            return None
        return y // CELL, x // CELL

    def draw_raised(self, draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
        left, top, right, bottom = box
        draw.rectangle(box, fill=BACKGROUND, outline=LINE)
        draw.line((left + 1, top + 1, right - 1, top + 1), fill="white", width=2)
        draw.line((left + 1, top + 1, left + 1, bottom - 1), fill="white", width=2)
        draw.line((left + 2, bottom - 1, right - 1, bottom - 1), fill=LINE, width=2)
        draw.line((right - 1, top + 2, right - 1, bottom - 1), fill=LINE, width=2)

    def screenshot(self) -> Image.Image:
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        game = self.game

        # Mines left counter, digital-display style
        text = f"{self.mines - len(game.flagged):03d}"
        x, y = MARGIN, HEADER // 2
        box = draw.textbbox((x, y), text, font=self.font, anchor="lm")
        draw.rectangle((box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4), fill="black")
        draw.text((x, y), text, fill="red", font=self.font, anchor="lm")

        # Face: smile while playing, sunglasses on a win, X eyes on a loss
        left, top, right, bottom = self.face
        self.draw_raised(draw, self.face)
        cx, cy = (left + right) // 2, (top + bottom) // 2
        draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill="yellow", outline="black")
        if game.status == "lost":
            for ex in (cx - 5, cx + 5):
                draw.line((ex - 2, cy - 6, ex + 2, cy - 2), fill="black", width=2)
                draw.line((ex - 2, cy - 2, ex + 2, cy - 6), fill="black", width=2)
            draw.arc((cx - 6, cy + 3, cx + 6, cy + 11), 200, 340, fill="black", width=2)
        else:
            if game.status == "won":
                draw.rectangle((cx - 9, cy - 6, cx + 9, cy - 2), fill="black")
            else:
                draw.ellipse((cx - 6, cy - 6, cx - 3, cy - 3), fill="black")
                draw.ellipse((cx + 3, cy - 6, cx + 6, cy - 3), fill="black")
            draw.arc((cx - 7, cy - 4, cx + 7, cy + 8), 20, 160, fill="black", width=2)

        # Draw every cell
        for cell in ((row, col) for row in range(self.rows) for col in range(self.cols)):
            left = self.grid_left + cell[1] * CELL
            top = self.grid_top + cell[0] * CELL
            box = (left, top, left + CELL - 1, top + CELL - 1)
            cx, cy = left + CELL // 2, top + CELL // 2
            lost = game.status == "lost"

            # Open cells, plus every unflagged mine once the game is lost
            if cell in game.revealed or (lost and cell in game.mines and cell not in game.flagged):
                fill = (255, 0, 0) if cell == game.exploded else OPEN
                draw.rectangle(box, fill=fill, outline=LINE)
                if cell in game.mines:
                    draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill="black")
                    draw.line((cx - 11, cy, cx + 11, cy), fill="black", width=2)
                    draw.line((cx, cy - 11, cx, cy + 11), fill="black", width=2)
                elif count := game.adjacent_mines(cell):
                    color = NUMBER_COLORS[count]
                    draw.text((cx, cy), str(count), fill=color, font=self.font, anchor="mm")
                continue

            # Hidden cell, possibly flagged; wrong flags are crossed out after a loss
            self.draw_raised(draw, box)
            if cell in game.flagged:
                draw.line((cx, top + 7, cx, top + 24), fill="black", width=2)
                draw.line((cx - 6, top + 25, cx + 6, top + 25), fill="black", width=2)
                draw.polygon([(cx, top + 7), (cx, top + 17), (cx - 10, top + 12)], fill="red")
                if lost and cell not in game.mines:
                    draw.line((left + 6, top + 6, left + 25, top + 25), fill="black", width=2)
                    draw.line((left + 25, top + 6, left + 6, top + 25), fill="black", width=2)
        if self.grid_labels:
            for row in range(self.rows):
                _, cy = self.cell_center((row, 0))
                draw.text(
                    (self.grid_left - 14, cy),
                    chr(65 + row),
                    fill="black",
                    font=self.font,
                    anchor="mm",
                )
            for col in range(self.cols):
                cx, _ = self.cell_center((0, col))
                draw.text(
                    (cx, self.grid_top - 14),
                    str(col + 1),
                    fill="black",
                    font=self.font,
                    anchor="mm",
                )
        for row, col in self.highlight:
            cx, cy = self.cell_center((row, col))
            draw.rectangle((cx - 14, cy - 14, cx + 13, cy + 13), outline="yellow", width=2)
        return image
