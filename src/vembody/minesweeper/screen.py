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
OPEN = (255, 255, 255)
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


class _Canvas:
    """Draw in base-size coordinates onto an image enlarged by an integer scale.

    Shapes and text are drawn at full resolution, so a larger window stays sharp.
    At scale 1 every call matches drawing on the image directly.
    """

    def __init__(self, image: Image.Image, scale: int, font: ImageFont.FreeTypeFont):
        self.draw = ImageDraw.Draw(image)
        self.s = scale
        self.font = font

    def _box(self, box):
        x0, y0, x1, y1 = box
        s = self.s
        return (x0 * s, y0 * s, (x1 + 1) * s - 1, (y1 + 1) * s - 1)

    def _points(self, points):
        flat = [v for p in points for v in p] if isinstance(points[0], tuple) else points
        return [v * self.s + self.s // 2 for v in flat]

    def rectangle(self, box, fill=None, outline=None, width=1):
        self.draw.rectangle(self._box(box), fill=fill, outline=outline, width=width * self.s)

    def ellipse(self, box, fill=None, outline=None, width=1):
        self.draw.ellipse(self._box(box), fill=fill, outline=outline, width=width * self.s)

    def arc(self, box, start, end, fill=None, width=1):
        self.draw.arc(self._box(box), start, end, fill=fill, width=width * self.s)

    def line(self, xy, fill=None, width=1):
        self.draw.line(self._points(xy), fill=fill, width=width * self.s)

    def polygon(self, points, fill=None):
        self.draw.polygon(self._points(points), fill=fill)

    def text(self, xy, text, fill=None, anchor=None):
        x, y = xy
        self.draw.text((x * self.s, y * self.s), text, fill=fill, font=self.font, anchor=anchor)

    def textbbox(self, xy, text, anchor=None):
        x, y = xy
        box = self.draw.textbbox((x * self.s, y * self.s), text, font=self.font, anchor=anchor)
        return tuple(v // self.s for v in box)


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
        scale: int = 3,
    ):
        if type(scale) is not int or not 1 <= scale <= 8:
            raise ValueError("scale must be an integer from 1 to 8")
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.seed = seed
        self.benchmark = benchmark
        self.grid_labels = grid_labels
        self.highlight = highlight
        self.scale = scale
        self.font = ImageFont.load_default(size=20 * scale)
        self.game = Minesweeper(rows, cols, mines, seed)

        # Layout at scale 1; drawing uses these and the canvas enlarges them
        self._left = MARGIN + (24 if benchmark else 0)
        self._top = HEADER + (24 if benchmark else 0)
        self._width = self._left + MARGIN + cols * CELL
        self._height = self._top + MARGIN + rows * CELL
        # Face button that restarts the game, centered in the header
        left = (self._width - FACE) // 2
        top = (HEADER - FACE) // 2
        self._face = (left, top, left + FACE - 1, top + FACE - 1)

        # Public geometry is in screenshot pixels
        self.cell = CELL * scale
        self.grid_left = self._left * scale
        self.grid_top = self._top * scale
        self.width = self._width * scale
        self.height = self._height * scale
        self.face = (
            left * scale,
            top * scale,
            (left + FACE) * scale - 1,
            (top + FACE) * scale - 1,
        )

    def locate(self, px: int, py: int) -> tuple[str, tuple[int, int] | None]:
        """Identify a permitted cell, human-play reset button, or off-target pixel."""
        cell = self.cell_at(px, py)
        if cell is not None:
            return "cell", cell

        left, top, right, bottom = self.face
        if not self.benchmark and left <= px <= right and top <= py <= bottom:
            return "face", None

        return "none", None

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

        target, cell = self.locate(px, py)
        if target == "face" and button == "left":
            self.game = Minesweeper(self.rows, self.cols, self.mines, self.seed)
            return

        # Clicks outside the grid do nothing
        if target != "cell":
            return
        if button == "left":
            self.game.reveal(cell)
        else:
            self.game.toggle_flag(cell)

    def _center(self, cell: tuple[int, int]) -> tuple[int, int]:
        row, col = cell
        return self._left + col * CELL + CELL // 2, self._top + row * CELL + CELL // 2

    def cell_center(self, cell: tuple[int, int]) -> tuple[int, int]:
        x, y = self._center(cell)
        return x * self.scale, y * self.scale

    def cell_at(self, px: int, py: int) -> tuple[int, int] | None:
        x, y = px // self.scale - self._left, py // self.scale - self._top
        if not (0 <= x < self.cols * CELL and 0 <= y < self.rows * CELL):
            return None
        if self.benchmark and not (2 <= x % CELL < CELL - 2 and 2 <= y % CELL < CELL - 2):
            return None
        return y // CELL, x // CELL

    def draw_raised(self, draw: _Canvas, box: tuple[int, int, int, int]) -> None:
        left, top, right, bottom = box
        draw.rectangle(box, fill=BACKGROUND, outline=LINE)
        draw.line((left + 1, top + 1, right - 1, top + 1), fill="white", width=2)
        draw.line((left + 1, top + 1, left + 1, bottom - 1), fill="white", width=2)
        draw.line((left + 2, bottom - 1, right - 1, bottom - 1), fill=LINE, width=2)
        draw.line((right - 1, top + 2, right - 1, bottom - 1), fill=LINE, width=2)

    def screenshot(self) -> Image.Image:
        image = Image.new("RGB", (self.width, self.height), BACKGROUND)
        draw = _Canvas(image, self.scale, self.font)
        game = self.game

        # Mines left counter, digital-display style
        text = f"{self.mines - len(game.flagged):03d}"
        x, y = MARGIN, HEADER // 2
        box = draw.textbbox((x, y), text, anchor="lm")
        draw.rectangle((box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4), fill="black")
        draw.text((x, y), text, fill="red", anchor="lm")

        # Face: smile while playing, sunglasses on a win, X eyes on a loss
        left, top, right, bottom = self._face
        self.draw_raised(draw, self._face)
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
            left = self._left + cell[1] * CELL
            top = self._top + cell[0] * CELL
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
                    draw.text((cx, cy), str(count), fill=color, anchor="mm")
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
                _, cy = self._center((row, 0))
                draw.text(
                    (self._left - 14, cy),
                    chr(65 + row),
                    fill="black",
                    anchor="mm",
                )
            for col in range(self.cols):
                cx, _ = self._center((0, col))
                draw.text(
                    (cx, self._top - 14),
                    str(col + 1),
                    fill="black",
                    anchor="mm",
                )
        for row, col in self.highlight:
            cx, cy = self._center((row, col))
            draw.rectangle((cx - 14, cy - 14, cx + 13, cy + 13), outline="yellow", width=2)
        return image
