import math
import itertools
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Literal, overload

# subsec internal
from guieval.utils.action_space import PREDICTION, UNIFIED_ACTION
from utils.utils import UTIL_BASE
from profiler.utils import fetch_background_color, get_most_visible_color


# section main
class Visualizer:
    @staticmethod
    def append_blank_top(screen: Image.Image,
                         blank_height: int, *,
                         fill_color: str = 'white',
                         split_line: bool = False) -> Image.Image:
        """
        Append a blank field on the top of the image.

        Args:
            screen: The original image
            blank_height: Height of the blank field to add at the top
            fill_color: Color of the blank field (default: 'white')

        Returns:
            A new image with blank space at the top
        """
        # Create a new image with increased height
        new_height = screen.size[1] + blank_height
        new_image = Image.new(screen.mode, (screen.size[0], new_height), fill_color)

        # Paste the original image at the bottom
        new_image.paste(screen, (0, blank_height))

        # Add split line
        if split_line:
            draw = ImageDraw.Draw(new_image)
            draw.line((0, blank_height, new_image.size[0], int(blank_height)), fill='black', width=3)

        return new_image

    @staticmethod
    def click(screen: Image.Image, prediction: PREDICTION, *,
              dot_ratio: float = 0.03, **kwargs):
        rela_x, rela_y = prediction['POINT'][0] / 1000, prediction['POINT'][1] / 1000
        x, y = screen.size[0] * rela_x, screen.size[1] * rela_y

        diagonal = np.linalg.norm(screen.size)
        dot_size = diagonal * dot_ratio

        draw = ImageDraw.Draw(screen)
        bb = ((x - dot_size / 2, y - dot_size / 2), (x + dot_size / 2, y + dot_size / 2))
        bg_color = fetch_background_color(screen, bb)
        fill_color = get_most_visible_color(bg_color=bg_color)
        draw.arc((x - dot_size / 4, y - dot_size / 4, x + dot_size / 4, y + dot_size / 4),
                 start=0, end=360,
                 fill=fill_color,
                 width=int(dot_size / 10))
        circ_cut = 2 ** 0.5 / 2 * dot_size / 4
        draw.line((x - dot_size / 2, y - dot_size / 2, x - circ_cut, y - circ_cut),
                  fill=fill_color, width=int(dot_size / 14))
        draw.line((x + dot_size / 2, y + dot_size / 2, x + circ_cut, y + circ_cut),
                  fill=fill_color, width=int(dot_size / 14))
        draw.line((x - dot_size / 2, y + dot_size / 2, x - circ_cut, y + circ_cut),
                  fill=fill_color, width=int(dot_size / 14))
        draw.line((x + dot_size / 2, y - dot_size / 2, x + circ_cut, y - circ_cut),
                  fill=fill_color, width=int(dot_size / 14))

        return screen

    @staticmethod
    def type_text(screen: Image.Image, prediction: PREDICTION, *,
             box_ratio: float = 0.03, **kwargs):
        text = prediction['TYPE']

        diagonal = np.linalg.norm(screen.size)

        text_box_hight = diagonal * box_ratio
        font = ImageFont.truetype((UTIL_BASE / 'fonts' / 'msyh.ttc'), size=int(text_box_hight * 0.7))

        text_width = font.getlength(text)
        line_num = math.ceil(text_width / screen.size[0])
        text_batch_size = int(len(text) / line_num)
        text_lines = 'TYPE:' + '\n' + '\n'.join(
            ''.join(text_batch)
            for text_batch in itertools.batched(text, text_batch_size))

        appended_screen = Visualizer.append_blank_top(screen,
                                               math.ceil(text_box_hight) * (line_num + 1),
                                               split_line=True)
        draw = ImageDraw.Draw(appended_screen)

        draw.rectangle((0, 0, screen.size[0], text_box_hight), fill='white')
        draw.text((0, 0), text_lines, fill='red', font=font)

        return appended_screen

    @staticmethod
    def scroll(screen: Image.Image, prediction: PREDICTION, *,
               arrow_ratio: float = 0.7, **kwargs):
        direction = prediction['to']

        diagonal = np.linalg.norm(screen.size)
        arrow_length = min(diagonal * arrow_ratio, *screen.size) / 2

        delta = ((0, -arrow_length)
                 if direction == 'down' else
                 (0, arrow_length)
                 if direction == 'up' else
                 (arrow_length, 0)
                 if direction == 'right' else
                 (-arrow_length, 0)
                 if direction == 'left' else
                 None)

        start = (screen.size[0] * 0.5, screen.size[1] * 0.5)
        end = (start[0] + delta[0], start[1] + delta[1])

        arrow_head_depth = arrow_length / 10
        arrow_head_width = arrow_head_depth * (2 ** 0.5)
        arrow_head_pivot0 = (start[0] + delta[0] * 0.9, start[1] + delta[1] * 0.9)
        if direction in ('down', 'up'):
            arrow_head_pivot1 = (arrow_head_pivot0[0] + arrow_head_width / 2, arrow_head_pivot0[1])
            arrow_head_pivot2 = (arrow_head_pivot0[0] - arrow_head_width / 2, arrow_head_pivot0[1])
        else:
            arrow_head_pivot1 = (arrow_head_pivot0[0], arrow_head_pivot0[1] - arrow_head_width / 2)
            arrow_head_pivot2 = (arrow_head_pivot0[0], arrow_head_pivot0[1] + arrow_head_width / 2)

        x_min = min([start[0], end[0], arrow_head_pivot1[0], arrow_head_pivot2[0]])
        x_max = max([start[0], end[0], arrow_head_pivot1[0], arrow_head_pivot2[0]])
        y_min = min([start[1], end[1], arrow_head_pivot1[1], arrow_head_pivot2[1]])
        y_max = max([start[1], end[1], arrow_head_pivot1[1], arrow_head_pivot2[1]])

        bb = ((x_min, y_min), (x_max, y_max))
        bg_color = fetch_background_color(screen, bb)
        fill_color = get_most_visible_color(bg_color=bg_color)

        draw = ImageDraw.Draw(screen)
        draw.line((start, end), fill=fill_color, width=int(arrow_length / 28))
        draw.polygon((end, arrow_head_pivot1, arrow_head_pivot2), fill=fill_color)

        return screen

    @staticmethod
    def press(screen: Image.Image, prediction: PREDICTION, *,
              box_ratio: float = 0.03, **kwargs):
        button = prediction['PRESS']

        diagonal = np.linalg.norm(screen.size)

        text_box_hight = diagonal * box_ratio
        font = ImageFont.truetype((UTIL_BASE / 'fonts' / 'msyh.ttc'), size=int(text_box_hight * 0.7))

        appended_screen = Visualizer.append_blank_top(screen, 1, split_line=True)
        draw = ImageDraw.Draw(appended_screen)

        draw.rectangle((0, 0, screen.size[0], text_box_hight), fill='white')
        draw.text((0, 0), f'PRESS: {button}', fill='red', font=font)

        return appended_screen

    @staticmethod
    def stop(screen: Image.Image, prediction: PREDICTION, *,
             icon_ratio: float = 0.14, **kwargs):
        diagonal = np.linalg.norm(screen.size)
        icon_size = diagonal * icon_ratio

        center = (screen.size[0] * 0.5, screen.size[1] * 0.5)
        left_top = (center[0] - icon_size / 2, center[1] - icon_size / 2)
        right_bottom = (center[0] + icon_size / 2, center[1] + icon_size / 2)
        inner_left_top = (center[0] - icon_size / 2.5, center[1] - icon_size / 2.5)
        inner_right_bottom = (center[0] + icon_size / 2.5, center[1] + icon_size / 2.5)

        pivot0 = (center[0] - icon_size / 5, center[1] - icon_size / 5)
        pivot1 = (center[0] + icon_size / 5, center[1] + icon_size / 5)

        bb = (left_top, right_bottom)
        bg_color = fetch_background_color(screen, bb)
        fill_color = get_most_visible_color(bg_color=bg_color)

        draw = ImageDraw.Draw(screen)
        draw.ellipse((left_top, right_bottom), fill=fill_color)
        draw.ellipse((inner_left_top, inner_right_bottom), fill='white')
        draw.rectangle((pivot0, pivot1), fill=fill_color)

        return screen

    @staticmethod
    def long_point(screen: Image.Image, prediction: PREDICTION, *,
                   dot_ratio: float = 0.03, **kwargs):
        rela_x, rela_y = prediction['POINT'][0] / 1000, prediction['POINT'][1] / 1000
        x, y = screen.size[0] * rela_x, screen.size[1] * rela_y

        diagonal = np.linalg.norm(screen.size)
        dot_size = diagonal * dot_ratio

        bb = ((x - dot_size / 2, y - dot_size / 2), (x + dot_size / 2, y + dot_size / 2))
        bg_color = fetch_background_color(screen, bb)
        fill_color = get_most_visible_color(bg_color=bg_color)

        draw = ImageDraw.Draw(screen)
        draw.ellipse((x - dot_size / 4, y - dot_size / 4, x + dot_size / 4, y + dot_size / 4), fill=fill_color)
        draw.line((x - dot_size / 2, y - dot_size / 2, x + dot_size / 2, y + dot_size / 2),
                  fill=fill_color, width=int(dot_size / 14))
        draw.line((x - dot_size / 2, y + dot_size / 2, x + dot_size / 2, y - dot_size / 2),
                  fill=fill_color, width=int(dot_size / 14))

        return screen

    @staticmethod
    def open(screen: Image.Image, prediction: PREDICTION, *,
             box_ratio: float = 0.03, **kwargs):
        app = prediction["OPEN_APP"]

        diagonal = np.linalg.norm(screen.size)

        text_box_hight = diagonal * box_ratio
        font = ImageFont.truetype((UTIL_BASE / 'fonts' / 'msyh.ttc'), size=int(text_box_hight * 0.7))

        appended_screen = Visualizer.append_blank_top(screen, 1, split_line=True)
        draw = ImageDraw.Draw(appended_screen)

        draw.rectangle((0, 0, screen.size[0], text_box_hight), fill='white')
        draw.text((0, 0), f'OPEN: {app}', fill='red', font=font)

        return appended_screen

    @staticmethod
    def wait(screen: Image.Image, prediction: PREDICTION, *,
             icon_ratio: float = 0.14, **kwargs):
        diagonal = np.linalg.norm(screen.size)
        icon_size = diagonal * icon_ratio

        center = (screen.size[0] * 0.5, screen.size[1] * 0.5)
        left_top = (center[0] - icon_size / 2, center[1] - icon_size / 2)
        right_bottom = (center[0] + icon_size / 2, center[1] + icon_size / 2)
        inner_left_top = (center[0] - icon_size / 2.5, center[1] - icon_size / 2.5)
        inner_right_bottom = (center[0] + icon_size / 2.5, center[1] + icon_size / 2.5)

        adjusted_center = (center[0] + icon_size / 14, center[1])
        pivot0 = (adjusted_center[0] - icon_size / 5, adjusted_center[1] - icon_size / 5)
        pivot1 = (adjusted_center[0] - icon_size / 5, adjusted_center[1] + icon_size / 5)
        pivot2 = (adjusted_center[0] + (icon_size * ((5 ** 0.5 + 1) / 2 - 1) / 5), adjusted_center[1])

        bb = (left_top, right_bottom)
        bg_color = fetch_background_color(screen, bb)
        fill_color = get_most_visible_color(bg_color=bg_color)

        draw = ImageDraw.Draw(screen)
        draw.ellipse((left_top, right_bottom), fill=fill_color)
        draw.ellipse((inner_left_top, inner_right_bottom), fill='white')
        draw.polygon((pivot0, pivot1, pivot2), fill=fill_color)

        return screen

    METHOD_MAP: dict[UNIFIED_ACTION, staticmethod | classmethod] = {
        'CLICK': click,
        'LONG_POINT': long_point,
        'OPEN': open,
        'PRESS': press,
        'SCROLL': scroll,
        'STOP': stop,
        'TYPE': type_text,
        'WAIT': wait
    }

    @overload  # noqa: E302
    @classmethod
    def visualize(cls, screen: Image.Image, action: Literal['CLICK'],
                  prediction: PREDICTION, *, dot_ratio: float = 0.03) -> Image.Image | None:
        '''
        Visualize the CLICK action.
        '''
    @overload  # noqa: E302
    @classmethod
    def visualize(cls, screen: Image.Image, action: Literal['LONG_POINT'],
                  prediction: PREDICTION, *, dot_ratio: float = 0.03) -> Image.Image | None:
        '''
        Visualize the LONG_POINT action.
        '''
    @overload  # noqa: E302
    @classmethod
    def visualize(cls, screen: Image.Image, action: Literal['OPEN'],
                  prediction: PREDICTION, *, box_ratio: float = 0.03) -> Image.Image | None:
        '''
        Visualize the OPEN action.
        '''
    @overload  # noqa: E302
    @classmethod
    def visualize(cls, screen: Image.Image, action: Literal['PRESS'],
                  prediction: PREDICTION, *, box_ratio: float = 0.03) -> Image.Image | None:
        '''
        Visualize the PRESS action.
        '''
    @overload  # noqa: E302
    @classmethod
    def visualize(cls, screen: Image.Image, action: Literal['SCROLL'],
                  prediction: PREDICTION, *, arrow_ratio: float = 0.7) -> Image.Image | None:
        '''
        Visualize the SCROLL action.
        '''
    @overload  # noqa: E302
    @classmethod
    def visualize(cls, screen: Image.Image, action: Literal['STOP'],
                  prediction: PREDICTION, *, icon_ratio: float = 0.14) -> Image.Image | None:
        '''
        Visualize the STOP action.
        '''
    @overload  # noqa: E302
    @classmethod
    def visualize(cls, screen: Image.Image, action: Literal['TYPE'],
                  prediction: PREDICTION, *, box_ratio: float = 0.03) -> Image.Image | None:
        '''
        Visualize the TYPE action.
        '''
    @overload  # noqa: E302
    @classmethod
    def visualize(cls, screen: Image.Image, action: Literal['WAIT'],
                  prediction: PREDICTION, *, icon_ratio: float = 0.14) -> Image.Image | None:
        '''
        Visualize the WAIT action.
        '''
    @classmethod
    def visualize(cls,
                  screen: Image.Image,
                  action: UNIFIED_ACTION,
                  prediction: PREDICTION,
                  *,
                  dot_ratio: float = 0.03,
                  box_ratio: float = 0.03,
                  arrow_ratio: float = 0.7,
                  icon_ratio: float = 0.14) -> Image.Image | None:
        try:
            return cls.METHOD_MAP[action](screen, prediction,
                                          dot_ratio=dot_ratio,
                                          box_ratio=box_ratio,
                                          arrow_ratio=arrow_ratio,
                                          icon_ratio=icon_ratio)
        except Exception:
            return None
