import colorsys
import importlib.resources as res
from PIL import Image

# subsec internal
from utils import get_logger


# section struc
logger = get_logger(root_name='GUIEvalKit',
                    name='Profiler',
                    level='INFO')
PROFILER_BASE = res.files('profiler')
RESOURCE_BASE = (PROFILER_BASE / 'rscs')


def rgb_to_hsv(rgb):
    """将RGB颜色转换为HSV"""
    r, g, b = [x / 255.0 for x in rgb]
    return colorsys.rgb_to_hsv(r, g, b)


def get_contrast_ratio(color1, color2):
    """
    计算两种颜色之间的对比度比率
    基于WCAG 2.0标准
    """
    def get_luminance(color):
        r, g, b = [x / 255.0 for x in color]
        # 转换为相对亮度
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    l1 = get_luminance(color1)
    l2 = get_luminance(color2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)


def get_most_visible_color(bg_color):
    """
    获取与背景对比度最高的颜色
    """
    if not bg_color:
        return (255, 0, 0)  # 默认返回红色

    # 候选的显眼颜色
    candidate_colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红色
        (0, 255, 255),  # 青色
        # (255, 255, 255), # 白色
        (255, 165, 0),  # 橙色
        (128, 0, 128)   # 紫色
    ]
    candidate_colors_preference = {
        (255, 0, 0): 1.0,
        (255, 165, 0): 0.9,
        (255, 255, 0): 0.8,
        (255, 0, 255): 0.75,
        (0, 255, 255): 0.6,
        (0, 0, 255): 0.5,
    }

    # 计算每个候选颜色与背景的对比度
    contrast_ratios = []
    for color in candidate_colors:
        ratio = get_contrast_ratio(bg_color, color)
        final_score = (ratio - 1) / 10 + candidate_colors_preference.get(color, 0.4)
        contrast_ratios.append((color, final_score))

    # 按对比度排序，选择最高的
    contrast_ratios.sort(key=lambda x: x[1], reverse=True)

    return contrast_ratios[0][0]


def fetch_background_color(image: Image.Image, bb: tuple[tuple[int, int], tuple[int, int]]):
    region = image.crop([*bb[0], *bb[1]])
    pixels = list(region.getdata())

    bg_r = sum(p[0] for p in pixels) // len(pixels)
    bg_g = sum(p[1] for p in pixels) // len(pixels)
    bg_b = sum(p[2] for p in pixels) // len(pixels)

    return (bg_r, bg_g, bg_b)
