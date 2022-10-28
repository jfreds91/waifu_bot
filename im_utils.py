from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import os

def shadow_text(draw, text, font_ttf, fontsize, x, y, offset=3):
    # adds text in both black and white to add a shadow
    # modifies draw object inplace
    font = ImageFont.truetype(font_ttf, fontsize)
    draw.text((x+offset, y+offset),text,(0,0,0),font=font)
    draw.text((x, y),text,(255,255,255),font=font)

def apply_watermark(im, seed):
    im = ImageEnhance.Color(im).enhance(1.4)
    draw = ImageDraw.Draw(im)
    
    # add title
    x_title = 20
    y_title = 450
    fontsize_title = 50
    font_title = os.path.join(os.path.dirname(os.path.abspath(__file__)),'josephsophia/josephsophia.ttf')

    if seed:
        shadow_text(draw, seed, font_title, fontsize_title, x_title, y_title, 2)
    return im