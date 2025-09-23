
def scale(old_width,old_height, width=None, height=None):
    """指定宽或高，得到按比例缩放后的宽高

    :param filePath:图片的绝对路径
    :param width:目标宽度
    :param height:目标高度
    :return:按比例缩放后的宽和高
    """
    if not width and not height:
        width, height = old_width,old_height  # 原图片宽高
    if not width or not height:
        _width, _height = old_width,old_height
        height = width * _height // _width if width else height
        width = height * _width // _height if height else width
    return width, height


def Scaling(old_width,old_height):
    global new_width, new_height
    length = 1024
    if old_width<=length or old_height<=length:
        new_width, new_height = scale(old_width,old_height)
    elif old_width>length and old_height>length:
        if old_width>=old_height:
            new_width, new_height = scale(old_width, old_height,height=length)
        else:
            new_width, new_height = scale(old_width, old_height,width=length)

    return new_width, new_height