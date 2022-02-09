from PIL import Image, ImageDraw, ImageTk

class CanvasUtils:

    @staticmethod
    def create_polygon(root, canvas, images, *args, **kwargs):
        if "alpha" in kwargs:         
            if "fill" in kwargs:
                # Get and process the input data
                fill = root.winfo_rgb(kwargs.pop("fill"))\
                    + (int(kwargs.pop("alpha") * 255),)
                outline = kwargs.pop("outline") if "outline" in kwargs else None

                # We need to find a rectangle the polygon is inscribed in
                # (max(args[::2]), max(args[1::2])) are x and y of the bottom right point of this rectangle
                # and they also are the width and height of it respectively (the image will be inserted into
                # (0, 0) coords for simplicity)
                image = Image.new("RGBA", (max(args[0][::2]), max(args[0][1::2])))
                ImageDraw.Draw(image).polygon(args[0], fill=fill, outline=outline)

                images.append(ImageTk.PhotoImage(image))  # prevent the Image from being garbage-collected
                return canvas.create_image(0, 0, image=images[-1], anchor="nw")  # insert the Image to the 0, 0 coords
            raise ValueError("fill color must be specified!")
        return canvas.create_polygon(*args, **kwargs)